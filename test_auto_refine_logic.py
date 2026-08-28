#!/usr/bin/env python3
"""
test_auto_refine_logic.py — exercises the staged-refinement / bounds-gating /
checkpoint-rollback control flow in gsas2_auto_refine.py WITHOUT depending on
a real GSAS-II install.

Per the project's directive on mock data for verification: this drives
RefinementRunner against a small fake in-memory project (a handful of
synthetic Rwp values and cell parameters) rather than real GSAS-II or real
experimental data. It is not a test of the crystallography — it is a test
that the deterministic control flow (checkpoint before each stage, accept
on convergence, roll back on divergence, never crash the whole run for one
bad stage) actually behaves as designed.

Run with: python3 test_auto_refine_logic.py
No pytest / GSAS-II dependency required.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gsas2_auto_refine import (  # noqa: E402
    Bounds,
    RefinementRunner,
    Stage,
    assess_fit_quality,
    build_protocol,
    cell_drift_ok,
    export_histogram_csv,
    get_phase_cells,
    import_gsasiiscriptable,
    profile_params_sane,
    rwp_improved_or_stable,
    seed_initial_scale,
)


# ---------------------------------------------------------------------------
# Fake GSASIIscriptable-shaped project — just enough surface area for
# RefinementRunner to drive, with a scripted sequence of outcomes so each
# stage's behavior (converge / diverge / error) is controlled by the test.
# ---------------------------------------------------------------------------

class FakeHist:
    def __init__(self, project):
        self.project = project

    def set_refinements(self, d):
        pass

    def residuals(self):
        return {"wR": self.project.rwp}


class FakePhase:
    def __init__(self, name, cell):
        self.name = name
        self.cell = list(cell)

    def set_refinements(self, d):
        pass

    def set_HAP_refinements(self, d):
        pass

    def get_cell(self):
        # Matches the real GSASIIscriptable.G2Phase.get_cell() shape —
        # confirmed against actual installed source: a dict, not a list —
        # see _cell()'s docstring-comment in gsas2_auto_refine.py for the
        # bug this real shape caught.
        keys = ["length_a", "length_b", "length_c",
                "angle_alpha", "angle_beta", "angle_gamma"]
        return dict(zip(keys, self.cell)) | {"volume": 0.0}

    def export_CIF(self, path):
        Path(path).write_text(f"# fake refined CIF for {self.name}\n")


_ACTIVE_OUTCOMES = []  # shared across a test's checkpoint reloads — see set_active_outcomes()


def set_active_outcomes(outcomes):
    """
    Point every FakeProject created for the rest of this test (including
    ones reconstructed by FakeG2scModule.G2Project() on rollback) at the
    same outcomes queue, so a checkpoint reload doesn't lose the script of
    "what happens on the remaining stages" — that queue models the real
    solver's future behavior, not per-object state.
    """
    global _ACTIVE_OUTCOMES
    _ACTIVE_OUTCOMES = list(outcomes)


class FakeProject:
    """
    A tiny stand-in for GSASIIscriptable.G2Project. Each do_refinements()
    call consumes one (rwp_after, cell_after) tuple off the shared
    _ACTIVE_OUTCOMES queue — that's how a test scripts "this stage
    converges, this one diverges" without touching real refinement math,
    and it survives a checkpoint-reload swapping in a fresh FakeProject
    instance mid-run, same as a real solver's future behavior would.
    """

    def __init__(self, rwp, phase_cells):
        self.rwp = rwp
        self._phases = [FakePhase(f"phase{i}", c) for i, c in enumerate(phase_cells)]
        self._hists = [FakeHist(self)]
        self.saved_to = []
        self.raise_on_next_refine = False
        # Mirrors the real G2Project.filename: whatever path save() was
        # last called with. Real GSASIIscriptable's do_refinements() ->
        # .refine() calls G2strMain.Refine(self.filename, ...), which
        # reads from AND writes back to this exact path as a side effect
        # of refining — see do_refinements() below, which simulates that.
        # This is what let a real bug slip past this whole test suite for
        # a while: without simulating it, "save a checkpoint, then refine"
        # looked perfectly safe here even though the real refine call was
        # silently overwriting the checkpoint with post-refinement state.
        self.filename = None

    # -- G2Project-shaped API used by RefinementRunner --
    def histogram(self, i):
        return self._hists[i]

    def histograms(self):
        return self._hists

    def phase(self, i):
        return self._phases[i]

    def phases(self):
        return self._phases

    def save(self, path):
        self.saved_to.append(path)
        self.filename = path
        self._write_state_to(path)

    def _write_state_to(self, path):
        state = {
            "rwp": self.rwp,
            "phases": [{"name": p.name, "cell": p.cell} for p in self._phases],
        }
        Path(path).write_text(json.dumps(state))

    def do_refinements(self, reflist):
        if self.raise_on_next_refine:
            raise RuntimeError("simulated solver blow-up")
        rwp_after, cell_after = _ACTIVE_OUTCOMES.pop(0)
        self.rwp = rwp_after
        if cell_after is not None:
            for p, c in zip(self._phases, cell_after):
                p.cell = list(c)
        # Simulate GSAS-II's real refine-time auto-save: it writes the
        # just-mutated (post-refinement) state back to whatever file is
        # currently self.filename. If a caller saved a checkpoint and then
        # refined without re-pointing self.filename elsewhere first, that
        # checkpoint gets silently overwritten right here — exactly the
        # bug this simulation exists to catch.
        if self.filename is not None:
            self._write_state_to(self.filename)


class ProfileFakeHist(FakeHist):
    """
    Extends FakeHist with the 'Instrument Parameters' shape
    RefinementRunner._profile_values() reads (a [dict, dict] pair, values
    at index 1) plus a .name — only used by
    test_runner_rolls_back_diverged_profile_params(), so the plain
    FakeHist used everywhere else stays minimal.
    """

    def __init__(self, project, name="PWDR fake histogram"):
        super().__init__(project)
        self.name = name
        self.data = {"Instrument Parameters": [
            {"U": [0.0, 0.0, True], "V": [0.0, 0.0, True], "W": [200.0, 200.0, True]},
            {},
        ]}


class ProfileFakeProject(FakeProject):
    """
    Extends FakeProject to also model Instrument Parameters U so a test can
    exercise profile_params_sane()/_profile_values() end to end — coverage
    for the real bug where U/V/W diverged to the millions while Rwp barely
    moved and the cell stayed put, which neither rwp_improved_or_stable()
    nor cell_drift_ok() could ever catch.
    """

    def __init__(self, rwp, phase_cells, u_values):
        super().__init__(rwp, phase_cells)
        self._hists = [ProfileFakeHist(self)]
        self._u_values = list(u_values)  # one consumed per do_refinements() call

    def do_refinements(self, reflist):
        super().do_refinements(reflist)
        if self._u_values:
            self.histogram(0).data["Instrument Parameters"][0]["U"][1] = self._u_values.pop(0)


class MustrainFallbackFakePhase(FakePhase):
    """Extends FakePhase to record what set_HAP_refinements() was last
    asked for, and to expose a 'Mustrain' values list at the real shape
    RefinementRunner._profile_values() reads — lets a test drive different
    do_refinements() outcomes depending on which Stage variant (primary vs.
    fallback) was applied, exercising the fallback ladder end to end
    rather than just the bounds check in isolation."""

    def __init__(self, name, cell, hist_name):
        super().__init__(name, cell)
        self.last_hap = {}
        self.data = {"Histograms": {hist_name: {"Mustrain": ["uniaxial", [1.0, 1.0]]}}}

    def set_HAP_refinements(self, d):
        self.last_hap = d


class MustrainFallbackFakeProject(FakeProject):
    """
    Regression coverage for the real bug found on real two-phase lab data
    (Data/MgO+MgBC): joint uniaxial-Mustrain + isotropic-Size refinement
    can be ~99% correlated — confirmed on real data, Mustrain ran away to
    -159,724 while Size collapsed to 0.001 and Rwp barely moved (27.2% ->
    27.2%), a bounds failure that reverting-and-moving-on alone would just
    waste the whole stage on. do_refinements() here diverges Mustrain
    whenever the *primary* config (uniaxial Mustrain) is applied — modeling
    that same real failure — so a test can confirm RefinementRunner's
    fallback ladder (see build_protocol()'s profile_microstrain_size
    fallbacks) actually falls through to a simpler model instead of just
    giving up.
    """

    def __init__(self, rwp, phase_cells):
        super().__init__(rwp, phase_cells)
        self._hists = [ProfileFakeHist(self, name="fake hist")]
        self._phases = [MustrainFallbackFakePhase(f"phase{i}", c, "fake hist")
                         for i, c in enumerate(phase_cells)]

    def do_refinements(self, reflist):
        hap = self._phases[0].last_hap
        mustrain_req = hap.get("Mustrain")
        mustrain_data = self._phases[0].data["Histograms"]["fake hist"]["Mustrain"]
        if mustrain_req and mustrain_req.get("type") == "uniaxial":
            self.rwp -= 0.001              # barely moves, just like on real data
            mustrain_data[1] = [999999.0, 1.0]
        else:
            self.rwp -= 2.0                # a real fallback genuinely converges
            mustrain_data[1] = [50.0, 1.0]
        if self.filename is not None:
            self._write_state_to(self.filename)


class AtomsFallbackFakePhase(FakePhase):
    """Extends FakePhase with a 'Mustrain' data shape (see
    MustrainFallbackFakePhase) plus a real clear_HAP_refinements() —
    tracks whether Mustrain was actually frozen, since the plain
    FakePhase stub used everywhere else doesn't define this method at
    all (confirmed: build_protocol()'s atoms-stage fallbacks are the
    first place this project calls clear_HAP_refinements(), so nothing
    would have caught a typo/wrong-method-name bug there without a fake
    that actually implements it — the same class of gap that let the
    --refine-atoms "Atoms": "all" bug and the --lebail HAP-vs-phase-key
    bug both ship unnoticed earlier this project)."""

    def __init__(self, name, cell, hist_name):
        super().__init__(name, cell)
        self.mustrain_cleared = False
        self.data = {"Histograms": {hist_name: {"Mustrain": ["uniaxial", [1.0, 1.0]]}}}

    def clear_HAP_refinements(self, refs):
        if "Mustrain" in refs:
            self.mustrain_cleared = True

    def set_HAP_refinements(self, d):
        pass


class AtomsFallbackFakeProject(FakeProject):
    """
    Regression coverage for the real bug found on real data (both
    Data/FeF3 and Data/MgO+MgBC): refining atom positions/Uiso while
    Mustrain is still free (every stage in this protocol is cumulative —
    see Stage.clear_hap's docstring) is correlated enough to send
    Mustrain to a runaway value (107,466 seen on real MgO+MgBC data)
    even when the underlying fit is genuinely, substantially better
    (Rwp 12.4% -> 9.8% on that same run) — a bounds-check failure that
    reverting-and-moving-on alone would just throw away a real
    improvement on. do_refinements() here models exactly that: the fit
    always improves, but Mustrain only stays sane if clear_hap actually
    froze it first.
    """

    def __init__(self, rwp, phase_cells):
        super().__init__(rwp, phase_cells)
        self._hists = [ProfileFakeHist(self, name="fake hist")]
        self._phases = [AtomsFallbackFakePhase(f"phase{i}", c, "fake hist")
                         for i, c in enumerate(phase_cells)]

    def do_refinements(self, reflist):
        mustrain_data = self._phases[0].data["Histograms"]["fake hist"]["Mustrain"]
        self.rwp -= 5.0  # the fit really is better either way...
        if self._phases[0].mustrain_cleared:
            mustrain_data[1] = [1.0, 1.0]        # ...and Mustrain stays sane
        else:
            mustrain_data[1] = [107466.0, 1.0]   # ...but Mustrain still blows up
        if self.filename is not None:
            self._write_state_to(self.filename)


class PrefOriFakePhase(FakePhase):
    """Extends FakePhase with the 'Pref.Ori.' shape
    RefinementRunner._attempt_variant() pokes the axis into directly
    (GSASIIscriptable's set_HAP_refinements() only exposes the refine
    flag for this key, not the axis — see Stage.pref_ori_axis)."""

    def __init__(self, name, cell, hist_name):
        super().__init__(name, cell)
        self.data = {"Histograms": {hist_name: {"Pref.Ori.": ["MD", 1.0, False, [0, 0, 1]]}}}


class PrefOriFakeProject(FakeProject):
    """
    Regression coverage for a real design gap: adding a
    preferred-orientation refinement essentially never fails bounds even
    for the *wrong* axis — the March-Dollase ratio just settles back
    toward 1.0 (no texture), which reads as "didn't get worse," not as
    "diverged." A plain non-worsening bar would therefore accept
    whichever axis is tried first regardless of whether it's the one
    that actually explains the data (see Bounds.min_optional_improvement_
    frac). Models a sample whose real texture only shows up along axis
    (1,1,0): every other axis barely moves Rwp (a harmless no-op), and
    only (1,1,0) produces a genuine improvement.
    """

    def __init__(self, rwp, phase_cells, hist_name="fake hist"):
        super().__init__(rwp, phase_cells)
        self._hists = [ProfileFakeHist(self, name=hist_name)]
        self._phases = [PrefOriFakePhase(f"phase{i}", c, hist_name)
                         for i, c in enumerate(phase_cells)]

    def do_refinements(self, reflist):
        hist_name = self._hists[0].name
        axis = tuple(self._phases[0].data["Histograms"][hist_name]["Pref.Ori."][3])
        if axis == (1, 1, 0):
            self.rwp -= 3.0       # the real texture axis: genuine improvement
        else:
            self.rwp -= 0.0001    # no real effect, same as a no-op axis
        if self.filename is not None:
            self._write_state_to(self.filename)


_ACTIVE_PROJECT_CLASS = FakeProject


def set_active_project_class(cls):
    """
    Points every FakeProject reconstructed by _reload() (via
    FakeG2scModule.G2Project(), same reload path a real checkpoint
    rollback or fallback-ladder retry takes) at `cls` instead of the
    plain FakeProject, for the rest of the current test. Needed by any
    test whose fake project models more than rwp/cell (e.g.
    MustrainFallbackFakeProject, PrefOriFakeProject) AND whose scenario
    needs that modeling to survive past the *first* reload — a run_stage()
    fallback ladder reloads the pre-stage checkpoint before every
    fallback attempt, not just once, so without this a test could only
    ever exercise a fallback's outcome via the generic
    _ACTIVE_OUTCOMES-driven plain FakeProject, never the richer modeling
    a specific real bug needs. Reset to FakeProject at the top of any
    test that doesn't need it (see test functions' setup).
    """
    global _ACTIVE_PROJECT_CLASS
    _ACTIVE_PROJECT_CLASS = cls


class FakeG2scModule:
    """Stands in for `import GSASIIscriptable as G2sc` inside _reload()."""

    @staticmethod
    def G2Project(path):
        state = json.loads(Path(path).read_text())
        cells = [tuple(p["cell"]) for p in state["phases"]]
        proj = _ACTIVE_PROJECT_CLASS(state["rwp"], cells)
        # Matches real G2Project(gpxfile) behavior: filename starts out
        # pointing at whatever file it was just loaded from.
        proj.filename = path
        return proj


sys.modules["GSASIIscriptable"] = FakeG2scModule()


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise SystemExit(1)


def test_bounds_helpers():
    b = Bounds(max_cell_drift_frac=0.15, rwp_worsen_tol_frac=0.02)

    check("cell within bounds accepted",
          cell_drift_ok((5.0, 5.0, 5.0, 90, 90, 90), (5.2, 5.1, 4.9, 90, 90, 90), b))
    check("cell drift beyond bound rejected",
          not cell_drift_ok((5.0, 5.0, 5.0, 90, 90, 90), (6.5, 5.0, 5.0, 90, 90, 90), b))

    check("rwp improvement accepted", rwp_improved_or_stable(20.0, 15.0, b))
    check("rwp small worsening within tolerance accepted", rwp_improved_or_stable(20.0, 20.3, b))
    check("rwp large worsening rejected", not rwp_improved_or_stable(20.0, 30.0, b))
    check("NaN rwp_after rejected", not rwp_improved_or_stable(20.0, float("nan"), b))
    check("None rwp_before (no baseline) accepted", rwp_improved_or_stable(None, 15.0, b))
    # Regression: rwp_before comes back NaN (not None) on the very first
    # stage of a real run, since GSAS-II hasn't computed a weighted
    # residual until at least one do_refinements() call has happened.
    # Treating that NaN as "worse than before" (NaN comparisons are always
    # False in Python) silently failed stage 1 on every single run —
    # including a run that produced a perfectly good Rwp — and rolled back
    # the phase's scale-factor refinement before it ever took effect. That
    # is exactly what produced the "no peaks in Calculated" bug reported
    # against real data: y_calc topped out at ~18% of y_obs's max because
    # Scale never actually got fit.
    check("NaN rwp_before (no baseline yet) accepted, same as None",
          rwp_improved_or_stable(float("nan"), 15.0, b))

    # Regression: U/V/W and Mustrain diverging to unphysical magnitudes
    # while Rwp barely moves and the cell stays put — confirmed on real
    # data (U: 0.0 -> 2,260,583.58; Mustrain -> 72,563.8) — was invisible
    # to both checks above. profile_params_sane() is the dedicated bounds
    # check for exactly this parameter family.
    check("profile values within bounds accepted",
          profile_params_sane([0.0, 12.5, -3.0, 200.0], b))
    check("one profile value beyond bounds rejected",
          not profile_params_sane([0.0, 12.5, 2_260_583.58], b))
    check("None entries skipped, not treated as failures",
          profile_params_sane([0.0, None, 5.0], b))
    check("NaN entries skipped, not treated as failures",
          profile_params_sane([0.0, float("nan"), 5.0], b))
    check("empty values list trivially accepted",
          profile_params_sane([], b))


def test_seed_initial_scale():
    """
    Regression test for the real bug found on real data (Data/FeF3):
    GSAS-II's raw default phase Scale (1.0) produced calculated peaks
    topping out at ~600 counts against a real pattern whose peaks
    reached ~80,000 — a ~130x magnitude mismatch that starved the very
    first refinement cycle's gradient and helped send Scale to 1e-12
    instead of up toward a real value. seed_initial_scale() should
    rescale a mismatch like that before any real stage runs, but leave a
    Scale that's already roughly in the right ballpark alone (that's
    what the real Scale refinement stage is for).
    """
    import numpy as np

    class FakeHistForScale:
        def __init__(self, yobs, ycalc, ybkg):
            self.name = "fake hist"
            self.data = {"data": [None, [None, yobs, None, ycalc, ybkg, None]]}

    class FakePhaseForScale:
        def __init__(self, scale):
            self.data = {"Histograms": {"fake hist": {"Scale": [scale, True]}}}

    class FakeGpxForScale:
        def __init__(self, phases):
            self._phases = phases

        def do_refinements(self, reflist):
            pass  # the fake hist's arrays are already "as if just calculated"

        def phases(self):
            return self._phases

    yobs = np.array([50.0, 79950.0 + 50.0, 50.0])
    ycalc = np.array([0.0, 550.0, 0.0])
    ybkg = np.array([50.0, 50.0, 50.0])
    phase = FakePhaseForScale(scale=1.0)
    logs = []
    seed_initial_scale(FakeGpxForScale([phase]), FakeHistForScale(yobs, ycalc, ybkg), logs.append)
    check("scale rescaled up toward the real magnitude",
          phase.data["Histograms"]["fake hist"]["Scale"][0] > 50)
    check("rescale logged", any("scale-seed" in m for m in logs))

    yobs2 = np.array([50.0, 1050.0, 50.0])
    ycalc2 = np.array([0.0, 900.0, 0.0])
    ybkg2 = np.array([50.0, 50.0, 50.0])
    phase2 = FakePhaseForScale(scale=1.0)
    logs2 = []
    seed_initial_scale(FakeGpxForScale([phase2]), FakeHistForScale(yobs2, ycalc2, ybkg2), logs2.append)
    check("close-enough scale left untouched",
          phase2.data["Histograms"]["fake hist"]["Scale"][0] == 1.0)
    check("no rescale logged for close-enough magnitude", logs2 == [])


def test_assess_fit_quality():
    """
    Regression test for the real bug found on real data (Data/FeF3,
    before its instrument/CIF mismatch was found): Rwp sat at a
    plausible-looking ~10.6% for five straight stages while the
    calculated pattern's correlation with the observed one was ~0.02 —
    indistinguishable from no relationship. Every other bounds check in
    this module can look fine in exactly that situation; this is the one
    that catches it directly.
    """
    import numpy as np

    class FakeHistForQuality:
        def __init__(self, x, yobs, ycalc, ybkg, limits=None):
            self.data = {"data": [None, [x, yobs, None, ycalc, ybkg, None]]}
            if limits is not None:
                self.data["Limits"] = limits

    x = np.linspace(0, 10, 50)
    yobs_peaked = 1000 + 500 * np.exp(-((x - 5) ** 2) / 0.5)
    ycalc_flat = np.full(50, 1000.0)
    ybkg = np.full(50, 1000.0)
    bad = assess_fit_quality(FakeHistForQuality(x, yobs_peaked, ycalc_flat, ybkg))
    check("a flat calculated pattern against a peaked real one is flagged for review",
          bad["needs_review"])

    ycalc_matching = yobs_peaked + np.random.RandomState(0).normal(0, 5, 50)
    good = assess_fit_quality(FakeHistForQuality(x, yobs_peaked, ycalc_matching, ybkg))
    check("a calculated pattern that tracks the real one is not flagged",
          not good["needs_review"] and good["calc_obs_correlation"] > 0.9)

    # Regression: with --tmin/--tmax trimming, GSAS-II still reports
    # y_calc/y_bkg across the *entire* raw pattern, but leaves everything
    # outside the fit range untouched (never refined) rather than
    # matching the data there — confirmed on real data: correlation
    # dropped from 0.94 (inside the actual fit range) to 0.375 (whole
    # pattern) purely because of points GSAS-II was never asked to fit,
    # wrongly flagging a genuinely good fit for review. Model that
    # directly: a good match everywhere inside [3, 8], and a calculated
    # pattern that's completely wrong (flat at 0) outside it.
    outside_bad_ycalc = np.where((x >= 3) & (x <= 8), ycalc_matching, 0.0)
    trimmed = assess_fit_quality(FakeHistForQuality(
        x, yobs_peaked, outside_bad_ycalc, ybkg, limits=[[0, 10], [3, 8]]))
    check("points outside --tmin/--tmax are excluded from the fit-quality check",
          not trimmed["needs_review"] and trimmed["calc_obs_correlation"] > 0.9)


def test_protocol_shape():
    stages = build_protocol(refine_atoms=False)
    check("8 stages without atom refinement", len(stages) == 8)
    # peak_asymmetry, extinction, and preferred_orientation are optional
    # even without --refine-atoms — see their docstrings in
    # build_protocol(): many phases genuinely show none of these effects,
    # and that's not a run failure.
    check("only peak_asymmetry, extinction, and preferred_orientation are optional",
          [s.name for s in stages if s.optional]
          == ["peak_asymmetry", "extinction", "preferred_orientation"])

    stages_atoms = build_protocol(refine_atoms=True)
    check("9 stages with atom refinement", len(stages_atoms) == 9)
    check("atoms stage is marked optional", stages_atoms[-1].optional and stages_atoms[-1].name == "atoms")
    # Regression: G2Phase.set_refinements()'s "Atoms" handler does
    # `value.items()` on whatever's passed here — it needs a dict of
    # {atom_label: refinement_flags}, not a bare string. Passing "all"
    # directly raised AttributeError("'str' object has no attribute
    # 'items'") on every real --refine-atoms run (confirmed against the
    # installed GSASIIscriptable.py source and on real data) — silently
    # swallowed as a normal optional-stage failure, so --refine-atoms
    # could never actually refine anything and nothing said so loudly.
    # The FakePhase.set_refinements() stub used everywhere else in this
    # test suite is a no-op that accepts any shape, so only an explicit
    # check like this one catches a bug like this.
    atoms_value = stages_atoms[-1].set_phase.get("Atoms")
    check("atoms stage passes a dict, not a bare string, for 'Atoms'",
          isinstance(atoms_value, dict))
    check("atoms stage's dict uses valid refinement flag characters (only F/X/U)",
          isinstance(atoms_value, dict)
          and all(c in " FXU" for v in atoms_value.values() for c in v))

    # Regression: refining both the histogram's own "Sample Parameters:
    # Scale" and the phase's HAP "Scale" (phase fraction) at the same time
    # is mathematically degenerate for a single-phase-in-one-histogram fit
    # — GSAS-II's own solver confirmed this on real data (SVD singularity,
    # "0:0:Scale and :0:Scale (@100.00%)" correlated), and it's why the
    # phase's Scale factor never actually moved off its untouched default
    # even though Rwp looked fine. sample_displacement must never
    # re-introduce histogram-level Scale refinement — DisplaceX only.
    displacement = next(s for s in stages if s.name == "sample_displacement")
    sample_params = displacement.set_hist.get("Sample Parameters", [])
    check("sample_displacement does not refine histogram-level Scale (degenerate with HAP Scale)",
          "Scale" not in sample_params)
    check("sample_displacement still refines DisplaceX",
          "DisplaceX" in sample_params)

    # Regression: DisplaceX (peak alignment) must be refined together with
    # Scale/Background in the FIRST stage, not deferred to a later one.
    # Confirmed on real data: refining Scale before peak position is
    # calibrated let GSAS-II's solver "fix" a misaligned-but-tall
    # calculated peak by shrinking it toward zero instead of moving it
    # into place — Scale converged to 1e-12. Position and intensity have
    # to be free to move together in that first pass.
    background_scale = next(s for s in stages if s.name == "background_scale")
    check("background_scale refines DisplaceX alongside Background/Scale",
          "DisplaceX" in background_scale.set_hist.get("Sample Parameters", []))
    check("background_scale still refines Background",
          "Background" in background_scale.set_hist)
    check("background_scale still refines phase HAP Scale",
          background_scale.set_hap.get("Scale") is True)

    # Regression: under --lebail, GSAS-II extracts each reflection's
    # intensity directly from the data instead of computing it from the
    # phase's atoms, so a phase-fraction HAP Scale factor is claiming the
    # same overall-intensity role a second time — the same kind of
    # degenerate-parameter trap already covered above for histogram Scale
    # vs. HAP Scale (confirmed on real data: leaving HAP Scale on
    # alongside LeBail reproduced the same SVD-singularity signature).
    lebail_stages = build_protocol(refine_atoms=False, lebail=True)
    lebail_background_scale = next(s for s in lebail_stages if s.name == "background_scale")
    check("lebail background_scale does not also refine phase HAP Scale",
          "Scale" not in lebail_background_scale.set_hap)
    check("lebail background_scale still refines DisplaceX and Background",
          "DisplaceX" in lebail_background_scale.set_hist.get("Sample Parameters", [])
          and "Background" in lebail_background_scale.set_hist)


def test_runner_accepts_converging_stages():
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        start_cell = (5.4668, 5.4666, 5.4666, 58.93, 58.93, 58.93)
        # every stage improves Rwp — mandatory stages by a little (well
        # inside default bounds), the optional peak_asymmetry/extinction
        # stages by a genuine >=1% (the bar Bounds.min_optional_
        # improvement_frac requires for an optional stage to be kept —
        # see run_stage/_attempt_variant). Unlike preferred_orientation
        # (which needs real 'Pref.Ori.' data the bare FakeProject/
        # FakePhase harness doesn't model, so every attempt errors out
        # before consuming an outcome), peak_asymmetry's Instrument
        # Parameters and extinction's HAP flag both go through
        # set_refinements()/set_HAP_refinements() no-op stubs that don't
        # raise, so they reach do_refinements() like any mandatory stage
        # — one scripted outcome each.
        set_active_outcomes([
            (18.0, [(5.47, 5.466, 5.466, 58.93, 58.93, 58.93)]),  # background_scale
            (17.5, None),                                         # sample_displacement
            (15.0, [(5.472, 5.468, 5.467, 58.92, 58.93, 58.93)]),  # unit_cell
            (14.2, None),                                         # profile_instrument
            (13.8, None),                                         # peak_asymmetry (optional, ~2.8% better)
            (13.0, None),                                         # profile_microstrain_size
            (12.7, None),                                         # extinction (optional, ~2.3% better)
        ])
        proj = FakeProject(rwp=20.0, phase_cells=[start_cell])
        stages = build_protocol(refine_atoms=False)
        runner = RefinementRunner(proj, outdir, Bounds(), log=lambda m: None)
        results = runner.run(stages)

        check("all 8 stages ran", len(results) == 8)
        check("all 5 mandatory stages ok",
              all(r.status == "ok" for r in results if not r.optional))
        check("peak_asymmetry and extinction (optional) accepted their genuine improvement",
              next(r for r in results if r.name == "peak_asymmetry").status == "ok"
              and next(r for r in results if r.name == "extinction").status == "ok")
        # preferred_orientation is the last stage and optional=True — the
        # bare FakeProject/FakePhase harness has no 'Pref.Ori.' data to
        # model (see PrefOriFakeProject for a harness that does model
        # that field properly), so every attempt errors out on a missing
        # attribute and the stage is expected to fail gracefully without
        # consuming any of the scripted outcomes above — see
        # test_runner_finds_correct_preferred_orientation_axis for a test
        # that models 'Pref.Ori.' properly and exercises the real ladder.
        check("preferred_orientation failed gracefully as optional, not counted as a run failure",
              results[-1].name == "preferred_orientation" and results[-1].optional
              and results[-1].status != "ok")
        check("rwp after the last mandatory stage (profile_microstrain_size) is the last real improvement",
              next(r for r in results if r.name == "profile_microstrain_size").rwp_after == 13.0)
        check("checkpoint files were written", len(proj.saved_to) >= 5)


def test_runner_accepts_first_stage_with_no_rwp_baseline():
    """
    Regression test for the real bug: a fresh project's first Rwp reading
    (before any do_refinements() call has ever run) is NaN, not None — see
    test_bounds_helpers' NaN-rwp_before check for the unit-level version.
    This is the integration-level version: drive a real RefinementRunner
    with a FakeProject whose initial rwp is NaN, exactly like a freshly
    add_phase()'d project, and confirm stage 1 is accepted (not rolled
    back) so its Scale refinement actually sticks.
    """
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        start_cell = (5.4668, 5.4666, 5.4666, 58.93, 58.93, 58.93)
        set_active_outcomes([
            (10.6, [(5.47, 5.466, 5.466, 58.93, 58.93, 58.93)]),  # background_scale: first real Rwp
            (10.6, None),  # sample_displacement
            (10.6, None),  # unit_cell
            (10.6, None),  # profile_instrument
            (10.6, None),  # peak_asymmetry (optional; flat, so it won't pass, but that's fine)
            (10.6, None),  # profile_microstrain_size
            (10.6, None),  # extinction (optional; flat, so it won't pass, but that's fine)
        ])
        proj = FakeProject(rwp=float("nan"), phase_cells=[start_cell])
        stages = build_protocol(refine_atoms=False)
        runner = RefinementRunner(proj, outdir, Bounds(), log=lambda m: None)
        results = runner.run(stages)

        check("stage 1 (background_scale, sets Scale) accepted, not rolled back",
              results[0].name == "background_scale" and results[0].status == "ok")
        check("stage 1's rwp_before recorded as NaN (informational, not a failure)",
              results[0].rwp_before != results[0].rwp_before)  # NaN != NaN
        check("all 5 mandatory stages ran without a spurious rollback",
              all(r.status == "ok" for r in results if not r.optional))


def test_runner_rolls_back_diverged_stage():
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        start_cell = (5.4668, 5.4666, 5.4666, 58.93, 58.93, 58.93)
        set_active_outcomes([
            (18.0, [(5.47, 5.466, 5.466, 58.93, 58.93, 58.93)]),  # background_scale: ok
            (17.5, None),  # sample_displacement: ok
            (12.0, [(9.0, 5.466, 5.466, 58.93, 58.93, 58.93)]),  # unit_cell: diverges
            (11.0, None),  # profile_instrument, runs post-rollback
            (10.5, None),  # peak_asymmetry (optional; ~4.5% better than 11.0 — passes)
            (10.0, None),  # profile_microstrain_size
            (9.7, None),   # extinction (optional; ~3% better than 10.0 — passes)
        ])
        proj = FakeProject(rwp=20.0, phase_cells=[start_cell])
        stages = build_protocol(refine_atoms=False)
        runner = RefinementRunner(proj, outdir, Bounds(max_cell_drift_frac=0.15), log=lambda m: None)
        results = runner.run(stages)

        check("8 stages recorded", len(results) == 8)
        check("stage 3 (unit_cell) failed bounds",
              results[2].name == "unit_cell" and results[2].status == "failed_bounds")
        check("subsequent mandatory stages still ran after rollback",
              all(r.status == "ok" for r in results[3:] if not r.optional))

        # Regression: the rolled-back project must actually contain the
        # pre-divergence cell (5.47...), not the diverged one (9.0...)
        # that triggered the rollback in the first place. This is the
        # check that catches the real bug found on actual data — GSAS-II's
        # refine-time auto-save silently overwriting the checkpoint file
        # with post-refinement state before a rollback could ever read the
        # pre-refinement state back out of it. FakeProject.do_refinements()
        # simulates that auto-save (see its comment); without
        # RefinementRunner re-pointing self.working_path before each
        # refine, this check fails.
        post_rollback_cell = runner._cell(0)
        check("post-rollback cell is the pre-divergence value, not the diverged one",
              abs(post_rollback_cell[0] - 5.47) < 1e-9)
        check("post-rollback cell is NOT the diverged value that failed bounds",
              abs(post_rollback_cell[0] - 9.0) > 1e-9)


def test_runner_rolls_back_diverged_profile_params():
    """
    Regression test for the real bug found via real GSAS-II diagnostics:
    U/V/W (and Mustrain) can diverge to unphysical magnitudes while Rwp
    barely moves and cell parameters stay put, because at that point in a
    fit these parameters have little leverage on the residual — so neither
    rwp_improved_or_stable() nor cell_drift_ok() ever flags it. That's
    exactly what left the calculated pattern peakless even after Scale
    was correctly fit: enormous U/V/W smear each reflection's intensity
    across a huge 2-theta range instead of a sharp peak. Drives a full
    RefinementRunner run where U blows up on profile_instrument (with Rwp
    still improving and the cell barely moving) and confirms that stage
    alone gets rolled back.
    """
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        start_cell = (5.4668, 5.4666, 5.4666, 58.93, 58.93, 58.93)
        # Rwp improves a little every stage and the cell barely moves — by
        # themselves these outcomes would pass every other bounds check on
        # every single stage.
        set_active_outcomes([
            (18.0, [(5.47, 5.466, 5.466, 58.93, 58.93, 58.93)]),  # background_scale
            (17.5, None),  # sample_displacement
            (17.0, None),  # unit_cell
            (16.8, None),  # profile_instrument: Rwp still improves...
            (16.5, None),  # peak_asymmetry (optional; ~2.9% better than 17.0 post-rollback — passes)
            (16.0, None),  # profile_microstrain_size
            (15.7, None),  # extinction (optional; ~1.9% better than 16.0 — passes)
        ])
        u_values = [0.0, 0.0, 0.0, 2_260_583.58]  # ...but U blows up right here
        proj = ProfileFakeProject(rwp=20.0, phase_cells=[start_cell], u_values=u_values)
        stages = build_protocol(refine_atoms=False)
        runner = RefinementRunner(proj, outdir, Bounds(), log=lambda m: None)
        results = runner.run(stages)

        check("8 stages recorded", len(results) == 8)
        profile_stage = next(r for r in results if r.name == "profile_instrument")
        check("profile_instrument rolled back despite Rwp improving and cell staying put",
              profile_stage.status == "failed_bounds")
        other_mandatory_stages = [r for r in results
                                   if r.name != "profile_instrument" and not r.optional]
        check("every other mandatory stage still ok",
              all(r.status == "ok" for r in other_mandatory_stages))


def test_runner_falls_back_to_simpler_profile_model():
    """
    Drives RefinementRunner.run_stage() with a Stage whose primary config
    (uniaxial Mustrain) genuinely diverges (via MustrainFallbackFakeProject
    — see its docstring for the real-data bug this models) and confirms
    the stage falls through to its first fallback (isotropic Mustrain)
    and reports "ok" using it, instead of just reverting and giving up.
    """
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        start_cell = (4.2, 4.2, 4.2, 90, 90, 90)

        stage = Stage(
            name="profile_microstrain_size",
            set_hap={
                "Size": {"type": "isotropic", "refine": True},
                "Mustrain": {"type": "uniaxial", "refine": True},
            },
            fallbacks=[
                Stage(name="isotropic_mustrain",
                      set_hap={"Mustrain": {"type": "isotropic", "refine": True}}),
            ],
        )

        # Consumed by the plain FakeProject that _reload() swaps in after
        # the primary attempt's bounds failure — models the fallback's
        # outcome (real FakeProject.do_refinements() has no Mustrain
        # concept of its own, only MustrainFallbackFakeProject does).
        set_active_outcomes([(12.0, None)])

        proj = MustrainFallbackFakeProject(rwp=20.0, phase_cells=[start_cell])
        runner = RefinementRunner(proj, outdir, Bounds(), log=lambda m: None)
        start_cells = {0: runner._cell(0)}
        result = runner.run_stage(stage, 1, start_cells)

        check("stage ok via fallback after primary diverged",
              result.status == "ok" and result.rwp_after == 12.0)
        check("detail names the fallback that succeeded",
              "isotropic_mustrain" in result.detail)


def test_runner_freezes_mustrain_for_atoms_fallback():
    """
    Drives RefinementRunner.run_stage() with the real "atoms" Stage from
    build_protocol(refine_atoms=True) against AtomsFallbackFakeProject
    (see its docstring), whose Mustrain only stays sane when clear_hap
    actually froze it. Confirms the primary attempt (Mustrain left free)
    fails bounds despite a genuinely better fit, and the
    "atoms_mustrain_frozen" fallback both keeps that improvement and
    keeps Mustrain sane — the exact recovery this fallback exists for.
    """
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        start_cell = (4.2, 4.2, 4.2, 90, 90, 90)

        atoms_stage = next(s for s in build_protocol(refine_atoms=True) if s.name == "atoms")

        # A fallback ladder reloads the pre-stage checkpoint before every
        # attempt — see set_active_project_class's docstring for why this
        # is needed for AtomsFallbackFakeProject's modeling to survive
        # past the primary (Mustrain-still-free) attempt.
        set_active_project_class(AtomsFallbackFakeProject)
        try:
            proj = AtomsFallbackFakeProject(rwp=20.0, phase_cells=[start_cell])
            runner = RefinementRunner(proj, outdir, Bounds(), log=lambda m: None)
            start_cells = {0: runner._cell(0)}
            result = runner.run_stage(atoms_stage, 1, start_cells)
        finally:
            set_active_project_class(FakeProject)

        check("atoms stage succeeded via the Mustrain-frozen fallback",
              result.status == "ok")
        check("detail names the atoms_mustrain_frozen fallback, not the primary",
              "atoms_mustrain_frozen" in result.detail)
        check("the genuine Rwp improvement was kept, not discarded",
              result.rwp_after == 15.0)


def test_runner_finds_correct_preferred_orientation_axis():
    """
    Drives RefinementRunner.run_stage() with the real preferred_orientation
    Stage from build_protocol() against PrefOriFakeProject (see its
    docstring), which only shows genuine improvement for axis (1,1,0).
    Confirms the ladder correctly skips the earlier no-op axes — which a
    plain non-worsening bar would have wrongly accepted — and lands on the
    one that actually helped.
    """
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        start_cell = (4.2, 4.2, 4.2, 90, 90, 90)

        po_stage = next(s for s in build_protocol(refine_atoms=False)
                         if s.name == "preferred_orientation")

        # A fallback ladder reloads the pre-stage checkpoint before every
        # attempt, not just once — see set_active_project_class's
        # docstring for why this is needed for PrefOriFakeProject's
        # modeling to survive past the first (0,0,1)-axis attempt.
        set_active_project_class(PrefOriFakeProject)
        try:
            proj = PrefOriFakeProject(rwp=20.0, phase_cells=[start_cell])
            runner = RefinementRunner(proj, outdir, Bounds(), log=lambda m: None)
            start_cells = {0: runner._cell(0)}
            result = runner.run_stage(po_stage, 1, start_cells)
        finally:
            set_active_project_class(FakeProject)

        check("preferred_orientation succeeded via the axis with real texture",
              result.status == "ok")
        check("detail names the (1,1,0) fallback, not the primary (0,0,1) no-op",
              "pref_ori_110" in result.detail)


def test_runner_survives_solver_exception():
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        start_cell = (5.0, 5.0, 5.0, 90, 90, 90)
        set_active_outcomes([(18.0, None)])
        stages = [Stage(name="background_scale", set_hist={"Background": {"refine": True}})]

        class ExplodingProject(FakeProject):
            def do_refinements(self, reflist):
                raise RuntimeError("simulated solver blow-up")

        proj2 = ExplodingProject(rwp=20.0, phase_cells=[start_cell])
        runner = RefinementRunner(proj2, outdir, Bounds(), log=lambda m: None)
        results = runner.run(stages)

        check("errored stage recorded as failed_error", results[0].status == "failed_error")
        check("run did not raise / crash the process", True)


def test_runner_catches_swallowed_solver_failure():
    """
    Regression test for the real bug found in results/FeF3_v3/run_v3.log:
    GSASIIscriptable's G2Project.refine() calls G2strMain.Refine(), which
    returns an (IfOK, Rvals) success flag that GSASIIscriptable itself
    discards — an internal solver failure (SVD inversion failure during
    the Hessian LM step) is only ever reported by printing "***** Refinement
    error *****" / "SVD inversion failure" to stdout, never by raising.
    On real data this let do_refinements() return "successfully" after an
    internal crash; since the failed cycle never updated the histogram's
    residuals, _rwp() read back the stale pre-refinement value, and
    rwp_improved_or_stable() judged that as "unchanged, therefore fine" —
    silently logging the stage "ok" when GSAS-II's own solver had already
    given up. This models exactly that: do_refinements() prints the known
    failure text and returns normally (Rwp unchanged), and confirms
    RefinementRunner catches it from the captured solver output anyway.
    """
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        start_cell = (5.0, 5.0, 5.0, 90, 90, 90)

        class SwallowedFailureProject(FakeProject):
            def do_refinements(self, reflist):
                print(" ***** Refinement error *****")
                print("Note refinement problem:\nSVD inversion failure")
                # Rwp left unchanged — exactly what a failed cycle does on
                # real data, since it never gets to update the residuals.

        proj = SwallowedFailureProject(rwp=10.613, phase_cells=[start_cell])
        stages = [Stage(name="profile_microstrain_size",
                         set_hap={"Mustrain": {"type": "uniaxial", "refine": True}})]
        runner = RefinementRunner(proj, outdir, Bounds(), log=lambda m: None)
        results = runner.run(stages)

        check("swallowed solver failure recorded as failed_error, not ok",
              results[0].status == "failed_error")


def test_rwp_handles_residuals_as_method_or_property():
    # Real GSASIIscriptable installs have been observed with `residuals`
    # both as a plain method (call it) and — on at least one installed
    # version — as a @property (a plain dict, calling it raises "'dict'
    # object is not callable"). RefinementRunner._rwp() must handle either
    # without knowing in advance which one a given install uses.
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)

        class MethodStyleHist:
            def residuals(self):
                return {"wR": 12.5}

        class PropertyStyleHist:
            @property
            def residuals(self):
                return {"wR": 12.5}

        for label, hist_cls in [("method-style", MethodStyleHist), ("property-style", PropertyStyleHist)]:
            proj = FakeProject(rwp=20.0, phase_cells=[(5.0, 5.0, 5.0, 90, 90, 90)])
            proj._hists = [hist_cls()]
            runner = RefinementRunner(proj, outdir, Bounds(), log=lambda m: None)
            check(f"_rwp() reads {label} residuals correctly", runner._rwp() == 12.5)


# ---------------------------------------------------------------------------
# import_gsasiiscriptable() — reproduces, on disk with fake modules, the two
# real GSAS-II layouts this function has to handle. This is what caught (and
# now guards against) the real "attempted relative import with no known
# parent package" error hit against an actual GSAS2MAIN install.
# ---------------------------------------------------------------------------

def _clear_module_and_path(module_name: str, path_entry: str):
    sys.modules.pop(module_name, None)
    top_level = module_name.split(".")[0]
    sys.modules.pop(top_level, None)
    if path_entry in sys.path:
        sys.path.remove(path_entry)


def test_import_gsasiiscriptable_package_layout_with_relative_import():
    # Reproduces the current GSAS2MAIN installer's layout: a "GSASII"
    # package directory whose GSASIIscriptable.py does a relative import
    # of a sibling module. Importing this directly as a top-level
    # "GSASIIscriptable" module (the old approach) is exactly what raised
    # "attempted relative import with no known parent package" against a
    # real install — this test fails the same way if that regresses.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "GSAS-II" / "GSASII"
        root.mkdir(parents=True)
        (root / "__init__.py").write_text("")
        (root / "_sibling_helper.py").write_text("MARKER = 'package-layout-ok'\n")
        (root / "GSASIIscriptable.py").write_text(
            "from . import _sibling_helper\nMARKER = _sibling_helper.MARKER\n"
        )
        try:
            module = import_gsasiiscriptable(root)
            check("package-layout module imported", module.MARKER == "package-layout-ok")
        finally:
            _clear_module_and_path("GSASII.GSASIIscriptable", str(root.parent))


def test_import_gsasiiscriptable_flat_layout_no_relative_import():
    # Reproduces an older/hand-built install: GSASIIscriptable.py sits
    # directly in the given folder with no relative imports.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "flat_gsas_install"
        root.mkdir(parents=True)
        (root / "GSASIIscriptable.py").write_text("MARKER = 'flat-layout-ok'\n")
        try:
            module = import_gsasiiscriptable(root)
            check("flat-layout module imported", module.MARKER == "flat-layout-ok")
        finally:
            _clear_module_and_path("GSASIIscriptable", str(root))
            _clear_module_and_path("flat_gsas_install.GSASIIscriptable", str(root.parent))


def test_export_histogram_csv_writes_expected_columns():
    import numpy as np

    class FakeHistForExport:
        def __init__(self):
            x = [10.0, 10.1, 10.2, 10.3]
            yobs = [100.0, 110.0, 105.0, np.nan]
            yobs_masked = np.ma.MaskedArray(yobs, mask=[False, False, False, True])
            self.data = {"data": [None, [x, yobs_masked, [1, 1, 1, 1],
                                          [98, 108, 103, 97], [10, 10, 10, 10], [2, 2, 2, 3]]]}

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pattern_raw.csv"
        export_histogram_csv(FakeHistForExport(), out, ["two_theta", "y_obs"])
        lines = out.read_text().strip().splitlines()
        check("csv header correct", lines[0] == "two_theta,y_obs")
        check("csv has one row per point", len(lines) == 5)  # header + 4 points
        check("masked point exported as nan", "nan" in lines[-1])
        check("first row values correct", lines[1] == "10,100")


def test_get_phase_cells_collects_all_phases():
    class FakePhaseForCells:
        def __init__(self, name, a):
            self.name = name
            self._a = a

        def get_cell(self):
            return {"length_a": self._a, "length_b": self._a, "length_c": self._a,
                    "angle_alpha": 90.0, "angle_beta": 90.0, "angle_gamma": 90.0, "volume": self._a ** 3}

        def get_cell_and_esd(self):
            return self.get_cell(), {k: 0.01 for k in self.get_cell()}

    class FakeGpxForCells:
        def phases(self):
            return [FakePhaseForCells("MgO", 4.2), FakePhaseForCells("MgBC", 3.1)]

    cells, esds = get_phase_cells(FakeGpxForCells())
    check("both phases present in cells dict", set(cells) == {"MgO", "MgBC"})
    check("cell values correct per phase", cells["MgO"]["length_a"] == 4.2 and cells["MgBC"]["length_a"] == 3.1)
    check("esds collected per phase", esds["MgO"]["length_a"] == 0.01)


def test_import_gsasiiscriptable_raises_with_details_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        bogus = Path(tmp) / "not_actually_gsas"
        raised = False
        try:
            import_gsasiiscriptable(bogus)
        except ImportError as exc:
            raised = True
            check("error message names the missing path", str(bogus) in str(exc))
        check("ImportError raised for a nonexistent GSAS-II path", raised)


if __name__ == "__main__":
    test_bounds_helpers()
    test_seed_initial_scale()
    test_assess_fit_quality()
    test_protocol_shape()
    test_runner_accepts_converging_stages()
    test_runner_accepts_first_stage_with_no_rwp_baseline()
    test_runner_rolls_back_diverged_stage()
    test_runner_rolls_back_diverged_profile_params()
    test_runner_falls_back_to_simpler_profile_model()
    test_runner_finds_correct_preferred_orientation_axis()
    test_runner_freezes_mustrain_for_atoms_fallback()
    test_runner_survives_solver_exception()
    test_runner_catches_swallowed_solver_failure()
    test_rwp_handles_residuals_as_method_or_property()
    test_export_histogram_csv_writes_expected_columns()
    test_get_phase_cells_collects_all_phases()
    test_import_gsasiiscriptable_package_layout_with_relative_import()
    test_import_gsasiiscriptable_flat_layout_no_relative_import()
    test_import_gsasiiscriptable_raises_with_details_when_missing()
    print("\nAll control-flow checks passed (no GSAS-II install required for this test).")
