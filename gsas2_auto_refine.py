#!/usr/bin/env python3
"""
gsas2_auto_refine.py — Turnkey, deterministic Rietveld refinement runner
built on GSAS-II's official scripting interface (GSASIIscriptable).

Purpose
-------
Wraps a fixed, hardcoded refinement protocol around GSASIIscriptable so a
scientist can point this at a pre-integrated 1D powder pattern + instrument
parameter file + one or more phase CIFs and get back a finished, checked
refinement — no manual GUI clicking, no interactive tuning.

This script handles Stage 2/3 of the pipeline described in
"GSAS-II Automation Pipeline: Architecture & Feasibility":
    pre-integrated pattern -> staged refinement -> export results

It does NOT do image integration (that's a separate, pyFAI/OpenCL-based
stage for raw 2D detector frames — out of scope here since both example
datasets in Data/ are already pre-integrated 1D patterns).

Refinement protocol (fixed, in order — see run_refinement()):
    1. Background + histogram scale factor
    2. Sample displacement / zero-shift
    3. Unit cell parameters
    4. Profile parameters (instrument U/V/W, then microstrain/crystallite size)
    5. Atom positions + isotropic thermal parameters   [optional, --refine-atoms]

Each stage is checkpointed (project saved to a stage-numbered .gpx) before
it runs. After a stage, results are validated against hardcoded bounds
(Rwp must not get worse beyond tolerance; cell parameters must not drift
more than --max-cell-drift from their starting values). If a stage fails
validation, the run reloads the last good checkpoint, records the stage as
FAILED, and continues to the next stage rather than compounding a bad fit
or waiting on a human to intervene.

GSAS-II is not a pip package — GSASIIscriptable.py lives inside a local
GSAS-II installation. Point --gsasii-path at that installation's top-level
directory (the one containing GSASIIscriptable.py) — the diretory is
inserted into sys.path before import. On import failure (module not found,
or GSAS-II not installed on this machine yet), the script prints a clear
error and exits — see also --dry-run below.

Testing without GSAS-II installed
----------------------------------
Use --dry-run to validate CLI arguments, confirm all input file paths
exist and are readable, print the exact staged-refinement plan that would
run, and exit — without importing GSASIIscriptable at all. This is the
"mock data for verification" path: use it to smoke-test a new dataset's
paths/config before running the real refinement on a machine with GSAS-II
installed. The control-flow logic itself (staging/checkpoint/bounds
gating) is exercised independently in test_auto_refine_logic.py against a
fake in-memory project, with no GSAS-II dependency at all.

Example — single-phase synchrotron data (Data/FeF3):
    python3 gsas2_auto_refine.py \\
        --gsasii-path /path/to/GSASII \\
        --pattern "Data/FeF3/r3c 20.txt" \\
        --instprm "Data/FeF3/ws2.prm" \\
        --cif "Data/FeF3/fef3 r3c.cif" \\
        --outdir results/FeF3

Example — two-phase lab data (Data/MgO+MgBC):
    python3 gsas2_auto_refine.py \\
        --gsasii-path /path/to/GSASII \\
        --pattern "Data/MgO+MgBC/h3_d1_120s_0p5mm_001_exported_exported.xy" \\
        --instprm "Data/MgO+MgBC/D8_parameter-1.prm" \\
        --cif "Data/MgO+MgBC/MgO.cif" \\
        --cif "Data/MgO+MgBC/MgBC.cif" \\
        --outdir results/MgO_MgBC

Note on the FeF3 CIF: it's a VESTA export with symmetry flattened to P1
(6 F + 2 Fe listed explicitly) even though the filename says r3c — GSAS-II
will happily refine it as P1 (just with more free parameters than a
proper R-3c setting would need). This script does not attempt to
re-detect/raise the symmetry; that's a modeling decision for the scientist,
not something to guess at silently.
"""

import os

# Must be set before numpy/GSASIIscriptable ever load MKL: on some Windows
# machines MKL's threaded (Intel OpenMP) codepath hard-crashes the process
# (access violation, no catchable Python exception) the first time any
# numpy.linalg/BLAS call actually runs — reproduced directly down to
# np.linalg.inv() and np.dot() on a stock identity matrix, unrelated to
# GSAS-II or any input data. Forcing MKL's sequential codepath avoids it.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse
import contextlib
import csv
import importlib
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


def import_gsasiiscriptable(gsasii_path: Path):
    """
    Locates and imports GSASIIscriptable, handling both GSAS-II layouts
    seen in the wild:

      - "package" layout (current GSAS2MAIN/pixi installs): --gsasii-path
        points at the GSASII directory itself, which is a proper Python
        package — GSASIIscriptable.py inside it uses relative imports
        (`from . import GSASIIpath`, etc.) and so MUST be imported as
        `GSASII.GSASIIscriptable`, with that folder's *parent* directory on
        sys.path. Importing it directly as a top-level module fails with
        "attempted relative import with no known parent package" — that's
        the error this function exists to avoid.
      - "flat" layout (older installs / hand-built checkouts): the folder
        given directly contains GSASIIscriptable.py meant to be imported as
        a plain top-level module, with that folder itself on sys.path.

    Tries the layout implied by the folder being named "GSASII" first
    (what today's installer produces — see install.html), then falls back
    to the flat style, so the same --gsasii-path value people are told to
    find via `find ~/g2main -name GSASIIscriptable.py` works either way.
    Raises ImportError with details from every attempt if none succeed.
    """
    attempts = [(f"{gsasii_path.name}.GSASIIscriptable", gsasii_path.parent)]
    if gsasii_path.name != "GSASII":
        attempts.append(("GSASII.GSASIIscriptable", gsasii_path.parent))
    attempts.append(("GSASIIscriptable", gsasii_path))

    errors = []
    for module_name, path_to_add in attempts:
        added = str(path_to_add) not in sys.path
        if added:
            sys.path.insert(0, str(path_to_add))
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"  import {module_name!r} (with {path_to_add} on sys.path): {exc}")
            if added:
                sys.path.remove(str(path_to_add))

    raise ImportError(
        "Could not import GSASIIscriptable from " + str(gsasii_path) +
        " using any known GSAS-II layout. Attempts:\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Fixed refinement protocol. This is intentionally hardcoded, not a config
# knob — per the project's automation directives, algorithmic control flow
# belongs in code, not in back-and-forth chat tuning.
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    name: str
    set_hist: dict = field(default_factory=dict)     # G2PwdrData.set_refinements
    set_phase: dict = field(default_factory=dict)    # G2Phase.set_refinements
    set_hap: dict = field(default_factory=dict)      # G2Phase.set_HAP_refinements
    optional: bool = False                            # e.g. atoms stage
    # Simpler (fewer-parameter, less mutually-correlated) alternative
    # configurations to try, in order, if this stage's primary
    # configuration fails its bounds check — see RefinementRunner.run_stage.
    # Each fallback's `name` is used only for logging which one succeeded;
    # the StageResult reported to callers always uses the stage's own name.
    fallbacks: list = field(default_factory=list)
    # Preferred-orientation (March-Dollase) axis, e.g. (0, 0, 1), applied to
    # every phase before set_hap — see build_protocol()'s
    # preferred_orientation stage. Not one of GSASIIscriptable's
    # set_HAP_refinements() keys (only the refine flag is — the axis has
    # no scriptable setter), so RefinementRunner._attempt_variant pokes it
    # directly into each phase's HAP data instead.
    pref_ori_axis: tuple = None
    # When set, pref_ori_axis and set_hap["Pref.Ori."] apply to ONLY this
    # phase index, not every phase — see build_protocol()'s
    # _pref_ori_fallbacks() for why: confirmed on real two-phase data
    # (Data/MgO+MgBC) that forcing the same texture axis onto every phase
    # at once made Rwp much worse for every candidate axis (12.4% ->
    # 27.3%), because two phases' real textures don't have to match —
    # MgO (cubic) and MgBC don't even share a crystal system.
    pref_ori_phase_index: int = None
    # HAP refinement flags to explicitly turn OFF (via G2Phase.
    # clear_HAP_refinements()) before applying set_hap/set_hist/set_phase
    # — see build_protocol()'s atoms-stage fallbacks. Every stage in this
    # protocol is cumulative (a parameter freed in an earlier stage stays
    # free in every later one, by design — see sample_displacement's
    # docstring), which is normally what makes joint refinement converge
    # properly by the end. It backfires specifically for atom positions:
    # confirmed on real data (both Data/FeF3 and Data/MgO+MgBC) that
    # atomic Uiso and Mustrain are correlated enough that refining atoms
    # while Mustrain is still free (carried over from
    # profile_microstrain_size) sends Mustrain to a runaway value
    # (107,466 seen on real MgO+MgBC data) — even though the underlying
    # fit was genuinely, substantially better (Rwp 12.4% -> 9.8% on that
    # same run) before the bounds check discarded it. Freezing Mustrain
    # at its already-converged value first removes that correlated
    # degree of freedom without giving up what profile_microstrain_size
    # already found.
    clear_hap: dict = field(default_factory=dict)


# Low-index axes real single-phase textures most often align with — see
# _pref_ori_fallbacks().
_PREF_ORI_AXES = [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)]


def _pref_ori_fallbacks(n_phases: int) -> list:
    """
    Builds the preferred_orientation stage's fallback ladder: first every
    candidate axis applied to every phase at once (the stage's own
    pref_ori_axis=(0,0,1) is the primary attempt — this list starts from
    the second axis), then — only for multi-phase samples — the same
    axes tried on ONE phase at a time.

    The per-phase fallbacks matter: confirmed on real two-phase data
    (Data/MgO+MgBC) that forcing the *same* texture axis onto every phase
    simultaneously made Rwp much worse for every single candidate axis
    (12.4% -> as high as 27.3%), because two phases' real textures (if
    they have any at all) don't have to match, or even exist in both —
    MgO (cubic rock salt) and MgBC don't share a crystal system, let
    alone a preferred texture direction. Trying each phase independently
    lets the ladder find "phase A is textured, phase B isn't" instead of
    only ever being able to accept or reject "both are textured the same
    way."
    """
    fallbacks = [
        Stage(name=f"pref_ori_{a}{b}{c}", pref_ori_axis=(a, b, c), set_hap={"Pref.Ori.": True})
        for a, b, c in _PREF_ORI_AXES[1:]  # [0] is the stage's own primary config
    ]
    if n_phases > 1:
        for phase_index in range(n_phases):
            for a, b, c in _PREF_ORI_AXES:
                fallbacks.append(Stage(
                    name=f"pref_ori_phase{phase_index}_{a}{b}{c}",
                    pref_ori_axis=(a, b, c),
                    pref_ori_phase_index=phase_index,
                    set_hap={"Pref.Ori.": True},
                ))
    return fallbacks


def build_protocol(refine_atoms: bool, lebail: bool = False, n_phases: int = 1) -> list:
    """
    lebail: when True, each stage's HAP dict skips "Scale" (phase-fraction
    scale factor). This is for --lebail runs, where GSAS-II extracts each
    reflection's intensity directly from the data (LeBail) instead of
    computing it from the phase's atoms — see the --lebail CLI help and
    the LeBail block in main() for why: a phase-fraction Scale factor and
    per-reflection LeBail-extracted intensities both claim ownership of
    the same overall intensity level, so refining both is the same kind
    of degenerate-parameter trap already documented for histogram Scale
    vs. HAP Scale below (sample_displacement's docstring) — confirmed by
    testing this pipeline against real data with LeBail on: leaving Scale
    on alongside it reproduced the same SVD-singularity/nonsense-shift
    signature.

    n_phases: how many phase CIFs this run will load (len(args.cif) in
    main() — known before any GSAS-II interaction). Only affects the
    preferred_orientation stage's fallback ladder — see
    _pref_ori_fallbacks() for why multi-phase samples need per-phase
    texture-axis attempts, not just one shared axis across every phase.
    """
    stages = [
        Stage(
            name="background_scale",
            set_hist={
                "Background": {"type": "chebyschev-1", "refine": True, "no. coeffs": 6},
                # DisplaceX (peak-position/alignment correction) is refined
                # HERE, together with Scale, not deferred to a later stage —
                # confirmed as necessary on real data. Refining peak height
                # (Scale) before peak position is calibrated is a classic
                # Rietveld ordering trap: if the calculated peaks start out
                # even slightly misaligned from the observed ones (the
                # normal starting condition, not a bug), a least-squares
                # solver sees a TALLER misaligned peak as making the fit
                # WORSE, not better — so it rationally drives Scale toward
                # zero rather than up. On real FeF3 data this collapsed
                # Scale to 1e-12 (Rwp still looked "reasonable" because the
                # fit was just background-only at that point). Position and
                # intensity have to be free to move together so the solver
                # can find "peak, in the right place" instead of getting
                # stuck at "no peak" as the locally-safer option.
                "Sample Parameters": ["DisplaceX"],
            },
            set_hap={} if lebail else {"Scale": True},
            # A 6-term Chebyshev polynomial doesn't have enough freedom
            # for every real background shape (broad amorphous humps,
            # multi-phase overlap, detector-specific curvature) — a
            # scientist's usual manual fix is just "add more background
            # terms." These fallbacks try progressively richer background
            # models automatically, in order, only if the 6-term default
            # fails its bounds check.
            fallbacks=[
                Stage(
                    name="background_10term",
                    set_hist={
                        "Background": {"type": "chebyschev-1", "refine": True, "no. coeffs": 10},
                        "Sample Parameters": ["DisplaceX"],
                    },
                    set_hap={} if lebail else {"Scale": True},
                ),
                Stage(
                    name="background_15term",
                    set_hist={
                        "Background": {"type": "chebyschev-1", "refine": True, "no. coeffs": 15},
                        "Sample Parameters": ["DisplaceX"],
                    },
                    set_hap={} if lebail else {"Scale": True},
                ),
            ],
        ),
        Stage(
            name="sample_displacement",
            # A second, focused pass on DisplaceX alone (now that Scale and
            # Background are already in the right ballpark from the stage
            # above) — cheap, and covers any further refinement the joint
            # first pass didn't fully settle.
            #
            # Deliberately NOT "Scale" here. GSAS-II has two multiplicative
            # intensity-scale parameters: this histogram's own overall
            # "Sample Parameters: Scale" and each phase's HAP "Scale"
            # (phase fraction, turned on in background_scale above via
            # set_hap). For one phase in one histogram those two are
            # mathematically 100% degenerate — GSAS-II's own solver log
            # confirmed this on real data: "Note highly correlated
            # parameters: 0:0:Scale and :0:Scale (@100.00%)" plus an SVD
            # singularity warning from the very first refinement cycle,
            # with a nonsense parameter shift (Maximum shift/esd ~625).
            # Refining both at once is why the phase's own Scale factor
            # never moved off its untouched default of 1.0 even though
            # Rwp looked reasonable — the fit was quietly putting whatever
            # intensity signal it found into the wrong (redundant)
            # parameter. Phase HAP Scale is the one that generalizes to
            # multi-phase samples (it becomes relative phase fraction), so
            # it's the one we keep; the histogram-level Scale is left
            # fixed at its default everywhere (see also the explicit
            # clear_refinements() call in main(), which stops relying on
            # that default rather than just never setting it here).
            set_hist={"Sample Parameters": ["DisplaceX"]},
        ),
        Stage(
            name="unit_cell",
            set_phase={"Cell": True},
        ),
        Stage(
            name="profile_instrument",
            set_hist={"Instrument Parameters": ["U", "V", "W"]},
        ),
        Stage(
            name="peak_asymmetry",
            optional=True,
            # SH/L (Finger-Cox-Jephcoat axial-divergence asymmetry) —
            # a common manual fix for lab data whose low-angle peaks
            # visibly lean to one side, something U/V/W (symmetric
            # Gaussian/Lorentzian broadening only) cannot represent at
            # all. Every .prm file in this project already carries an
            # SH/L value (it's a real, fittable Instrument Parameters
            # field — 'SH/L' is in the same [value, esd, refine-flag]
            # shape as U/V/W, confirmed against the installed
            # GSASIIscriptable.py source), it's just never been refined
            # by this pipeline before. optional=True and requires a
            # genuine improvement (see Bounds.min_optional_improvement_
            # frac): plenty of instrument geometries have negligible
            # asymmetry, and SH/L refining right back to its starting
            # value is the expected, harmless outcome there, not a
            # pipeline failure.
            set_hist={"Instrument Parameters": ["SH/L"]},
        ),
        Stage(
            name="profile_microstrain_size",
            set_hap={
                "Size": {"type": "isotropic", "refine": True},
                "Mustrain": {"type": "uniaxial", "refine": True},
            },
            # Real two-phase lab data (Data/MgO+MgBC) confirmed uniaxial
            # Mustrain and isotropic Size can be ~99% correlated: the
            # solver sends Mustrain to a runaway value (-159,724 seen on
            # real data) while Size collapses toward zero (0.001), a
            # bounds-check failure that reverting-and-moving-on would
            # otherwise just give up on. Each fallback below is a
            # strictly simpler (fewer/less-correlated free parameters)
            # model tried automatically, in order, before giving up — see
            # RefinementRunner.run_stage.
            fallbacks=[
                Stage(
                    name="isotropic_mustrain",
                    set_hap={
                        "Size": {"type": "isotropic", "refine": True},
                        "Mustrain": {"type": "isotropic", "refine": True},
                    },
                ),
                Stage(
                    name="size_only",
                    set_hap={"Size": {"type": "isotropic", "refine": True}},
                ),
                Stage(
                    name="mustrain_only",
                    set_hap={"Mustrain": {"type": "isotropic", "refine": True}},
                ),
            ],
        ),
        Stage(
            name="extinction",
            optional=True,
            # Secondary extinction — a common manual fix specifically for
            # well-crystallized samples whose *strongest* peak(s) come out
            # systematically shorter than the calculated (kinematic)
            # intensity predicts, while everything else fits fine. That's
            # exactly the residual pattern confirmed on real data
            # (Data/MgO+MgBC's strongest peak, 2theta=42.85 deg, stayed
            # calculated ~35% too tall through every earlier stage) —
            # extinction is the standard physical explanation for that
            # specific signature (as opposed to preferred orientation,
            # which affects a whole reflection *family* sharing an axis,
            # not necessarily just the single most intense line).
            # GSASIIscriptable's 'Extinction' HAP field is a plain
            # [value, refine-flag] pair starting at 0.0 (confirmed
            # against the installed source) — no axis or type choice
            # needed the way Pref.Ori. has, so there's just one config to
            # try. optional=True and requires a genuine improvement (see
            # Bounds.min_optional_improvement_frac): most phases show no
            # measurable extinction, and it refining right back to 0.0 is
            # the expected, harmless outcome there.
            set_hap={"Extinction": True},
        ),
        Stage(
            name="preferred_orientation",
            optional=True,
            # March-Dollase preferred orientation (texture) — a common
            # manual fix when peak *positions* and profile widths already
            # fit well but a few specific reflections are still
            # systematically over- or under-predicted (confirmed on real
            # data: Data/MgO+MgBC's strongest peak, 2theta=42.85 deg,
            # stayed ~35% overestimated after every other stage
            # converged). GSASIIscriptable's set_HAP_refinements() only
            # exposes the refine flag for "Pref.Ori.", not the axis
            # itself (see Stage.pref_ori_axis), so this tries the
            # low-index axes real textures most often align with, one at
            # a time via the fallback ladder, and keeps whichever
            # actually improves Rwp. optional=True: for many phases no
            # single low-index axis will help, and that's a legitimate
            # "this phase isn't textured" outcome, not a pipeline
            # failure — see main()'s failed-stage summary, which doesn't
            # count an all-fallbacks-failed optional stage against the
            # run.
            pref_ori_axis=(0, 0, 1),
            set_hap={"Pref.Ori.": True},
            fallbacks=_pref_ori_fallbacks(n_phases),
        ),
    ]
    if refine_atoms:
        stages.append(
            Stage(
                name="atoms",
                # G2Phase.set_refinements()'s "Atoms" handler does
                # `value.items()` — it wants a dict of {atom_label:
                # refinement_flags}, not a bare string. "all" -> every
                # atom; "XU" -> confirmed against the installed source
                # (G2AtomRecord.refinement_flags's setter) as position
                # (X) + isotropic displacement parameter (U); only 'F'
                # (occupancy), 'X', and 'U' are valid flag characters.
                # Passing the bare string "all" here raised
                # AttributeError("'str' object has no attribute 'items'")
                # on every real --refine-atoms run — caught by
                # RefinementRunner's broad exception handling and
                # reported as a normal failed_error (optional, so it
                # didn't fail the run), but it meant --refine-atoms could
                # never actually refine anything.
                set_phase={"Atoms": {"all": "XU"}},
                optional=True,
                # Confirmed on real data (both Data/FeF3 and
                # Data/MgO+MgBC): atomic Uiso is correlated enough with
                # Mustrain (still free here — every stage in this
                # protocol is cumulative) that refining atoms alongside
                # it sends Mustrain to a runaway value even when the
                # underlying fit is genuinely much better (Rwp 12.4% ->
                # 9.8% on real MgO+MgBC data, discarded by the bounds
                # check before this fallback existed). See Stage.
                # clear_hap's docstring for the full evidence.
                fallbacks=[
                    Stage(
                        name="atoms_mustrain_frozen",
                        set_phase={"Atoms": {"all": "XU"}},
                        clear_hap={"Mustrain": True},
                    ),
                    Stage(
                        name="atoms_mustrain_and_size_frozen",
                        set_phase={"Atoms": {"all": "XU"}},
                        clear_hap={"Mustrain": True, "Size": True},
                    ),
                ],
            )
        )
    return stages


# ---------------------------------------------------------------------------
# Bounds / convergence gating — deterministic, hardcoded thresholds.
# ---------------------------------------------------------------------------

@dataclass
class Bounds:
    max_cell_drift_frac: float = 0.15   # cell edge must stay within +/-15% of start
    rwp_worsen_tol_frac: float = 0.02   # allow Rwp to get up to 2% relatively worse
                                         # before flagging the stage as failed
    max_rwp_absolute: float = 60.0      # a "successful" refinement above this Rwp%
                                         # is still suspect; flagged, not fatal
    max_profile_param_abs: float = 10000.0
    min_optional_improvement_frac: float = 0.01
    # An *optional* stage (see Stage.optional) needs a stricter bar than
    # "didn't get worse": adding a free parameter that has no real effect
    # on this sample essentially never makes Rwp worse either (it just
    # settles back to its no-op value — e.g. a preferred-orientation
    # ratio refining right back to 1.0 = no texture), so
    # rwp_improved_or_stable()'s ordinary "non-worsening" bar would
    # accept the *first* candidate tried regardless of whether it's the
    # one that actually explains the data. Requiring a real relative
    # improvement is what lets RefinementRunner tell "this candidate
    # helped" apart from "this candidate was a harmless no-op" — and
    # therefore what lets the fallback ladder (see build_protocol()'s
    # preferred_orientation stage) usefully try several candidates
    # instead of always keeping the first.
    # Absolute (not relative) ceiling for the Caglioti peak-width terms
    # (Instrument Parameters U, V, W) and the HAP Mustrain terms. Relative
    # drift doesn't work for these the way it does for cell edges: U and V
    # legitimately start at exactly 0.0, so "percent change from start" is
    # undefined/meaningless for them. This is a coarse, deliberately
    # generous absolute sanity ceiling instead — real physical values for
    # these are essentially always well under 100 in GSAS-II's internal
    # units. It exists because Rwp alone doesn't reliably catch this
    # failure mode: on real data, profile_instrument's U/V/W diverged to
    # the millions (U: 0.0 -> 2,260,583; Mustrain later: -> 72,563) while
    # Rwp barely moved at all, because with peaks not yet well-scaled these
    # parameters had little leverage on the fit and were free to wander to
    # nonsense values without any Rwp penalty. The practical symptom was a
    # calculated pattern with no visible peaks at all: enormous U/V/W
    # smears each reflection's intensity across a huge 2-theta range
    # instead of a sharp peak, regardless of how well-scaled the total
    # intensity is.


@dataclass
class StageResult:
    name: str
    status: str          # "ok" | "failed_bounds" | "failed_error" | "skipped"
    rwp_before: float
    rwp_after: float
    detail: str = ""
    # Mirrors Stage.optional — an optional stage whose primary config and
    # every fallback all failed is a legitimate "this enhancement doesn't
    # apply here" outcome (e.g. a phase that genuinely has no preferred
    # orientation), not a run failure. See main()'s failed-stage summary.
    optional: bool = False


@dataclass
class _AttemptOutcome:
    """Result of trying one Stage variant (the primary config or one of its
    fallbacks) — see RefinementRunner._attempt_variant."""
    ok: bool
    rwp_after: float
    error: Exception = None


def cell_drift_ok(start_cell, new_cell, bounds: Bounds) -> bool:
    """start_cell/new_cell: (a, b, c, alpha, beta, gamma)."""
    for s, n in zip(start_cell[:3], new_cell[:3]):
        if s == 0:
            continue
        if abs(n - s) / abs(s) > bounds.max_cell_drift_frac:
            return False
    return True


def profile_params_sane(values, bounds: Bounds) -> bool:
    """
    values: a flat iterable of numbers to sanity-check (Instrument
    Parameters U/V/W and/or HAP Mustrain terms — see RefinementRunner
    callers). Absolute-magnitude check, not relative drift — see the
    Bounds.max_profile_param_abs docstring for why. None/NaN entries are
    skipped rather than treated as failures (a value GSAS-II hasn't
    populated yet isn't the same as a value that has blown up).
    """
    for v in values:
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN check without numpy
            continue
        if abs(v) > bounds.max_profile_param_abs:
            return False
    return True


def rwp_improved_or_stable(rwp_before, rwp_after, bounds: Bounds) -> bool:
    if rwp_after is None or rwp_after != rwp_after:  # NaN check without numpy
        return False
    # No usable baseline to compare against — None (never measured) or NaN
    # (measured, but no residual exists yet, which is exactly what happens
    # on the very first stage: GSAS-II hasn't computed a weighted residual
    # until at least one do_refinements() call has run). Either way there's
    # nothing to have "gotten worse" relative to, so accept and let the
    # stage's own result stand. Missing the NaN half of this check was a
    # real bug: it silently failed stage 1 on every run (NaN comparisons
    # are always False), rolling back the phase's scale-factor refinement
    # before it could ever take effect — which is why the calculated
    # pattern's peaks stayed far too short relative to the observed data.
    if rwp_before is None or rwp_before != rwp_before:
        return True
    if rwp_after <= rwp_before:
        return True
    # allow a small relative worsening (refining new parameters can transiently
    # bump Rwp before the next stage recovers it)
    return (rwp_after - rwp_before) / max(rwp_before, 1e-9) <= bounds.rwp_worsen_tol_frac


def seed_initial_scale(gpx, hist, log) -> None:
    """
    Automates a common manual fix: GSAS-II's raw default phase Scale
    (1.0) can be many orders of magnitude smaller than what the real
    data's intensity actually needs, which starves the very first
    Hessian LM step of a usable gradient — confirmed on real data
    (Data/FeF3): at Scale=1 the calculated peaks topped out at ~600
    counts against a pattern whose real peaks reached ~80,000, and the
    first refinement cycle sent Scale to 1e-12 instead of up toward a
    real value, because a taller but still slightly *misaligned* peak
    looks like it makes the fit worse, not better, to a local optimizer.
    This runs one lightweight forward calculation (0 refined variables)
    at the phases' current Scale values, measures how far off the
    overall magnitude is, and rescales every phase's Scale by the same
    factor before any real refinement stage begins. It's a numerical
    pre-conditioning step, not a scientific judgment call about the
    sample — the real Scale refinement stage still determines the
    actual relative phase fractions from here; this only makes sure it
    starts from a magnitude where the solver has a usable gradient to
    follow instead of a flat/misleading one.
    """
    import numpy as np  # local: only needed on a real run, never for --dry-run

    try:
        gpx.do_refinements([{}])  # 0 variables: just (re)compute Icalc
        yobs = np.asarray(hist.data["data"][1][1], dtype=float)
        ycalc = np.asarray(hist.data["data"][1][3], dtype=float)
        ybkg = np.asarray(hist.data["data"][1][4], dtype=float)
        obs_span = np.nanmax(yobs - ybkg)
        calc_span = np.nanmax(ycalc - ybkg)
        if not (np.isfinite(obs_span) and obs_span > 0
                and np.isfinite(calc_span) and calc_span > 0):
            return
        factor = obs_span / calc_span
        if not (np.isfinite(factor) and factor > 0):
            return
        # Only bother if it's off by an order-of-magnitude-ish amount --
        # smaller differences are exactly what the real Scale refinement
        # stage (background_scale) is for, not something to pre-correct.
        if 0.2 <= factor <= 5.0:
            return
        for phase in gpx.phases():
            hap = phase.data["Histograms"][hist.name]
            hap["Scale"][0] = hap["Scale"][0] * factor
        log(f"  [scale-seed] starting phase Scale(s) rescaled {factor:.3g}x "
            f"(calculated peak heights were "
            f"{'too small' if factor > 1 else 'too large'} relative to the data)")
    except Exception:  # noqa: BLE001 — best-effort preconditioning, never fatal
        pass


# ---------------------------------------------------------------------------
# GSASIIscriptable-facing runner
# ---------------------------------------------------------------------------

# GSASIIscriptable's G2Project.refine() calls G2strMain.Refine(), which
# returns an (IfOK, Rvals) success flag that GSASIIscriptable itself
# discards — an internal solver failure (e.g. an SVD inversion failure
# during the Hessian LM step) is only ever reported by printing
# "***** Refinement error *****" / "SVD inversion failure" to stdout via
# G2Print, not by raising. Confirmed on real data (FeF3_v3/run_v3.log):
# do_refinements() completed "successfully" from Python's point of view
# after an internal SVD inversion failure, and because the failed cycle
# never updated the histogram's residuals, _rwp() just read back the
# stale pre-refinement value, which rwp_improved_or_stable() then judged
# as "unchanged, therefore fine" — silently logging the stage "ok" when
# GSAS-II's own solver had already given up on it. Scanning the captured
# solver output for these known failure phrases is the only way to catch
# this from the scripting layer.
_SOLVER_FAILURE_RE = re.compile(
    r"Refinement failed|SVD inversion failure|Refinement error", re.IGNORECASE
)


class _Tee(io.TextIOBase):
    """Writes to every stream in `streams`; used to capture do_refinements()'s
    stdout for failure-scanning while still printing it live as before."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


class RefinementRunner:
    """
    Thin, deterministic orchestrator around a GSASIIscriptable.G2Project.
    Kept separate from CLI/argparse so test_auto_refine_logic.py can drive
    it against a fake project object without GSAS-II installed.
    """

    def __init__(self, gpx, outdir: Path, bounds: Bounds, log, on_event=None, g2sc=None):
        self.gpx = gpx
        self.outdir = outdir
        self.bounds = bounds
        self.log = log
        # The project's normal, non-checkpoint save path. Every stage
        # re-saves here immediately after writing its checkpoint (see
        # run_stage) — confirmed against the real installed source that
        # G2Project.do_refinements() -> .refine() calls
        # G2strMain.Refine(self.filename, ...), which reads from AND
        # writes back to whichever file G2Project.save() last pointed
        # self.filename at. Without this second save, do_refinements()
        # would auto-save the post-refinement result over the checkpoint
        # we'd just written, destroying the exact pre-stage snapshot
        # rollback depends on — which is exactly what was happening
        # before this fix (confirmed on real data: a stage correctly
        # flagged failed_bounds, followed by "reverting to checkpoint",
        # whose diverged cell parameters then showed up anyway in the
        # final output).
        self.working_path = outdir / "run.gpx"
        # The already-imported GSASIIscriptable module, reused on checkpoint
        # rollback (_reload) instead of re-importing — re-importing would
        # hit the same package/flat layout ambiguity that
        # import_gsasiiscriptable() exists to resolve once, up front.
        self.g2sc = g2sc
        # Optional callback(dict) fired after each stage — structured, GUI-
        # friendly progress reporting alongside the human-readable log lines.
        # None (the default) is a no-op; CLI-only use never needs to set it.
        self.on_event = on_event or (lambda event: None)
        self.results: list = []

    def _rwp(self) -> float:
        hist = self.gpx.histogram(0)
        # residuals is a plain method in some GSASIIscriptable versions and
        # a @property in others (the "'dict' object is not callable" crash
        # this replaced came from assuming method-only on a version where
        # it's a property) — accept either without needing to know which.
        attr = hist.residuals
        r = attr() if callable(attr) else attr
        return float(r.get("wR", r.get("Rwp", float("nan"))))

    def _cell(self, phase_idx: int):
        # G2Phase.get_cell() returns a dict keyed by 'length_a', 'length_b',
        # 'length_c', 'angle_alpha', 'angle_beta', 'angle_gamma', 'volume' —
        # confirmed against the actual installed GSASIIscriptable.py source
        # (not a plain list/tuple, which is what the earlier, doc-summary-
        # based version of this method wrongly assumed — that mismatch is
        # what produced the "KeyError(slice(None, 6, None))" crash).
        phase = self.gpx.phase(phase_idx)
        cell = phase.get_cell()
        return (cell["length_a"], cell["length_b"], cell["length_c"],
                cell["angle_alpha"], cell["angle_beta"], cell["angle_gamma"])

    def _profile_values(self) -> list:
        """
        Flat list of the profile-broadening parameters currently in play —
        Instrument Parameters U/V/W for the histogram, and HAP Mustrain
        terms for every phase — fed to profile_params_sane() after every
        stage (see run_stage). Best-effort: real GSASIIscriptable data
        confirmed via the same real-source-checking this whole module has
        relied on ('Instrument Parameters' is [dict, dict] with values at
        index 1; a phase's per-histogram 'Mustrain' is
        [type, [values...], [refine flags...], ...] with values at index
        1). The FakeProject test harness doesn't model this level of
        GSAS-II's internal structure, so any missing piece just
        contributes nothing here rather than raising — this is a sanity
        net layered on top of the real control flow, not something the
        control flow depends on.
        """
        values = []
        try:
            hist = self.gpx.histogram(0)
            instprm = hist.data["Instrument Parameters"][0]
            for key in ("U", "V", "W"):
                entry = instprm.get(key)
                if entry:
                    values.append(entry[1])
        except (AttributeError, KeyError, IndexError, TypeError):
            pass

        try:
            hist_name = self.gpx.histogram(0).name
        except (AttributeError, IndexError):
            hist_name = None
        if hist_name is not None:
            for phase in self.gpx.phases():
                try:
                    mustrain = phase.data["Histograms"][hist_name]["Mustrain"]
                    values.extend(mustrain[1])
                except (AttributeError, KeyError, IndexError, TypeError):
                    pass
        return values

    def run_stage(self, stage: Stage, stage_idx: int, start_cells: dict) -> StageResult:
        checkpoint = self.outdir / f"checkpoint_{stage_idx:02d}_pre_{stage.name}.gpx"
        self.gpx.save(str(checkpoint))
        rwp_before = self._rwp()
        self.on_event({"event": "stage_start", "index": stage_idx, "name": stage.name})

        # Try the stage's primary configuration first, then — only if that
        # fails its bounds/error check — each fallback in order, reloading
        # the same pre-stage checkpoint before every attempt so a failed
        # attempt's partial mutations never leak into the next one. This is
        # a deterministic, hardcoded retry ladder (see build_protocol()'s
        # profile_microstrain_size fallbacks for the motivating real-data
        # case: joint uniaxial-Mustrain + isotropic-Size refinement is
        # ~99% correlated for some real samples, so the solver sends one
        # parameter to a runaway value while the other collapses toward
        # zero — a strictly *simpler*, less-correlated model often
        # converges fine where the richer one can't), not an unbounded or
        # interactive search — every attempt still goes through the exact
        # same bounds checks as a lone stage would.
        attempts = [stage] + list(stage.fallbacks)
        last_failure = None
        for i, variant in enumerate(attempts):
            if i > 0:
                self._reload(checkpoint)
            # Re-point the project's active save file away from the
            # checkpoint BEFORE refining — see the comment on
            # self.working_path in __init__ for why this is load-bearing,
            # not just tidiness.
            self.gpx.save(str(self.working_path))

            outcome = self._attempt_variant(variant, rwp_before, start_cells,
                                             require_improvement=stage.optional)
            label = stage.name if i == 0 else f"{stage.name} (fallback: {variant.name})"

            if outcome.ok:
                self.log(f"  [{label}] ok (Rwp {rwp_before:.3f} -> {outcome.rwp_after:.3f})")
                detail = "" if i == 0 else f"used fallback {variant.name!r}"
                result = StageResult(stage.name, "ok", rwp_before, outcome.rwp_after, detail,
                                      optional=stage.optional)
                self._emit_result(stage_idx, result)
                return result

            last_failure = (label, outcome)

        # Every attempt (primary + all fallbacks) failed — report using the
        # last one tried and revert to the untouched pre-stage checkpoint.
        label, outcome = last_failure
        status = "failed_error" if outcome.error else "failed_bounds"
        reason = f"raised {outcome.error!r}" if outcome.error else (
            f"failed bounds check (Rwp {rwp_before:.3f} -> {outcome.rwp_after:.3f})")
        self.log(f"  [{label}] {reason} - reverting to checkpoint")
        self._reload(checkpoint)
        result = StageResult(stage.name, status, rwp_before, outcome.rwp_after, str(outcome.error or ""),
                              optional=stage.optional)
        self._emit_result(stage_idx, result)
        return result

    def _attempt_variant(self, stage: Stage, rwp_before: float, start_cells: dict,
                          require_improvement: bool = False) -> _AttemptOutcome:
        """Applies one Stage's refinement settings to self.gpx (already
        pointed at the right pre-stage state by the caller) and reports
        whether the result passes every bounds check. Never raises —
        a solver exception is captured into the returned outcome instead,
        same as a bounds failure, so run_stage's fallback loop can treat
        both uniformly.

        require_improvement: set for an optional Stage (see
        Bounds.min_optional_improvement_frac's docstring) — a plain
        "didn't get worse" bar isn't enough there, since a candidate with
        no real effect on the sample (e.g. the wrong preferred-orientation
        axis) typically doesn't make Rwp worse either, it just settles
        back to a no-op value."""
        try:
            # Pref.Ori. targeting: every phase by default, or just one —
            # see Stage.pref_ori_phase_index's docstring. Only affects
            # the axis poke and set_hap below, never set_hist/set_phase
            # (those apply to every phase/histogram regardless, same as
            # any other stage).
            pref_ori_phases = (
                [self.gpx.phase(stage.pref_ori_phase_index)]
                if stage.pref_ori_phase_index is not None
                else list(self.gpx.phases())
            )

            if stage.pref_ori_axis is not None:
                # Not one of set_HAP_refinements()'s recognized keys — see
                # Stage.pref_ori_axis's docstring — so poke the axis in
                # directly before turning the refine flag on below.
                hist_name = self.gpx.histogram(0).name
                for p in pref_ori_phases:
                    p.data["Histograms"][hist_name]["Pref.Ori."][3] = list(stage.pref_ori_axis)
            if stage.clear_hap:
                # Applied before set_hap/set_hist/set_phase below — see
                # Stage.clear_hap's docstring.
                for p in self.gpx.phases():
                    p.clear_HAP_refinements(stage.clear_hap)
            if stage.set_hist:
                for h in self.gpx.histograms():
                    h.set_refinements(stage.set_hist)
            if stage.set_phase:
                for p in self.gpx.phases():
                    p.set_refinements(stage.set_phase)
            if stage.set_hap:
                target_phases = pref_ori_phases if "Pref.Ori." in stage.set_hap else self.gpx.phases()
                for p in target_phases:
                    p.set_HAP_refinements(stage.set_hap)

            solver_output = io.StringIO()
            with contextlib.redirect_stdout(_Tee(sys.stdout, solver_output)):
                self.gpx.do_refinements([{}])
            solver_text = solver_output.getvalue()
            failure = _SOLVER_FAILURE_RE.search(solver_text)
            if failure:
                # See _SOLVER_FAILURE_RE docstring above: do_refinements()
                # returns normally even when GSAS-II's own solver gave up
                # internally, so this is the only way to catch it here.
                raise RuntimeError(
                    f"GSAS-II solver reported an internal failure "
                    f"({failure.group(0)!r}) that do_refinements() did not "
                    f"raise on"
                )
            rwp_after = self._rwp()
        except Exception as exc:  # noqa: BLE001 — deliberately broad: any solver
            # failure here means "this attempt failed, try the next one /
            # revert", not "crash the run"
            return _AttemptOutcome(ok=False, rwp_after=float("nan"), error=exc)

        ok = rwp_improved_or_stable(rwp_before, rwp_after, self.bounds)
        if ok and require_improvement:
            ok = (rwp_before == rwp_before and rwp_before > 0  # not NaN/zero
                  and (rwp_before - rwp_after) / rwp_before
                  >= self.bounds.min_optional_improvement_frac)
        if ok:
            for i in range(len(self.gpx.phases())):
                if i in start_cells:
                    ok = ok and cell_drift_ok(start_cells[i], self._cell(i), self.bounds)
        if ok:
            # Catches the failure mode cell-drift alone misses: U/V/W and
            # Mustrain wandering to unphysical magnitudes (or, as seen on
            # real two-phase lab data, Mustrain and Size trading off
            # against each other — one runs away while the other
            # collapses toward zero, ~99% correlated) while the cell
            # stays put — see Bounds.max_profile_param_abs and
            # _profile_values() for the real-data evidence that motivated
            # this.
            ok = ok and profile_params_sane(self._profile_values(), self.bounds)
        return _AttemptOutcome(ok=ok, rwp_after=rwp_after, error=None)

    def _emit_result(self, stage_idx: int, result: "StageResult"):
        self.on_event({
            "event": "stage_result",
            "index": stage_idx,
            "name": result.name,
            "status": result.status,
            "rwp_before": result.rwp_before,
            "rwp_after": result.rwp_after,
            "detail": result.detail,
            "optional": result.optional,
        })

    def _reload(self, checkpoint: Path):
        # GSASIIscriptable projects are mutated in place; the deterministic
        # rollback is to reopen the last known-good checkpoint file rather
        # than trying to hand-unwind partial in-memory state. Reuses the
        # module handed in at construction (see g2sc above) rather than
        # re-importing, since a fresh `import GSASIIscriptable` can fail
        # depending on which of the two GSAS-II layouts is installed.
        if self.g2sc is None:
            import GSASIIscriptable as g2sc  # test/mock path — see test_auto_refine_logic.py
            self.g2sc = g2sc
        self.gpx = self.g2sc.G2Project(str(checkpoint))

    def run(self, stages: list) -> list:
        start_cells = {i: self._cell(i) for i in range(len(self.gpx.phases()))}
        for idx, stage in enumerate(stages, start=1):
            result = self.run_stage(stage, idx, start_cells)
            self.results.append(result)
        return self.results


# ---------------------------------------------------------------------------
# Plot-data export — plain CSV/dict output for the GUI (or anything else)
# to read and plot, so GSAS-II-specific data access lives in exactly one
# place instead of being duplicated in the GUI. Column layout confirmed
# against the real installed GSASIIscriptable.py source: a histogram's
# hist.data['data'][1] is [X, Yobs, Yweight, Ycalc, Ybkg, Ydiff] — the same
# six arrays G2PwdrData's own .plot() method uses.
# ---------------------------------------------------------------------------

_DATA_COLS = {"two_theta": 0, "y_obs": 1, "y_weight": 2, "y_calc": 3, "y_bkg": 4, "y_diff": 5}


def _mask_to_plain(arr):
    """Numpy masked arrays show up in GSAS-II histogram data; convert to a
    plain array with masked points as NaN so csv.writer doesn't choke on
    numpy.ma.masked sentinels."""
    import numpy as np  # local: only needed on a real run, never for --dry-run

    if np.ma.isMaskedArray(arr):
        return np.ma.filled(arr.astype(float), np.nan)
    return np.asarray(arr, dtype=float)


def export_histogram_csv(hist, path: Path, colnames: list) -> None:
    """Writes the requested columns (names from _DATA_COLS) from a
    histogram's data arrays to a CSV at `path`, one row per point."""
    raw = hist.data["data"][1]
    arrays = [_mask_to_plain(raw[_DATA_COLS[name]]) for name in colnames]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(colnames)
        for row in zip(*arrays):
            w.writerow(f"{v:.6g}" for v in row)


def get_phase_cells(gpx) -> dict:
    """Returns {phase_name: {cell dict}} and {phase_name: {esd dict}} for
    every phase in the project, using G2Phase.get_cell()/get_cell_and_esd()
    — both confirmed to return dicts keyed by 'length_a'..'angle_gamma'/
    'volume' against the real installed source (see RefinementRunner._cell
    for the bug this caught)."""
    cells, esds = {}, {}
    for phase in gpx.phases():
        cells[phase.name] = dict(phase.get_cell())
        try:
            _c, esd = phase.get_cell_and_esd()
            esds[phase.name] = dict(esd)
        except Exception:  # noqa: BLE001 — esd is a nice-to-have, never fatal
            esds[phase.name] = {}
    return cells, esds


def assess_fit_quality(hist) -> dict:
    """
    Automates the single most important manual check a scientist does
    before trusting a refinement's Rwp at all: does the calculated
    pattern actually track the real one, peak for peak? A converged,
    in-bounds Rwp is not proof of that — confirmed directly on real data
    (Data/FeF3, before its instrument/CIF mismatch was found): Rwp sat at
    a plausible-looking ~10.6% for five straight stages while the
    calculated pattern's correlation with the observed one was ~0.02,
    indistinguishable from no relationship at all (the model's tallest
    calculated peaks landed where the real data was completely flat).
    Every other bounds check in this module (Rwp trend, cell drift,
    profile-parameter sanity) can look perfectly fine in exactly that
    situation — this is the one check that would have caught it
    immediately, and it's cheap enough to run on every single output
    rather than requiring a human to eyeball each plot. Returns a dict
    (always present in summary.json) with a correlation coefficient and
    a `needs_review` flag a batch process can filter on directly instead
    of a person opening every plot.

    Only the *active fit range* (hist.data["Limits"]'s applied [tmin,
    tmax] — see --tmin/--tmax) is considered. Confirmed as a real bug on
    real data: with --tmin/--tmax trimming applied, GSAS-II still reports
    y_calc/y_bkg across the *entire* raw pattern, but leaves everything
    outside the fit range at a flat, never-refined value (y_calc-y_bkg
    was exactly 0.0 there) — scoring the whole pattern measured a
    correlation of 0.375 (wrongly flagged "needs review") when the
    correlation inside the actual fit range was 0.94, matching every
    other diagnostic for that same run.
    """
    import numpy as np  # local: only needed on a real run, never for --dry-run

    try:
        x = np.asarray(hist.data["data"][1][0], dtype=float)
        yobs = np.asarray(hist.data["data"][1][1], dtype=float)
        ycalc = np.asarray(hist.data["data"][1][3], dtype=float)
        ybkg = np.asarray(hist.data["data"][1][4], dtype=float)
        mask = np.isfinite(x) & np.isfinite(yobs) & np.isfinite(ycalc) & np.isfinite(ybkg)
        try:
            _limits_full, limits_applied = hist.data["Limits"]
            tmin, tmax = limits_applied
            mask &= (x >= tmin) & (x <= tmax)
        except (KeyError, TypeError, ValueError):
            pass  # no Limits info available — fall back to the whole pattern
        corr = float(np.corrcoef((yobs - ybkg)[mask], (ycalc - ybkg)[mask])[0, 1])
    except Exception:  # noqa: BLE001 — the fit itself already succeeded or
        # failed independently of this check; never let it crash the run
        return {"calc_obs_correlation": None, "needs_review": True,
                "reason": "could not compute calc/obs correlation"}

    if corr != corr:  # NaN (e.g. a totally flat calculated pattern)
        return {"calc_obs_correlation": None, "needs_review": True,
                "reason": "calc/obs correlation is undefined (flat calculated pattern)"}
    # Below ~0.5, "converged" is not the same as "correct" — see this
    # function's docstring: real bad-model data measured ~0.02, real good
    # fits in this project measured 0.91-0.96, so 0.5 sits well clear of
    # both without being close enough to a genuine borderline case to
    # matter in practice.
    needs_review = corr < 0.5
    reason = ("calculated pattern does not track the observed one "
              "(wrong phase, wrong wavelength, or severe preferred "
              "orientation are the usual causes)") if needs_review else ""
    return {"calc_obs_correlation": corr, "needs_review": needs_review, "reason": reason}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Turnkey GSAS-II Rietveld refinement runner (GSASIIscriptable-based).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pattern", required=True, type=Path,
                    help="Path to the pre-integrated 1D pattern (.xy/.dat/.fxye/.txt).")
    p.add_argument("--instprm", required=True, type=Path,
                    help="Path to the GSAS instrument parameter file (.prm/.instprm).")
    p.add_argument("--cif", required=True, action="append", type=Path,
                    help="Path to a phase CIF. Repeat --cif for multi-phase samples.")
    p.add_argument("--outdir", required=True, type=Path,
                    help="Directory for checkpoints, exported results, and the run log.")
    p.add_argument("--gsasii-path", type=Path, default=None,
                    help="Path to the local GSAS-II install (dir containing "
                         "GSASIIscriptable.py). Required unless --dry-run.")
    p.add_argument("--refine-atoms", action="store_true",
                    help="Add the optional atom-position/thermal-parameter stage. "
                         "Off by default — atom refinement without restraints can be "
                         "unstable and is a modeling decision, not a default.")
    p.add_argument("--lebail", action="store_true",
                    help="Extract each reflection's intensity directly from the data "
                         "(LeBail) instead of computing it from the phase's atom "
                         "positions. Use this when cell/profile refinement won't "
                         "converge because the calculated pattern's peak intensities "
                         "don't resemble the data (wrong/incomplete/uncertain "
                         "structural model) even though the phase and instrument "
                         "file are otherwise right for this data — LeBail lets cell, "
                         "profile, and background still refine against the real peak "
                         "positions/shapes without depending on the atom-position "
                         "structure factor being correct. Mutually exclusive with "
                         "--refine-atoms (atom positions aren't meaningful to refine "
                         "when reflection intensities aren't being computed from them).")
    p.add_argument("--max-cell-drift", type=float, default=0.15,
                    help="Max fractional change allowed in a,b,c before a stage is "
                         "considered diverged and rolled back. Default 0.15 (15%%).")
    p.add_argument("--tmin", type=float, default=None,
                    help="Lower 2-theta (or TOF) bound to fit against - trims the tail "
                         "below this value out of the refinement. Must be given together "
                         "with --tmax. Omit both to use the instrument/data file's full "
                         "range unchanged.")
    p.add_argument("--tmax", type=float, default=None,
                    help="Upper 2-theta (or TOF) bound to fit against. See --tmin.")
    p.add_argument("--dry-run", action="store_true",
                    help="Validate arguments and input files, print the refinement "
                         "plan, and exit. Does not import GSASIIscriptable.")
    p.add_argument("--emit-events", action="store_true",
                    help="Also print one JSON line per event (plan/stage_start/"
                         "stage_result/done) to stdout, alongside the normal human-"
                         "readable log. Intended for a GUI or other tool driving this "
                         "script as a subprocess — see gsas2_gui.py.")
    return p.parse_args(argv)


def check_inputs_exist(args) -> list:
    problems = []
    for label, path in [("--pattern", args.pattern), ("--instprm", args.instprm)]:
        if not path.is_file():
            problems.append(f"{label}: no such file: {path}")
    for path in args.cif:
        if not path.is_file():
            problems.append(f"--cif: no such file: {path}")
    return problems


def print_plan(args, stages: list, emit=None):
    print("Refinement plan")
    print("----------------")
    print(f"  pattern : {args.pattern}")
    print(f"  instprm : {args.instprm}")
    for c in args.cif:
        print(f"  phase   : {c}")
    print(f"  outdir  : {args.outdir}")
    print(f"  bounds  : max_cell_drift={args.max_cell_drift:.0%}")
    if args.lebail:
        print("  lebail  : on (reflection intensities extracted from data, not atoms)")
    print("  stages  :")
    for i, s in enumerate(stages, 1):
        tag = " (optional)" if s.optional else ""
        print(f"    {i}. {s.name}{tag}")
    if emit:
        emit({
            "event": "plan",
            "pattern": str(args.pattern),
            "instprm": str(args.instprm),
            "phases": [str(c) for c in args.cif],
            "outdir": str(args.outdir),
            "stages": [{"index": i, "name": s.name, "optional": s.optional}
                       for i, s in enumerate(stages, 1)],
        })


def main(argv=None):
    args = parse_args(argv)
    stages = build_protocol(args.refine_atoms, lebail=args.lebail, n_phases=len(args.cif))

    problems = check_inputs_exist(args)
    if (args.tmin is None) != (args.tmax is None):
        problems.append("--tmin and --tmax must be given together (or neither).")
    elif args.tmin is not None and args.tmin >= args.tmax:
        problems.append(f"--tmin ({args.tmin}) must be less than --tmax ({args.tmax}).")
    if args.lebail and args.refine_atoms:
        problems.append("--lebail and --refine-atoms are mutually exclusive: LeBail "
                         "extracts reflection intensities directly from the data, so "
                         "there's nothing for atom-position refinement to act on.")
    if problems:
        for msg in problems:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    def emit(event: dict):
        if args.emit_events:
            print(json.dumps(event), flush=True)

    if args.dry_run:
        print_plan(args, stages, emit=emit)
        print("\n--dry-run: no refinement performed, GSASIIscriptable not imported.")
        return 0

    if args.gsasii_path is None:
        print("ERROR: --gsasii-path is required for a real run (omit only with --dry-run).",
              file=sys.stderr)
        return 2

    try:
        G2sc = import_gsasiiscriptable(args.gsasii_path)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    log_lines = []

    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    print_plan(args, stages, emit=emit)

    # Project setup (loading the pattern, instrument file, and phase CIFs)
    # isn't covered by RefinementRunner's per-stage error handling — guard
    # it here too, so a bad/unexpected input file produces a clean "done"
    # event and a readable error instead of a raw traceback. This matters
    # most for a GUI or other tool driving this script as a subprocess.
    try:
        project_path = args.outdir / "run.gpx"
        gpx = G2sc.G2Project(newgpx=str(project_path))
        hist = gpx.add_powder_histogram(str(args.pattern), str(args.instprm))
        if hist is None:
            # Some GSASIIscriptable versions return the new histogram
            # object from add_powder_histogram, others return None and
            # expect it to be looked up afterward — handle both rather
            # than assume, same as the import-layout/residuals fixes above.
            hist = gpx.histogram(0)

        # Export the full raw pattern (unaffected by any --tmin/--tmax trim
        # below — hist.data['Limits'] only restricts what feeds the least-
        # squares fit, the underlying X/Yobs arrays always hold the whole
        # measured range) as early as possible, so it's available for the
        # GUI to plot even if a later step (a bad phase CIF, a solver
        # crash) stops the run before refinement finishes.
        export_histogram_csv(hist, args.outdir / "pattern_raw.csv", ["two_theta", "y_obs"])

        # Explicitly force the histogram's own "Sample Parameters: Scale"
        # refinement flag off, rather than relying on it defaulting to off.
        # It doesn't, at least on the installed version this was diagnosed
        # against: a freshly-loaded histogram already has it enabled, and
        # it is mathematically 100% degenerate with the phase's HAP
        # "Scale" (phase fraction) that build_protocol()'s background_scale
        # stage turns on — refining both at once produced an SVD-singular
        # Hessian (confirmed via GSAS-II's own solver log: "0:0:Scale and
        # :0:Scale (@100.00%)" correlated, "Maximum shift/esd = 625")
        # and the phase's real intensity scale never actually got fit as a
        # result. clear_refinements() is a no-op if it's already off, so
        # this is safe regardless of the installed version's default.
        hist.clear_refinements({"Sample Parameters": ["Scale"]})

        if args.tmin is not None:
            hist.set_refinements({"Limits": [args.tmin, args.tmax]})
            log(f"  [limits] fitting restricted to [{args.tmin}, {args.tmax}]")

        for cif in args.cif:
            gpx.add_phase(str(cif), phasename=cif.stem, histograms=gpx.histograms())

        if args.lebail:
            # See --lebail's help text and build_protocol()'s lebail
            # docstring: each reflection's intensity is extracted directly
            # from the data every refinement cycle (GSAS-II's own solver
            # already does this internally once LeBail is on — no separate
            # extraction pass needed from the scripting layer), instead of
            # being computed from the phase's atom positions.
            #
            # G2Phase.set_refinements() (phase-level), NOT
            # set_HAP_refinements() — confirmed against the installed
            # source: "LeBail" is in G2Phase.is_valid_refinement_key()'s
            # list, not is_valid_HAP_refinement_key()'s. Calling
            # set_HAP_refinements({"LeBail": True}) doesn't raise (it just
            # silently doesn't match any of that method's known keys) —
            # it's a no-op that looks like it worked. Confirmed on real
            # data: the resulting gpx had LeBail still False, and five
            # stages of refinement left Rwp completely unmoved (~10.61%
            # throughout) because reflection intensities were still being
            # computed from the phase's atoms as if --lebail had never
            # been passed.
            for p in gpx.phases():
                p.set_refinements({"LeBail": True})
            log("  [lebail] reflection intensities will be extracted from the "
                "data, not computed from atom positions")
        else:
            # Not meaningful under --lebail: LeBail extracts each
            # reflection's intensity directly from the data, so a phase
            # Scale factor isn't the thing standing between a flat
            # starting guess and the real peak heights the way it is in
            # ordinary Rietveld mode.
            seed_initial_scale(gpx, hist, log)

        gpx.save()

        bounds = Bounds(max_cell_drift_frac=args.max_cell_drift)
        runner = RefinementRunner(gpx, args.outdir, bounds, log, on_event=emit, g2sc=G2sc)
        results = runner.run(stages)
    except Exception as exc:  # noqa: BLE001 — setup/solver failures should end
        # the run cleanly (readable error + a "done" event), not crash out
        # with a raw traceback that a GUI subprocess consumer can't parse.
        msg = f"ERROR: refinement setup failed: {exc!r}"
        print(msg, file=sys.stderr)
        log_lines.append(msg)
        (args.outdir / "run.log").write_text("\n".join(log_lines), encoding="utf-8")
        emit({
            "event": "done", "ok": False, "failed_stages": ["setup"],
            "summary_path": None, "refined_cifs": [], "outdir": str(args.outdir),
            "error": str(exc),
        })
        return 1

    final_gpx = args.outdir / "final.gpx"
    runner.gpx.save(str(final_gpx))

    # Re-fetch the histogram/phases from runner.gpx (not the earlier `hist`
    # variable) — a checkpoint rollback during refinement can have replaced
    # runner.gpx with a freshly-reloaded G2Project, whose objects aren't the
    # same Python instances even though they represent the same data.
    final_hist = runner.gpx.histogram(0)
    export_histogram_csv(final_hist, args.outdir / "fit_final.csv",
                          ["two_theta", "y_obs", "y_calc", "y_bkg", "y_diff"])
    cells, cell_esds = get_phase_cells(runner.gpx)
    limits_full, limits_applied = final_hist.data["Limits"]
    fit_quality = assess_fit_quality(final_hist)

    summary = {
        "pattern": str(args.pattern),
        "instprm": str(args.instprm),
        "phases": [str(c) for c in args.cif],
        "stages": [
            {"name": r.name, "status": r.status, "rwp_before": r.rwp_before,
             "rwp_after": r.rwp_after, "detail": r.detail, "optional": r.optional}
            for r in results
        ],
        # The *actual* current Rwp of runner.gpx, not results[-1].rwp_after
        # — those aren't always the same thing. A failed/reverted stage's
        # rwp_after is only informational (what that rejected attempt
        # measured before being rolled back); if the LAST stage in the
        # sequence fails, results[-1].rwp_after would report that
        # rejected, discarded attempt's Rwp instead of the real one the
        # saved project ends up with. Trivial before every stage was
        # mandatory, but preferred_orientation being optional and last
        # makes "last stage failed and reverted, run still succeeded"
        # a routine, not rare, case.
        "final_rwp": runner._rwp(),
        "fit_quality": fit_quality,
        "cells": cells,
        "cell_esds": cell_esds,
        "limits_full_range": list(limits_full),
        "limits_applied": list(limits_applied),
        "pattern_raw_csv": str(args.outdir / "pattern_raw.csv"),
        "fit_final_csv": str(args.outdir / "fit_final.csv"),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if fit_quality["needs_review"]:
        log(f"  [fit-quality] NEEDS REVIEW: {fit_quality['reason']} "
            f"(calc/obs correlation: {fit_quality['calc_obs_correlation']})")

    with (args.outdir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "status", "rwp_before", "rwp_after"])
        for r in results:
            w.writerow([r.name, r.status, r.rwp_before, r.rwp_after])

    # export_CIF() reads project covariance data that may not be fully
    # populated (e.g. every stage failed and rolled back, or a phase never
    # picked up refined values) — guard per-phase so one bad export can't
    # crash a run that otherwise completed and already has a summary worth
    # keeping. A skipped export is noted in the log, not silently dropped.
    refined_cifs = []
    for i, phase in enumerate(runner.gpx.phases()):
        cif_path = args.outdir / f"refined_phase_{i}_{phase.name}.cif"
        try:
            phase.export_CIF(str(cif_path))
            refined_cifs.append(str(cif_path))
        except Exception as exc:  # noqa: BLE001
            log(f"  [export] could not export refined CIF for phase {phase.name!r}: {exc!r}")
    summary["refined_cifs"] = refined_cifs
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (args.outdir / "run.log").write_text("\n".join(log_lines), encoding="utf-8")

    # An optional stage (atoms, preferred_orientation) whose primary config
    # and every fallback all failed is a legitimate "doesn't apply to this
    # phase" outcome, not a run failure — see StageResult.optional. It's
    # still visible in results/summary.json/run.log either way, just not
    # counted against the run's overall pass/fail.
    failed = [r for r in results if r.status != "ok" and not r.optional]
    # needs_review folds into the same pass/fail signal as a failed stage
    # — see assess_fit_quality()'s docstring: a plausible Rwp with a
    # calculated pattern that doesn't actually track the data is exactly
    # the failure mode a batch process (running unattended across many
    # samples, with no one watching each plot) most needs surfaced by
    # exit code / the "ok" event field alone, not buried in a file only a
    # human who thinks to open it would catch.
    ok = not failed and not fit_quality["needs_review"]
    emit({
        "event": "done",
        "ok": ok,
        "failed_stages": [r.name for r in failed],
        "needs_review": fit_quality["needs_review"],
        "summary_path": str(args.outdir / "summary.json"),
        "refined_cifs": refined_cifs,
        "outdir": str(args.outdir),
    })

    if failed:
        print(f"\nCompleted with {len(failed)} stage(s) not converged: "
              f"{', '.join(r.name for r in failed)}")
        return 1
    if fit_quality["needs_review"]:
        print(f"\nCompleted with all stages 'ok', but flagged for review: "
              f"{fit_quality['reason']} "
              f"(calc/obs correlation: {fit_quality['calc_obs_correlation']:.3f})")
        return 1

    print("\nRefinement complete. See summary.json / summary.csv / refined_phase_*.cif in", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
