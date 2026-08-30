#!/usr/bin/env python3
"""
gsas2_swarm_worker.py — evaluates ONE candidate starting point for the
Size/Mustrain continuous parameters and reports the Rwp GSAS-II's own
Hessian LM refinement converges to from there.

This is the CPU-side "fitness function" gsas2_swarm_optimize.py's swarm
calls once per particle per generation, in parallel (see that script's
module docstring for why the swarm's own bookkeeping stays on plain CPU
numpy rather than needing a GPU — this evaluation, a full GSAS-II
refinement call, is what actually costs time, not the swarm math).

Loads a checkpoint .gpx (produced by gsas2_auto_refine.py — normally
checkpoint_0N_pre_profile_microstrain_size.gpx from a real run, so the
swarm starts from the same already-converged Background/Scale/Cell/
DisplaceX/U-V-W state the deterministic pipeline would hand off from),
sets ONE candidate starting point for every phase's isotropic Size and
uniaxial Mustrain (equatorial + axial), refines once, and reports the
result. The checkpoint file itself is never written to — this script
saves its own working copy in --outdir, so many workers can safely
evaluate different particles against the same checkpoint in parallel
(the same pattern gsas2_candidate_sweep.py and gsas2_batch_run.py
already use: every parallel unit gets its own untouched-by-others
output location).

Prints exactly one JSON line to stdout:
    {"rwp": float|null, "sane": bool, "error": str|null,
     "final": {"<phase index>": {"size":.., "mustrain_eq":.., "mustrain_ax":..}, ...} | null}

`sane` reuses gsas2_auto_refine.py's own bounds checks (Rwp trend, cell
drift, profile-parameter sanity) — the same standard every stage of the
deterministic pipeline is held to, so a particle that "improves" Rwp by
sending Mustrain to a runaway value is scored exactly as unsound here as
it would be there, not treated as a free win just because a human isn't
watching this one.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gsas2_auto_refine import (  # noqa: E402
    Bounds,
    _SOLVER_FAILURE_RE,
    _Tee,
    cell_drift_ok,
    import_gsasiiscriptable,
    profile_params_sane,
    rwp_improved_or_stable,
)
from gsas2_swarm_logic import HIGH_ANGLE_CUTOFF_PARAM, LOW_ANGLE_CUTOFF_PARAM  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate one Size/Mustrain starting point against a checkpoint .gpx "
                    "and report the Rwp GSAS-II's own refinement converges to.",
    )
    p.add_argument("--checkpoint", required=True, type=Path,
                    help="A checkpoint_0N_pre_profile_microstrain_size.gpx from a real "
                         "gsas2_auto_refine.py run. Read-only — never modified.")
    p.add_argument("--gsasii-path", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path,
                    help="Unique scratch directory for this one evaluation's working .gpx "
                         "— must not be shared with any other concurrently-running worker.")
    p.add_argument("--values", required=True,
                    help='JSON dict: {"<phase index>": {"size": .., "mustrain": ..}, ..., '
                         '"low_angle_cutoff": .., "high_angle_cutoff": ..} — one entry per '
                         'phase, plus optional whole-histogram "low_angle_cutoff"/'
                         '"high_angle_cutoff" (degrees of data to discard from the start/end '
                         'of the fit range). Size is always isotropic; Mustrain is isotropic '
                         '(one "mustrain" value) if that key is present, or uniaxial (two '
                         'values, "mustrain_eq"/"mustrain_ax") if those are present instead — '
                         'see gsas2_swarm_logic.build_param_specs\' mustrain_type for why '
                         'isotropic is the default.')
    return p.parse_args(argv)


def _rwp(hist) -> float:
    # Mirrors RefinementRunner._rwp() in gsas2_auto_refine.py exactly —
    # see its docstring-comment for why `residuals` needs the
    # method-or-property check.
    attr = hist.residuals
    r = attr() if callable(attr) else attr
    return float(r.get("wR", r.get("Rwp", float("nan"))))


def peak_amplitude_error(hist) -> "float | None":
    """
    Mean RELATIVE intensity mismatch across every calculated reflection
    from every phase in the histogram, weighting every peak EQUALLY
    regardless of its size — unlike Rwp, which is intensity-weighted and
    so is dominated by whichever peak is largest (confirmed on real FeF3
    data: a fit can nail one huge peak, drag Rwp down, while several
    smaller peaks are comparatively poorly matched — Rwp alone doesn't
    surface that). This is a secondary signal used to break near-ties
    between candidates with similar Rwp — see gsas2_swarm_optimize.py's
    --tie-break-rwp-margin — not a replacement for Rwp itself.

    Per-reflection Iobs/Icalc are derived from GSAS-II's own reflection
    list per its documented column layout (docs/source/objvarorg.rst,
    "Powder Reflection Data Structure"): Iobs = Icorr * Fobs^2, Icalc =
    Icorr * Fcalc^2, where Icorr is the reflection's intensity-correction
    column. Column indices shift by one for 3+1 superspace phases (the
    'Super' flag) — both are handled.

    Returns None only if the histogram has no phases/reflections to
    compare at all (never raises).
    """
    reflection_lists = hist.reflections()
    relative_errors = []
    for refl in reflection_lists.values():
        reflist = np.asarray(refl.get("RefList", []), dtype=float)
        if reflist.size == 0:
            continue
        fobs2_col, fcalc2_col, icorr_col = (9, 10, 12) if refl.get("Super") else (8, 9, 11)
        iobs = reflist[:, icorr_col] * reflist[:, fobs2_col]
        icalc = reflist[:, icorr_col] * reflist[:, fcalc2_col]
        # Normalize by whichever of Iobs/Icalc is larger — bounds the
        # relative error even when one side is near zero (a weak-to-
        # absent reflection), rather than dividing by a near-zero Icalc
        # and blowing up.
        denom = np.maximum(np.abs(iobs), np.abs(icalc))
        denom = np.where(denom == 0, 1.0, denom)  # both exactly 0 -> perfect match, error 0
        relative_errors.extend((np.abs(iobs - icalc) / denom).tolist())

    if not relative_errors:
        return None
    return float(np.mean(relative_errors))


def _cell(gpx, phase_idx: int):
    cell = gpx.phase(phase_idx).get_cell()
    return (cell["length_a"], cell["length_b"], cell["length_c"],
            cell["angle_alpha"], cell["angle_beta"], cell["angle_gamma"])


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        values = json.loads(args.values)
        if not isinstance(values, dict) or not values:
            raise ValueError("must be a non-empty JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"rwp": None, "sane": False, "error": f"bad --values: {exc}", "final": None}))
        return 2

    try:
        G2sc = import_gsasiiscriptable(args.gsasii_path)
    except ImportError as exc:
        print(json.dumps({"rwp": None, "sane": False, "error": str(exc), "final": None}))
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    bounds = Bounds()

    phase_values = {k: v for k, v in values.items()
                     if k not in (LOW_ANGLE_CUTOFF_PARAM, HIGH_ANGLE_CUTOFF_PARAM)}
    low_angle_cutoff = values.get(LOW_ANGLE_CUTOFF_PARAM)
    high_angle_cutoff = values.get(HIGH_ANGLE_CUTOFF_PARAM)

    try:
        gpx = G2sc.G2Project(str(args.checkpoint))
        hist = gpx.histogram(0)
        start_cells = {i: _cell(gpx, i) for i in range(len(gpx.phases()))}
        rwp_before = _rwp(hist)

        limits_applied = None
        if low_angle_cutoff is not None or high_angle_cutoff is not None:
            # hist.data["Limits"] is (full_range, currently_applied_range)
            # — trim from the pattern's own natural start/end, keeping
            # whichever bound wasn't given as whatever is already in
            # effect (respecting any earlier trim from the deterministic
            # pipeline this checkpoint came from), same mechanism
            # gsas2_auto_refine.py's --tmin/--tmax use. See build_param_
            # specs' low_angle_cutoff_bounds/high_angle_cutoff_bounds
            # docstring for why Rwp isn't strictly comparable across
            # different cutoff choices.
            limits_full, limits_currently_applied = hist.data["Limits"]
            new_tmin = (limits_full[0] + float(low_angle_cutoff)
                        if low_angle_cutoff is not None else limits_currently_applied[0])
            new_tmax = (limits_full[1] - float(high_angle_cutoff)
                        if high_angle_cutoff is not None else limits_currently_applied[1])
            if new_tmin >= new_tmax:
                raise RuntimeError(f"low_angle_cutoff={low_angle_cutoff}/"
                                    f"high_angle_cutoff={high_angle_cutoff} would leave no fit "
                                    f"range at all (tmin={new_tmin}, tmax={new_tmax})")
            hist.set_refinements({"Limits": [new_tmin, new_tmax]})
            limits_applied = [new_tmin, new_tmax]

        for phase_idx_str, v in phase_values.items():
            phase = gpx.phase(int(phase_idx_str))
            hap = phase.data["Histograms"][hist.name]
            hap["Size"][0] = "isotropic"
            hap["Size"][1][0] = float(v["size"])
            hap["Size"][2][0] = True
            # "mustrain" (isotropic, one free value) vs "mustrain_eq"/
            # "mustrain_ax" (uniaxial, two) — which key(s) are present
            # tells us which gsas2_swarm_logic.build_param_specs()
            # mustrain_type built this candidate. Shapes verified against
            # a real project via phase.set_HAP_refinements(): isotropic
            # sets only value[0]/refine[0]; uniaxial sets both [0] and [1].
            if "mustrain" in v:
                hap["Mustrain"][0] = "isotropic"
                hap["Mustrain"][1][0] = float(v["mustrain"])
                hap["Mustrain"][2][0] = True
            else:
                hap["Mustrain"][0] = "uniaxial"
                hap["Mustrain"][1][0] = float(v["mustrain_eq"])
                hap["Mustrain"][1][1] = float(v["mustrain_ax"])
                hap["Mustrain"][2][0] = True
                hap["Mustrain"][2][1] = True

        # Re-point the project's save file to our OWN working copy before
        # refining, not the checkpoint — do_refinements() reads from and
        # writes back to whatever file is currently pointed at, so
        # refining without this would corrupt the shared, read-only
        # checkpoint out from under every other concurrently-running
        # worker (the identical concurrency hazard RefinementRunner.
        # working_path exists to avoid in the main pipeline).
        working_path = args.outdir / "working.gpx"
        gpx.save(str(working_path))

        solver_output = io.StringIO()
        with contextlib.redirect_stdout(_Tee(sys.stdout, solver_output)):
            gpx.do_refinements([{}])
        solver_text = solver_output.getvalue()
        failure = _SOLVER_FAILURE_RE.search(solver_text)
        if failure:
            raise RuntimeError(f"GSAS-II solver reported an internal failure: {failure.group(0)!r}")

        rwp_after = _rwp(hist)
        peak_error = peak_amplitude_error(hist)

        sane = rwp_improved_or_stable(rwp_before, rwp_after, bounds)
        if sane:
            for i in range(len(gpx.phases())):
                if i in start_cells:
                    sane = sane and cell_drift_ok(start_cells[i], _cell(gpx, i), bounds)
        if sane:
            profile_values = []
            for phase_idx_str in phase_values:
                phase = gpx.phase(int(phase_idx_str))
                hap = phase.data["Histograms"][hist.name]
                profile_values.extend(hap["Mustrain"][1])
                profile_values.extend(hap["Size"][1])
            sane = sane and profile_params_sane(profile_values, bounds)

        final = {}
        for phase_idx_str, v in phase_values.items():
            phase = gpx.phase(int(phase_idx_str))
            hap = phase.data["Histograms"][hist.name]
            if "mustrain" in v:
                final[phase_idx_str] = {"size": hap["Size"][1][0], "mustrain": hap["Mustrain"][1][0]}
            else:
                final[phase_idx_str] = {
                    "size": hap["Size"][1][0],
                    "mustrain_eq": hap["Mustrain"][1][0],
                    "mustrain_ax": hap["Mustrain"][1][1],
                }
        if limits_applied is not None:
            final["limits_applied"] = limits_applied

        print(json.dumps({"rwp": rwp_after, "sane": bool(sane), "error": None, "final": final,
                           "peak_amplitude_error": peak_error}))
        return 0

    except Exception as exc:  # noqa: BLE001 — always report a parseable result, never a raw traceback
        print(json.dumps({"rwp": None, "sane": False, "error": repr(exc), "final": None,
                           "peak_amplitude_error": None}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
