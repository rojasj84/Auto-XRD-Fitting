#!/usr/bin/env python3
"""
gsas2_batch_run_logic.py — pure, subprocess-free logic for
gsas2_batch_run.py: refine every experiment subfolder under one root
directory, unattended, and flag which results need a human to look at.

An "experiment" is exactly what gsas2_gui_logic.scan_dataset_subfolders()
already discovers for the GUI's "Load bundled example" feature: an
immediate subfolder containing one pattern file, one instrument-parameter
file, and one or more phase CIFs (multiple CIFs in a subfolder mean a
multi-phase sample, not ambiguity — see scan_dataset_subfolders()'s
docstring for the distinction). This module reuses that discovery
convention directly (in strict mode — an experiment folder with more than
one candidate pattern or instprm file is skipped as ambiguous rather than
silently guessing, since nobody is watching an unattended batch run to
notice a wrong guess), rather than inventing a second one.

Each experiment's own subfolder may additionally contain a "params.json"
overriding this batch's global run options (max_cell_drift, refine_atoms,
tmin, tmax) for just that one experiment — see load_experiment_params().
This is entirely optional; an experiment with no params.json just uses
the batch's global defaults.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gsas2_gui_logic import RunConfig, scan_dataset_subfolders

_OVERRIDABLE_PARAMS = ("max_cell_drift", "refine_atoms", "tmin", "tmax")


@dataclass
class BatchDefaults:
    gsasii_path: str = ""
    max_cell_drift: float = 0.15
    refine_atoms: bool = False
    tmin: Optional[float] = None
    tmax: Optional[float] = None
    rwp_threshold: float = 10.0


@dataclass
class Experiment:
    name: str
    pattern: str
    instprm: str
    cifs: list
    max_cell_drift: float
    refine_atoms: bool
    tmin: Optional[float]
    tmax: Optional[float]


def load_experiment_params(subfolder, defaults: BatchDefaults) -> dict:
    """
    Reads <subfolder>/params.json, if present, and returns
    {max_cell_drift, refine_atoms, tmin, tmax} with any keys it sets
    overriding `defaults`. Never raises — a missing params.json (the
    common case: most experiments just want the batch's defaults) or a
    corrupt/malformed one both just fall back to `defaults` untouched,
    the same as gsas2_gui_logic.read_summary()'s "never let a bad file
    take down the whole run" convention. One experiment's typo in its
    own params.json shouldn't cost the rest of the batch anything.
    """
    result = {key: getattr(defaults, key) for key in _OVERRIDABLE_PARAMS}
    params_path = Path(subfolder) / "params.json"
    if not params_path.is_file():
        return result
    try:
        overrides = json.loads(params_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return result
    if not isinstance(overrides, dict):
        return result
    for key in _OVERRIDABLE_PARAMS:
        if key in overrides:
            result[key] = overrides[key]
    return result


def discover_experiments(root: str, defaults: BatchDefaults) -> tuple:
    """
    Returns (experiments: list[Experiment], skipped: dict) for every
    subfolder of `root` — see scan_dataset_subfolders(..., strict=True)
    for what makes a subfolder qualify (and `skipped`'s reasons for the
    ones that don't). Experiments are sorted by name for a deterministic
    run order across repeated batches over the same folder.
    """
    found, skipped = scan_dataset_subfolders(root, strict=True)
    experiments = []
    for name in sorted(found):
        entry = found[name]
        params = load_experiment_params(Path(root) / name, defaults)
        experiments.append(Experiment(
            name=name,
            pattern=entry["pattern"],
            instprm=entry["instprm"],
            cifs=list(entry["cifs"]),
            **params,
        ))
    return experiments, skipped


def experiment_to_run_config(experiment: Experiment, gsasii_path: str, outdir) -> RunConfig:
    """Builds the RunConfig for one experiment's gsas2_auto_refine.py
    invocation — the same config shape the GUI and the candidate-sweep
    tool both use (see gsas2_gui_logic.RunConfig / build_command), just
    constructed from a discovered experiment instead of widget values or
    a manifest entry."""
    return RunConfig(
        pattern=experiment.pattern,
        instprm=experiment.instprm,
        cifs=list(experiment.cifs),
        outdir=str(outdir),
        gsasii_path=gsasii_path,
        refine_atoms=experiment.refine_atoms,
        max_cell_drift=experiment.max_cell_drift,
        tmin=experiment.tmin,
        tmax=experiment.tmax,
    )


def classify_result(summary, rwp_threshold: float) -> dict:
    """
    The single triage decision this whole tool exists to automate: is
    this result trustworthy enough to skip, or does a human need to
    look at it? Combines TWO independent signals, either one enough to
    flag on its own:

      1. gsas2_auto_refine.py's own fit_quality.needs_review — the
         calculated/observed correlation check (see assess_fit_quality()
         there). Catches a plausible-looking Rwp on a model that doesn't
         actually explain the data — confirmed as a real failure mode on
         this project's own FeF3 data (Rwp ~10.6% while correlation was
         ~0.02) that Rwp alone would never have caught.
      2. final_rwp >= rwp_threshold — the operational bar a scientist
         actually works to day to day.

    A summary of None (the run crashed before producing one — a bad
    file, a GSAS-II import failure, anything upstream of the pipeline's
    own checks) is always flagged; it tells you nothing, the same
    reasoning gsas2_candidate_sweep.py's ranking already uses.
    """
    if not summary:
        return {"needs_review": True,
                "reason": "run produced no summary.json (crashed or setup failed)",
                "final_rwp": None, "calc_obs_correlation": None}

    fit_quality = summary.get("fit_quality") or {}
    rwp = summary.get("final_rwp")
    reasons = []
    if fit_quality.get("needs_review", True):
        reasons.append(fit_quality.get("reason")
                        or "calculated pattern does not track the observed one")
    if not isinstance(rwp, (int, float)):
        reasons.append("final Rwp is missing")
    elif rwp >= rwp_threshold:
        reasons.append(f"final Rwp {rwp:.3f} is at or above the {rwp_threshold:g}% threshold")

    return {
        "needs_review": bool(reasons),
        "reason": "; ".join(reasons),
        "final_rwp": rwp,
        "calc_obs_correlation": fit_quality.get("calc_obs_correlation"),
    }


def build_batch_row(name: str, outdir: str, returncode: int, summary, rwp_threshold: float) -> dict:
    """One row of the batch's results table — see classify_result() for
    the review-flagging logic. `failed_stages` (mandatory stages that
    didn't converge) is separate from needs_review: a run can have every
    mandatory stage "ok" and still be flagged (Rwp above threshold), or
    have a failed optional stage and still be fine (see StageResult.
    optional in gsas2_auto_refine.py) — this column is diagnostic detail
    for a human reviewing the row, not itself a review trigger."""
    classification = classify_result(summary, rwp_threshold)
    failed_stages = [s["name"] for s in (summary or {}).get("stages", [])
                      if s["status"] != "ok" and not s.get("optional")]
    return {
        "name": name,
        "outdir": outdir,
        "returncode": returncode,
        "final_rwp": classification["final_rwp"],
        "calc_obs_correlation": classification["calc_obs_correlation"],
        "needs_review": classification["needs_review"],
        "reason": classification["reason"],
        "failed_stages": failed_stages,
    }
