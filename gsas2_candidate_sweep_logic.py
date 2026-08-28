#!/usr/bin/env python3
"""
gsas2_candidate_sweep_logic.py — pure, subprocess-free logic for
gsas2_candidate_sweep.py.

Automates a real diagnostic workflow from this project's own history: when
it isn't clear which instrument file / crystal structure actually matches a
dataset (the FeF3 wavelength-and-cell mismatch this whole pipeline was
originally debugged against), the fix was to try several candidate
(instprm, CIF) pairs and see which one's calculated pattern actually tracks
the real data — assess_fit_quality()'s correlation check in
gsas2_auto_refine.py is exactly the automatic version of "does this
candidate's fit look real." This module owns the parts of that workflow
that don't need a subprocess or GSAS-II to test: manifest parsing/
validation, building one gsas2_auto_refine.py RunConfig per candidate, and
ranking finished results. gsas2_candidate_sweep.py owns launching those
RunConfigs as parallel subprocesses and printing/writing the outcome.

Kept separate from gsas2_candidate_sweep.py the same way gsas2_gui_logic.py
is kept separate from gsas2_gui.py — see test_candidate_sweep_logic.py for
tests that need neither GSAS-II nor a real subprocess.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from gsas2_gui_logic import RunConfig, build_command  # noqa: F401 — re-exported for callers


@dataclass
class Candidate:
    """One hypothesis to test: a specific instrument-parameter file paired
    with a specific phase CIF (or CIFs, for a multi-phase guess), against
    the one pattern every candidate in a sweep shares. Everything here
    ends up as one gsas2_auto_refine.py invocation — see
    candidate_to_run_config()."""
    name: str
    instprm: str
    cifs: list
    max_cell_drift: float = 0.15
    refine_atoms: bool = False
    tmin: Optional[float] = None
    tmax: Optional[float] = None


class ManifestError(ValueError):
    """Raised for any structurally invalid sweep manifest — always with a
    human-readable message naming what's wrong and (where possible) which
    candidate, never a raw KeyError/TypeError a caller has to decode."""


def load_manifest(path) -> tuple:
    """
    Reads and validates a candidate-sweep manifest JSON file. Returns
    (pattern: str, gsasii_path: str, candidates: list[Candidate]).

    Expected shape::

        {
          "pattern": "Data/FeF3/r3c 20.txt",
          "gsasii_path": "/path/to/GSASII",
          "max_cell_drift": 0.15,
          "candidates": [
            {"name": "cu_ambient", "instprm": "...", "cif": ["..."]},
            {"name": "mo_guess",   "instprm": "...", "cif": ["...", "..."]}
          ]
        }

    "gsasii_path" and "max_cell_drift" (and "refine_atoms", "tmin", "tmax")
    are shared defaults applied to every candidate; any candidate may
    override them by setting the same key on itself. "cif" may be a single
    string (one phase) or a list of strings (multi-phase). Raises
    ManifestError — never a raw json/KeyError/TypeError — for anything
    structurally wrong, so a caller can show the message directly to a
    user instead of a traceback.
    """
    try:
        raw = json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise ManifestError(f"Manifest file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest {path} is not valid JSON: {exc}")

    if not isinstance(raw, dict):
        raise ManifestError(f"Manifest {path} must be a JSON object, not {type(raw).__name__}")

    pattern = raw.get("pattern")
    if not pattern or not isinstance(pattern, str):
        raise ManifestError("Manifest must have a top-level string \"pattern\" "
                             "(the one dataset every candidate is tested against).")

    gsasii_path = raw.get("gsasii_path", "")

    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ManifestError("Manifest must have a non-empty \"candidates\" list.")

    defaults = {
        "max_cell_drift": raw.get("max_cell_drift", 0.15),
        "refine_atoms": raw.get("refine_atoms", False),
        "tmin": raw.get("tmin"),
        "tmax": raw.get("tmax"),
    }

    seen_names = set()
    candidates = []
    for i, c in enumerate(candidates_raw):
        if not isinstance(c, dict):
            raise ManifestError(f"candidates[{i}] must be a JSON object, not {type(c).__name__}")
        name = c.get("name") or f"candidate_{i + 1}"
        if name in seen_names:
            raise ManifestError(f"Duplicate candidate name {name!r} — "
                                 "each candidate's outdir is named after it, so names must be unique.")
        seen_names.add(name)

        instprm = c.get("instprm")
        if not instprm or not isinstance(instprm, str):
            raise ManifestError(f"candidate {name!r}: missing or invalid \"instprm\".")

        cif = c.get("cif")
        if isinstance(cif, str):
            cifs = [cif]
        elif isinstance(cif, list) and cif and all(isinstance(x, str) for x in cif):
            cifs = list(cif)
        else:
            raise ManifestError(f"candidate {name!r}: \"cif\" must be a non-empty string "
                                 "or list of strings.")

        candidates.append(Candidate(
            name=name,
            instprm=instprm,
            cifs=cifs,
            max_cell_drift=c.get("max_cell_drift", defaults["max_cell_drift"]),
            refine_atoms=c.get("refine_atoms", defaults["refine_atoms"]),
            tmin=c.get("tmin", defaults["tmin"]),
            tmax=c.get("tmax", defaults["tmax"]),
        ))

    return pattern, gsasii_path, candidates


def candidate_to_run_config(pattern: str, gsasii_path: str, candidate: Candidate,
                             outdir) -> RunConfig:
    """Builds the RunConfig for one candidate's gsas2_auto_refine.py
    invocation — the same config shape the GUI uses (see
    gsas2_gui_logic.RunConfig / build_command), just constructed from a
    manifest entry instead of widget values."""
    return RunConfig(
        pattern=pattern,
        instprm=candidate.instprm,
        cifs=list(candidate.cifs),
        outdir=str(outdir),
        gsasii_path=gsasii_path,
        refine_atoms=candidate.refine_atoms,
        max_cell_drift=candidate.max_cell_drift,
        tmin=candidate.tmin,
        tmax=candidate.tmax,
    )


def score_result(entry: dict) -> tuple:
    """
    Sort key for one candidate's finished result — LARGER is BETTER, so
    ranking is `sorted(results, key=score_result, reverse=True)`.
    `entry` is {"name": str, "returncode": int, "summary": dict | None}
    (summary is gsas2_auto_refine.py's summary.json content, or None if
    the candidate's run never produced one at all — e.g. it crashed
    during setup before any stage ran).

    Ranked, most significant first:
      1. did it produce a summary at all (a candidate that crashed outright
         is always worse than one that merely fit poorly — it tells you
         nothing, e.g. a bad file path or missing phase multiplicities)
      2. fit_quality.needs_review is False — see assess_fit_quality() in
         gsas2_auto_refine.py: this is the check that actually
         distinguishes "the calculated pattern tracks the data" from "Rwp
         merely looks plausible," which is the whole point of comparing
         candidates in the first place rather than trusting Rwp alone
      3. higher calc/obs correlation (a finer-grained tiebreaker within
         "good" or within "needs review" alike)
      4. lower final Rwp
    """
    summary = entry.get("summary")
    if not summary:
        return (0, 0, -1.0, float("-inf"))

    fit_quality = summary.get("fit_quality") or {}
    not_needs_review = 1 if not fit_quality.get("needs_review", True) else 0
    corr = fit_quality.get("calc_obs_correlation")
    corr = corr if isinstance(corr, (int, float)) else -1.0
    rwp = summary.get("final_rwp")
    neg_rwp = -rwp if isinstance(rwp, (int, float)) else float("-inf")
    return (1, not_needs_review, corr, neg_rwp)


def rank_results(results: list) -> list:
    """Returns `results` sorted best-first by score_result. Stable for
    equal scores (Python's sort is stable), so candidates that tie keep
    their manifest order rather than shuffling run to run."""
    return sorted(results, key=score_result, reverse=True)


def format_ranking_table(ranked: list) -> str:
    """Plain-text ranking table for CLI/log output — one row per
    candidate, best first, with the columns a human deciding between
    candidates actually needs: correlation, Rwp, and whether it's
    flagged for review or crashed outright."""
    lines = [f"{'name':<24} {'status':<14} {'correlation':>12} {'final_rwp':>10}"]
    lines.append("-" * len(lines[0]))
    for entry in ranked:
        summary = entry.get("summary")
        if not summary:
            lines.append(f"{entry['name']:<24} {'crashed':<14} {'':>12} {'':>10}")
            continue
        fit_quality = summary.get("fit_quality") or {}
        needs_review = fit_quality.get("needs_review", True)
        corr = fit_quality.get("calc_obs_correlation")
        rwp = summary.get("final_rwp")
        status = "needs review" if needs_review else "ok"
        corr_str = f"{corr:.4f}" if isinstance(corr, (int, float)) else "n/a"
        rwp_str = f"{rwp:.3f}" if isinstance(rwp, (int, float)) else "n/a"
        lines.append(f"{entry['name']:<24} {status:<14} {corr_str:>12} {rwp_str:>10}")
    return "\n".join(lines)
