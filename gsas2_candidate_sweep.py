#!/usr/bin/env python3
"""
gsas2_candidate_sweep.py — run several (instrument-parameter, phase CIF)
hypotheses against one pattern in parallel, and automatically rank which
one actually explains the data.

Why this exists
----------------
When it isn't clear which wavelength or crystal structure actually matches
a dataset — this project's own FeF3 data was originally paired with the
wrong instrument file and an unverified CIF, and diagnosing that took
manual indexing, a literature search, and several trial refinements — the
fix each time was the same: try a candidate, look at whether the
calculated pattern actually tracks the real one (not just whether Rwp
looks plausible), and repeat. gsas2_auto_refine.py's own
assess_fit_quality() check (a calculated/observed correlation, plus a
`needs_review` flag) already automates the "does this look real" judgment
for a single run. This script automates the "try several and compare"
part: it launches one gsas2_auto_refine.py subprocess per candidate (in
parallel — each run typically takes single-digit seconds, see the
Bounds/-timing evidence in gsas2_auto_refine.py's own docstrings, so
running N of them concurrently costs little), then ranks the finished
results by that same fit-quality signal plus final Rwp.

This does NOT try to invent candidates for you — every (instprm, CIF) pair
still has to come from somewhere real (a beamline log, a materials
database, a scientist's judgment call), same as gsas2_auto_refine.py never
guesses a phase or wavelength on its own. It only removes the manual,
serial trial-and-error of testing hypotheses you already have in hand.

Manifest format (JSON) — see gsas2_candidate_sweep_logic.load_manifest()
for the full schema:

    {
      "pattern": "Data/FeF3/r3c 20.txt",
      "gsasii_path": "/path/to/GSASII",
      "candidates": [
        {"name": "cu_ambient", "instprm": "Data/FeF3/ws2.cu.prm",
         "cif": ["Data/FeF3/fef3_r3c_ambient.cif"]},
        {"name": "mo_guess", "instprm": "Data/FeF3/ws2.mo.prm",
         "cif": ["Data/FeF3/fef3_r3c_cod2100647.cif"]}
      ]
    }

Example:
    python3 gsas2_candidate_sweep.py \\
        --manifest sweep.json \\
        --outdir results/fef3_sweep

Each candidate's full gsas2_auto_refine.py output lands in
<outdir>/<candidate name>/ exactly as a standalone run would; this script
additionally writes <outdir>/sweep_summary.json (every candidate's
outcome, ranked) and prints a ranking table to stdout.
"""

import os

# Must be set before numpy/GSASIIscriptable ever load MKL — see
# gsas2_auto_refine.py's matching comment for the full crash trace this
# works around. Set here too so subprocesses spawned below inherit it.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

from gsas2_gui_logic import validate_run_config, read_summary, resolve_gsasii_python
from gsas2_candidate_sweep_logic import (
    build_command,
    candidate_to_run_config,
    format_ranking_table,
    load_manifest,
    rank_results,
    ManifestError,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFINE_SCRIPT = SCRIPT_DIR / "gsas2_auto_refine.py"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run several (instprm, CIF) candidates against one pattern in "
                    "parallel and rank them by fit quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True, type=Path,
                    help="JSON file listing the pattern and candidate (instprm, CIF) pairs. "
                         "See this script's module docstring for the format.")
    p.add_argument("--outdir", required=True, type=Path,
                    help="Parent directory - each candidate gets its own "
                         "<outdir>/<candidate name>/ subfolder.")
    p.add_argument("--gsasii-path", type=Path, default=None,
                    help="Overrides the manifest's \"gsasii_path\" for every candidate, "
                         "if given.")
    p.add_argument("--max-workers", type=int, default=None,
                    help="Max candidates to run at once. Default: run all of them "
                         "concurrently (each is a lightweight subprocess - see this "
                         "script's module docstring on typical per-run cost).")
    p.add_argument("--emit-events", action="store_true",
                    help="Also print one JSON line per event (sweep_plan/candidate_start/"
                         "candidate_done/sweep_done) to stdout, alongside the normal "
                         "human-readable log. Intended for a GUI or other tool driving this "
                         "script as a subprocess - see gsas2_gui.py's Sweep tab.")
    return p.parse_args(argv)


def _no_window_kwargs() -> dict:
    """Extra subprocess.run kwargs to suppress the console window
    Windows otherwise briefly flashes open for every subprocess — matters
    doubly here: when this script is itself launched from the GUI's
    Sweep tab with the same suppression, each gsas2_auto_refine.py it
    spawns in turn would otherwise flash its own separate console window
    on top. CREATE_NO_WINDOW only exists in the subprocess module on
    Windows, so this is a no-op dict everywhere else. Defined locally
    rather than imported from gsas2_gui.py so this stays runnable without
    tkinter/matplotlib installed — see this module's own docstring."""
    if sys.platform.startswith("win"):
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def run_one_candidate(cmd: list, outdir: Path) -> dict:
    """Runs a single candidate's gsas2_auto_refine.py subprocess to
    completion and reads back its summary.json. Never raises — a
    candidate that crashes outright (bad path, GSAS-II import failure)
    is reported via returncode/summary=None, exactly like one that ran
    but produced a poor fit; see score_result() for why that distinction
    still matters for ranking."""
    # stdin=DEVNULL: confirmed as a real hang, not a theoretical one — a
    # subprocess.run() call here without this blocked for minutes at ~0
    # CPU usage (i.e. waiting on a read, not computing) instead of the
    # single-digit seconds a real run takes. gsas2_auto_refine.py is
    # fully non-interactive and should never read from stdin, but without
    # explicitly closing it the child inherits whatever this process's
    # own stdin is, which isn't always a terminal that sends EOF.
    proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           **_no_window_kwargs())
    return {
        "returncode": proc.returncode,
        "summary": read_summary(str(outdir)),
        "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-10:]) if proc.stderr else "",
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    def emit(event: dict):
        if args.emit_events:
            print(json.dumps(event), flush=True)

    try:
        pattern, gsasii_path, candidates = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        emit({"event": "sweep_done", "ok": False, "winner": None, "error": str(exc)})
        return 2

    if args.gsasii_path is not None:
        gsasii_path = str(args.gsasii_path)

    args.outdir.mkdir(parents=True, exist_ok=True)

    jobs = []  # (name, cmd, outdir)
    problems = []
    for candidate in candidates:
        outdir = args.outdir / candidate.name
        cfg = candidate_to_run_config(pattern, gsasii_path, candidate, outdir)
        cfg_problems = validate_run_config(cfg)
        if cfg_problems:
            problems.append(f"candidate {candidate.name!r}: " + "; ".join(cfg_problems))
            continue
        python_exe = resolve_gsasii_python(cfg.gsasii_path, sys.executable)
        cmd = build_command(cfg, script_path=str(REFINE_SCRIPT), python_exe=python_exe)
        jobs.append((candidate.name, cmd, outdir))

    if problems:
        for p in problems:
            print(f"ERROR: {p}", file=sys.stderr)
        emit({"event": "sweep_done", "ok": False, "winner": None, "error": "; ".join(problems)})
        return 2

    print(f"Running {len(jobs)} candidate(s) against {pattern!r}"
          + (f" (up to {args.max_workers} at a time)" if args.max_workers else " (in parallel)")
          + " ...", flush=True)
    emit({
        "event": "sweep_plan",
        "pattern": pattern,
        "outdir": str(args.outdir),
        "candidates": [{"name": name, "outdir": str(outdir)} for name, _cmd, outdir in jobs],
    })
    for name, _cmd, outdir in jobs:
        emit({"event": "candidate_start", "name": name, "outdir": str(outdir)})

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_to_name = {
            pool.submit(run_one_candidate, cmd, outdir): name
            for name, cmd, outdir in jobs
        }
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            outcome = future.result()
            results.append({"name": name, **outcome})
            summary = outcome["summary"]
            if summary:
                fq = summary.get("fit_quality") or {}
                print(f"  [{name}] done - Rwp={summary.get('final_rwp')}, "
                      f"correlation={fq.get('calc_obs_correlation')}, "
                      f"needs_review={fq.get('needs_review')}", flush=True)
            else:
                print(f"  [{name}] produced no summary.json (return code "
                      f"{outcome['returncode']}) - see {args.outdir / name / 'run.log'} "
                      f"if it exists, or: {outcome['stderr_tail']}", flush=True)
            emit({"event": "candidate_done", "name": name,
                  "returncode": outcome["returncode"], "summary": summary})

    ranked = rank_results(results)

    print("\n" + format_ranking_table(ranked), flush=True)

    winner = ranked[0] if ranked else None
    winner_ok = bool(
        winner and winner["summary"]
        and not (winner["summary"].get("fit_quality") or {}).get("needs_review", True)
    )

    sweep_summary = {
        "pattern": pattern,
        "outdir": str(args.outdir),
        "ranking": [r["name"] for r in ranked],
        "winner": winner["name"] if winner_ok else None,
        "results": ranked,
    }
    (args.outdir / "sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")

    if winner_ok:
        print(f"\nBest candidate: {winner['name']} "
              f"(see {args.outdir / winner['name'] / 'summary.json'})", flush=True)
        emit({"event": "sweep_done", "ok": True, "winner": winner["name"],
              "ranking": sweep_summary["ranking"],
              "sweep_summary_path": str(args.outdir / "sweep_summary.json")})
        return 0

    print("\nNo candidate produced a fit that passed the fit-quality check - "
          "the best-ranked one is still recorded above and in sweep_summary.json, "
          "but none of these candidates should be trusted without review.", flush=True)
    emit({"event": "sweep_done", "ok": False,
          "winner": ranked[0]["name"] if ranked else None,
          "ranking": sweep_summary["ranking"],
          "sweep_summary_path": str(args.outdir / "sweep_summary.json")})
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
