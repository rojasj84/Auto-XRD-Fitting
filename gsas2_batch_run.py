#!/usr/bin/env python3
"""
gsas2_batch_run.py — refine every experiment under one root directory,
unattended, in parallel, and flag which results a human should look at.

Point this at a folder whose immediate subfolders are each one experiment
(a pattern file, an instrument-parameter file, and one or more phase
CIFs — the same convention as the GUI's "Load bundled example" feature,
see gsas2_gui_logic.scan_dataset_subfolders()). Every experiment is run
as its own gsas2_auto_refine.py invocation, all in parallel (each run
typically takes single-digit seconds — see gsas2_candidate_sweep.py's
module docstring for the timing evidence this project's own runs
produced), landing in its own subfolder under --outdir. A results table
(both CSV and JSON) is written alongside them, flagging every experiment
that either didn't produce a trustworthy fit (gsas2_auto_refine.py's own
calculated/observed correlation check — see assess_fit_quality() there)
or came in at or above --rwp-threshold, so a scientist reviewing hundreds
of results can filter straight to the ones that actually need a look
instead of opening every single output.

Per-experiment overrides: drop a "params.json" in any experiment's own
subfolder to override this batch's global --max-cell-drift/--refine-atoms/
--tmin/--tmax for just that one experiment — see
gsas2_batch_run_logic.load_experiment_params() for the exact format.
Entirely optional; most experiments won't need one.

Example:
    python3 gsas2_batch_run.py \\
        --root /data/xrd_experiments \\
        --outdir results/big_batch \\
        --gsasii-path /path/to/GSASII

This does NOT invent experiments, phases, or wavelengths for you — same
as gsas2_auto_refine.py and gsas2_candidate_sweep.py, every experiment's
pattern/instprm/CIF still has to be real files a scientist put there. It
only removes the manual, serial work of running and triaging many of them.
"""

import argparse
import concurrent.futures
import csv
import json
import sys
from pathlib import Path

from gsas2_gui_logic import validate_run_config
from gsas2_candidate_sweep import run_one_candidate
from gsas2_candidate_sweep_logic import build_command
from gsas2_batch_run_logic import (
    BatchDefaults,
    build_batch_row,
    discover_experiments,
    experiment_to_run_config,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFINE_SCRIPT = SCRIPT_DIR / "gsas2_auto_refine.py"

_CSV_FIELDS = ["name", "needs_review", "final_rwp", "calc_obs_correlation",
               "reason", "failed_stages", "returncode", "outdir"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Refine every experiment subfolder under one root directory, "
                    "unattended, in parallel, and flag which results need review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", required=True, type=Path,
                    help="Folder whose immediate subfolders are each one experiment "
                         "(pattern + instrument-parameter file + CIF(s)) - same "
                         "discovery convention as the GUI's 'Load bundled example.'")
    p.add_argument("--outdir", required=True, type=Path,
                    help="Parent output folder - each experiment gets its own "
                         "<outdir>/<experiment name>/ subfolder.")
    p.add_argument("--gsasii-path", required=True, type=Path,
                    help="Path to the local GSAS-II install, used for every experiment.")
    p.add_argument("--rwp-threshold", type=float, default=10.0,
                    help="Final Rwp at or above this percent is flagged for manual "
                         "review (default 10.0).")
    p.add_argument("--max-cell-drift", type=float, default=0.15,
                    help="Default for every experiment; a per-experiment params.json "
                         "can override it (see this script's module docstring).")
    p.add_argument("--refine-atoms", action="store_true",
                    help="Default for every experiment; a per-experiment params.json "
                         "can override it.")
    p.add_argument("--tmin", type=float, default=None,
                    help="Default 2-theta trim lower bound for every experiment; a "
                         "per-experiment params.json can override it. Must be given "
                         "together with --tmax.")
    p.add_argument("--tmax", type=float, default=None, help="See --tmin.")
    p.add_argument("--max-workers", type=int, default=None,
                    help="Max experiments to run at once. Default: run all of them "
                         "concurrently.")
    p.add_argument("--emit-events", action="store_true",
                    help="Also print one JSON line per event (batch_plan/"
                         "experiment_start/experiment_done/batch_done) to stdout, "
                         "alongside the normal human-readable log. Intended for a GUI "
                         "or other tool driving this script as a subprocess.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    def emit(event: dict):
        if args.emit_events:
            print(json.dumps(event), flush=True)

    if (args.tmin is None) != (args.tmax is None):
        print("ERROR: --tmin and --tmax must be given together (or neither).", file=sys.stderr)
        return 2
    if not args.root.is_dir():
        print(f"ERROR: --root is not a directory: {args.root}", file=sys.stderr)
        return 2
    if not args.gsasii_path.is_dir():
        print(f"ERROR: --gsasii-path not found: {args.gsasii_path}", file=sys.stderr)
        return 2

    defaults = BatchDefaults(
        gsasii_path=str(args.gsasii_path),
        max_cell_drift=args.max_cell_drift,
        refine_atoms=args.refine_atoms,
        tmin=args.tmin,
        tmax=args.tmax,
        rwp_threshold=args.rwp_threshold,
    )
    experiments, skipped = discover_experiments(str(args.root), defaults)

    for name, reason in sorted(skipped.items()):
        print(f"  [skipped] {name}: {reason}")

    if not experiments:
        print(f"ERROR: no usable experiment subfolders found under {args.root}.",
              file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    jobs = []  # (name, cmd, outdir)
    problems = []
    for exp in experiments:
        outdir = args.outdir / exp.name
        cfg = experiment_to_run_config(exp, defaults.gsasii_path, outdir)
        cfg_problems = validate_run_config(cfg)
        if cfg_problems:
            problems.append(f"experiment {exp.name!r}: " + "; ".join(cfg_problems))
            continue
        cmd = build_command(cfg, script_path=str(REFINE_SCRIPT), python_exe=sys.executable)
        jobs.append((exp.name, cmd, outdir))

    if problems:
        for p in problems:
            print(f"ERROR: {p}", file=sys.stderr)
        emit({"event": "batch_done", "ok": False, "error": "; ".join(problems)})
        return 2

    print(f"Running {len(jobs)} experiment(s) from {args.root} (in parallel) ...", flush=True)
    emit({
        "event": "batch_plan",
        "root": str(args.root),
        "outdir": str(args.outdir),
        "skipped": skipped,
        "experiments": [{"name": name, "outdir": str(outdir)} for name, _cmd, outdir in jobs],
    })
    for name, _cmd, outdir in jobs:
        emit({"event": "experiment_start", "name": name, "outdir": str(outdir)})

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_to_job = {pool.submit(run_one_candidate, cmd, outdir): (name, outdir)
                          for name, cmd, outdir in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            name, outdir = future_to_job[future]
            outcome = future.result()
            row = build_batch_row(name, str(outdir), outcome["returncode"],
                                   outcome["summary"], args.rwp_threshold)
            rows.append(row)
            flag = "NEEDS REVIEW" if row["needs_review"] else "ok"
            print(f"  [{name}] {flag} - Rwp={row['final_rwp']}, "
                  f"correlation={row['calc_obs_correlation']}", flush=True)
            emit({"event": "experiment_done", **row})

    rows.sort(key=lambda r: r["name"])
    n_flagged = sum(1 for r in rows if r["needs_review"])

    results_csv = args.outdir / "batch_results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "failed_stages": ";".join(r["failed_stages"])})

    results_json = args.outdir / "batch_results.json"
    results_json.write_text(json.dumps({
        "root": str(args.root),
        "outdir": str(args.outdir),
        "rwp_threshold": args.rwp_threshold,
        "n_total": len(rows),
        "n_flagged": n_flagged,
        "skipped": skipped,
        "results": rows,
    }, indent=2), encoding="utf-8")

    print(f"\n{len(rows)} experiment(s) run: {len(rows) - n_flagged} ok, "
          f"{n_flagged} flagged for review.")
    if skipped:
        print(f"{len(skipped)} subfolder(s) skipped (see above / batch_results.json).")
    print(f"See {results_csv} / {results_json}")

    emit({
        "event": "batch_done",
        "ok": n_flagged == 0,
        "n_total": len(rows),
        "n_flagged": n_flagged,
        "results_csv": str(results_csv),
        "results_json": str(results_json),
    })

    return 1 if n_flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
