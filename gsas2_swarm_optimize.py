#!/usr/bin/env python3
"""
gsas2_swarm_optimize.py — surrogate-assisted search over the Size/
Mustrain continuous parameter space, starting from a real
gsas2_auto_refine.py checkpoint, in place of that pipeline's fixed
single starting point.

Why this exists, and why this specific algorithm
--------------------------------------------------
gsas2_auto_refine.py's profile_microstrain_size stage (and its fallback
ladder) always starts Size/Mustrain refinement from the same fixed
values and tries a small, hand-picked set of simpler models when that
starting point leads somewhere unsound. That's worked well enough to
build a whole session of real fixes on, but it can only ever find what
GSAS-II's own local (Hessian LM) refinement reaches from THAT one
starting point (or one of a few fallback ones).

An earlier version of this script searched with plain PSO sampling
starting points uniformly across the whole bounds. Confirmed as a real
problem on real data, not theoretical: most of that space is badly
conditioned for GSAS-II's own solver (one randomly-sampled point had its
Size parameter dropped as insensitive; another's Mustrain blew up to
41,297) — random-across-everything wastes most evaluations on points
that were never going to converge to anything useful.

This version instead: perturbs a batch of candidates around the current
best point (a mix of small and large perturbations — see gsas2_swarm_
logic.perturb_points), evaluates all of them for real via GSAS-II in
parallel across CPU workers, fits a cheap polynomial surrogate Rwp(x) to
every REAL (sane) evaluation collected so far (the dataset only grows —
see gsas2_swarm_logic.fit_surrogate), searches that surrogate with PSO
(in-memory, no subprocesses — see search_surrogate(), and that module's
docstring for where GPU work, --backend gpu, actually has a legitimate
job to do: this step, not the raw swarm bookkeeping and not GSAS-II
itself, which stays CPU-only regardless), and VERIFIES whatever the
surrogate proposes with one more real GSAS-II evaluation before ever
trusting it. Repeats for --outer-iterations, each time perturbing around
whatever the best verified point is so far.

This does NOT replace gsas2_auto_refine.py — it picks up from one of its
checkpoints (normally checkpoint_0N_pre_profile_microstrain_size.gpx)
and hands back a candidate replacement for just that one stage's result.

Example:
    python3 gsas2_swarm_optimize.py \\
        --checkpoint results/sample/checkpoint_06_pre_profile_microstrain_size.gpx \\
        --gsasii-path /path/to/GSASII \\
        --outdir results/sample_swarm \\
        --outer-iterations 20 --perturbations 50 --backend auto
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import gsas2_swarm_logic as logic

SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "gsas2_swarm_worker.py"


def _no_window_kwargs() -> dict:
    """See gsas2_candidate_sweep.py's identical helper — same reasoning,
    duplicated locally so this script has no import-time dependency on
    that one."""
    if sys.platform.startswith("win"):
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Surrogate-assisted search over Size/Mustrain starting points, "
                    "picking up from a gsas2_auto_refine.py checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, type=Path,
                    help="A checkpoint_0N_pre_profile_microstrain_size.gpx from a real "
                         "gsas2_auto_refine.py run. Never modified.")
    p.add_argument("--gsasii-path", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path,
                    help="Every real evaluation gets its own subfolder here "
                         "(evaluations/iter<N>_<label>/) plus the final best result.")
    p.add_argument("--outer-iterations", type=int, default=20,
                    help="How many perturb-evaluate-fit-propose-verify rounds to run "
                         "(default 20). Compute time is cheap here by design — see this "
                         "script's module docstring — so this defaults generous rather "
                         "than minimal; --patience below still stops early on plateau.")
    p.add_argument("--perturbations", type=int, default=50,
                    help="How many real candidates to evaluate around the current best "
                         "point each outer iteration (default 50).")
    p.add_argument("--close-frac", type=float, default=0.6,
                    help="Fraction of each round's perturbations that are 'close' "
                         "(exploit near the current best) vs 'far' (explore) — default 0.6.")
    p.add_argument("--close-sigma", type=float, default=0.15)
    p.add_argument("--far-sigma", type=float, default=0.75)
    p.add_argument("--surrogate-degree", type=int, default=2, choices=(1, 2),
                    help="Polynomial degree for the surrogate fit (default 2, quadratic).")
    p.add_argument("--surrogate-particles", type=int, default=200,
                    help="Swarm size for searching the (cheap, in-memory) surrogate — can "
                         "be large since it costs no real GSAS-II time (default 200).")
    p.add_argument("--surrogate-generations", type=int, default=150)
    p.add_argument("--surrogate-ridge", type=float, default=1.0,
                    help="L2 (ridge) regularization strength for the surrogate fit (default "
                         "1.0; 0 = plain least squares). Confirmed necessary on real data — an "
                         "unregularized fit through a sparse, partly-insane-masked dataset can "
                         "invent a wildly wrong 'minimum' between training points even while "
                         "matching every one of them exactly. See gsas2_swarm_logic.fit_surrogate's "
                         "docstring.")
    p.add_argument("--surrogate-min-multiplier", type=float, default=2.0,
                    help="Don't fit/trust the surrogate at all until the dataset has at least "
                         "this many times the bare-minimum point count for the chosen degree "
                         "(default 2.0x) — a fit right at the bare minimum is barely better than "
                         "an interpolation table, not a trustworthy model of the landscape.")
    p.add_argument("--surrogate-candidates", type=int, default=1,
                    help="How many distinct proposals to extract from each surrogate search "
                         "and verify for real per outer iteration (default 1). >1 costs that "
                         "many extra real GSAS-II evaluations in exchange for checking "
                         "multiple distinct promising regions instead of just the single best "
                         "one — only useful together with --surrogate-explorer-frac > 0 (see "
                         "gsas2_swarm_logic.search_surrogate's docstring for why).")
    p.add_argument("--surrogate-explorer-frac", type=float, default=0.0,
                    help="Fraction of surrogate-search particles (default 0.0 = off) with "
                         "reduced pull toward the best-known point(s) — see --surrogate-"
                         "explorer-c-scale — so they keep wandering rather than collapsing "
                         "onto a single minimum. Makes --surrogate-candidates > 1 actually find "
                         "distinct points instead of near-duplicates of one basin.")
    p.add_argument("--surrogate-explorer-c-scale", type=float, default=0.2,
                    help="How much weaker explorer particles' pull toward pbest/gbest is, as a "
                         "fraction of the normal --c1/--c2 (default 0.2 = 20%% strength — still "
                         "pulled somewhat, not a pure random walk).")
    p.add_argument("--surrogate-min-separation", type=float, default=1.0,
                    help="Minimum distance between distinct --surrogate-candidates proposals, "
                         "measured in the surrogate's own normalized coordinate space (roughly "
                         "standard deviations) so it's meaningful across dimensions of very "
                         "different physical scale like Size vs. Mustrain (default 1.0).")
    p.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto",
                    help="Where to fit/search the surrogate. 'gpu' uses PyTorch + CUDA and "
                         "raises if unavailable rather than silently using CPU instead; "
                         "'auto' picks 'gpu' only if that's genuinely available, else 'cpu' "
                         "silently. See gsas2_swarm_logic.py's module docstring for why the "
                         "surrogate step, not GSAS-II itself or the swarm bookkeeping, is "
                         "the one place GPU work actually has a job to do here.")
    p.add_argument("--size-bounds", type=float, nargs=2, default=(0.01, 1000.0),
                    metavar=("LO", "HI"), help="Search bounds for isotropic Size.")
    p.add_argument("--mustrain-bounds", type=float, nargs=2, default=(0.01, 9000.0),
                    metavar=("LO", "HI"),
                    help="Search bounds for uniaxial Mustrain (each of equatorial/axial). "
                         "Kept under profile_params_sane's 10000 abs ceiling by default.")
    p.add_argument("--low-angle-cutoff-bounds", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="Also search how many degrees of low-angle data to discard from the "
                         "start of the fit range (e.g. '0 15') — real low-angle regions are "
                         "sometimes dominated by beamstop shadow/detector-edge artifacts rather "
                         "than genuine signal. Omit (the default) to leave the fit range exactly "
                         "as the checkpoint has it, like every earlier version of this script.")
    p.add_argument("--high-angle-cutoff-bounds", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="Also search how many degrees of high-angle data to discard from the "
                         "end of the fit range (e.g. '0 10') — high-angle regions are sometimes "
                         "dominated by vanishing peak statistics/background rather than genuine "
                         "signal. Omit (the default) to leave the fit range's upper bound exactly "
                         "as the checkpoint has it. NOTE for both of the above: Rwp isn't "
                         "strictly comparable across different cutoffs — trimming more points "
                         "can mechanically lower Rwp somewhat independent of whether the "
                         "retained-range fit genuinely improved; keep these bounds tight and to "
                         "whatever's independently scientifically justified — see "
                         "gsas2_swarm_logic.build_param_specs' docstring for a real example of "
                         "this going wrong on real data.")
    p.add_argument("--w", type=float, default=0.6, help="PSO inertia weight.")
    p.add_argument("--c1", type=float, default=1.6, help="PSO cognitive (pbest) coefficient.")
    p.add_argument("--c2", type=float, default=1.6, help="PSO social (gbest) coefficient.")
    p.add_argument("--patience", type=int, default=6,
                    help="Stop early if the best verified Rwp hasn't improved by "
                         "--min-improvement over this many outer iterations (default 6).")
    p.add_argument("--min-improvement", type=float, default=0.002,
                    help="Relative improvement threshold for --patience (default 0.002 = 0.2%%).")
    p.add_argument("--seed", type=int, default=None, help="RNG seed, for reproducible runs.")
    p.add_argument("--max-workers", type=int, default=None,
                    help="Max real evaluations to run at once. Default: all of "
                         "--perturbations concurrently.")
    p.add_argument("--emit-events", action="store_true",
                    help="Also print one JSON line per event (iteration_result/"
                         "swarm_done) to stdout, alongside the normal human-readable log.")
    return p.parse_args(argv)


def _count_phases(checkpoint: Path, gsasii_path: Path) -> int:
    sys.path.insert(0, str(SCRIPT_DIR))
    from gsas2_auto_refine import import_gsasiiscriptable
    G2sc = import_gsasiiscriptable(gsasii_path)
    gpx = G2sc.G2Project(str(checkpoint))
    return len(gpx.phases())


def evaluate_point(checkpoint: Path, gsasii_path: Path, outdir: Path, values: dict) -> dict:
    """Runs one gsas2_swarm_worker.py subprocess for one candidate point
    and returns its parsed JSON result. Never raises — a subprocess that
    fails to produce parseable output is reported as an error result,
    same convention as gsas2_candidate_sweep.py's run_one_candidate."""
    cmd = [sys.executable, "-u", str(WORKER_SCRIPT),
           "--checkpoint", str(checkpoint), "--gsasii-path", str(gsasii_path),
           "--outdir", str(outdir), "--values", json.dumps(values)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                               **_no_window_kwargs())
        for line in reversed(proc.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"rwp": None, "sane": False,
                "error": f"worker produced no JSON result (stderr: {proc.stderr[-300:]!r})",
                "final": None}
    except Exception as exc:  # noqa: BLE001
        return {"rwp": None, "sane": False, "error": repr(exc), "final": None}


def evaluate_batch(checkpoint: Path, gsasii_path: Path, batch_outdir: Path,
                    points: np.ndarray, param_specs: list, max_workers) -> list:
    """Evaluates every row of `points` in parallel, each in its own
    batch_outdir/point<i>/ subfolder. Returns results in the SAME order
    as `points` (not completion order), so callers can zip them back up
    with the positions that produced them."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(evaluate_point, checkpoint, gsasii_path,
                        batch_outdir / f"point{i:03d}",
                        logic.position_to_values(pos, param_specs))
            for i, pos in enumerate(points)
        ]
        return [f.result() for f in futures]


def main(argv=None) -> int:
    args = parse_args(argv)

    def emit(event: dict):
        if args.emit_events:
            print(json.dumps(event), flush=True)

    if not args.checkpoint.is_file():
        print(f"ERROR: --checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.gsasii_path.is_dir():
        print(f"ERROR: --gsasii-path not found: {args.gsasii_path}", file=sys.stderr)
        return 2

    # Size and Mustrain are physically positive, scale-varying quantities
    # (see perturb_points()'s multiplicative sampling) — a zero, negative,
    # or inverted bound here isn't a valid search space and would only
    # surface later as a confusing degenerate fit/NaN, not a clear error.
    for flag, bounds in (("--size-bounds", args.size_bounds), ("--mustrain-bounds", args.mustrain_bounds)):
        lo, hi = bounds
        if lo <= 0 or hi <= lo:
            print(f"ERROR: {flag} must satisfy 0 < LO < HI, got ({lo}, {hi})", file=sys.stderr)
            return 2

    # Angle cutoffs are degrees to discard, so 0 is a legitimate lower
    # bound (unlike Size/Mustrain above) — only reject negative or
    # inverted ranges.
    for flag, bounds in (("--low-angle-cutoff-bounds", args.low_angle_cutoff_bounds),
                          ("--high-angle-cutoff-bounds", args.high_angle_cutoff_bounds)):
        if bounds is None:
            continue
        lo, hi = bounds
        if lo < 0 or hi <= lo:
            print(f"ERROR: {flag} must satisfy 0 <= LO < HI, got ({lo}, {hi})", file=sys.stderr)
            return 2

    try:
        backend = logic.pick_backend(args.backend)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        n_phases = _count_phases(args.checkpoint, args.gsasii_path)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not open --checkpoint: {exc!r}", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    eval_dir = args.outdir / "evaluations"
    low_angle_cutoff_bounds = (tuple(args.low_angle_cutoff_bounds)
                                if args.low_angle_cutoff_bounds is not None else None)
    high_angle_cutoff_bounds = (tuple(args.high_angle_cutoff_bounds)
                                 if args.high_angle_cutoff_bounds is not None else None)
    param_specs = logic.build_param_specs(n_phases, tuple(args.size_bounds),
                                           tuple(args.mustrain_bounds),
                                           low_angle_cutoff_bounds, high_angle_cutoff_bounds)
    n_dims = len(param_specs)
    rng = np.random.default_rng(args.seed)

    print(f"Searching {n_dims} dimension(s) across {n_phases} phase(s) from "
          f"{args.checkpoint}, backend={backend!r} ...", flush=True)

    # Seed point: the same starting values gsas2_auto_refine.py's own
    # profile_microstrain_size primary config uses (isotropic Size=10,
    # uniaxial Mustrain=1000/1000 per phase) — so the first thing this
    # script reports is directly comparable to what the deterministic
    # pipeline alone would have found from this checkpoint. The seed for
    # low/high_angle_cutoff (when enabled) is 0.0 — no extra trim beyond
    # whatever the checkpoint already has — for the same reason. Appended
    # in the SAME order build_param_specs adds them (low before high).
    x0 = np.tile([10.0, 1000.0, 1000.0], n_phases)
    if low_angle_cutoff_bounds is not None:
        x0 = np.append(x0, 0.0)
    if high_angle_cutoff_bounds is not None:
        x0 = np.append(x0, 0.0)
    seed_result = evaluate_point(args.checkpoint, args.gsasii_path, eval_dir / "seed",
                                  logic.position_to_values(x0, param_specs))
    seed_fitness = logic.evaluation_to_fitness(seed_result)
    print(f"  seed point: Rwp={seed_result.get('rwp')}  sane={seed_result.get('sane')}", flush=True)

    # worst_sane_rwp anchors the bounded penalty training_target() gives
    # insane-but-computed points — see that function's docstring for why
    # the surrogate needs SOME signal near the sane/insane boundary
    # rather than being blind to it (confirmed as a real cause of
    # systematically-too-optimistic predictions on real data).
    dataset_X, dataset_y = [], []
    worst_sane_rwp = None
    if seed_fitness < logic.UNSOUND_PENALTY:
        dataset_X.append(x0)
        dataset_y.append(seed_fitness)
        worst_sane_rwp = seed_fitness
    best_x, best_fitness, best_result = x0, seed_fitness, seed_result

    history = [best_fitness]
    for outer_iter in range(args.outer_iterations):
        iter_dir = eval_dir / f"iter{outer_iter:03d}"

        candidates = logic.perturb_points(best_x, param_specs, args.perturbations, rng,
                                           args.close_frac, args.close_sigma, args.far_sigma)
        results = evaluate_batch(args.checkpoint, args.gsasii_path, iter_dir / "perturb",
                                  candidates, param_specs, args.max_workers)
        fitnesses = np.array([logic.evaluation_to_fitness(r) for r in results])

        # Pass 1: find this batch's best/worst-sane points first, so pass
        # 2's insane-point penalties (which key off worst_sane_rwp) see
        # every sane point in the batch, not just ones earlier in the list.
        n_sane = 0
        for point, fit, result in zip(candidates, fitnesses, results):
            if fit >= logic.UNSOUND_PENALTY:
                continue
            n_sane += 1
            if fit < best_fitness:
                best_x, best_fitness, best_result = point, fit, result
            if worst_sane_rwp is None or fit > worst_sane_rwp:
                worst_sane_rwp = fit

        # Pass 2: every real evaluation with a usable Rwp — sane or not —
        # becomes surrogate training data (see logic.training_target).
        for point, result in zip(candidates, results):
            target = logic.training_target(result, worst_sane_rwp)
            if target is not None:
                dataset_X.append(point)
                dataset_y.append(target)

        proposed_note = ""
        min_dataset_points = round(
            logic.min_points_for_degree(n_dims, args.surrogate_degree, margin=0)
            * args.surrogate_min_multiplier)
        if len(dataset_X) >= min_dataset_points:
            dataset_X_arr = np.array(dataset_X)
            model = logic.fit_surrogate(dataset_X_arr, np.array(dataset_y),
                                         args.surrogate_degree, backend,
                                         ridge_alpha=args.surrogate_ridge)
            proposed_candidates = logic.search_surrogate(
                model, param_specs, dataset_X_arr, args.surrogate_particles,
                args.surrogate_generations, rng, args.w, args.c1, args.c2,
                explorer_frac=args.surrogate_explorer_frac,
                explorer_c_scale=args.surrogate_explorer_c_scale,
                n_candidates=args.surrogate_candidates,
                min_separation=args.surrogate_min_separation)
            proposed_positions = np.array([c[0] for c in proposed_candidates])
            predicted_fitnesses = [c[1] for c in proposed_candidates]
            proposed_results = evaluate_batch(
                args.checkpoint, args.gsasii_path, iter_dir / "surrogate_proposals",
                proposed_positions, param_specs, args.max_workers)
            proposed_fitnesses = [logic.evaluation_to_fitness(r) for r in proposed_results]

            # Same two-pass pattern as the perturbation batch above: find
            # every sane candidate (updating best_x/worst_sane_rwp) before
            # computing any insane candidate's bounded training penalty,
            # so it reflects every sane point THIS batch found too.
            for proposed_x, fit, result in zip(proposed_positions, proposed_fitnesses, proposed_results):
                if fit >= logic.UNSOUND_PENALTY:
                    continue
                if worst_sane_rwp is None or fit > worst_sane_rwp:
                    worst_sane_rwp = fit
                if fit < best_fitness:
                    best_x, best_fitness, best_result = proposed_x, fit, result

            notes = []
            for proposed_x, predicted_fitness, result in zip(
                    proposed_positions, predicted_fitnesses, proposed_results):
                target = logic.training_target(result, worst_sane_rwp)
                if target is not None:
                    dataset_X.append(proposed_x)
                    dataset_y.append(target)
                notes.append(f"Rwp={result.get('rwp')} (predicted {predicted_fitness:.4f}, "
                             f"sane={result.get('sane')})")
            proposed_note = ", surrogate proposed " + "; ".join(notes) if notes else ""

        history.append(best_fitness)
        print(f"  iter {outer_iter:3d}: {n_sane}/{args.perturbations} sane perturbations, "
              f"best so far = {best_fitness:.4f}{proposed_note}", flush=True)
        emit({"event": "iteration_result", "iteration": outer_iter, "n_sane": n_sane,
              "best_fitness": best_fitness, "dataset_size": len(dataset_X)})

        if logic.has_converged(history, args.patience, args.min_improvement):
            print(f"  converged (no >{args.min_improvement:.2%} improvement over "
                  f"the last {args.patience} iterations)", flush=True)
            break

    if best_fitness >= logic.UNSOUND_PENALTY:
        print("\nNo evaluation ever converged to a sane result — nothing trustworthy to "
              "report. Try wider bounds, more perturbations, or check that the checkpoint "
              "itself is from a healthy run.", flush=True)
        emit({"event": "swarm_done", "ok": False})
        return 1

    best_values = logic.position_to_values(best_x, param_specs)
    summary = {
        "checkpoint": str(args.checkpoint),
        "n_phases": n_phases,
        "backend": backend,
        "outer_iterations_run": len(history) - 1,
        "dataset_size": len(dataset_X),
        "fitness_history": history,
        "best_values": best_values,
        "best_result": best_result,
    }
    (args.outdir / "swarm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nBest found: Rwp={best_result.get('rwp')}  sane={best_result.get('sane')}")
    print(f"Starting values: {json.dumps(best_values)}")
    print(f"See {args.outdir / 'swarm_summary.json'} for the full history and every "
          f"evaluation's own output folder under {eval_dir}.")

    emit({"event": "swarm_done", "ok": bool(best_result.get("sane")),
          "best_rwp": best_result.get("rwp"), "summary_path": str(args.outdir / "swarm_summary.json")})

    return 0 if best_result.get("sane") else 1


if __name__ == "__main__":
    raise SystemExit(main())
