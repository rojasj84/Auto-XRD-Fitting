#!/usr/bin/env python3
"""
gsas2_swarm_gui_logic.py — pure, tkinter-free logic used by gsas2_gui.py's
Swarm tab: validation and command-building for gsas2_swarm_optimize.py,
same separation as gsas2_gui_logic.py (see that module's docstring) so this
stays unit-testable without tkinter or GSAS-II — see test_swarm_gui_logic.py.

Every knob gsas2_swarm_optimize.py exposes for controlling run speed
(perturbations per generation, number of generations, surrogate particles/
generations, backend, worker count) is surfaced here as a plain field so the
GUI tab can be a thin wiring layer over it, matching the split every other
tool in this project already uses.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

GPX_FILETYPES = [
    ("GSAS-II project files", "*.gpx"),
    ("All files", "*.*"),
]


@dataclass
class SwarmRunConfig:
    """Everything the GUI needs to launch one swarm-optimize run."""
    checkpoint: str = ""
    gsasii_path: str = ""
    outdir: str = ""
    outer_iterations: int = 20
    perturbations: int = 50
    surrogate_particles: int = 200
    surrogate_generations: int = 150
    backend: str = "auto"
    seed: Optional[int] = None
    max_workers: Optional[int] = None
    # "isotropic" (default) or "uniaxial" — see gsas2_swarm_logic.
    # build_param_specs' mustrain_type docstring: isotropic is the
    # default because uniaxial Mustrain and isotropic Size are confirmed
    # ~98% correlated on real data, causing most of a swarm run's
    # "insane" perturbations.
    mustrain_type: str = "isotropic"
    # "microstrain_size" (default) or "cell" — see gsas2_swarm_optimize.
    # py's --target help. Picks which checkpoint stage/parameter space
    # this run searches; the two are separate searches, not combined.
    target: str = "microstrain_size"
    # --target cell only — see gsas2_swarm_optimize.py's --cell-length-
    # drift/--cell-angle-bounds help for why these default to 0.15/2.0.
    cell_length_drift: float = 0.15
    cell_angle_bounds: float = 2.0
    # Off by default: every candidate's saved GSAS-II project gets
    # deleted immediately once it's known to have lost (only the current
    # best is kept, then copied to a clean best.gpx at the end) — a real
    # run's full evaluation tree otherwise reaches hundreds of MB to
    # multiple GB, confirmed on real runs. See gsas2_swarm_optimize.py's
    # --keep-evaluations help text.
    keep_evaluations: bool = False
    # Fit-range trimming (see gsas2_swarm_logic.build_param_specs's
    # docstring for why these default to None/off, and why the bounds
    # matter a lot: too wide a range lets the search cherry-pick around a
    # genuinely hard-to-fit peak rather than trimming real bad data).
    low_angle_cutoff_bounds: Optional[Tuple[float, float]] = None
    high_angle_cutoff_bounds: Optional[Tuple[float, float]] = None


def validate_swarm_config(cfg: SwarmRunConfig) -> list:
    """Same "collect every problem, don't stop at the first" convention as
    gsas2_gui_logic.validate_run_config, so the GUI can show the user
    everything wrong with one dialog instead of one round-trip per field."""
    problems = []

    if not cfg.checkpoint:
        problems.append("A checkpoint .gpx file is required.")
    elif not Path(cfg.checkpoint).is_file():
        problems.append(f"Checkpoint file not found: {cfg.checkpoint}")

    if not cfg.gsasii_path:
        problems.append("GSAS-II install path is required.")
    elif not Path(cfg.gsasii_path).is_dir():
        problems.append(f"GSAS-II install path not found: {cfg.gsasii_path}")

    if not cfg.outdir:
        problems.append("Output folder is required.")

    if cfg.outer_iterations < 1:
        problems.append("Generations (outer iterations) must be at least 1.")
    if cfg.perturbations < 1:
        problems.append("Perturbations per generation must be at least 1.")
    if cfg.surrogate_particles < 1:
        problems.append("Surrogate particles must be at least 1.")
    if cfg.surrogate_generations < 1:
        problems.append("Surrogate generations must be at least 1.")
    if cfg.max_workers is not None and cfg.max_workers < 1:
        problems.append("Max parallel workers must be at least 1 (or left blank for no limit).")

    if cfg.target == "cell":
        if not (0.0 < cfg.cell_length_drift < 1.0):
            problems.append("Cell length drift must be a fraction between 0 and 1 "
                             "(e.g. 0.15 for 15%).")
        if cfg.cell_angle_bounds <= 0:
            problems.append("Cell angle bounds must be greater than 0.")

    for label, bounds in (("Low-angle cutoff", cfg.low_angle_cutoff_bounds),
                           ("High-angle cutoff", cfg.high_angle_cutoff_bounds)):
        if bounds is None:
            continue
        lo, hi = bounds
        if lo < 0 or hi <= lo:
            problems.append(f"{label} bounds must satisfy 0 <= LO < HI, got ({lo}, {hi}).")

    return problems


def build_swarm_command(cfg: SwarmRunConfig, script_path: str, python_exe: str = "python3") -> list:
    """Builds the gsas2_swarm_optimize.py CLI invocation for `cfg` —
    always with --emit-events so the GUI can parse structured progress
    the same way gsas2_gui_logic.parse_event_line already does for every
    other tool (that function is generic over any {"event": ...} JSON
    line, so it's reused as-is rather than duplicated here)."""
    cmd = [python_exe, "-u", script_path,
           "--checkpoint", cfg.checkpoint,
           "--gsasii-path", cfg.gsasii_path,
           "--outdir", cfg.outdir,
           "--target", cfg.target,
           "--outer-iterations", str(cfg.outer_iterations),
           "--perturbations", str(cfg.perturbations),
           "--surrogate-particles", str(cfg.surrogate_particles),
           "--surrogate-generations", str(cfg.surrogate_generations),
           "--backend", cfg.backend]
    if cfg.target == "cell":
        cmd += ["--cell-length-drift", str(cfg.cell_length_drift),
                "--cell-angle-bounds", str(cfg.cell_angle_bounds)]
    else:
        cmd += ["--mustrain-type", cfg.mustrain_type]
    if cfg.seed is not None:
        cmd += ["--seed", str(cfg.seed)]
    if cfg.max_workers is not None:
        cmd += ["--max-workers", str(cfg.max_workers)]
    if cfg.low_angle_cutoff_bounds is not None:
        cmd += ["--low-angle-cutoff-bounds",
                str(cfg.low_angle_cutoff_bounds[0]), str(cfg.low_angle_cutoff_bounds[1])]
    if cfg.high_angle_cutoff_bounds is not None:
        cmd += ["--high-angle-cutoff-bounds",
                str(cfg.high_angle_cutoff_bounds[0]), str(cfg.high_angle_cutoff_bounds[1])]
    if cfg.keep_evaluations:
        cmd.append("--keep-evaluations")
    cmd.append("--emit-events")
    return cmd


def auto_swarm_outdir(checkpoint_path: str, now: Optional[datetime] = None) -> str:
    """A fresh, timestamped default output folder next to the checkpoint
    file itself — same reasoning as gsas2_gui_logic.auto_outdir: every run
    gets its own folder so re-running never overwrites a previous one's
    evaluations/summary. `now` is a parameter purely so this stays
    testable without mocking the clock."""
    if not checkpoint_path:
        return ""
    now = now or datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    return str(Path(checkpoint_path).resolve().parent / f"swarm_{stamp}")
