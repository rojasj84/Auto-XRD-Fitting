#!/usr/bin/env python3
"""
test_swarm_logic.py — tests for gsas2_swarm_logic.py. No GSAS-II, no
subprocess, no display required — exercises the PSO math and
parameter-space bookkeeping in isolation, plus a full synthetic-function
convergence test (the strongest way to validate the optimizer's own
correctness independent of anything GSAS-II-specific).

Run with: python3 test_swarm_logic.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import gsas2_swarm_logic as logic  # noqa: E402


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise SystemExit(1)


def test_build_param_specs():
    """
    mustrain_type defaults to "isotropic" (one free Mustrain value per
    phase) -- see build_param_specs' docstring for why: uniaxial Mustrain
    and isotropic Size are confirmed ~98%% correlated on real data, and
    isotropic-by-default removes that degenerate pairing at the source.
    See test_build_param_specs_uniaxial_mustrain for the explicit opt-in.
    """
    specs = logic.build_param_specs(2, size_bounds=(0.1, 100.0), mustrain_bounds=(1.0, 5000.0))
    check("2 params per phase (isotropic default) x 2 phases = 4 specs", len(specs) == 4)
    check("order is phase0(size,mustrain), phase1(size,mustrain)",
          [(s.phase_index, s.name) for s in specs] == [
              (0, "size"), (0, "mustrain"),
              (1, "size"), (1, "mustrain"),
          ])
    check("size bounds applied", specs[0].lo == 0.1 and specs[0].hi == 100.0)
    check("mustrain bounds applied", specs[1].lo == 1.0 and specs[1].hi == 5000.0)
    check("every Size/Mustrain spec defaults to multiplicative perturbation",
          all(s.kind == "multiplicative" for s in specs))

    check("low_angle_cutoff_bounds=None (the default) adds no extra dimension",
          len(logic.build_param_specs(2)) == 4)

    try:
        logic.build_param_specs(1, mustrain_type="quadraxial")
        check("an unknown mustrain_type raises ValueError", False)
    except ValueError:
        check("an unknown mustrain_type raises ValueError", True)


def test_build_param_specs_uniaxial_mustrain():
    """The explicit opt-in for anisotropic microstrain broadening --
    reproduces the pre-isotropic-default 3-params-per-phase shape."""
    specs = logic.build_param_specs(2, size_bounds=(0.1, 100.0), mustrain_bounds=(1.0, 5000.0),
                                     mustrain_type="uniaxial")
    check("3 params per phase (uniaxial) x 2 phases = 6 specs", len(specs) == 6)
    check("order is phase0(size,eq,ax), phase1(size,eq,ax)",
          [(s.phase_index, s.name) for s in specs] == [
              (0, "size"), (0, "mustrain_eq"), (0, "mustrain_ax"),
              (1, "size"), (1, "mustrain_eq"), (1, "mustrain_ax"),
          ])
    check("mustrain bounds applied to both eq and ax",
          specs[1].lo == 1.0 and specs[1].hi == 5000.0
          and specs[2].lo == 1.0 and specs[2].hi == 5000.0)


def test_build_param_specs_low_angle_cutoff():
    specs = logic.build_param_specs(2, low_angle_cutoff_bounds=(0.0, 30.0))
    check("adds exactly one extra whole-histogram dimension "
          "(4 base isotropic-default dims + 1 cutoff)",
          len(specs) == 5)
    cutoff_spec = specs[-1]
    check("the extra dimension is phase_index=None (not tied to any one phase)",
          cutoff_spec.phase_index is None)
    check("the extra dimension is named LOW_ANGLE_CUTOFF_PARAM",
          cutoff_spec.name == logic.LOW_ANGLE_CUTOFF_PARAM)
    check("the extra dimension's bounds are as given", cutoff_spec.lo == 0.0 and cutoff_spec.hi == 30.0)
    check("the extra dimension uses ADDITIVE perturbation, not multiplicative "
          "(multiplying a 0.0 seed by anything is always 0.0 -- see perturb_points)",
          cutoff_spec.kind == "additive")


def test_position_to_values():
    specs = logic.build_param_specs(2)  # isotropic default: 2 dims/phase
    position = [10.0, 200.0, 20.0, 400.0]
    values = logic.position_to_values(position, specs)
    check("both phases present", set(values) == {"0", "1"})
    check("phase 0 values correct", values["0"] == {"size": 10.0, "mustrain": 200.0})
    check("phase 1 values correct", values["1"] == {"size": 20.0, "mustrain": 400.0})


def test_position_to_values_uniaxial_mustrain():
    specs = logic.build_param_specs(2, mustrain_type="uniaxial")
    position = [10.0, 200.0, 300.0, 20.0, 400.0, 500.0]
    values = logic.position_to_values(position, specs)
    check("phase 0 values correct",
          values["0"] == {"size": 10.0, "mustrain_eq": 200.0, "mustrain_ax": 300.0})
    check("phase 1 values correct",
          values["1"] == {"size": 20.0, "mustrain_eq": 400.0, "mustrain_ax": 500.0})


def test_position_to_values_low_angle_cutoff():
    specs = logic.build_param_specs(1, low_angle_cutoff_bounds=(0.0, 30.0))
    position = [10.0, 200.0, 12.5]
    values = logic.position_to_values(position, specs)
    check("phase 0 values still nested under its phase key",
          values["0"] == {"size": 10.0, "mustrain": 200.0})
    check("low_angle_cutoff lands at the TOP level, not nested under a phase",
          values[logic.LOW_ANGLE_CUTOFF_PARAM] == 12.5)


def test_perturb_points_low_angle_cutoff_moves_away_from_a_zero_seed():
    """
    Regression coverage for the exact bug that motivated ParamSpec.kind:
    multiplicative perturbation of a 0.0 seed (0.0 * exp(noise)) is
    ALWAYS 0.0 no matter the noise, so a naive reuse of Size/Mustrain's
    scheme would make low_angle_cutoff permanently stuck at 0 -- never
    actually searched despite being a requested dimension.
    """
    specs = logic.build_param_specs(1, low_angle_cutoff_bounds=(0.0, 30.0))
    x0 = np.array([10.0, 1000.0, 0.0])  # the real seed gsas2_swarm_optimize.py uses (isotropic default)
    rng = np.random.default_rng(4)
    points = logic.perturb_points(x0, specs, n=200, rng=rng, far_sigma=0.9)
    cutoff_values = points[:, -1]
    check("low_angle_cutoff actually moves away from its 0.0 seed",
          cutoff_values.std() > 0.0)
    check("a meaningful spread of low_angle_cutoff values gets sampled, not just noise near 0",
          cutoff_values.max() > 3.0)
    check("every sampled low_angle_cutoff stays within its declared bounds",
          bool(np.all((cutoff_values >= 0.0) & (cutoff_values <= 30.0))))
    check("Size/Mustrain dimensions are unaffected (still positive, still perturbed normally)",
          bool(np.all(points[:, :2] > 0)))


def test_build_param_specs_both_angle_cutoffs_together():
    """
    Both low_angle_cutoff and high_angle_cutoff can be enabled at once —
    low is always appended before high (gsas2_swarm_optimize.py's seed
    construction depends on this exact order matching).
    """
    specs = logic.build_param_specs(1, low_angle_cutoff_bounds=(0.0, 15.0),
                                     high_angle_cutoff_bounds=(0.0, 10.0))
    check("adds exactly two extra whole-histogram dimensions "
          "(2 base isotropic-default dims + 2 cutoffs)",
          len(specs) == 4)
    check("low_angle_cutoff comes before high_angle_cutoff",
          specs[-2].name == logic.LOW_ANGLE_CUTOFF_PARAM
          and specs[-1].name == logic.HIGH_ANGLE_CUTOFF_PARAM)
    check("high_angle_cutoff also uses additive perturbation",
          specs[-1].kind == "additive")
    check("high_angle_cutoff's bounds are as given",
          specs[-1].lo == 0.0 and specs[-1].hi == 10.0)

    position = [10.0, 1000.0, 8.0, 4.0]
    values = logic.position_to_values(position, specs)
    check("both cutoffs land at the top level with their own values",
          values[logic.LOW_ANGLE_CUTOFF_PARAM] == 8.0
          and values[logic.HIGH_ANGLE_CUTOFF_PARAM] == 4.0)

    x0 = np.array([10.0, 1000.0, 0.0, 0.0])
    points = logic.perturb_points(x0, specs, n=200, rng=np.random.default_rng(6), far_sigma=0.9)
    check("both cutoff dimensions independently move away from their 0.0 seeds",
          points[:, 2].std() > 0.0 and points[:, 3].std() > 0.0)
    check("both cutoff dimensions stay within their own (different) bounds",
          bool(np.all((points[:, 2] >= 0.0) & (points[:, 2] <= 15.0)))
          and bool(np.all((points[:, 3] >= 0.0) & (points[:, 3] <= 10.0))))


def test_cell_free_params():
    """
    Every (SGLaue, SGUniq) pairing here was confirmed directly against a
    real GSAS-II install this session (GSASIIspc.SpcGroup() on one
    representative space group per row: Fm-3m, P4/mmm, P21/c, C2/m, Pnma,
    P-1, R-3c in both hexagonal and rhombohedral axes settings) and cross-
    checked against the installed GSASIIstrIO.cellVary() source, which
    every real GSAS-II refine call uses to build the same cell-parameter
    equivalences.
    """
    check("cubic (m3m): only a is free",
          logic.cell_free_params({"SGLaue": "m3m", "SGUniq": ""}) == ["length_a"])
    check("cubic (m3): only a is free",
          logic.cell_free_params({"SGLaue": "m3", "SGUniq": ""}) == ["length_a"])
    check("tetragonal (4/mmm): a and c are free, b is not",
          logic.cell_free_params({"SGLaue": "4/mmm", "SGUniq": ""})
          == ["length_a", "length_c"])
    check("hexagonal-setting trigonal (3m1, e.g. R-3c hexagonal axes): a and c free",
          logic.cell_free_params({"SGLaue": "3m1", "SGUniq": ""})
          == ["length_a", "length_c"])
    check("rhombohedral-setting trigonal (3mR, e.g. R-3c rhombohedral axes): a and alpha free",
          logic.cell_free_params({"SGLaue": "3mR", "SGUniq": ""})
          == ["length_a", "angle_alpha"])
    check("orthorhombic (mmm): a, b, c free, no angles",
          logic.cell_free_params({"SGLaue": "mmm", "SGUniq": ""})
          == ["length_a", "length_b", "length_c"])
    check("monoclinic unique b (2/m, SGUniq='b', the common default): a, b, c, beta free",
          logic.cell_free_params({"SGLaue": "2/m", "SGUniq": "b"})
          == ["length_a", "length_b", "length_c", "angle_beta"])
    check("monoclinic unique a: the free angle is alpha, not beta",
          logic.cell_free_params({"SGLaue": "2/m", "SGUniq": "a"})
          == ["length_a", "length_b", "length_c", "angle_alpha"])
    check("monoclinic unique c: the free angle is gamma, not beta",
          logic.cell_free_params({"SGLaue": "2/m", "SGUniq": "c"})
          == ["length_a", "length_b", "length_c", "angle_gamma"])
    check("triclinic (-1): every parameter is free",
          logic.cell_free_params({"SGLaue": "-1", "SGUniq": ""})
          == ["length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"])

    try:
        logic.cell_free_params({"SGLaue": "not-a-real-laue-class", "SGUniq": ""})
        check("an unrecognized SGLaue raises ValueError rather than guessing", False)
    except ValueError:
        check("an unrecognized SGLaue raises ValueError rather than guessing", True)


def test_reconstruct_cell():
    """reconstruct_cell() must derive every symmetry-dependent value
    (b=a, fixed angles, etc.) instead of leaving it at start_cell's own
    value — the whole point of respecting the phase's crystal system
    rather than treating every one of a/b/c/alpha/beta/gamma as
    independently searchable."""
    start = (5.0, 5.1, 5.2, 89.0, 91.0, 92.0)  # deliberately "wrong" on every
    # dependent slot, so a test that accidentally reads from start_cell
    # instead of deriving the real value would be caught immediately.

    check("cubic: b and c forced equal to a; angles forced to 90",
          logic.reconstruct_cell({"length_a": 4.2}, {"SGLaue": "m3m", "SGUniq": ""}, start)
          == (4.2, 4.2, 4.2, 90.0, 90.0, 90.0))

    check("tetragonal: b forced equal to a; c independent; angles forced to 90",
          logic.reconstruct_cell({"length_a": 4.2, "length_c": 7.5},
                                  {"SGLaue": "4/mmm", "SGUniq": ""}, start)
          == (4.2, 4.2, 7.5, 90.0, 90.0, 90.0))

    check("hexagonal-setting trigonal: b=a, gamma forced to 120, alpha/beta to 90",
          logic.reconstruct_cell({"length_a": 5.2, "length_c": 13.3},
                                  {"SGLaue": "3m1", "SGUniq": ""}, start)
          == (5.2, 5.2, 13.3, 90.0, 90.0, 120.0))

    check("rhombohedral-setting trigonal: b=c=a, beta=gamma=alpha",
          logic.reconstruct_cell({"length_a": 5.4, "angle_alpha": 58.9},
                                  {"SGLaue": "3mR", "SGUniq": ""}, start)
          == (5.4, 5.4, 5.4, 58.9, 58.9, 58.9))

    check("orthorhombic: a, b, c independent; angles forced to 90",
          logic.reconstruct_cell({"length_a": 3.0, "length_b": 4.0, "length_c": 5.0},
                                  {"SGLaue": "mmm", "SGUniq": ""}, start)
          == (3.0, 4.0, 5.0, 90.0, 90.0, 90.0))

    check("monoclinic unique b: a, b, c, beta independent; alpha/gamma forced to 90",
          logic.reconstruct_cell(
              {"length_a": 6.0, "length_b": 7.0, "length_c": 8.0, "angle_beta": 95.0},
              {"SGLaue": "2/m", "SGUniq": "b"}, start)
          == (6.0, 7.0, 8.0, 90.0, 95.0, 90.0))

    check("triclinic: every value comes straight from free_values, none derived",
          logic.reconstruct_cell(
              {"length_a": 1.0, "length_b": 2.0, "length_c": 3.0,
               "angle_alpha": 70.0, "angle_beta": 80.0, "angle_gamma": 100.0},
              {"SGLaue": "-1", "SGUniq": ""}, start)
          == (1.0, 2.0, 3.0, 70.0, 80.0, 100.0))

    check("a partial dict falls back to start_cell for anything missing",
          logic.reconstruct_cell({}, {"SGLaue": "m3m", "SGUniq": ""}, start)
          == (5.0, 5.0, 5.0, 90.0, 90.0, 90.0))


def test_build_cell_param_specs_and_seed():
    start_cells = {0: (5.196, 5.196, 13.309, 90.0, 90.0, 120.0)}  # real FeF3 R-3c cell
    free_params = {0: logic.cell_free_params({"SGLaue": "3m1", "SGUniq": ""})}

    specs = logic.build_cell_param_specs(free_params, start_cells,
                                          length_drift_frac=0.15, angle_bounds_deg=2.0)
    check("one ParamSpec per free cell parameter (a, c) — hexagonal-setting trigonal",
          [(s.phase_index, s.name) for s in specs] == [(0, "length_a"), (0, "length_c")])
    check("length dimensions use multiplicative perturbation",
          all(s.kind == "multiplicative" for s in specs))
    check("length bounds are +/- length_drift_frac around the checkpoint's own cell",
          abs(specs[0].lo - 5.196 * 0.85) < 1e-9 and abs(specs[0].hi - 5.196 * 1.15) < 1e-9
          and abs(specs[1].lo - 13.309 * 0.85) < 1e-9 and abs(specs[1].hi - 13.309 * 1.15) < 1e-9)

    seed = logic.cell_seed_position(free_params, start_cells)
    check("cell_seed_position returns the checkpoint's own current cell, in the same "
          "dimension order as build_cell_param_specs (not an arbitrary universal guess "
          "the way Size/Mustrain's fixed seed is)",
          np.allclose(seed, [5.196, 13.309]))

    # A monoclinic phase's free angle dimension must use ADDITIVE perturbation
    # (an angle isn't a scale-varying quantity the way a length is — same
    # reasoning as LOW_ANGLE_CUTOFF_PARAM's kind="additive").
    mono_free = {0: logic.cell_free_params({"SGLaue": "2/m", "SGUniq": "b"})}
    mono_cells = {0: (6.0, 7.0, 8.0, 90.0, 95.0, 90.0)}
    mono_specs = logic.build_cell_param_specs(mono_free, mono_cells, angle_bounds_deg=2.0)
    angle_spec = next(s for s in mono_specs if s.name == "angle_beta")
    check("a free cell angle uses additive perturbation, not multiplicative",
          angle_spec.kind == "additive")
    check("angle bounds are +/- angle_bounds_deg around the checkpoint's own value",
          abs(angle_spec.lo - 93.0) < 1e-9 and abs(angle_spec.hi - 97.0) < 1e-9)

    # Multi-phase ordering: specs/seed must walk phases in sorted-index
    # order, matching position_to_values()'s own phase-key convention.
    two_phase_free = {1: ["length_a"], 0: ["length_a", "length_c"]}
    two_phase_cells = {0: (5.0, 5.0, 6.0, 90.0, 90.0, 90.0), 1: (3.0, 3.0, 3.0, 90.0, 90.0, 90.0)}
    two_phase_specs = logic.build_cell_param_specs(two_phase_free, two_phase_cells)
    check("multi-phase specs are ordered by phase index, not insertion order",
          [(s.phase_index, s.name) for s in two_phase_specs]
          == [(0, "length_a"), (0, "length_c"), (1, "length_a")])
    two_phase_seed = logic.cell_seed_position(two_phase_free, two_phase_cells)
    check("multi-phase seed matches that same order",
          np.allclose(two_phase_seed, [5.0, 6.0, 3.0]))


def test_position_to_values_cell():
    """Cell ParamSpecs reuse position_to_values() unchanged — same nested-
    under-phase-key shape gsas2_swarm_worker.py's --values already expects
    for Size/Mustrain, just with cell parameter names instead."""
    start_cells = {0: (5.196, 5.196, 13.309, 90.0, 90.0, 120.0)}
    free_params = {0: logic.cell_free_params({"SGLaue": "3m1", "SGUniq": ""})}
    specs = logic.build_cell_param_specs(free_params, start_cells)
    values = logic.position_to_values([5.3, 13.5], specs)
    check("cell values land nested under the phase key, same as Size/Mustrain",
          values == {"0": {"length_a": 5.3, "length_c": 13.5}})


def test_evaluation_to_fitness():
    check("a sane result's fitness is its own Rwp",
          logic.evaluation_to_fitness({"rwp": 6.07, "sane": True, "error": None}) == 6.07)
    check("an insane result (real trap: lower Rwp via a runaway parameter) is penalized",
          logic.evaluation_to_fitness({"rwp": 5.99, "sane": False, "error": None})
          == logic.UNSOUND_PENALTY)
    check("a crashed evaluation (error set) is penalized",
          logic.evaluation_to_fitness({"rwp": None, "sane": False, "error": "boom"})
          == logic.UNSOUND_PENALTY)
    check("the penalty is worse (higher) than any realistic Rwp",
          logic.UNSOUND_PENALTY > 100.0)


def test_training_target():
    """
    Regression coverage for a real bug found on real data: excluding
    every insane point from the surrogate's training set entirely (the
    original behavior) left the surrogate blind to the sane/insane
    boundary, causing it to repeatedly propose the same doomed point and
    predict systematically-too-optimistic Rwp values. training_target()
    is a DIFFERENT function from evaluation_to_fitness() -- this one
    decides what the surrogate should LEARN from a real evaluation, not
    whether to accept it as the run's best answer.
    """
    check("a sane result's training target is its own real Rwp",
          logic.training_target({"rwp": 6.07, "sane": True, "error": None}, worst_sane_rwp=6.5)
          == 6.07)
    check("a crashed evaluation (error set) has no usable training target",
          logic.training_target({"rwp": None, "sane": False, "error": "boom"}, worst_sane_rwp=6.5)
          is None)
    check("an insane evaluation with no rwp computed has no usable training target",
          logic.training_target({"rwp": None, "sane": False, "error": None}, worst_sane_rwp=6.5)
          is None)
    check("an insane evaluation with no sane point seen yet has no usable training target "
          "(nothing to anchor the bounded penalty against)",
          logic.training_target({"rwp": 5.99, "sane": False, "error": None}, worst_sane_rwp=None)
          is None)

    insane_eval = {"rwp": 5.99, "sane": False, "error": None}  # the real, documented trap:
    # a runaway Mustrain blowup produced a deceptively LOW raw Rwp (5.99) --
    # lower than the sane baseline (6.5) used here, mirroring the real
    # FeF3 case (blowup -> 5.997% vs a sane optimum around 6%).
    target = logic.training_target(insane_eval, worst_sane_rwp=6.5)
    check("an insane point's training target is NOT its own (deceptively low) raw Rwp",
          target != insane_eval["rwp"])
    check("an insane point's training target is worse (higher) than the worst sane Rwp seen",
          target > 6.5)
    check("an insane point's training target is exactly worst_sane_rwp + SANE_PENALTY_MARGIN "
          "(a bounded penalty on the real Rwp scale, not UNSOUND_PENALTY's 10,000)",
          target == 6.5 + logic.SANE_PENALTY_MARGIN)


def test_is_better_candidate():
    """
    Covers the peak-amplitude-error tie-breaker: Rwp stays the PRIMARY
    criterion (a clearly-better or clearly-worse Rwp always wins/loses
    outright, regardless of peak match), but when two candidates' Rwp
    values are within tie_margin of each other, the one with the lower
    peak_amplitude_error wins instead -- addresses a real scientist
    concern that Rwp (intensity-weighted) can be dominated by one huge
    peak while other peaks fit worse.
    """
    check("a clearly better Rwp wins outright, even with worse peak match",
          logic.is_better_candidate(fit=5.0, peak_error=0.5,
                                     best_fitness=6.0, best_peak_error=0.1,
                                     tie_margin=0.05) is True)
    check("a clearly worse Rwp loses outright, even with better peak match "
          "(peak match can never override a real Rwp difference outside the margin)",
          logic.is_better_candidate(fit=6.0, peak_error=0.05,
                                     best_fitness=5.0, best_peak_error=0.5,
                                     tie_margin=0.05) is False)

    check("within tie_margin, lower peak_amplitude_error wins even though its Rwp "
          "is slightly worse",
          logic.is_better_candidate(fit=6.02, peak_error=0.10,
                                     best_fitness=6.00, best_peak_error=0.30,
                                     tie_margin=0.05) is True)
    check("within tie_margin, higher peak_amplitude_error loses even though its Rwp "
          "is slightly better",
          logic.is_better_candidate(fit=6.00, peak_error=0.30,
                                     best_fitness=6.02, best_peak_error=0.10,
                                     tie_margin=0.05) is False)

    check("exactly at the tie_margin boundary falls back to plain Rwp comparison "
          "when peak_error is missing",
          logic.is_better_candidate(fit=5.99, peak_error=None,
                                     best_fitness=6.00, best_peak_error=None,
                                     tie_margin=0.05) is True)
    check("missing peak_error on either side falls back to plain Rwp comparison, "
          "even within the tie margin",
          logic.is_better_candidate(fit=6.02, peak_error=0.05,
                                     best_fitness=6.00, best_peak_error=None,
                                     tie_margin=0.05) is False)

    check("tie_margin=0 disables the tie-breaker entirely (pure Rwp comparison)",
          logic.is_better_candidate(fit=6.001, peak_error=0.01,
                                     best_fitness=6.000, best_peak_error=0.99,
                                     tie_margin=0.0) is False)


def test_init_swarm_respects_bounds():
    specs = logic.build_param_specs(1, size_bounds=(1.0, 10.0), mustrain_bounds=(100.0, 200.0))
    rng = np.random.default_rng(0)
    swarm = logic.init_swarm(specs, n_particles=50, rng=rng)
    check("positions shape is (n_particles, n_dims)", swarm.positions.shape == (50, 2))
    check("every initial position is within its dimension's bounds",
          bool(np.all(swarm.positions >= swarm.lo) and np.all(swarm.positions <= swarm.hi)))
    check("pbest starts as a copy of the initial positions",
          np.array_equal(swarm.pbest_positions, swarm.positions))
    check("pbest_fitness starts at +inf (nothing evaluated yet)",
          bool(np.all(np.isinf(swarm.pbest_fitness))))


def test_update_swarm_tracks_best():
    specs = logic.build_param_specs(1)
    rng = np.random.default_rng(1)
    swarm = logic.init_swarm(specs, n_particles=5, rng=rng)
    fitness = np.array([10.0, 5.0, 8.0, 3.0, 9.0])
    swarm = logic.update_swarm(swarm, fitness, w=0.5, c1=1.5, c2=1.5, rng=rng)

    check("gbest fitness is the minimum of this generation's fitness",
          swarm.gbest_fitness == 3.0)
    check("gbest position is particle 3's (the one with fitness 3.0)",
          np.array_equal(swarm.gbest_position, swarm.pbest_positions[3]))
    check("every particle's pbest was set on the first generation (all improved from inf)",
          bool(np.all(np.isfinite(swarm.pbest_fitness))))
    check("positions stay within bounds after the velocity update",
          bool(np.all(swarm.positions >= swarm.lo) and np.all(swarm.positions <= swarm.hi)))

    # A worse fitness on generation 2 must not overwrite a better pbest/gbest.
    worse_fitness = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
    prev_gbest_fitness = swarm.gbest_fitness
    swarm = logic.update_swarm(swarm, worse_fitness, w=0.5, c1=1.5, c2=1.5, rng=rng)
    check("gbest never gets worse from one generation to the next",
          swarm.gbest_fitness == prev_gbest_fitness)


def test_has_converged():
    check("still-improving history is not converged",
          not logic.has_converged([10.0, 8.0, 6.0, 4.0, 2.0], patience=3, min_improvement_frac=0.01))
    check("a flat history over the patience window is converged",
          logic.has_converged([5.0, 5.0, 5.0, 5.0, 5.0], patience=3, min_improvement_frac=0.01))
    check("too short a history is never converged yet",
          not logic.has_converged([5.0, 5.0], patience=3, min_improvement_frac=0.01))
    check("a tiny improvement below the threshold still counts as converged",
          logic.has_converged([5.0, 4.999, 4.998, 4.997], patience=3, min_improvement_frac=0.01))


def test_pso_converges_on_a_known_function():
    """
    The strongest correctness check for the optimizer itself, independent
    of anything GSAS-II-specific: run the full PSO loop against a plain
    synthetic function with a known minimum (a shifted 2D sphere/bowl —
    same dimensionality as one real phase's Size/Mustrain search space
    under the isotropic-Mustrain default) and confirm it actually finds
    it. If this test passes, the PSO math itself is correct; gsas2_swarm_
    worker.py is then the only thing standing between this and a real
    refinement.
    """
    true_minimum = np.array([300.0, 3000.0])  # plausible real Size/Mustrain values
    specs = logic.build_param_specs(1, size_bounds=(0.01, 1000.0), mustrain_bounds=(0.01, 9000.0))

    def fitness_fn(positions: np.ndarray) -> np.ndarray:
        # Normalized per-dimension so no one axis's larger numeric range
        # dominates the "distance" — mirrors how real Rwp doesn't care
        # that Mustrain's scale is ~1000x Size's.
        span = np.array([spec.hi - spec.lo for spec in specs])
        return np.sum(((positions - true_minimum) / span) ** 2, axis=1)

    rng = np.random.default_rng(42)
    swarm = logic.init_swarm(specs, n_particles=40, rng=rng)
    history = []
    for _ in range(60):
        fitness = fitness_fn(swarm.positions)
        swarm = logic.update_swarm(swarm, fitness, w=0.6, c1=1.6, c2=1.6, rng=rng)
        history.append(swarm.gbest_fitness)
        if logic.has_converged(history, patience=8, min_improvement_frac=0.001):
            break

    check("the swarm's fitness history is monotonically non-increasing (best-so-far never gets worse)",
          all(history[i] >= history[i + 1] - 1e-9 for i in range(len(history) - 1)))
    check("the swarm found a near-zero-fitness point (i.e. found the true minimum)",
          swarm.gbest_fitness < 1e-3)
    relative_error = np.abs(swarm.gbest_position - true_minimum) / (np.array(
        [spec.hi - spec.lo for spec in specs]))
    check("the found position is within 2% of the true minimum in every dimension",
          bool(np.all(relative_error < 0.02)))


def test_perturb_points():
    specs = logic.build_param_specs(1, size_bounds=(0.01, 1000.0), mustrain_bounds=(0.01, 9000.0))
    x0 = np.array([10.0, 1000.0])
    rng = np.random.default_rng(3)
    points = logic.perturb_points(x0, specs, n=200, rng=rng, frac_close=0.6,
                                   close_sigma=0.15, far_sigma=0.75)

    check("returns the requested number of points", points.shape == (200, 2))
    lo = np.array([s.lo for s in specs])
    hi = np.array([s.hi for s in specs])
    check("every perturbed point stays within bounds",
          bool(np.all(points >= lo) and np.all(points <= hi)))
    check("perturbed points are all positive (multiplicative, not additive)",
          bool(np.all(points > 0)))
    # The "close" majority should, on average, land nearer x0 (in log-space,
    # matching the multiplicative perturbation) than the "far" minority.
    n_close = int(round(200 * 0.6))
    log_dist_close = np.mean(np.abs(np.log(points[:n_close]) - np.log(x0)))
    log_dist_far = np.mean(np.abs(np.log(points[n_close:]) - np.log(x0)))
    check("the 'close' perturbations land nearer x0 (in log-space) than the 'far' ones on average",
          log_dist_close < log_dist_far)


def test_pick_backend():
    check("explicit 'cpu' always resolves to 'cpu'", logic.pick_backend("cpu") == "cpu")

    # Whether a real GPU backend is available varies by machine (this
    # project has been run both with no torch installed at all, and with
    # a real ROCm/ AMD GPU set up) — so this checks pick_backend()'s
    # BEHAVIOR relative to actual availability, not a hardcoded assumption
    # about what any one environment has installed.
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    check("'auto' resolves to 'gpu' iff a real GPU backend is genuinely available",
          logic.pick_backend("auto") == ("gpu" if has_gpu else "cpu"))

    if has_gpu:
        check("explicit 'gpu' request succeeds when a real GPU backend is available",
              logic.pick_backend("gpu") == "gpu")
    else:
        try:
            logic.pick_backend("gpu")
            check("explicit 'gpu' request raises when unavailable, not silently downgrades", False)
        except RuntimeError:
            check("explicit 'gpu' request raises when unavailable, not silently downgrades", True)

    try:
        logic.pick_backend("quantum")
        check("an unknown backend name raises ValueError", False)
    except ValueError:
        check("an unknown backend name raises ValueError", True)


def test_min_points_for_degree():
    # degree 2, 3 dims: 1 intercept + 3 linear + 6 quadratic/cross = 10 terms.
    check("degree-2, 3-dim term count matches the known closed form",
          logic.min_points_for_degree(3, 2, margin=0) == 10)
    check("margin adds directly to the requirement",
          logic.min_points_for_degree(3, 2, margin=5) == 15)
    check("degree 1 needs far fewer points than degree 2 for the same dimensionality",
          logic.min_points_for_degree(3, 1, margin=0) < logic.min_points_for_degree(3, 2, margin=0))


def test_fit_surrogate_ridge_regularization():
    """
    Regression coverage for a real bug found on real data: an
    unregularized quadratic (10 free coefficients for a 3-dim, one-phase
    problem) fit to a SPARSE dataset -- right at the bare minimum point
    count, the exact regime a real run hit at iter 1 with only 14/100
    sane perturbations -- swung wildly between training points: the
    surrogate predicted Rwp=2.92 for a point that verified at Rwp=6.11.
    ridge_alpha is the fix (see fit_surrogate()'s docstring); this proves
    both that it actually shrinks the fit, and that a first ridge
    implementation's real bug (penalizing the intercept term, which
    biased every prediction toward 0 regardless of the data's actual
    ~6% baseline) is fixed -- ridge_alpha=5.0 on data centered at 6.0
    must NOT drag predictions toward 0.
    """
    rng = np.random.default_rng(3)
    # mustrain_type="uniaxial" explicitly here: this test's whole point is
    # exercising a specific 3-dim/10-term sparse-fit regime (see n_terms
    # below), independent of which mustrain_type happens to be the
    # swarm's own default.
    specs = logic.build_param_specs(1, size_bounds=(0.01, 1000.0), mustrain_bounds=(0.01, 9000.0),
                                     mustrain_type="uniaxial")
    n_terms = len(logic._poly_terms(3, 2))  # 10

    x0 = np.array([300.0, 3000.0, 5000.0])
    # n_terms + 2 points for 10 coefficients -- deliberately sparse, not
    # the generous 60-point regime test_fit_and_predict_surrogate_
    # recovers_a_known_quadratic uses.
    X = logic.perturb_points(x0, specs, n=n_terms + 2, rng=rng,
                              frac_close=0.5, close_sigma=0.05, far_sigma=0.15)
    y = rng.normal(6.0, 0.3, size=X.shape[0])  # noisy "Rwp" around a ~6% baseline

    model_unreg = logic.fit_surrogate(X, y, degree=2, backend="cpu", ridge_alpha=0.0)
    model_reg = logic.fit_surrogate(X, y, degree=2, backend="cpu", ridge_alpha=5.0)

    check("ridge_alpha=0 reproduces plain (manual) OLS exactly (fit against log(y) — "
          "see fit_surrogate's docstring for why)",
          np.allclose(model_unreg.coeffs,
                      np.linalg.lstsq(logic._poly_features_numpy(
                          (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0)),
                          logic._poly_terms(3, 2)), np.log(y), rcond=None)[0]))
    check("ridge regularization shrinks the fitted coefficients overall",
          np.linalg.norm(model_reg.coeffs) < np.linalg.norm(model_unreg.coeffs))
    check("ridge regularization does NOT drag the (log-space) intercept toward "
          "exp(0)=1 (it must stay near the data's actual ~6.0 baseline)",
          abs(np.exp(model_reg.coeffs[0]) - 6.0) < 1.0)
    check("ridge regularization increases training-point error (the standard "
          "bias/variance tradeoff -- it must not be a free lunch)",
          np.sqrt(np.mean((logic.predict_surrogate(model_reg, X) - y) ** 2))
          > np.sqrt(np.mean((logic.predict_surrogate(model_unreg, X) - y) ** 2)))

    probe = (x0 + np.array([5.0, -20.0, 15.0]))[None, :]
    pred_reg = logic.predict_surrogate(model_reg, probe)[0]
    check("a regularized prediction near the training cloud stays physically "
          "plausible (no wild interpolation-instability excursion like the real Rwp=2.92 miss)",
          4.0 < pred_reg < 8.0)


def test_fit_and_predict_surrogate_recovers_a_known_quadratic():
    """
    The strongest correctness check for the surrogate itself: sample a
    KNOWN function (no noise) at enough points, fit a degree-2 surrogate
    to those samples, and confirm it predicts that same function
    accurately at NEW, unseen points — i.e. the fit actually recovered
    the underlying shape, not just memorized the training points.

    The true function is exp(quadratic), not a bare quadratic — matching
    what fit_surrogate() actually models now (see its docstring: it fits
    log(y) ~ polynomial(x) so every prediction is positive by
    construction, since real Rwp values can never be negative). log(this
    true_fn) is EXACTLY a quadratic in x, so a correct log-space fit
    should recover it just as precisely as the old direct-quadratic test
    did — this isn't a loosened tolerance, it's testing against the
    actual functional form being fit. ridge_alpha=0 (plain OLS)
    deliberately — with 60 clean, noiseless points for 10 coefficients
    there's nothing to regularize against, and this test's whole point is
    isolating "does the design-matrix/least-squares math itself work"
    from the bias/variance tradeoff ridge_alpha introduces (see
    test_fit_surrogate_ridge_regularization for that).
    """
    rng = np.random.default_rng(7)
    true_minimum = np.array([300.0, 3000.0, 5000.0])
    curvature = np.array([1.0, 0.5, 2.0])
    # mustrain_type="uniaxial": this test's 3-dim shape/curvature array
    # predate the isotropic default and aren't about mustrain semantics.
    specs = logic.build_param_specs(1, size_bounds=(0.01, 1000.0), mustrain_bounds=(0.01, 9000.0),
                                     mustrain_type="uniaxial")
    span = np.array([s.hi - s.lo for s in specs])

    def true_fn(X: np.ndarray) -> np.ndarray:
        # Normalized deviation keeps the exponent's argument modest (no
        # overflow) regardless of Size/Mustrain's very different raw
        # scales — 6.0 ~ a baseline Rwp at the true minimum itself.
        return 6.0 * np.exp(np.sum(curvature * ((X - true_minimum) / span) ** 2, axis=1))

    X_train = logic.perturb_points(true_minimum, specs, n=60, rng=rng,
                                    frac_close=0.5, close_sigma=0.2, far_sigma=0.6)
    y_train = true_fn(X_train)

    model = logic.fit_surrogate(X_train, y_train, degree=2, backend="cpu", ridge_alpha=0.0)
    check("fit_surrogate returns coefficients for every degree-2 term",
          model.coeffs.shape[0] == logic.min_points_for_degree(3, 2, margin=0))

    X_test = logic.perturb_points(true_minimum, specs, n=30, rng=rng,
                                   frac_close=0.5, close_sigma=0.2, far_sigma=0.6)
    y_test_true = true_fn(X_test)
    y_test_pred = logic.predict_surrogate(model, X_test)
    max_relative_error = np.max(np.abs(y_test_pred - y_test_true) / y_test_true)
    check("the surrogate predicts unseen points within 5% of the true (noiseless) function",
          max_relative_error < 0.05)
    check("every prediction is strictly positive (guaranteed by construction, not just "
          "true for this particular function)",
          bool(np.all(y_test_pred > 0)))


def test_search_surrogate_finds_the_fitted_minimum():
    """
    End-to-end check of search_surrogate(): fit a surrogate to samples
    of a known bowl function, then confirm PSO-over-the-surrogate
    actually finds a point close to that function's true minimum —
    exercising fit + predict + the in-memory PSO loop together, the
    same combination gsas2_swarm_optimize.py's outer loop uses every
    iteration.
    """
    rng = np.random.default_rng(11)
    true_minimum = np.array([300.0, 3000.0, 5000.0])
    # mustrain_type="uniaxial": this test's 3-dim true_minimum predates the
    # isotropic default and isn't about mustrain semantics.
    specs = logic.build_param_specs(1, size_bounds=(0.01, 1000.0), mustrain_bounds=(0.01, 9000.0),
                                     mustrain_type="uniaxial")
    baseline = 0.01  # a small but strictly positive "Rwp at the true minimum" —
    # fit_surrogate requires y > 0 (see its docstring: it fits log(y)), so this
    # can't be exactly 0 the way a bare quadratic bowl's minimum would be.

    def true_fn(X: np.ndarray) -> np.ndarray:
        span = np.array([s.hi - s.lo for s in specs])
        return baseline * np.exp(np.sum(((X - true_minimum) / span) ** 2, axis=1))

    X_train = logic.perturb_points(true_minimum, specs, n=60, rng=rng,
                                    frac_close=0.5, close_sigma=0.3, far_sigma=0.9)
    y_train = true_fn(X_train)
    # ridge_alpha=0 -- see test_fit_and_predict_surrogate_recovers_a_known_quadratic's
    # docstring for why this test isolates fit+PSO correctness from the
    # regularization tradeoff.
    model = logic.fit_surrogate(X_train, y_train, degree=2, backend="cpu", ridge_alpha=0.0)

    candidates = logic.search_surrogate(
        model, specs, X_train, n_particles=30, n_generations=40, rng=rng)
    check("search_surrogate returns exactly 1 candidate by default (n_candidates=1)",
          len(candidates) == 1)
    best_position, best_predicted = candidates[0]

    span = np.array([s.hi - s.lo for s in specs])
    relative_error = np.abs(best_position - true_minimum) / span
    check("search_surrogate finds a point within 5% of the true minimum (via the fitted surrogate)",
          bool(np.all(relative_error < 0.05)))
    check("the predicted fitness at that point is close to the true minimum's own "
          f"value ({baseline}), not near some unrelated number",
          best_predicted < baseline * 3)
    check("the predicted fitness is strictly positive (guaranteed by construction)",
          best_predicted > 0)


def test_explorer_particles_pull_more_weakly():
    """
    Direct mechanism test for --surrogate-explorer-frac: explorer
    particles must move less per generation than normal ones toward the
    SAME pbest/gbest targets (weaker pull, not zero pull -- see
    update_swarm's docstring), which is what lets them keep wandering
    instead of collapsing onto the swarm's single attractor.
    """
    specs = logic.build_param_specs(1, size_bounds=(0.01, 1000.0), mustrain_bounds=(0.01, 9000.0))
    rng = np.random.default_rng(5)
    swarm = logic.init_swarm(specs, n_particles=200, rng=rng, explorer_frac=0.5)
    check("roughly half the particles are flagged as explorers",
          0.3 < swarm.is_explorer.mean() < 0.7)

    # Force every particle's pbest and the shared gbest away from its
    # current position by the same amount, and zero out inertia (w=0), so
    # the ONLY thing driving movement this step is the pbest/gbest pull
    # -- isolates explorer_c_scale's effect cleanly.
    swarm.pbest_positions = swarm.positions + 50.0
    swarm.pbest_fitness = np.full(200, 5.0)
    swarm.gbest_position = swarm.positions[0] + 50.0
    swarm.gbest_fitness = 1.0
    swarm.velocities[:] = 0.0
    fitness = np.full(200, 10.0)  # worse than pbest everywhere -> pbest unchanged this step

    updated = logic.update_swarm(swarm, fitness, w=0.0, c1=1.6, c2=1.6, rng=rng, explorer_c_scale=0.2)
    step_size = np.linalg.norm(updated.velocities, axis=1)
    explorer_step = step_size[updated.is_explorer].mean()
    normal_step = step_size[~updated.is_explorer].mean()

    check("explorer particles move noticeably less per step than normal particles "
          "toward the same targets", explorer_step < normal_step * 0.5)
    check("explorer particles still move somewhat (weakened, not a pure random walk)",
          explorer_step > 0)

    # explorer_c_scale=1.0 (the default) must be a complete no-op relative
    # to plain PSO -- confirms nothing changed for existing callers that
    # never opt into explorer_frac/explorer_c_scale at all.
    swarm2 = logic.init_swarm(specs, n_particles=50, rng=np.random.default_rng(5), explorer_frac=0.0)
    check("explorer_frac=0.0 flags no particles as explorers (the default, backward-compatible)",
          not swarm2.is_explorer.any())


def test_select_diverse_candidates():
    """
    Direct, deterministic test of the greedy diverse-candidate selection
    search_surrogate() uses to turn --surrogate-candidates > 1 into
    genuinely distinct proposals rather than near-duplicates of the same
    basin. Uses hand-built positions/fitness rather than a full PSO run,
    since the selection LOGIC itself (not emergent swarm dynamics) is
    what this needs to verify reliably.
    """
    # mustrain_type="uniaxial": this test's 3-dim hand-built positions
    # below predate the isotropic default and aren't about mustrain
    # semantics.
    specs = logic.build_param_specs(1, size_bounds=(0.01, 1000.0), mustrain_bounds=(0.01, 9000.0),
                                     mustrain_type="uniaxial")
    model = logic.fit_surrogate(
        logic.perturb_points(np.array([300.0, 3000.0, 5000.0]), specs, n=20,
                              rng=np.random.default_rng(1)),
        np.linspace(6.0, 6.5, 20), degree=1, backend="cpu")

    # Two tight clusters (near-duplicates within each) plus one lone
    # point, in NORMALIZED units so the separation threshold is directly
    # interpretable: cluster A ~(0,0,0), cluster B ~(5,5,5) normalized.
    raw_a = model.x_mean + np.array([0.0, 0.0, 0.0]) * model.x_scale
    raw_a2 = model.x_mean + np.array([0.01, 0.0, 0.0]) * model.x_scale
    raw_b = model.x_mean + np.array([5.0, 5.0, 5.0]) * model.x_scale
    raw_b2 = model.x_mean + np.array([5.01, 5.0, 5.0]) * model.x_scale
    raw_c = model.x_mean + np.array([-5.0, 5.0, -5.0]) * model.x_scale
    positions = np.array([raw_a, raw_a2, raw_b, raw_b2, raw_c])
    fitness = np.array([1.0, 1.1, 2.0, 2.1, 3.0])  # a (best), a2, b, b2, c (worst)

    picked_pos, picked_fit = logic._select_diverse_candidates(
        positions, fitness, model, n_candidates=3, min_separation=1.0)
    check("picks the best point first", np.allclose(picked_pos[0], raw_a))
    check("skips the near-duplicate of the best point (too close)",
          not any(np.allclose(p, raw_a2) for p in picked_pos))
    check("picks the next sufficiently-separated point (cluster b's best)",
          any(np.allclose(p, raw_b) for p in picked_pos))
    check("picks the third sufficiently-separated point (the lone point)",
          any(np.allclose(p, raw_c) for p in picked_pos))
    check("returns exactly n_candidates when that many distinct points exist",
          len(picked_pos) == 3)

    check("asking for more candidates than exist returns only the distinct ones found",
          len(logic._select_diverse_candidates(positions, fitness, model,
                                                n_candidates=10, min_separation=1.0)[0]) == 3)
    check("an absurdly large separation threshold still returns at least the single best point",
          len(logic._select_diverse_candidates(positions, fitness, model,
                                                n_candidates=3, min_separation=1e6)[0]) == 1)


def test_search_surrogate_n_candidates_wiring():
    """
    Mechanical check that search_surrogate's n_candidates/min_separation
    parameters are actually threaded through end-to-end (fit -> PSO ->
    selection), independent of whether a given seed's PSO run happens to
    produce a lot of natural spread — see test_select_diverse_candidates
    for the selection logic itself and test_explorer_particles_pull_more_
    weakly for the pull mechanism.
    """
    true_minimum = np.array([300.0, 3000.0, 5000.0])
    # mustrain_type="uniaxial": this test's 3-dim true_minimum predates
    # the isotropic default and isn't about mustrain semantics.
    specs = logic.build_param_specs(1, size_bounds=(0.01, 1000.0), mustrain_bounds=(0.01, 9000.0),
                                     mustrain_type="uniaxial")
    span = np.array([s.hi - s.lo for s in specs])

    def true_fn(X):
        # baseline > 0 -- fit_surrogate requires strictly positive
        # targets (it fits log(y), see its docstring); a bare quadratic
        # bowl would hit exactly 0 at true_minimum.
        return 0.01 * np.exp(np.sum(((X - true_minimum) / span) ** 2, axis=1))

    X_train = logic.perturb_points(true_minimum, specs, n=60, rng=np.random.default_rng(9),
                                    frac_close=0.5, close_sigma=0.3, far_sigma=0.9)
    model = logic.fit_surrogate(X_train, true_fn(X_train), degree=2, backend="cpu", ridge_alpha=0.0)

    default_candidates = logic.search_surrogate(
        model, specs, X_train, n_particles=50, n_generations=50, rng=np.random.default_rng(9))
    check("default n_candidates=1 returns exactly one candidate",
          len(default_candidates) == 1)

    many_candidates = logic.search_surrogate(
        model, specs, X_train, n_particles=50, n_generations=50, rng=np.random.default_rng(9),
        n_candidates=4, min_separation=1e-6)
    check("a large n_candidates with a negligible separation threshold returns "
          "that many candidates", len(many_candidates) == 4)
    check("returned candidates are sorted best (lowest predicted fitness) first",
          all(many_candidates[i][1] <= many_candidates[i + 1][1]
              for i in range(len(many_candidates) - 1)))


def test_trust_region_specs():
    """
    Regression coverage for the real bug found on real data: without
    narrowing to where the surrogate actually has training data,
    search_surrogate() explored the FULL original bounds and proposed
    physically-impossible negative "Rwp" — pure extrapolation nonsense —
    because a quadratic fit to one small data patch is meaningless far
    outside it.
    """
    # mustrain_type="uniaxial": this test's 3-dim hand-built training
    # data predates the isotropic default and isn't about mustrain
    # semantics.
    specs = logic.build_param_specs(1, size_bounds=(0.01, 1000.0), mustrain_bounds=(0.01, 9000.0),
                                     mustrain_type="uniaxial")
    # Training data all clustered in a small patch, far from the full bounds.
    X = np.array([
        [295.0, 2950.0, 4950.0],
        [305.0, 3050.0, 5050.0],
        [300.0, 3000.0, 5000.0],
        [310.0, 2900.0, 4900.0],
    ])
    region = logic.trust_region_specs(specs, X, margin_frac=0.25)

    check("trust region is narrower than the original bounds in every dimension",
          all(r.hi - r.lo < s.hi - s.lo for r, s in zip(region, specs)))
    check("trust region still contains every training point",
          all(np.all(X[:, i] >= region[i].lo) and np.all(X[:, i] <= region[i].hi)
              for i in range(3)))
    check("trust region never exceeds the original physical bounds",
          all(r.lo >= s.lo and r.hi <= s.hi for r, s in zip(region, specs)))

    # A single-point dataset (zero spread) must not produce a zero-width
    # (or inverted) region -- the 1e-9 span floor in trust_region_specs.
    single_point = np.array([[300.0, 3000.0, 5000.0]])
    region_single = logic.trust_region_specs(specs, single_point, margin_frac=0.25)
    check("a single-point dataset still yields a valid (non-empty) region",
          all(r.hi > r.lo for r in region_single))


if __name__ == "__main__":
    test_build_param_specs()
    test_build_param_specs_uniaxial_mustrain()
    test_build_param_specs_low_angle_cutoff()
    test_position_to_values()
    test_position_to_values_uniaxial_mustrain()
    test_position_to_values_low_angle_cutoff()
    test_perturb_points_low_angle_cutoff_moves_away_from_a_zero_seed()
    test_build_param_specs_both_angle_cutoffs_together()
    test_cell_free_params()
    test_reconstruct_cell()
    test_build_cell_param_specs_and_seed()
    test_position_to_values_cell()
    test_evaluation_to_fitness()
    test_training_target()
    test_is_better_candidate()
    test_init_swarm_respects_bounds()
    test_update_swarm_tracks_best()
    test_has_converged()
    test_pso_converges_on_a_known_function()
    test_perturb_points()
    test_pick_backend()
    test_min_points_for_degree()
    test_fit_surrogate_ridge_regularization()
    test_fit_and_predict_surrogate_recovers_a_known_quadratic()
    test_search_surrogate_finds_the_fitted_minimum()
    test_explorer_particles_pull_more_weakly()
    test_select_diverse_candidates()
    test_search_surrogate_n_candidates_wiring()
    test_trust_region_specs()
    print("\nAll swarm logic checks passed (no GSAS-II/subprocess required for this test).")
