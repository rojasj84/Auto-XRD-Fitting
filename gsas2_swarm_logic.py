#!/usr/bin/env python3
"""
gsas2_swarm_logic.py — pure, subprocess-free optimization math for
gsas2_swarm_optimize.py: perturbation sampling, a polynomial surrogate
model (CPU/numpy or GPU/torch), and particle swarm optimization (PSO),
used together as a surrogate-assisted search loop.

Searches the *continuous* starting-value space for one or more phases'
isotropic Size and uniaxial Mustrain (equatorial + axial) parameters —
exactly the parameters gsas2_auto_refine.py's own profile_microstrain_size
stage refines, and exactly the ones this project spent an entire session
hand-building individual escape routes for (Size/Mustrain trading off
against each other, one running away while the other collapses — see
that stage's docstring in gsas2_auto_refine.py for the real-data
evidence). Rather than adding one more hand-coded fallback per new trap
discovered, this generalizes: try many different starting points, keep
whichever one GSAS-II's own refinement converges to the best *and
physically sane* result from.

Why a surrogate model, not just raw PSO over real evaluations
---------------------------------------------------------------
An earlier version of this module ran PSO directly against real GSAS-II
evaluations, sampling starting points uniformly across the whole bounded
space. Confirmed as a real problem on real data, not theoretical: most of
that space is badly conditioned (GSAS-II's own solver dropped a Size
parameter as insensitive at one randomly-sampled point, and Mustrain blew
up to 41,297 at another) — random-across-everything wastes most
evaluations on points that were never going to converge to anything
useful. The fix is the approach the user actually specified: run a real
GSAS-II evaluation only to *verify* promising points; do the exploration
by fitting a cheap polynomial surrogate Rwp(x) to every (params, Rwp)
pair evaluated so far, then run PSO against that surrogate (a matrix
multiply, not a subprocess) to propose the next point to actually verify.
The dataset — and therefore the surrogate's accuracy — grows every outer
iteration.

This is where GPU work (see fit_surrogate()/predict_surrogate(),
backend="gpu" via torch) actually has a job to do: the surrogate fit/
search is decoupled from GSAS-II entirely, so it can be scaled up (a much
larger in-memory PSO over the surrogate, run for many generations) for
free relative to the real evaluations' cost. Practically, at the
dimensionality this problem has (a handful of parameters per phase), CPU/
numpy would likely finish the surrogate step just as fast in wall-clock
terms — but the point isn't squeezing out the last millisecond, it's
using hardware that would otherwise sit idle while a scientist starts
this and walks away (their framing, not an assumption on my part).
backend="cpu" (numpy) and backend="gpu" (torch) run identical design-
matrix/least-squares math; both are now verified against real hardware,
including a real AMD/ROCm GPU (torch's `torch.cuda` namespace is a
compatibility shim that covers ROCm too, not just NVIDIA/CUDA).

A second real-data failure, distinct from the one above: even once
search_surrogate() only searches within the trained region (see
trust_region_specs()), an UNREGULARIZED fit through a sparse, noisy
dataset can still invent wild excursions BETWEEN training points while
matching them exactly — an interpolation instability, not an
extrapolation one. Confirmed on real data: the surrogate predicted
Rwp=2.92 for a point that verified at Rwp=6.11. fit_surrogate()'s
`ridge_alpha` (on by default) fixes this the standard way — shrinking
coefficients toward zero — and gsas2_swarm_optimize.py additionally
raises min_points_for_degree()'s own bar before ever trusting a fit at
all. See both functions' docstrings.

A third real-data failure: even with ridge_alpha, an unconstrained
linear-space fit occasionally predicted a physically impossible NEGATIVE
Rwp (e.g. -3.45) for a proposed point — a real evaluation was spent
verifying a candidate the surrogate was confidently wrong about in a way
that couldn't even be true. fit_surrogate() now fits log(Rwp), not Rwp
directly, and predict_surrogate() exponentiates back — every prediction
is positive BY CONSTRUCTION, not by clipping after the fact.
"""

from dataclasses import dataclass, field, replace

import numpy as np

# Size + Mustrain per phase, in this fixed order — see ParamSpec /
# build_param_specs(). Mustrain is either one "mustrain" value
# (mustrain_type="isotropic", the default) or two "mustrain_eq"/
# "mustrain_ax" values (mustrain_type="uniaxial") — see build_param_specs'
# docstring for why isotropic is the default.
MUSTRAIN_TYPES = ("isotropic", "uniaxial")

# The optional whole-histogram (not per-phase) dimensions: how many
# degrees of data to discard from the START/END of the fit range — see
# build_param_specs' low_angle_cutoff_bounds/high_angle_cutoff_bounds and
# gsas2_swarm_worker.py for how these actually get applied
# (hist.set_refinements({"Limits": ...})).
LOW_ANGLE_CUTOFF_PARAM = "low_angle_cutoff"
HIGH_ANGLE_CUTOFF_PARAM = "high_angle_cutoff"


@dataclass
class ParamSpec:
    """One search dimension. `phase_index` is which phase this belongs to
    for a per-phase parameter (Size/Mustrain), or None for a whole-
    histogram parameter like LOW_ANGLE_CUTOFF_PARAM that isn't tied to
    any one phase — see position_to_values(). `lo`/`hi` bound the
    STARTING point the swarm is allowed to place here (GSAS-II's own LM
    refinement still moves values from there during evaluation — see
    gsas2_swarm_worker.py — so these bound the search's starting guesses,
    not the converged result; profile_params_sane already bounds the
    *result* for Size/Mustrain).

    `kind` controls how perturb_points() samples around a starting point:
    "multiplicative" (log-normal, the default — appropriate for Size/
    Mustrain, positive scale-varying quantities where "a 10% change"
    means the same thing at any magnitude) or "additive" (Gaussian,
    scaled by the dimension's own span — for a linear-scale quantity
    whose natural starting point can legitimately BE zero, like
    LOW_ANGLE_CUTOFF_PARAM/HIGH_ANGLE_CUTOFF_PARAM: multiplicative
    perturbation of exactly 0.0 is always 0.0 regardless of noise, so it
    would never be searched at all)."""
    phase_index: "int | None"
    name: str  # "size", "mustrain" (isotropic) or "mustrain_eq"/"mustrain_ax"
               # (uniaxial), LOW_ANGLE_CUTOFF_PARAM, or HIGH_ANGLE_CUTOFF_PARAM
    lo: float
    hi: float
    kind: str = "multiplicative"  # "multiplicative" or "additive"


def build_param_specs(n_phases: int, size_bounds=(0.01, 1000.0),
                       mustrain_bounds=(0.01, 9000.0),
                       mustrain_type: str = "isotropic",
                       low_angle_cutoff_bounds=None,
                       high_angle_cutoff_bounds=None) -> list:
    """One (size, mustrain...) group of ParamSpecs per phase index
    0..n_phases-1, in a fixed, deterministic order — this order is what
    every position/velocity array's columns mean throughout the rest of
    this module. low_angle_cutoff (if given) is always appended before
    high_angle_cutoff (if given) — callers that build a seed/starting
    position must append their own extra values in that same order (see
    gsas2_swarm_optimize.py's x0 construction).

    `mustrain_type` is "isotropic" (the default — one "mustrain"
    dimension per phase) or "uniaxial" (two dimensions per phase,
    "mustrain_eq"/"mustrain_ax"). isotropic is the default because
    uniaxial Mustrain and isotropic Size are confirmed, not theoretically
    possible, to be ~98% correlated for real data (see gsas2_auto_
    refine.py's profile_microstrain_size stage, which already documents
    this and falls back to isotropic Mustrain for exactly this reason):
    diagnosed directly on a real FeF3 checkpoint, EVERY insane swarm
    perturbation (26/26, both close- and far-tier) failed for the same
    reason — Size collapsed to exactly 10.0 while Mustrain exploded to
    unphysical magnitudes (some over a billion), a signature of the
    solver running off along that ~98%-correlated near-degenerate
    direction rather than a badly-chosen starting point per se. Reducing
    Mustrain to one free parameter removes that degenerate pairing at
    the source, instead of generating candidates doomed to fail it and
    then recovering (or not) after the fact. uniaxial is preserved as an
    explicit opt-in for datasets/phases where the extra anisotropic
    microstrain freedom is worth that risk.

    `low_angle_cutoff_bounds`/`high_angle_cutoff_bounds` (e.g. (0.0,
    20.0) / (0.0, 10.0)) each additionally append ONE whole-histogram
    dimension searching how many degrees of data to discard from the
    start/end of the fit range respectively — real low- and high-angle
    regions are sometimes dominated by beamstop shadow/detector-edge
    artifacts, background curvature, or vanishing peak statistics rather
    than genuine signal, and this lets the swarm discover whether
    trimming some of that helps. Either or both default to None (off),
    exactly reproducing prior behavior.

    NOTE both of these change what data Rwp is even computed OVER, so
    Rwp values across different cutoffs aren't strictly apples-to-apples
    the way they are for Size/Mustrain alone — trimming more points can
    mechanically improve Rwp somewhat independent of whether the
    retained-range fit genuinely got better. Confirmed as a REAL risk on
    real data, not theoretical: an earlier (0, 30) degree low-angle
    ceiling let a swarm run discard 40% of a real FeF3 pattern —
    including its single strongest peak (96,000 counts at 2θ≈24°, versus
    a max of 27,775 anywhere in the retained range) — because that peak's
    fit was worse than the rest of the pattern (8.8% vs. 6.4% relative
    RMS error), not because that region was actually bad data. Keep these
    bounds tight, and to whatever range is independently scientifically
    justified (known instrument limitations decided BEFORE looking at
    Rwp) — not wide enough to let the search cherry-pick around a
    genuinely hard-to-fit peak."""
    if mustrain_type not in MUSTRAIN_TYPES:
        raise ValueError(f"mustrain_type must be one of {MUSTRAIN_TYPES}, got {mustrain_type!r}")
    specs = []
    for phase_index in range(n_phases):
        specs.append(ParamSpec(phase_index, "size", *size_bounds))
        if mustrain_type == "isotropic":
            specs.append(ParamSpec(phase_index, "mustrain", *mustrain_bounds))
        else:
            specs.append(ParamSpec(phase_index, "mustrain_eq", *mustrain_bounds))
            specs.append(ParamSpec(phase_index, "mustrain_ax", *mustrain_bounds))
    if low_angle_cutoff_bounds is not None:
        specs.append(ParamSpec(None, LOW_ANGLE_CUTOFF_PARAM, *low_angle_cutoff_bounds,
                                kind="additive"))
    if high_angle_cutoff_bounds is not None:
        specs.append(ParamSpec(None, HIGH_ANGLE_CUTOFF_PARAM, *high_angle_cutoff_bounds,
                                kind="additive"))
    return specs


def position_to_values(position, param_specs: list) -> dict:
    """Converts one particle's flat position vector (len == len(param_specs))
    into the {"<phase index>": {"size":.., "mustrain_eq":.., "mustrain_ax":..},
    ..., "low_angle_cutoff": .., "high_angle_cutoff": ..} shape
    gsas2_swarm_worker.py's --values expects — a whole-histogram spec
    (phase_index=None) lands at the top level under its own name instead
    of nested under a phase key."""
    values: dict = {}
    for spec, coord in zip(param_specs, position):
        if spec.phase_index is None:
            values[spec.name] = float(coord)
            continue
        phase_key = str(spec.phase_index)
        values.setdefault(phase_key, {})[spec.name] = float(coord)
    return values


@dataclass
class Swarm:
    positions: np.ndarray       # (n_particles, n_dims)
    velocities: np.ndarray      # (n_particles, n_dims)
    pbest_positions: np.ndarray  # (n_particles, n_dims)
    pbest_fitness: np.ndarray   # (n_particles,) — lower is better
    lo: np.ndarray              # (n_dims,)
    hi: np.ndarray              # (n_dims,)
    gbest_position: np.ndarray = None   # (n_dims,)
    gbest_fitness: float = field(default=float("inf"))
    # (n_particles,) bool — particles with weakened pbest/gbest pull (see
    # update_swarm's `explorer_c_scale`), so they keep wandering rather
    # than collapsing onto the single best-known point. All-False (the
    # default from init_swarm's explorer_frac=0.0) reproduces plain
    # single-attractor PSO exactly — see search_surrogate()'s docstring
    # for why this exists: without it, --surrogate-candidates > 1 has
    # nothing distinct to find, since every particle converges to the
    # same basin.
    is_explorer: np.ndarray = None


# A fitness value no real (sane) Rwp — a percentage, essentially always
# well under 100 for anything worth reporting — could ever reach. Used
# for both outright failures (the worker's subprocess crashed, GSAS-II's
# solver errored) and "sane": false results (see gsas2_swarm_worker.py —
# Rwp trend / cell drift / profile-parameter bounds), so the swarm can
# never be lured toward a numerically-lower Rwp that's actually a
# runaway parameter — confirmed as a real risk, not theoretical: on real
# FeF3 data, a Mustrain blowup to 49,650 produced an *even lower* Rwp
# (5.997%) than the sane optimum, exactly the trap this exists to avoid.
UNSOUND_PENALTY = 10_000.0


def evaluation_to_fitness(evaluation: dict) -> float:
    """Maps one gsas2_swarm_worker.py JSON result to a scalar to
    MINIMIZE. See UNSOUND_PENALTY's docstring for why unsound results
    are penalized rather than trusted at face value."""
    if evaluation.get("error") is not None or evaluation.get("rwp") is None:
        return UNSOUND_PENALTY
    if not evaluation.get("sane", False):
        return UNSOUND_PENALTY
    return float(evaluation["rwp"])


# How much worse than the worst known-sane Rwp an insane point's surrogate
# TRAINING target is set to — see training_target()'s docstring for why
# this exists as a separate, bounded value rather than either excluding
# insane points outright or using UNSOUND_PENALTY (10,000) itself.
SANE_PENALTY_MARGIN = 2.0


def training_target(evaluation: dict, worst_sane_rwp) -> "float | None":
    """
    Maps one gsas2_swarm_worker.py JSON result to the value fed into the
    surrogate's TRAINING SET — a different question from
    evaluation_to_fitness()'s "should this become our new best answer,"
    which must stay UNSOUND_PENALTY-gated. This one answers "what should
    the surrogate learn from this real evaluation."

    Confirmed necessary on real data, not precautionary: excluding every
    insane point entirely (the original behavior) starves the surrogate
    of any signal near the sane/insane boundary, so a smooth polynomial
    happily interpolates a good-looking value right up to and past a
    cliff it's never been shown — observed as the surrogate proposing the
    same doomed point repeatedly across outer iterations, always
    predicting a lower Rwp than the real, verified one.

    The fix is NOT to feed the insane point's own raw Rwp back in either
    — that number is frequently deceptively LOW, a documented real trap
    (see UNSOUND_PENALTY's docstring: a Mustrain blowup to 49,650
    produced Rwp=5.997%, actually lower than the sane optimum). Doing
    that would train the surrogate to chase the exact trap the sanity
    checks exist to avoid. Instead, every insane-but-computed point gets
    a single BOUNDED penalty — `worst_sane_rwp + SANE_PENALTY_MARGIN` —
    on the same numeric scale as real Rwp values (unlike UNSOUND_PENALTY,
    which would wreck a quadratic fit's conditioning by orders of
    magnitude): far enough above every known-good point to steer the
    surrogate away, without claiming to know exactly how bad it is.

    Returns None (exclude entirely — genuinely no usable information) for
    a crashed/errored evaluation, or for an insane one when no sane point
    has been seen yet at all (nothing to anchor the penalty against).
    `worst_sane_rwp` should be updated by the caller as sane points are
    found — see gsas2_swarm_optimize.py's accumulation loop.
    """
    if evaluation.get("error") is not None or evaluation.get("rwp") is None:
        return None
    if evaluation.get("sane", False):
        return float(evaluation["rwp"])
    if worst_sane_rwp is None:
        return None
    return float(worst_sane_rwp) + SANE_PENALTY_MARGIN


def is_better_candidate(fit: float, peak_error, best_fitness: float, best_peak_error,
                         tie_margin: float) -> bool:
    """
    Decides whether a new (already-confirmed-sane) candidate should
    replace the current best one. Rwp stays the PRIMARY criterion — the
    surrogate/PSO machinery is untouched and keeps optimizing Rwp only,
    since it's cheap and already validated — but when two candidates'
    Rwp values are within `tie_margin` of each other (a near-wash by that
    metric), this breaks the tie using peak_amplitude_error instead.

    Why: Rwp is intensity-weighted, so it's dominated by whichever peak
    happens to be largest — a real scientist's concern, confirmed on real
    FeF3 data: one huge peak can be fit very well (pulling Rwp down)
    while several smaller peaks are comparatively poorly matched, and
    Rwp alone never surfaces that. peak_amplitude_error weighs every
    peak equally instead (see gsas2_swarm_worker.peak_amplitude_error),
    so among near-equally-good-by-Rwp candidates, this prefers the one
    that's actually more evenly accurate across the whole pattern —
    without ever letting a worse-Rwp candidate win outright; that would
    reopen exactly the "deceptively good-looking number" trap
    UNSOUND_PENALTY exists to avoid.

    Falls back to a plain Rwp comparison if either peak_error value is
    unavailable (e.g. a histogram with no reflections at all).
    """
    if fit < best_fitness - tie_margin:
        return True
    if fit > best_fitness + tie_margin:
        return False
    if peak_error is None or best_peak_error is None:
        return fit < best_fitness
    return peak_error < best_peak_error


def init_swarm(param_specs: list, n_particles: int, rng: np.random.Generator,
                explorer_frac: float = 0.0) -> Swarm:
    """`explorer_frac` (default 0.0 = none, exactly reproduces the
    original plain single-attractor PSO) marks that fraction of
    particles, chosen at random, as explorers for the swarm's whole
    lifetime — see update_swarm()'s `explorer_c_scale` for what that
    means, and Swarm.is_explorer's docstring for why a stable subgroup
    (not re-randomized every generation) is what actually preserves
    diversity."""
    lo = np.array([s.lo for s in param_specs], dtype=float)
    hi = np.array([s.hi for s in param_specs], dtype=float)
    span = hi - lo
    positions = lo + rng.random((n_particles, len(param_specs))) * span
    # Initial velocities: a modest fraction of each dimension's span, in
    # a random direction — standard PSO practice (starting at exactly 0
    # would leave every particle motionless until pbest/gbest first
    # differ from its own position).
    velocities = (rng.random((n_particles, len(param_specs))) - 0.5) * span * 0.2
    is_explorer = rng.random(n_particles) < explorer_frac
    return Swarm(
        positions=positions,
        velocities=velocities,
        pbest_positions=positions.copy(),
        pbest_fitness=np.full(n_particles, float("inf")),
        lo=lo,
        hi=hi,
        is_explorer=is_explorer,
    )


def update_swarm(swarm: Swarm, fitness: np.ndarray, w: float, c1: float, c2: float,
                  rng: np.random.Generator, explorer_c_scale: float = 1.0) -> Swarm:
    """One PSO generation step: absorb this generation's fitness values
    (updating personal/global bests), then move every particle per the
    standard velocity-update rule, clamped back into bounds. `fitness`
    must be aligned with swarm.positions (fitness[i] is particle i's
    result this generation) — see evaluation_to_fitness().

    `explorer_c_scale` (default 1.0 = no effect) multiplies BOTH the
    cognitive (c1, pull toward own pbest) and social (c2, pull toward
    gbest) terms for particles swarm.is_explorer flags — weakening,
    not zeroing, their pull toward the best-known point(s) so they still
    drift generally useful directions while remaining much freer to keep
    wandering rather than collapsing into the same basin as everyone
    else. Only takes effect when swarm.is_explorer has any True entries
    (see init_swarm's explorer_frac) — with none, this is numerically
    identical to the original scalar c1/c2 behavior."""
    improved = fitness < swarm.pbest_fitness
    swarm.pbest_fitness = np.where(improved, fitness, swarm.pbest_fitness)
    swarm.pbest_positions[improved] = swarm.positions[improved]

    best_idx = int(np.argmin(swarm.pbest_fitness))
    if swarm.pbest_fitness[best_idx] < swarm.gbest_fitness:
        swarm.gbest_fitness = float(swarm.pbest_fitness[best_idx])
        swarm.gbest_position = swarm.pbest_positions[best_idx].copy()

    n_particles, n_dims = swarm.positions.shape
    r1 = rng.random((n_particles, n_dims))
    r2 = rng.random((n_particles, n_dims))
    if swarm.is_explorer is not None and np.any(swarm.is_explorer):
        c1_vec = np.where(swarm.is_explorer, c1 * explorer_c_scale, c1)
        c2_vec = np.where(swarm.is_explorer, c2 * explorer_c_scale, c2)
        c1_term = c1_vec[:, None] * r1 * (swarm.pbest_positions - swarm.positions)
        c2_term = c2_vec[:, None] * r2 * (swarm.gbest_position - swarm.positions)
    else:
        c1_term = c1 * r1 * (swarm.pbest_positions - swarm.positions)
        c2_term = c2 * r2 * (swarm.gbest_position - swarm.positions)
    swarm.velocities = w * swarm.velocities + c1_term + c2_term
    swarm.positions = np.clip(swarm.positions + swarm.velocities, swarm.lo, swarm.hi)
    return swarm


def has_converged(fitness_history: list, patience: int, min_improvement_frac: float) -> bool:
    """True once the best-so-far fitness hasn't improved by at least
    `min_improvement_frac` (relative) over the last `patience`
    generations — the swarm's equivalent of gsas2_auto_refine.py's
    Bounds.min_optional_improvement_frac: a tiny numerical wobble isn't
    a reason to keep spending CPU time on more generations."""
    if len(fitness_history) <= patience:
        return False
    window = fitness_history[-(patience + 1):]
    best_before = min(window[:-1])
    best_now = min(window)
    if best_before <= 0:
        return best_now >= best_before
    return (best_before - best_now) / best_before < min_improvement_frac


# ---------------------------------------------------------------------------
# Perturbation sampling — generates the real-evaluation candidates around
# the current best point each outer iteration (see gsas2_swarm_optimize.py).
# ---------------------------------------------------------------------------

def perturb_points(x0: np.ndarray, param_specs: list, n: int, rng: np.random.Generator,
                    frac_close: float = 0.6, close_sigma: float = 0.15,
                    far_sigma: float = 0.75) -> np.ndarray:
    """
    Generates `n` candidate points around x0. Each dimension is
    perturbed according to its own ParamSpec.kind:

    - "multiplicative" (log-normal) — Size/Mustrain and any other
      positive, scale-varying quantity where "a 10% change" means the
      same thing whether the value is 10 or 10,000; an additive Gaussian
      perturbation would not have that property (and could also go
      negative).
    - "additive" (Gaussian, scaled by the dimension's own span) — a
      linear-scale quantity whose natural starting point can legitimately
      BE zero, like LOW_ANGLE_CUTOFF_PARAM: multiplicative perturbation of
      exactly 0.0 is always 0.0 no matter the noise, so it would never
      get searched at all under the multiplicative scheme. Scaling by
      span (not a fixed sigma) keeps close_sigma/far_sigma meaning
      roughly "this fraction of the allowed range" for both kinds.

    A majority (`frac_close`) of points get a small perturbation
    (`close_sigma`) to refine near the current best; the rest get a
    larger one (`far_sigma`) so the surrogate also sees points further
    out, rather than only ever tightening around wherever the first
    guess happened to land — the mix the user specified ("some close,
    some far"). Every point is clipped to each dimension's search
    bounds afterward.
    """
    x0 = np.asarray(x0, dtype=float)
    lo = np.array([s.lo for s in param_specs])
    hi = np.array([s.hi for s in param_specs])
    span = hi - lo
    n_close = int(round(n * frac_close))
    sigmas = np.array([close_sigma] * n_close + [far_sigma] * (n - n_close))
    noise = rng.normal(0.0, 1.0, size=(n, len(param_specs))) * sigmas[:, None]

    multiplicative_points = x0[None, :] * np.exp(noise)
    additive_points = x0[None, :] + noise * span[None, :]
    is_additive = np.array([s.kind == "additive" for s in param_specs])
    points = np.where(is_additive[None, :], additive_points, multiplicative_points)
    return np.clip(points, lo, hi)


# ---------------------------------------------------------------------------
# Polynomial surrogate model — fit to the accumulated (params, Rwp) points
# from every real evaluation so far, and searched with PSO in place of real
# GSAS-II calls. See this module's docstring for why the surrogate step (not
# raw PSO over real evaluations, and not the swarm bookkeeping itself) is
# where GPU work actually has a legitimate, if modest, job to do.
# ---------------------------------------------------------------------------

@dataclass
class SurrogateModel:
    coeffs: np.ndarray    # always plain numpy, regardless of which backend fit it
    degree: int           # 1 (linear) or 2 (quadratic, with cross-terms)
    n_dims: int
    x_mean: np.ndarray    # inputs are normalized before fitting/predicting —
    x_scale: np.ndarray   # polynomial features blow up fast on raw Size/Mustrain scales


def pick_backend(requested: str) -> str:
    """
    Resolves 'auto'/'cpu'/'gpu' to an actual backend, 'cpu' or 'gpu'.
    'auto' picks 'gpu' only if PyTorch AND a CUDA device are both
    genuinely available, else 'cpu', silently — it's a preference, not
    a request. An explicit 'gpu' request that can't be satisfied raises
    rather than quietly downgrading to CPU — the same "fail loud, don't
    silently guess" convention this whole project follows elsewhere
    (e.g. gsas2_auto_refine.py's Bounds checks): a scientist who asked
    for their GPU to be used should find out immediately if it wasn't,
    not discover it later from an unexpectedly-slow-but-still-correct
    run.
    """
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    if requested == "gpu":
        if not has_gpu:
            raise RuntimeError("backend='gpu' requested but PyTorch with a CUDA device "
                                "isn't available in this environment.")
        return "gpu"
    if requested == "auto":
        return "gpu" if has_gpu else "cpu"
    raise ValueError(f"unknown backend {requested!r}, expected 'cpu', 'gpu', or 'auto'")


def _poly_terms(n_dims: int, degree: int) -> list:
    """Every term in the polynomial as a tuple of dimension indices:
    () for the intercept, (i,) for a linear term, (i, j) with i<=j for a
    degree-2 term (i==j is x_i^2). Only degree 1/2 are supported — this
    problem's dimensionality (a handful of parameters per phase) and the
    perturbation-around-a-point sampling strategy don't realistically
    supply enough data to fit a higher-degree surface without severe
    overfitting."""
    if degree not in (1, 2):
        raise ValueError(f"degree must be 1 or 2, got {degree}")
    terms = [()] + [(i,) for i in range(n_dims)]
    if degree == 2:
        terms += [(i, j) for i in range(n_dims) for j in range(i, n_dims)]
    return terms


def _poly_features_numpy(Xn: np.ndarray, terms: list) -> np.ndarray:
    cols = []
    for term in terms:
        if not term:
            cols.append(np.ones(Xn.shape[0]))
        elif len(term) == 1:
            cols.append(Xn[:, term[0]])
        else:
            cols.append(Xn[:, term[0]] * Xn[:, term[1]])
    return np.stack(cols, axis=1)


def _ridge_augment_numpy(design: np.ndarray, y: np.ndarray, ridge_alpha: float):
    """Standard augmented-design-matrix trick for L2 (ridge/Tikhonov)
    regularized least squares: appending sqrt(ridge_alpha) * I to the
    design matrix and zeros to the targets, then solving with PLAIN least
    squares on the augmented system, is mathematically equivalent to
    solving argmin ||Xc - y||^2 + ridge_alpha * ||c||^2 directly — without
    needing a separate ridge solver (or forming X^T X, which squares the
    matrix's condition number and is the less numerically stable way to
    do this). Used identically by both the numpy and torch fit paths —
    see fit_surrogate()'s docstring for why this exists at all.

    The intercept term (always column 0 — see _poly_terms()) is NEVER
    penalized: it carries the fit's baseline Rwp level (typically ~5-10),
    and shrinking it toward zero along with the shape coefficients would
    bias every prediction toward 0 regardless of alpha's magnitude —
    caught empirically while adding this: a naive full-I penalty predicted
    Rwp=3.2 for noisy data centered on 6.0, a huge, purely mechanical bias
    with nothing to do with the actual landscape."""
    if ridge_alpha <= 0:
        return design, y
    n_terms = design.shape[1]
    penalty = np.ones(n_terms)
    penalty[0] = 0.0  # intercept — see docstring above
    reg_rows = np.sqrt(ridge_alpha) * np.diag(penalty)
    return np.vstack([design, reg_rows]), np.concatenate([y, np.zeros(n_terms)])


def _fit_surrogate_torch(Xn: np.ndarray, y: np.ndarray, terms: list, ridge_alpha: float) -> np.ndarray:
    """Same least-squares fit as _poly_features_numpy + np.linalg.lstsq
    (including the same ridge augmentation — see _ridge_augment_numpy's
    docstring), done with torch tensors on a CUDA device if one is
    available (CPU otherwise, which still exercises this code path's
    numerics but without the speed benefit an actual GPU would give).
    Raises ImportError with a clear message if torch isn't installed —
    never silently falls back to the numpy path, since backend='gpu' is
    an explicit request (see pick_backend()). Verified against real ROCm
    (AMD) hardware — torch's `torch.cuda` namespace is a compatibility
    shim that covers ROCm-backed AMD GPUs too, not just NVIDIA/CUDA
    ones, so `torch.cuda.is_available()` and this function both work
    unchanged on either."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("backend='gpu' requires PyTorch (pip install torch); "
                           "not installed in this environment.") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(Xn, dtype=torch.float64, device=device)
    cols = []
    for term in terms:
        if not term:
            cols.append(torch.ones(Xt.shape[0], dtype=torch.float64, device=device))
        elif len(term) == 1:
            cols.append(Xt[:, term[0]])
        else:
            cols.append(Xt[:, term[0]] * Xt[:, term[1]])
    design = torch.stack(cols, dim=1)
    yt = torch.as_tensor(y, dtype=torch.float64, device=device)
    if ridge_alpha > 0:
        # See _ridge_augment_numpy's docstring — same augmented-matrix
        # trick, and the same "never penalize the intercept" fix.
        n_terms = design.shape[1]
        penalty = torch.ones(n_terms, dtype=torch.float64, device=device)
        penalty[0] = 0.0
        reg_rows = torch.diag(penalty) * (ridge_alpha ** 0.5)
        design = torch.cat([design, reg_rows], dim=0)
        yt = torch.cat([yt, torch.zeros(n_terms, dtype=torch.float64, device=device)])
    solution = torch.linalg.lstsq(design, yt.unsqueeze(1)).solution
    return solution.squeeze(1).cpu().numpy()


def min_points_for_degree(n_dims: int, degree: int, margin: int = 5) -> int:
    """How many (params, Rwp) points are needed before fit_surrogate()
    is worth calling at all — fewer than this and the least-squares fit
    is under-determined or so close to it that the "fit" is really just
    interpolation noise. `margin` extra points beyond the bare minimum
    (one per coefficient) is a standard, conservative regression
    heuristic, not a hard requirement GSAS-II or the math imposes. This
    is necessary but NOT sufficient on its own for a trustworthy fit —
    see fit_surrogate()'s `ridge_alpha` for the other half of the actual
    fix confirmed necessary on real data (gsas2_swarm_optimize.py raises
    this function's own bar further, to 2x the raw term count, before
    ever calling fit_surrogate — see that script's `min_dataset_points`)."""
    return len(_poly_terms(n_dims, degree)) + margin


def fit_surrogate(X: np.ndarray, y: np.ndarray, degree: int = 2, backend: str = "cpu",
                   ridge_alpha: float = 1.0) -> SurrogateModel:
    """
    Fits log(Rwp) ~ polynomial(params) by (ridge-regularized) least
    squares over every (params, Rwp) point evaluated so far (`y` must be
    strictly positive — always true here: real Rwp values and
    training_target()'s bounded penalty are both positive by
    construction). predict_surrogate() exponentiates back, so every
    prediction is POSITIVE BY CONSTRUCTION rather than by clipping —
    confirmed as a real, not just cosmetic, problem on real data: an
    unconstrained linear-space fit predicted Rwp=-3.45 for a proposed
    point, an impossible value search_surrogate's PSO would happily
    "exploit" as if it were genuinely the best point found. `backend` is
    "cpu" (numpy) or "gpu" (torch — verified against real ROCm hardware).
    Returns a SurrogateModel whose coefficients are always plain numpy
    regardless of backend, so predict_surrogate() and every caller never
    need to know which one did the fitting.

    `ridge_alpha` (see _ridge_augment_numpy's docstring for the mechanism)
    shrinks coefficients toward zero, trading a little training-point
    accuracy for a much flatter, more trustworthy surface BETWEEN them.
    Confirmed necessary, not precautionary, on real data: an unregularized
    quadratic (10 free coefficients for a 3-dimensional, one-phase
    problem) fit to a bare-minimum-sized, sparse dataset — sparse because
    a large fraction of any perturbed batch is often physically insane
    and excluded entirely, see UNSOUND_PENALTY — produced a real
    interpolation-instability failure: the surrogate predicted Rwp=2.92
    for a point real GSAS-II verified at Rwp=6.11, a >100% relative miss.
    This is a DIFFERENT failure mode from the one trust_region_specs()
    guards against (extrapolation outside the trained region); ridge_alpha
    guards the region *inside* it. Set to 0 to recover plain OLS. The
    intercept term (see _ridge_augment_numpy) is exempt from shrinkage in
    EITHER case — in log-space it's log(baseline Rwp), and shrinking it
    would bias every prediction toward exp(0)=1% regardless of the data's
    actual baseline, the same bias problem already fixed once for the
    linear-space intercept.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("fit_surrogate requires strictly positive targets (Rwp values are "
                          "always > 0) — got a non-positive value, which log(y) can't handle.")
    log_y = np.log(y)
    n_dims = X.shape[1]
    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0)
    x_scale = np.where(x_scale == 0, 1.0, x_scale)
    Xn = (X - x_mean) / x_scale
    terms = _poly_terms(n_dims, degree)

    if backend == "cpu":
        design = _poly_features_numpy(Xn, terms)
        design_aug, y_aug = _ridge_augment_numpy(design, log_y, ridge_alpha)
        coeffs, *_ = np.linalg.lstsq(design_aug, y_aug, rcond=None)
    elif backend == "gpu":
        coeffs = _fit_surrogate_torch(Xn, log_y, terms, ridge_alpha)
    else:
        raise ValueError(f"unknown backend {backend!r}, expected 'cpu' or 'gpu'")

    return SurrogateModel(coeffs=np.asarray(coeffs), degree=degree, n_dims=n_dims,
                           x_mean=x_mean, x_scale=x_scale)


def predict_surrogate(model: SurrogateModel, X: np.ndarray) -> np.ndarray:
    """Evaluates a fitted SurrogateModel at (possibly many) points —
    always plain numpy in, plain numpy out, regardless of which backend
    fit the model. Cheap enough (a matrix multiply plus an exp()) to call
    every generation of an in-memory PSO search — see search_surrogate().
    Returns exp(polynomial(x)) — see fit_surrogate()'s docstring for why
    the model is fit in log-space: this guarantees every prediction is
    strictly positive, never the physically-impossible negative "Rwp"
    values an unconstrained linear-space fit produced on real data."""
    X = np.asarray(X, dtype=float)
    Xn = (X - model.x_mean) / model.x_scale
    terms = _poly_terms(model.n_dims, model.degree)
    design = _poly_features_numpy(Xn, terms)
    return np.exp(design @ model.coeffs)


def trust_region_specs(param_specs: list, X: np.ndarray, margin_frac: float = 0.25) -> list:
    """
    Narrows each dimension's search bounds to the region the surrogate
    was actually trained on — the observed data's own min/max in that
    dimension, padded by `margin_frac` of that range, then clipped back
    to the original param_specs bounds (never wider than the physically
    sane range those already encode).

    Confirmed as necessary, not precautionary, on real data: without
    this, search_surrogate()'s PSO explored the FULL original bounds
    (Size up to 1000, Mustrain up to 9000) even when every training
    point sat in a small, tightly-clustered patch of that space — a
    quadratic surface fit to one small patch extrapolates to nonsense
    (negative predicted Rwp, which is physically impossible) the moment
    it's evaluated far outside that patch, and PSO will happily
    "exploit" that nonsense minimum since nothing tells it the
    prediction there is meaningless. This is the standard trust-region
    fix for exactly this failure mode in surrogate-based optimization:
    only trust — and only search — where the model has actually seen
    data.
    """
    X = np.asarray(X, dtype=float)
    data_lo = X.min(axis=0)
    data_hi = X.max(axis=0)
    span = np.maximum(data_hi - data_lo, 1e-9)
    padded_lo = data_lo - margin_frac * span
    padded_hi = data_hi + margin_frac * span

    narrowed = []
    for spec, lo, hi in zip(param_specs, padded_lo, padded_hi):
        narrowed.append(replace(spec, lo=float(max(lo, spec.lo)), hi=float(min(hi, spec.hi))))
    return narrowed


def _select_diverse_candidates(positions: np.ndarray, fitness: np.ndarray, model: SurrogateModel,
                                n_candidates: int, min_separation: float):
    """Greedily picks up to `n_candidates` points from `positions` (best
    predicted fitness first), skipping any candidate too close — in the
    surrogate's own NORMALIZED coordinate space (model.x_mean/x_scale),
    since raw Size/Mustrain units aren't comparable across dimensions —
    to one already picked. This is what makes explorer particles (see
    init_swarm's explorer_frac) actually useful: without both this filter
    AND explorers to produce genuinely separated candidates, every
    particle converges toward the same basin and this would just return
    near-duplicates of the single best point n_candidates times.
    Always returns at least one candidate (the single best), even if
    nothing else clears the separation threshold."""
    order = np.argsort(fitness)
    normalized = (positions - model.x_mean) / model.x_scale
    selected = []
    for idx in order:
        if len(selected) >= n_candidates:
            break
        if all(np.linalg.norm(normalized[idx] - normalized[j]) >= min_separation for j in selected):
            selected.append(int(idx))
    if not selected:
        selected = [int(order[0])]
    return positions[selected], fitness[selected]


def search_surrogate(model: SurrogateModel, param_specs: list, X: np.ndarray, n_particles: int,
                      n_generations: int, rng: np.random.Generator, w: float = 0.6,
                      c1: float = 1.6, c2: float = 1.6, patience: int = 10,
                      min_improvement_frac: float = 1e-4, margin_frac: float = 0.25,
                      explorer_frac: float = 0.0, explorer_c_scale: float = 0.2,
                      n_candidates: int = 1, min_separation: float = 1.0) -> list:
    """
    Runs PSO entirely in-memory against the surrogate's PREDICTED Rwp —
    no subprocesses, no real GSAS-II calls — so this can use a much
    larger population/generation budget than the real-evaluation outer
    loop without adding meaningfully to wall-clock time; see this
    module's docstring for where GPU work belongs (this step, not the
    swarm bookkeeping and not GSAS-II itself). Searches only within
    trust_region_specs(param_specs, X, margin_frac) — see that
    function's docstring for why unrestricted search over a surrogate
    trained on a small data patch produces meaningless extrapolated
    proposals. `X` should be the same training data `model` was fit on.

    Returns a list of (position: np.ndarray, predicted_fitness: float)
    tuples, best first, length <= n_candidates (fewer only if the
    swarm's final population didn't have that many sufficiently distinct
    points — see _select_diverse_candidates). n_candidates=1 (the
    default) reproduces the original single-best-point behavior exactly.
    The caller (gsas2_swarm_optimize.py) is responsible for VERIFYING
    every returned candidate with a real GSAS-II evaluation before
    trusting it; a surrogate fit to a handful of points is a guess at the
    landscape's shape, not a substitute for the real fitness function.

    explorer_frac/explorer_c_scale (see init_swarm/update_swarm): with
    the default 0.0, every particle behaves identically and converges
    toward one shared best point, so n_candidates > 1 would just find
    near-duplicates of it. Set explorer_frac > 0 to keep a subset of
    particles wandering more broadly, so there's something genuinely
    distinct for a higher n_candidates to actually find.
    """
    region_specs = trust_region_specs(param_specs, X, margin_frac)
    swarm = init_swarm(region_specs, n_particles, rng, explorer_frac=explorer_frac)
    history = []
    for _ in range(n_generations):
        fitness = predict_surrogate(model, swarm.positions)
        swarm = update_swarm(swarm, fitness, w, c1, c2, rng, explorer_c_scale=explorer_c_scale)
        history.append(swarm.gbest_fitness)
        if has_converged(history, patience, min_improvement_frac):
            break

    positions, fitnesses = _select_diverse_candidates(
        swarm.pbest_positions, swarm.pbest_fitness, model, n_candidates, min_separation)
    return list(zip(positions, (float(f) for f in fitnesses)))
