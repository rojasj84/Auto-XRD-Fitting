# Progress Notes

Running log of where this project stands: what's solid, what's still open, and
what's currently blocking further work. Not a replacement for README.md (which
documents what the tools do) — this is status and direction.

Last updated: 2026-08-30

## Currently blocked on

**Waiting on the scientist for corrected data.** The `Data/FeF3-2/` folder she
provided (unaltered files) has two real data-quality problems that make the
deterministic pipeline hang rather than converge — see "Known issues" below.
Plan: flag this to her, she'll send a bulk scan list (multiple datasets) to
try once the instrument-parameter question is sorted out.

## What's hitting the mark

- **Deterministic pipeline** (`gsas2_auto_refine.py`) — staged refinement with
  checkpointing, rollback, and fallback ladders for known correlation traps
  (Size/Mustrain, preferred orientation, atoms-vs-Mustrain). Validated on
  real FeF3 and MgO+MgBC data.
- **GUI** (`gsas2_gui.py`) — Data/Phases/Options/Swarm tabs, unnumbered
  Results tab, live progress, plotting. Sweep tab hidden per request (code
  intact, one line to re-enable).
- **Batch run tool** — unattended folder-of-experiments processing with
  triage flags for manual review.
- **Swarm optimizer** (`gsas2_swarm_optimize.py`) — surrogate-assisted PSO
  search over the continuous Size/Mustrain starting-point space. Built up
  through a long chain of real-data-driven fixes this session:
  - Log-space, ridge-regularized surrogate fitting — predictions can no
    longer go negative or swing wildly between sparse training points.
  - `training_target()` bounded penalty — insane (unsound) points now teach
    the surrogate something instead of being invisible to it.
  - Explorer particles + multi-candidate proposals per iteration, instead of
    betting everything on one surrogate guess.
  - **Isotropic Mustrain as the default** (was uniaxial) — fixed the real
    root cause of most "insane" perturbations: uniaxial Mustrain and
    isotropic Size are ~98% correlated. Sane-perturbation rate went from
    ~20-50% to a confirmed 100% across multiple live runs.
  - Low/high-angle fit-range cutoff search, with real guardrails after
    confirming the naive version could cherry-pick around a hard-to-fit peak
    instead of trimming genuinely bad data.
  - `peak_amplitude_error` tie-breaker — addresses the real complaint that
    Rwp is intensity-weighted and can look great from one dominant peak
    while smaller peaks are poorly matched.
  - Aggressive disk cleanup — a run's final output is now `best.gpx` +
    2 CSVs + a summary (~1MB), not a multi-GB tree of every discarded
    candidate's full GSAS-II project.
  - GPU backend (ROCm/AMD) verified working end-to-end, not just CPU.
  - Full test suite: 108 swarm-logic checks + GUI logic checks, all passing.

## What needs improvement

- **Preferred orientation is the next real lever, not cell parameters.**
  Traced a real case where Size/Mustrain optimization barely moved
  peak-amplitude matching (some peaks stuck at 40-90% relative error even
  after 160 real evaluations), and cell parameters were ruled out (peak
  positions already accurate, <0.16° off). But the deterministic pipeline's
  own preferred-orientation search found a genuine Rwp improvement
  (6.055%→6.045%) that got discarded for missing a 1%-relative-improvement
  bar — same "single/few starting points isn't enough" problem the swarm
  already solved for Size/Mustrain. Not yet built: a swarm-style multi-start
  search over texture axis/model.
- **`scan_dataset_subfolders()` file-selection bug (found, not yet fixed).**
  When a folder has more than one candidate `.prm`/pattern file, the
  non-strict path (used by the GUI's "Load example") doesn't actually sort
  before picking — contrary to its own docstring — so the choice is
  filesystem-order-dependent, not even reliably "alphabetical." This is
  confirmed as the reason `results/swarm_scaffold_test` and other FeF3
  checkpoints this session were built with the Cu Kα instrument file
  (`ws2.cu.prm`) instead of the correct synchrotron one (`ws2.prm`, 0.3344 Å)
  — the two happened to sit in the same `Data/FeF3/` folder. Two fix options
  discussed, not yet decided: match `discover_experiments()`'s existing
  `strict=True` behavior (refuse and report ambiguity) vs. just fixing the
  sort so the pick is at least deterministic. Revisit once the scientist's
  new data is in.

## Known issues (data, not code)

Found while testing `Data/FeF3-2/` (the scientist's unaltered files):

1. **CIF is in space group P1**, not the correct R-3c — almost certainly a
   raw VESTA export that never had symmetry assigned. Not "wrong" data, just
   unreduced: refining in P1 requires ~12x more individually-computed
   reflections than R-3c, which is what actually causes the apparent hang
   (traced directly to `do_refinements()`, confirmed not an infinite loop,
   just computationally impractical at that resolution). Workaround exists:
   `Data/FeF3/fef3_r3c_ambient.cif` already has the correct R-3c setting for
   this structure.
2. **Instrument-parameter file has `W = 200.0`** — a Caglioti Gaussian
   broadening term that should realistically be ~0.001-0.05. This is
   unrelated to the CIF issue and only surfaces once the P1 problem is
   worked around: the pipeline's own bounds check correctly rejects
   `profile_instrument`'s attempt to fix it (can't improve Rwp enough from
   such a bad starting point in one local-optimizer step), so `W` stays
   stuck at 200 and poisons the `peak_asymmetry` (SH/L) stage until it
   never converges. Confirmed directly: manually overriding `W` to 2.0 took
   the same refinement from "hangs indefinitely" to 3.2 seconds. Needs the
   scientist's actual calibrated value — not something to guess at.

## Next steps

1. Scientist confirms/corrects the instrument-parameter file and CIF
   symmetry for the FeF3-2 measurement; sends a bulk scan list to test
   against more broadly.
2. Decide and apply the `scan_dataset_subfolders()` ambiguous-file fix.
3. Once there's a validated, working checkpoint again: build the
   preferred-orientation multi-start search, following the same
   diagnose-with-real-data-first approach used for everything above.
