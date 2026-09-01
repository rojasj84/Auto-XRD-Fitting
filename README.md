# Auto XRD Fitting

Turnkey Rietveld refinement automation on top of [GSAS-II](https://subversion.xray.aps.anl.gov/trac/pyGSAS). Point it at a powder diffraction pattern, an instrument-parameter file, and a phase CIF, and it runs a full staged refinement — background, scale, cell, profile, microstrain/size, thermal parameters, peak asymmetry, extinction, preferred orientation, atom positions — unattended, with no manual parameter tweaking.

## Why this exists

Rietveld refinement in GSAS-II normally means a scientist manually clicking through stages, watching for divergence, rolling back bad steps, and nudging individual parameters until the fit converges. This project automates that workflow end to end:

- **Deterministic staged refinement** with checkpointing and automatic rollback — a stage that diverges reverts cleanly instead of corrupting the fit.
- **Automated recovery, not just failure detection.** When a stage hits a known correlation trap (e.g. Mustrain and Size fighting for the same degree of freedom, or a texture axis that helps one phase and hurts another), it automatically retries with a simpler or differently-targeted model instead of just giving up.
- **A fit-quality check that catches what Rwp alone can't.** A converged, in-bounds Rwp is not proof the model is right — this project was built around a real case where Rwp looked like a plausible ~10% for five straight stages while the calculated pattern had essentially zero correlation with the real data. Every run reports a calculated/observed correlation and a `needs_review` flag alongside Rwp.
- **Parallel tooling for when you have real alternatives to test or many datasets to run** — compare candidate structures/wavelengths against one pattern, or batch-refine an entire folder of experiments unattended, both with automatic triage of which results need a human look.

## What's here

| | |
|---|---|
| `gsas2_auto_refine.py` | Core CLI — one dataset, one full staged refinement. |
| `gsas2_gui.py` / `gsas2_gui_logic.py` | Desktop GUI (Tkinter) — pick files, watch live stage progress, view plots. |
| `gsas2_plots.py` | Shared plotting helpers for the GUI's Results tab. |
| `gsas2_candidate_sweep.py` / `..._logic.py` | Run several (instrument file, CIF) hypotheses against one pattern in parallel and rank them by fit quality. |
| `gsas2_batch_run.py` / `..._logic.py` | Refine every experiment in a folder tree unattended, in parallel, with a CSV/JSON results table flagging what needs review. |
| `test_*.py` | Unit tests for every module's logic, independent of a real GSAS-II install (self-contained fake project objects). |
| `Data/` | Example datasets (FeF3, MgO+MgBC) used for development and the test suite. |

## Requirements

- Python 3.9+
- A local [GSAS-II](https://gsas-ii.readthedocs.io/) installation (not a pip package — point `--gsasii-path` at it)
- `matplotlib` (GUI plots only; the CLI tools have no extra dependencies beyond the standard library)
- Tkinter (ships with standard Python; only needed for the GUI)

Cross-platform: developed on Linux, with Windows compatibility (encoding, console-window suppression, path handling) built in and verified.

## Quick start

```bash
# One dataset, full refinement
python3 gsas2_auto_refine.py \
    --gsasii-path /path/to/GSASII \
    --pattern data/sample.xy \
    --instprm data/sample.instprm \
    --cif data/phase.cif \
    --outdir results/sample

# Desktop GUI
python3 gsas2_gui.py

# Compare candidate structures/wavelengths against one pattern
python3 gsas2_candidate_sweep.py --manifest sweep.json --outdir results/sweep

# Refine an entire folder of experiments unattended
python3 gsas2_batch_run.py --root /data/experiments --outdir results/batch --gsasii-path /path/to/GSASII
```

Run `--help` on any script for full options, or `--dry-run` on `gsas2_auto_refine.py` to validate inputs without needing GSAS-II installed at all.

## Testing

```bash
python3 test_auto_refine_logic.py
python3 test_gui_logic.py
python3 test_candidate_sweep_logic.py
python3 test_batch_run_logic.py
python3 test_plots.py
```

No GSAS-II install or display required — the control-flow logic is tested against fake in-memory project objects that model the real GSASIIscriptable API shapes.
