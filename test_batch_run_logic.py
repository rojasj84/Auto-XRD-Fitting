#!/usr/bin/env python3
"""
test_batch_run_logic.py — tests for gsas2_batch_run_logic.py. No
tkinter, no GSAS-II, no subprocess required — this exercises experiment
discovery, per-experiment param overrides, and review-flagging in
isolation, the same "mock data for verification" spirit as
test_auto_refine_logic.py / test_gui_logic.py / test_candidate_sweep_logic.py.

Run with: python3 test_batch_run_logic.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gsas2_batch_run_logic as logic  # noqa: E402


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise SystemExit(1)


def _make_experiment_folder(root: Path, name: str, n_cifs: int = 1, params: dict = None):
    sub = root / name
    sub.mkdir(parents=True)
    (sub / "pattern.xy").write_text("x")
    (sub / "inst.prm").write_text("x")
    for i in range(n_cifs):
        (sub / f"phase{i}.cif").write_text("x")
    if params is not None:
        (sub / "params.json").write_text(json.dumps(params))
    return sub


def test_load_experiment_params():
    with tempfile.TemporaryDirectory() as tmp:
        defaults = logic.BatchDefaults(max_cell_drift=0.15, refine_atoms=False,
                                        tmin=None, tmax=None)

        no_override = _make_experiment_folder(Path(tmp), "NoOverride")
        result = logic.load_experiment_params(no_override, defaults)
        check("no params.json falls back to defaults entirely",
              result == {"max_cell_drift": 0.15, "refine_atoms": False, "tmin": None, "tmax": None})

        partial = _make_experiment_folder(Path(tmp), "Partial",
                                           params={"refine_atoms": True, "tmin": 20.0, "tmax": 80.0})
        result = logic.load_experiment_params(partial, defaults)
        check("params.json overrides only the keys it sets",
              result["max_cell_drift"] == 0.15  # untouched, from defaults
              and result["refine_atoms"] is True
              and result["tmin"] == 20.0 and result["tmax"] == 80.0)

        corrupt = Path(tmp) / "Corrupt"
        corrupt.mkdir()
        (corrupt / "params.json").write_text("{not valid json")
        result = logic.load_experiment_params(corrupt, defaults)
        check("corrupt params.json falls back to defaults, doesn't raise",
              result == {"max_cell_drift": 0.15, "refine_atoms": False, "tmin": None, "tmax": None})

        not_a_dict = Path(tmp) / "NotADict"
        not_a_dict.mkdir()
        (not_a_dict / "params.json").write_text("[1, 2, 3]")
        result = logic.load_experiment_params(not_a_dict, defaults)
        check("a params.json that isn't a JSON object falls back to defaults",
              result == {"max_cell_drift": 0.15, "refine_atoms": False, "tmin": None, "tmax": None})


def test_discover_experiments():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_experiment_folder(root, "Alpha")
        _make_experiment_folder(root, "Beta", n_cifs=2, params={"refine_atoms": True})
        # Ambiguous: two pattern files — see scan_dataset_subfolders(strict=True).
        gamma = root / "Gamma"
        gamma.mkdir()
        (gamma / "a.xy").write_text("x")
        (gamma / "b.xy").write_text("x")
        (gamma / "inst.prm").write_text("x")
        (gamma / "phase.cif").write_text("x")

        defaults = logic.BatchDefaults(gsasii_path="/gsas2", max_cell_drift=0.15,
                                        refine_atoms=False, tmin=None, tmax=None)
        experiments, skipped = logic.discover_experiments(str(root), defaults)

        names = [e.name for e in experiments]
        check("both well-formed experiments discovered, sorted by name",
              names == ["Alpha", "Beta"])
        check("the ambiguous folder is skipped, not silently resolved",
              "Gamma" in skipped and "Gamma" not in names)
        beta = next(e for e in experiments if e.name == "Beta")
        check("Beta's multi-phase CIFs are all present", len(beta.cifs) == 2)
        check("Beta's params.json override took effect", beta.refine_atoms is True)
        alpha = next(e for e in experiments if e.name == "Alpha")
        check("Alpha (no params.json) uses the batch defaults", alpha.refine_atoms is False)


def test_experiment_to_run_config():
    experiment = logic.Experiment(name="Alpha", pattern="p.xy", instprm="i.prm",
                                   cifs=["a.cif", "b.cif"], max_cell_drift=0.2,
                                   refine_atoms=True, tmin=10.0, tmax=90.0)
    cfg = logic.experiment_to_run_config(experiment, "/gsas2", "/out/Alpha")
    check("pattern/instprm/cifs carried through",
          cfg.pattern == "p.xy" and cfg.instprm == "i.prm" and cfg.cifs == ["a.cif", "b.cif"])
    check("outdir is the experiment's own subfolder", cfg.outdir == "/out/Alpha")
    check("gsasii_path carried through", cfg.gsasii_path == "/gsas2")
    check("per-experiment overrides carried through",
          cfg.refine_atoms is True and cfg.max_cell_drift == 0.2
          and cfg.tmin == 10.0 and cfg.tmax == 90.0)


def test_classify_result():
    # Regression coverage for the real motivating scenario (Data/FeF3):
    # a plausible-looking Rwp with a calculated pattern that doesn't
    # actually track the data must still be flagged.
    good = logic.classify_result(
        {"final_rwp": 6.07, "fit_quality": {"calc_obs_correlation": 0.94, "needs_review": False}},
        rwp_threshold=10.0)
    check("a good fit under the Rwp threshold with good correlation is not flagged",
          not good["needs_review"])

    plausible_but_wrong = logic.classify_result(
        {"final_rwp": 4.5, "fit_quality": {"calc_obs_correlation": 0.02, "needs_review": True}},
        rwp_threshold=10.0)
    check("a low Rwp with bad correlation is still flagged (Rwp alone isn't proof)",
          plausible_but_wrong["needs_review"])

    over_threshold = logic.classify_result(
        {"final_rwp": 12.4, "fit_quality": {"calc_obs_correlation": 0.96, "needs_review": False}},
        rwp_threshold=10.0)
    check("good correlation but Rwp at/above threshold is flagged",
          over_threshold["needs_review"])
    check("the reason names the threshold, not just 'needs review'",
          "threshold" in over_threshold["reason"])

    check("a crashed run (no summary at all) is always flagged",
          logic.classify_result(None, rwp_threshold=10.0)["needs_review"])

    exactly_at_threshold = logic.classify_result(
        {"final_rwp": 10.0, "fit_quality": {"calc_obs_correlation": 0.96, "needs_review": False}},
        rwp_threshold=10.0)
    check("Rwp exactly at the threshold is flagged (>=, not >)",
          exactly_at_threshold["needs_review"])


def test_build_batch_row():
    summary = {
        "final_rwp": 12.4,
        "fit_quality": {"calc_obs_correlation": 0.96, "needs_review": False},
        "stages": [
            {"name": "background_scale", "status": "ok", "optional": False},
            {"name": "unit_cell", "status": "failed_bounds", "optional": False},
            {"name": "preferred_orientation", "status": "failed_bounds", "optional": True},
        ],
    }
    row = logic.build_batch_row("Alpha", "/out/Alpha", 0, summary, rwp_threshold=10.0)
    check("row carries name/outdir/returncode through",
          row["name"] == "Alpha" and row["outdir"] == "/out/Alpha" and row["returncode"] == 0)
    check("row is flagged (Rwp over threshold)", row["needs_review"])
    check("failed_stages lists only the failed MANDATORY stage",
          row["failed_stages"] == ["unit_cell"])


if __name__ == "__main__":
    test_load_experiment_params()
    test_discover_experiments()
    test_experiment_to_run_config()
    test_classify_result()
    test_build_batch_row()
    print("\nAll batch-run logic checks passed (no GSAS-II/subprocess required for this test).")
