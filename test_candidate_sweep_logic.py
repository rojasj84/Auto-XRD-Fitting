#!/usr/bin/env python3
"""
test_candidate_sweep_logic.py — tests for gsas2_candidate_sweep_logic.py.
No tkinter, no GSAS-II, no subprocess required — this exercises manifest
parsing/validation and result ranking in isolation, the same "mock data for
verification" spirit as test_auto_refine_logic.py / test_gui_logic.py.

Run with: python3 test_candidate_sweep_logic.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gsas2_candidate_sweep_logic as logic  # noqa: E402


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise SystemExit(1)


def _write_manifest(tmp, data) -> Path:
    p = Path(tmp) / "manifest.json"
    p.write_text(json.dumps(data))
    return p


def test_load_manifest_happy_path():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _write_manifest(tmp, {
            "pattern": "Data/FeF3/r3c 20.txt",
            "gsasii_path": "/path/to/GSASII",
            "max_cell_drift": 0.10,
            "candidates": [
                {"name": "cu_ambient", "instprm": "ws2.cu.prm", "cif": "fef3_ambient.cif"},
                {"name": "mo_guess", "instprm": "ws2.mo.prm", "cif": ["a.cif", "b.cif"],
                 "max_cell_drift": 0.25, "refine_atoms": True},
            ],
        })
        pattern, gsasii_path, candidates = logic.load_manifest(manifest)

        check("pattern read correctly", pattern == "Data/FeF3/r3c 20.txt")
        check("gsasii_path read correctly", gsasii_path == "/path/to/GSASII")
        check("both candidates parsed", len(candidates) == 2)
        check("a bare string 'cif' becomes a one-element list",
              candidates[0].cifs == ["fef3_ambient.cif"])
        check("a list 'cif' is kept as-is", candidates[1].cifs == ["a.cif", "b.cif"])
        check("candidate without its own max_cell_drift inherits the shared default",
              candidates[0].max_cell_drift == 0.10)
        check("candidate's own max_cell_drift overrides the shared default",
              candidates[1].max_cell_drift == 0.25)
        check("refine_atoms defaults to False when not set",
              candidates[0].refine_atoms is False)
        check("candidate's own refine_atoms is honored",
              candidates[1].refine_atoms is True)


def test_load_manifest_rejects_structural_problems():
    with tempfile.TemporaryDirectory() as tmp:
        missing_pattern = _write_manifest(tmp, {"candidates": [{"name": "a", "instprm": "x", "cif": "y"}]})
        try:
            logic.load_manifest(missing_pattern)
            check("missing pattern raises ManifestError", False)
        except logic.ManifestError:
            check("missing pattern raises ManifestError", True)

        no_candidates = _write_manifest(tmp, {"pattern": "p.txt", "candidates": []})
        try:
            logic.load_manifest(no_candidates)
            check("empty candidates list raises ManifestError", False)
        except logic.ManifestError:
            check("empty candidates list raises ManifestError", True)

        dup_names = _write_manifest(tmp, {
            "pattern": "p.txt",
            "candidates": [
                {"name": "a", "instprm": "x", "cif": "y"},
                {"name": "a", "instprm": "x2", "cif": "y2"},
            ],
        })
        try:
            logic.load_manifest(dup_names)
            check("duplicate candidate names raise ManifestError", False)
        except logic.ManifestError:
            check("duplicate candidate names raise ManifestError", True)

        missing_instprm = _write_manifest(tmp, {
            "pattern": "p.txt", "candidates": [{"name": "a", "cif": "y"}],
        })
        try:
            logic.load_manifest(missing_instprm)
            check("candidate missing instprm raises ManifestError", False)
        except logic.ManifestError:
            check("candidate missing instprm raises ManifestError", True)

        missing_cif = _write_manifest(tmp, {
            "pattern": "p.txt", "candidates": [{"name": "a", "instprm": "x"}],
        })
        try:
            logic.load_manifest(missing_cif)
            check("candidate missing cif raises ManifestError", False)
        except logic.ManifestError:
            check("candidate missing cif raises ManifestError", True)

        not_json = Path(tmp) / "bad.json"
        not_json.write_text("{not valid json")
        try:
            logic.load_manifest(not_json)
            check("invalid JSON raises ManifestError, not a raw exception", False)
        except logic.ManifestError:
            check("invalid JSON raises ManifestError, not a raw exception", True)

        try:
            logic.load_manifest(Path(tmp) / "does_not_exist.json")
            check("missing manifest file raises ManifestError", False)
        except logic.ManifestError:
            check("missing manifest file raises ManifestError", True)


def test_candidate_to_run_config():
    candidate = logic.Candidate(name="cu_ambient", instprm="ws2.cu.prm", cifs=["a.cif", "b.cif"],
                                 max_cell_drift=0.2, refine_atoms=True, tmin=10.0, tmax=90.0)
    cfg = logic.candidate_to_run_config("pattern.txt", "/gsas2", candidate, "/out/cu_ambient")

    check("pattern carried through", cfg.pattern == "pattern.txt")
    check("instprm from the candidate", cfg.instprm == "ws2.cu.prm")
    check("multi-phase cif list carried through", cfg.cifs == ["a.cif", "b.cif"])
    check("outdir is the candidate's own subfolder", cfg.outdir == "/out/cu_ambient")
    check("gsasii_path carried through", cfg.gsasii_path == "/gsas2")
    check("per-candidate overrides carried through",
          cfg.refine_atoms is True and cfg.max_cell_drift == 0.2
          and cfg.tmin == 10.0 and cfg.tmax == 90.0)


def _result(name, correlation=None, needs_review=None, rwp=None, no_summary=False):
    if no_summary:
        return {"name": name, "returncode": 1, "summary": None}
    return {
        "name": name,
        "returncode": 0,
        "summary": {
            "final_rwp": rwp,
            "fit_quality": {"calc_obs_correlation": correlation, "needs_review": needs_review},
        },
    }


def test_rank_results():
    # Regression coverage for the real motivating scenario (Data/FeF3):
    # a candidate can have a *lower* Rwp than another while its calculated
    # pattern doesn't actually track the data at all (needs_review=True) —
    # ranking must prefer the trustworthy one, not the lower-Rwp one.
    crashed = _result("crashed", no_summary=True)
    bad_but_low_rwp = _result("bad_but_low_rwp", correlation=0.05, needs_review=True, rwp=4.0)
    good = _result("good", correlation=0.94, needs_review=False, rwp=6.1)
    good_but_higher_rwp = _result("good_but_higher_rwp", correlation=0.91, needs_review=False, rwp=8.0)

    ranked = logic.rank_results([crashed, bad_but_low_rwp, good_but_higher_rwp, good])
    names = [r["name"] for r in ranked]

    check("a genuinely good fit outranks a lower-Rwp fit flagged for review",
          names.index("good") < names.index("bad_but_low_rwp"))
    check("among two 'ok' fits, lower Rwp / higher correlation wins",
          names.index("good") < names.index("good_but_higher_rwp"))
    check("a crashed candidate (no summary at all) ranks last",
          names[-1] == "crashed")
    check("full order is good, good_but_higher_rwp, bad_but_low_rwp, crashed",
          names == ["good", "good_but_higher_rwp", "bad_but_low_rwp", "crashed"])


def test_format_ranking_table():
    ranked = [
        _result("good", correlation=0.94, needs_review=False, rwp=6.1),
        _result("crashed", no_summary=True),
    ]
    table = logic.format_ranking_table(ranked)
    check("table includes the good candidate's name and Rwp",
          "good" in table and "6.100" in table)
    check("table marks a crashed candidate distinctly", "crashed" in table)


if __name__ == "__main__":
    test_load_manifest_happy_path()
    test_load_manifest_rejects_structural_problems()
    test_candidate_to_run_config()
    test_rank_results()
    test_format_ranking_table()
    print("\nAll candidate-sweep logic checks passed (no GSAS-II/subprocess required for this test).")
