#!/usr/bin/env python3
"""
test_gui_logic.py — tests for gsas2_gui_logic.py. No tkinter, no GSAS-II,
no display required — this exercises the GUI's pure logic (validation,
command building, event-line parsing, config persistence, example
discovery) in isolation, the same "mock data for verification" spirit as
test_auto_refine_logic.py.

Run with: python3 test_gui_logic.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gsas2_gui_logic as logic  # noqa: E402


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise SystemExit(1)


def test_validate_run_config_catches_missing_fields():
    cfg = logic.RunConfig()
    problems = logic.validate_run_config(cfg)
    check("empty config has multiple problems", len(problems) >= 4)
    check("missing pattern flagged", any("pattern file" in p.lower() for p in problems))
    check("missing cif flagged", any("cif" in p.lower() for p in problems))
    check("missing outdir flagged", any("output folder" in p.lower() for p in problems))
    check("missing gsasii path flagged (non-dry-run)",
          any("gsas-ii install path is required" in p.lower() for p in problems))


def test_validate_run_config_dry_run_skips_gsasii_requirement():
    cfg = logic.RunConfig(pattern="", instprm="", cifs=[], outdir="", dry_run=True)
    problems = logic.validate_run_config(cfg)
    check("dry run does not require gsasii path",
          not any("gsas-ii install path is required" in p.lower() for p in problems))


def test_validate_run_config_accepts_real_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pattern = tmp / "pattern.xy"
        instprm = tmp / "inst.prm"
        cif = tmp / "phase.cif"
        gsasdir = tmp / "GSASII"
        for f in (pattern, instprm, cif):
            f.write_text("x")
        gsasdir.mkdir()

        cfg = logic.RunConfig(
            pattern=str(pattern), instprm=str(instprm), cifs=[str(cif)],
            outdir=str(tmp / "out"), gsasii_path=str(gsasdir), max_cell_drift=0.15,
        )
        problems = logic.validate_run_config(cfg)
        check("fully-populated valid config has no problems", problems == [])


def test_validate_run_config_rejects_bad_cell_drift():
    cfg = logic.RunConfig(max_cell_drift=1.5, dry_run=True)
    problems = logic.validate_run_config(cfg)
    check("out-of-range cell drift flagged", any("cell drift" in p.lower() for p in problems))


def test_build_command_shape():
    cfg = logic.RunConfig(
        pattern="Data/FeF3/r3c 20.txt",
        instprm="Data/FeF3/ws2.prm",
        cifs=["Data/FeF3/fef3 r3c.cif"],
        outdir="results/FeF3",
        gsasii_path="/opt/GSASII",
        refine_atoms=True,
        max_cell_drift=0.2,
    )
    cmd = logic.build_command(cfg, script_path="/opt/tools/gsas2_auto_refine.py")

    check("starts with python3 -u <script>",
          cmd[:3] == ["python3", "-u", "/opt/tools/gsas2_auto_refine.py"])
    check("pattern passed through", "Data/FeF3/r3c 20.txt" in cmd)
    check("cif passed through", "Data/FeF3/fef3 r3c.cif" in cmd)
    check("--refine-atoms present when enabled", "--refine-atoms" in cmd)
    check("--max-cell-drift value present", "0.2" in cmd)
    check("--emit-events always appended", cmd[-1] == "--emit-events")
    check("--dry-run absent when not requested", "--dry-run" not in cmd)


def test_build_command_multi_phase_and_dry_run():
    cfg = logic.RunConfig(
        pattern="p.xy", instprm="i.prm",
        cifs=["a.cif", "b.cif"],
        outdir="out", dry_run=True,
    )
    cmd = logic.build_command(cfg, script_path="script.py")
    check("both cifs present", cmd.count("--cif") == 2)
    check("--dry-run present", "--dry-run" in cmd)
    check("--gsasii-path absent when not set", "--gsasii-path" not in cmd)


def test_validate_run_config_trim_range():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pattern, instprm, cif = tmp / "p.xy", tmp / "i.prm", tmp / "c.cif"
        for f in (pattern, instprm, cif):
            f.write_text("x")

        one_sided = logic.RunConfig(pattern=str(pattern), instprm=str(instprm), cifs=[str(cif)],
                                     outdir=str(tmp / "out"), dry_run=True, tmin=10.0, tmax=None)
        check("tmin without tmax flagged",
              any("both" in p.lower() for p in logic.validate_run_config(one_sided)))

        backwards = logic.RunConfig(pattern=str(pattern), instprm=str(instprm), cifs=[str(cif)],
                                     outdir=str(tmp / "out"), dry_run=True, tmin=100.0, tmax=10.0)
        check("tmin >= tmax flagged",
              any("less than" in p.lower() for p in logic.validate_run_config(backwards)))

        valid = logic.RunConfig(pattern=str(pattern), instprm=str(instprm), cifs=[str(cif)],
                                 outdir=str(tmp / "out"), dry_run=True, tmin=10.0, tmax=100.0)
        check("valid trim range accepted", logic.validate_run_config(valid) == [])


def test_build_command_includes_trim_range():
    cfg = logic.RunConfig(pattern="p.xy", instprm="i.prm", cifs=["a.cif"],
                           outdir="out", tmin=15.0, tmax=95.0)
    cmd = logic.build_command(cfg, script_path="script.py")
    check("--tmin present with correct value",
          cmd[cmd.index("--tmin") + 1] == "15.0")
    check("--tmax present with correct value",
          cmd[cmd.index("--tmax") + 1] == "95.0")

    no_trim = logic.RunConfig(pattern="p.xy", instprm="i.prm", cifs=["a.cif"], outdir="out")
    cmd2 = logic.build_command(no_trim, script_path="script.py")
    check("--tmin absent when not set", "--tmin" not in cmd2)


def test_parse_event_line_json_vs_plain_text():
    plain = logic.parse_event_line("  [background_scale] ok (Rwp 20.000 -> 18.000)")
    check("plain text becomes a log event", plain["event"] == "log")
    check("plain text preserved verbatim",
          plain["text"] == "  [background_scale] ok (Rwp 20.000 -> 18.000)")

    payload = {"event": "stage_result", "name": "unit_cell", "status": "ok",
               "rwp_before": 18.0, "rwp_after": 15.0}
    parsed = logic.parse_event_line(json.dumps(payload) + "\n")
    check("json line parsed as structured event", parsed["event"] == "stage_result")
    check("json fields preserved", parsed["name"] == "unit_cell" and parsed["status"] == "ok")

    malformed = logic.parse_event_line("{not actually json}")
    check("malformed brace-wrapped text falls back to log event", malformed["event"] == "log")


def test_config_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        original_path = logic.CONFIG_PATH
        logic.CONFIG_PATH = Path(tmp) / "config.json"
        try:
            check("missing config file returns empty dict", logic.load_config() == {})
            values = {"gsasii_path": "/opt/GSASII", "last_outdir": "results"}
            logic.save_config(values)
            check("config round-trips", logic.load_config() == values)
        finally:
            logic.CONFIG_PATH = original_path


def test_auto_outdir():
    from datetime import datetime

    now = datetime(2026, 8, 28, 14, 30, 22)
    check("derives a timestamped folder next to the pattern file",
          logic.auto_outdir("Data/FeF3/r3c 20.txt", now=now)
          == str(Path("Data/FeF3").resolve() / "results_20260828_143022"))
    check("empty pattern yields empty suggestion", logic.auto_outdir("", now=now) == "")

    later = datetime(2026, 8, 28, 14, 31, 0)
    check("a later call gets a different (fresh) folder",
          logic.auto_outdir("Data/FeF3/r3c 20.txt", now=now)
          != logic.auto_outdir("Data/FeF3/r3c 20.txt", now=later))


def test_find_example_datasets():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fef3 = tmp / "Data" / "FeF3"
        fef3.mkdir(parents=True)
        (fef3 / "r3c 20.txt").write_text("x")
        (fef3 / "ws2.prm").write_text("x")
        (fef3 / "fef3 r3c.cif").write_text("x")

        incomplete = tmp / "Data" / "Incomplete"
        incomplete.mkdir(parents=True)
        (incomplete / "only_a_cif.cif").write_text("x")

        examples = logic.find_example_datasets(str(tmp))
        check("complete example discovered", "FeF3" in examples)
        check("incomplete folder (missing pattern/instprm) skipped", "Incomplete" not in examples)
        check("discovered example has all three file kinds",
              all(k in examples["FeF3"] for k in ("pattern", "instprm", "cifs")))

    check("no Data/ folder returns empty dict",
          logic.find_example_datasets(tempfile.mkdtemp()) == {})


def test_scan_dataset_subfolders_strict_mode():
    """
    Regression coverage for gsas2_batch_run_logic's discovery needs:
    strict=True must flag an ambiguous subfolder (more than one
    candidate pattern or instprm file) rather than silently guessing —
    confirmed as a real, not just theoretical, concern: this project's
    own Data/FeF3 folder accumulated a second .prm file over the course
    of development and would have been silently mis-resolved by the
    lenient (strict=False) behavior alone. strict=False (used by
    find_example_datasets(), the GUI's "Load example" picker) must keep
    tolerating that — a human sees what loaded there and can fix it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        clean = tmp / "Clean"
        clean.mkdir()
        (clean / "pattern.xy").write_text("x")
        (clean / "inst.prm").write_text("x")
        (clean / "phase.cif").write_text("x")

        ambiguous = tmp / "Ambiguous"
        ambiguous.mkdir()
        (ambiguous / "pattern.xy").write_text("x")
        (ambiguous / "inst_a.prm").write_text("x")
        (ambiguous / "inst_b.prm").write_text("x")
        (ambiguous / "phase.cif").write_text("x")

        multiphase = tmp / "MultiPhase"
        multiphase.mkdir()
        (multiphase / "pattern.xy").write_text("x")
        (multiphase / "inst.prm").write_text("x")
        (multiphase / "phase_a.cif").write_text("x")
        (multiphase / "phase_b.cif").write_text("x")

        found_lenient, skipped_lenient = logic.scan_dataset_subfolders(str(tmp), strict=False)
        check("lenient mode resolves the ambiguous folder anyway (picks one)",
              "Ambiguous" in found_lenient)
        check("lenient mode reports nothing skipped for a folder with an attempt",
              "Ambiguous" not in skipped_lenient)

        found_strict, skipped_strict = logic.scan_dataset_subfolders(str(tmp), strict=True)
        check("strict mode still finds the clean folder", "Clean" in found_strict)
        check("strict mode still allows multiple CIFs (multi-phase, not ambiguous)",
              "MultiPhase" in found_strict and len(found_strict["MultiPhase"]["cifs"]) == 2)
        check("strict mode skips the ambiguous folder instead of guessing",
              "Ambiguous" in skipped_strict and "Ambiguous" not in found_strict)
        check("strict mode's skip reason names what was ambiguous",
              "instrument-parameter" in skipped_strict["Ambiguous"])


def test_read_xy_csv():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "pattern_raw.csv"
        p.write_text("two_theta,y_obs\n10,100\n10.1,110\n10.2,nan\n")
        cols = logic.read_xy_csv(str(p))
        check("both columns present", set(cols) == {"two_theta", "y_obs"})
        check("values parsed as floats", cols["two_theta"] == [10.0, 10.1, 10.2])
        check("nan handled without raising", cols["y_obs"][2] != cols["y_obs"][2])  # NaN != NaN

        check("missing file returns empty dict",
              logic.read_xy_csv(str(Path(tmp) / "nope.csv")) == {})


def test_read_summary():
    with tempfile.TemporaryDirectory() as tmp:
        check("missing summary.json returns None", logic.read_summary(tmp) is None)
        (Path(tmp) / "summary.json").write_text(json.dumps({"final_rwp": 12.3}))
        check("existing summary.json parsed", logic.read_summary(tmp)["final_rwp"] == 12.3)

        (Path(tmp) / "summary.json").write_text("{not valid json")
        check("corrupt summary.json returns None, not raise", logic.read_summary(tmp) is None)


def test_format_cell_rows():
    summary = {
        "cells": {
            "FeF3": {"length_a": 5.4668, "length_b": 5.4666, "length_c": 5.4666,
                     "angle_alpha": 58.93, "angle_beta": 58.93, "angle_gamma": 58.93,
                     "volume": 118.2},
        },
        "cell_esds": {
            "FeF3": {"length_a": 0.0003, "length_b": 0.0003, "length_c": 0.0003,
                     "angle_alpha": 0.01, "angle_beta": 0.01, "angle_gamma": 0.01,
                     "volume": 0.05},
        },
    }
    rows = logic.format_cell_rows(summary)
    check("one row per cell parameter", len(rows) == 7)
    check("phase name attached to every row", all(r["phase"] == "FeF3" for r in rows))
    a_row = next(r for r in rows if r["param"] == "a")
    check("value correct", a_row["value"] == 5.4668)
    check("esd attached", a_row["esd"] == 0.0003)

    check("no cell data returns empty list", logic.format_cell_rows({}) == [])
    check("None summary returns empty list, not raise", logic.format_cell_rows(None) == [])


if __name__ == "__main__":
    test_validate_run_config_catches_missing_fields()
    test_validate_run_config_dry_run_skips_gsasii_requirement()
    test_validate_run_config_accepts_real_files()
    test_validate_run_config_rejects_bad_cell_drift()
    test_build_command_shape()
    test_build_command_multi_phase_and_dry_run()
    test_validate_run_config_trim_range()
    test_build_command_includes_trim_range()
    test_parse_event_line_json_vs_plain_text()
    test_config_round_trip()
    test_auto_outdir()
    test_find_example_datasets()
    test_scan_dataset_subfolders_strict_mode()
    test_read_xy_csv()
    test_read_summary()
    test_format_cell_rows()
    print("\nAll GUI logic checks passed (no tkinter/display required for this test).")
