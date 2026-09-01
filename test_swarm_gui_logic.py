#!/usr/bin/env python3
"""
test_swarm_gui_logic.py — tests for gsas2_swarm_gui_logic.py. No tkinter,
no GSAS-II, no display required. Same "mock data for verification" spirit
as test_gui_logic.py.

Run with: python3 test_swarm_gui_logic.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gsas2_swarm_gui_logic as logic  # noqa: E402


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise SystemExit(1)


def test_validate_swarm_config_catches_missing_fields():
    cfg = logic.SwarmRunConfig()
    problems = logic.validate_swarm_config(cfg)
    check("empty config has multiple problems", len(problems) >= 3)
    check("missing checkpoint flagged", any("checkpoint" in p.lower() for p in problems))
    check("missing gsasii path flagged", any("gsas-ii install path is required" in p.lower() for p in problems))
    check("missing outdir flagged", any("output folder" in p.lower() for p in problems))


def test_validate_swarm_config_catches_bad_counts():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        checkpoint = tmp / "checkpoint.gpx"
        checkpoint.write_text("x")
        gsasdir = tmp / "GSASII"
        gsasdir.mkdir()

        cfg = logic.SwarmRunConfig(
            checkpoint=str(checkpoint), gsasii_path=str(gsasdir), outdir=str(tmp / "out"),
            outer_iterations=0, perturbations=0, surrogate_particles=0,
            surrogate_generations=0, max_workers=0,
        )
        problems = logic.validate_swarm_config(cfg)
        check("zero generations flagged", any("generations (outer iterations)" in p.lower() for p in problems))
        check("zero perturbations flagged", any("perturbations per generation" in p.lower() for p in problems))
        check("zero surrogate particles flagged", any("surrogate particles" in p.lower() for p in problems))
        check("zero surrogate generations flagged", any("surrogate generations" in p.lower() for p in problems))
        check("zero max workers flagged", any("max parallel workers" in p.lower() for p in problems))


def test_validate_swarm_config_accepts_real_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        checkpoint = tmp / "checkpoint.gpx"
        checkpoint.write_text("x")
        gsasdir = tmp / "GSASII"
        gsasdir.mkdir()

        cfg = logic.SwarmRunConfig(
            checkpoint=str(checkpoint), gsasii_path=str(gsasdir), outdir=str(tmp / "out"),
        )
        problems = logic.validate_swarm_config(cfg)
        check("a fully-specified config with sane defaults has no problems", problems == [])


def test_validate_swarm_config_allows_blank_max_workers():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        checkpoint = tmp / "checkpoint.gpx"
        checkpoint.write_text("x")
        gsasdir = tmp / "GSASII"
        gsasdir.mkdir()

        cfg = logic.SwarmRunConfig(
            checkpoint=str(checkpoint), gsasii_path=str(gsasdir), outdir=str(tmp / "out"),
            max_workers=None,
        )
        problems = logic.validate_swarm_config(cfg)
        check("max_workers=None (no limit) is not flagged as a problem",
              not any("max parallel workers" in p.lower() for p in problems))


def test_validate_swarm_config_catches_bad_angle_cutoff_bounds():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        checkpoint = tmp / "checkpoint.gpx"
        checkpoint.write_text("x")
        gsasdir = tmp / "GSASII"
        gsasdir.mkdir()

        cfg = logic.SwarmRunConfig(
            checkpoint=str(checkpoint), gsasii_path=str(gsasdir), outdir=str(tmp / "out"),
            low_angle_cutoff_bounds=(-1.0, 15.0), high_angle_cutoff_bounds=(10.0, 5.0),
        )
        problems = logic.validate_swarm_config(cfg)
        check("negative low-angle-cutoff lower bound flagged",
              any("low-angle cutoff" in p.lower() for p in problems))
        check("inverted high-angle-cutoff bounds flagged",
              any("high-angle cutoff" in p.lower() for p in problems))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        checkpoint = tmp / "checkpoint.gpx"
        checkpoint.write_text("x")
        gsasdir = tmp / "GSASII"
        gsasdir.mkdir()

        cfg = logic.SwarmRunConfig(
            checkpoint=str(checkpoint), gsasii_path=str(gsasdir), outdir=str(tmp / "out"),
            low_angle_cutoff_bounds=(0.0, 15.0), high_angle_cutoff_bounds=(0.0, 10.0),
        )
        problems = logic.validate_swarm_config(cfg)
        check("a lower bound of exactly 0.0 is valid (it's a degrees-to-trim value, not "
              "a positive scale like Size/Mustrain)", problems == [])


def test_validate_swarm_config_catches_bad_cell_bounds():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        checkpoint = tmp / "checkpoint.gpx"
        checkpoint.write_text("x")
        gsasdir = tmp / "GSASII"
        gsasdir.mkdir()

        cfg = logic.SwarmRunConfig(
            checkpoint=str(checkpoint), gsasii_path=str(gsasdir), outdir=str(tmp / "out"),
            target="cell", cell_length_drift=1.5, cell_angle_bounds=-1.0,
        )
        problems = logic.validate_swarm_config(cfg)
        check("cell length drift outside (0, 1) flagged",
              any("cell length drift" in p.lower() for p in problems))
        check("non-positive cell angle bounds flagged",
              any("cell angle bounds" in p.lower() for p in problems))

        # Same out-of-range values are NOT flagged when target isn't
        # "cell" — they're simply unused/irrelevant then, not invalid.
        cfg_unused = logic.SwarmRunConfig(
            checkpoint=str(checkpoint), gsasii_path=str(gsasdir), outdir=str(tmp / "out"),
            target="microstrain_size", cell_length_drift=1.5, cell_angle_bounds=-1.0,
        )
        problems_unused = logic.validate_swarm_config(cfg_unused)
        check("cell bounds are only validated when target='cell'",
              not any("cell length drift" in p.lower() or "cell angle bounds" in p.lower()
                      for p in problems_unused))


def test_build_swarm_command_includes_every_knob():
    cfg = logic.SwarmRunConfig(
        checkpoint="checkpoint.gpx", gsasii_path="/opt/GSASII", outdir="out",
        outer_iterations=7, perturbations=13, surrogate_particles=99,
        surrogate_generations=42, backend="gpu", seed=123, max_workers=8,
    )
    cmd = logic.build_swarm_command(cfg, script_path="gsas2_swarm_optimize.py", python_exe="python3")

    check("script invoked unbuffered", cmd[:2] == ["python3", "-u"])
    check("checkpoint passed through", "--checkpoint" in cmd and "checkpoint.gpx" in cmd)
    check("gsasii path passed through", "--gsasii-path" in cmd and "/opt/GSASII" in cmd)
    check("outdir passed through", "--outdir" in cmd and "out" in cmd)
    check("outer-iterations passed through", "--outer-iterations" in cmd and "7" in cmd)
    check("perturbations passed through", "--perturbations" in cmd and "13" in cmd)
    check("surrogate-particles passed through", "--surrogate-particles" in cmd and "99" in cmd)
    check("surrogate-generations passed through", "--surrogate-generations" in cmd and "42" in cmd)
    check("backend passed through", "--backend" in cmd and "gpu" in cmd)
    check("seed passed through", "--seed" in cmd and "123" in cmd)
    check("max-workers passed through", "--max-workers" in cmd and "8" in cmd)
    check("--emit-events always included", "--emit-events" in cmd)
    check("mustrain_type defaults to isotropic and is always passed through",
          "--mustrain-type" in cmd and "isotropic" in cmd)


def test_build_swarm_command_uniaxial_mustrain_type():
    cfg = logic.SwarmRunConfig(checkpoint="c.gpx", gsasii_path="/opt/GSASII", outdir="out",
                                mustrain_type="uniaxial")
    cmd = logic.build_swarm_command(cfg, script_path="gsas2_swarm_optimize.py")
    check("uniaxial mustrain_type passed through",
          "--mustrain-type" in cmd and "uniaxial" in cmd)


def test_build_swarm_command_target_cell():
    cfg = logic.SwarmRunConfig(
        checkpoint="checkpoint_03_pre_unit_cell.gpx", gsasii_path="/opt/GSASII", outdir="out",
        target="cell", cell_length_drift=0.1, cell_angle_bounds=3.0,
    )
    cmd = logic.build_swarm_command(cfg, script_path="gsas2_swarm_optimize.py")
    check("--target cell passed through", "--target" in cmd and "cell" in cmd)
    check("--cell-length-drift passed through", "--cell-length-drift" in cmd and "0.1" in cmd)
    check("--cell-angle-bounds passed through", "--cell-angle-bounds" in cmd and "3.0" in cmd)
    check("--mustrain-type is NOT passed for --target cell (it's meaningless there)",
          "--mustrain-type" not in cmd)


def test_build_swarm_command_target_microstrain_size_default():
    cfg = logic.SwarmRunConfig(checkpoint="c.gpx", gsasii_path="/opt/GSASII", outdir="out")
    cmd = logic.build_swarm_command(cfg, script_path="gsas2_swarm_optimize.py")
    check("--target microstrain_size passed through (the default)",
          "--target" in cmd and "microstrain_size" in cmd)
    check("--mustrain-type is still passed for the default target",
          "--mustrain-type" in cmd)
    check("--cell-length-drift/--cell-angle-bounds are NOT passed for the default target",
          "--cell-length-drift" not in cmd and "--cell-angle-bounds" not in cmd)


def test_build_swarm_command_keep_evaluations():
    cfg_default = logic.SwarmRunConfig(checkpoint="c.gpx", gsasii_path="/opt/GSASII", outdir="out")
    cmd_default = logic.build_swarm_command(cfg_default, script_path="gsas2_swarm_optimize.py")
    check("keep_evaluations=False (the default) omits --keep-evaluations",
          "--keep-evaluations" not in cmd_default)

    cfg_keep = logic.SwarmRunConfig(checkpoint="c.gpx", gsasii_path="/opt/GSASII", outdir="out",
                                     keep_evaluations=True)
    cmd_keep = logic.build_swarm_command(cfg_keep, script_path="gsas2_swarm_optimize.py")
    check("keep_evaluations=True includes --keep-evaluations",
          "--keep-evaluations" in cmd_keep)


def test_build_swarm_command_omits_optional_none_fields():
    cfg = logic.SwarmRunConfig(checkpoint="c.gpx", gsasii_path="/opt/GSASII", outdir="out",
                                seed=None, max_workers=None)
    cmd = logic.build_swarm_command(cfg, script_path="gsas2_swarm_optimize.py")
    check("no --seed flag when seed is None", "--seed" not in cmd)
    check("no --max-workers flag when max_workers is None", "--max-workers" not in cmd)
    check("no --low-angle-cutoff-bounds flag when it's None",
          "--low-angle-cutoff-bounds" not in cmd)
    check("no --high-angle-cutoff-bounds flag when it's None",
          "--high-angle-cutoff-bounds" not in cmd)


def test_build_swarm_command_includes_angle_cutoff_bounds():
    cfg = logic.SwarmRunConfig(
        checkpoint="c.gpx", gsasii_path="/opt/GSASII", outdir="out",
        low_angle_cutoff_bounds=(0.0, 15.0), high_angle_cutoff_bounds=(0.0, 10.0),
    )
    cmd = logic.build_swarm_command(cfg, script_path="gsas2_swarm_optimize.py")
    check("--low-angle-cutoff-bounds passed through with both values",
          "--low-angle-cutoff-bounds" in cmd and "0.0" in cmd and "15.0" in cmd)
    check("--high-angle-cutoff-bounds passed through with both values",
          "--high-angle-cutoff-bounds" in cmd and "10.0" in cmd)


def test_auto_swarm_outdir_is_next_to_the_checkpoint_and_timestamped():
    from datetime import datetime
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "checkpoint_06_pre_profile_microstrain_size.gpx"
        checkpoint.write_text("x")
        stamp_time = datetime(2026, 8, 30, 14, 30, 22)
        outdir = logic.auto_swarm_outdir(str(checkpoint), now=stamp_time)
        check("outdir sits next to the checkpoint file",
              Path(outdir).parent == checkpoint.resolve().parent)
        check("outdir name is timestamped", Path(outdir).name == "swarm_20260830_143022")


def test_auto_swarm_outdir_blank_when_no_checkpoint():
    check("blank checkpoint path yields a blank outdir", logic.auto_swarm_outdir("") == "")


if __name__ == "__main__":
    test_validate_swarm_config_catches_missing_fields()
    test_validate_swarm_config_catches_bad_counts()
    test_validate_swarm_config_accepts_real_files()
    test_validate_swarm_config_allows_blank_max_workers()
    test_validate_swarm_config_catches_bad_angle_cutoff_bounds()
    test_validate_swarm_config_catches_bad_cell_bounds()
    test_build_swarm_command_includes_every_knob()
    test_build_swarm_command_uniaxial_mustrain_type()
    test_build_swarm_command_target_cell()
    test_build_swarm_command_target_microstrain_size_default()
    test_build_swarm_command_keep_evaluations()
    test_build_swarm_command_omits_optional_none_fields()
    test_build_swarm_command_includes_angle_cutoff_bounds()
    test_auto_swarm_outdir_is_next_to_the_checkpoint_and_timestamped()
    test_auto_swarm_outdir_blank_when_no_checkpoint()
    print("\nAll swarm GUI logic checks passed (no tkinter/GSAS-II required for this test).")
