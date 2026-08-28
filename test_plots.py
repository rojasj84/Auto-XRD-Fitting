#!/usr/bin/env python3
"""
test_plots.py — tests for gsas2_plots.py. Runs headlessly (Agg backend,
set inside gsas2_plots.py itself) — no display needed, same spirit as the
other two test files.

Run with: python3 test_plots.py
"""

import math
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display in this sandbox/CI environment

sys.path.insert(0, str(Path(__file__).parent))
import gsas2_plots as plots  # noqa: E402


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise SystemExit(1)


def _sample_raw():
    x = [10.0 + 0.1 * i for i in range(50)]
    y = [100 + 5 * math.sin(i / 3.0) for i in range(50)]
    return {"two_theta": x, "y_obs": y}


def _sample_fit():
    x = [10.0 + 0.1 * i for i in range(50)]
    y_obs = [100 + 5 * math.sin(i / 3.0) for i in range(50)]
    y_calc = [98 + 5 * math.sin(i / 3.0) for i in range(50)]
    y_bkg = [20.0 for _ in range(50)]
    y_diff = [o - c for o, c in zip(y_obs, y_calc)]
    return {"two_theta": x, "y_obs": y_obs, "y_calc": y_calc, "y_bkg": y_bkg, "y_diff": y_diff}


def test_raw_pattern_figure_empty_data_shows_placeholder():
    fig = plots.make_raw_pattern_figure({})
    check("empty dict yields a figure, not an exception", fig is not None)
    check("empty-data figure has exactly one axes", len(fig.axes) == 1)


def test_raw_pattern_figure_with_data():
    fig = plots.make_raw_pattern_figure(_sample_raw())
    ax = fig.axes[0]
    check("one line drawn for the pattern", len(ax.lines) == 1)
    check("x data round-trips", list(ax.lines[0].get_xdata()) == _sample_raw()["two_theta"])
    check("title set", ax.get_title() == "Raw pattern")


def test_raw_pattern_figure_with_trim_range_shades_region():
    fig = plots.make_raw_pattern_figure(_sample_raw(), tmin=11.0, tmax=13.0)
    ax = fig.axes[0]
    check("trim range adds axvspan patch", len(ax.patches) >= 1)
    check("trim range adds two boundary lines (plus the pattern line)", len(ax.lines) == 3)


def test_raw_pattern_figure_no_trim_range_no_shading():
    fig = plots.make_raw_pattern_figure(_sample_raw())
    ax = fig.axes[0]
    check("no trim range means no shading patch", len(ax.patches) == 0)


def test_fit_overlay_figure_empty_data_shows_placeholder():
    fig = plots.make_fit_overlay_figure({})
    check("empty dict yields a figure, not an exception", fig is not None)
    check("empty-data figure has exactly one axes", len(fig.axes) == 1)


def test_fit_overlay_figure_with_full_data():
    fig = plots.make_fit_overlay_figure(_sample_fit())
    check("two axes (fit + diff panels)", len(fig.axes) == 2)
    ax_fit, ax_diff = fig.axes
    check("fit panel has observed + calc + bkg lines", len(ax_fit.lines) == 3)
    check("diff panel has diff line + zero line", len(ax_diff.lines) == 2)
    check("fit panel title set", ax_fit.get_title() == "Fit overlay")


def test_fit_overlay_figure_missing_optional_columns_degrades_gracefully():
    partial = {"two_theta": [10.0, 10.1, 10.2], "y_obs": [100.0, 101.0, 102.0]}
    fig = plots.make_fit_overlay_figure(partial)
    check("still builds with only obs data", fig is not None)
    ax_fit, ax_diff = fig.axes
    check("fit panel has only the observed line", len(ax_fit.lines) == 1)
    check("diff panel has only the zero line (no diff data)", len(ax_diff.lines) == 1)


def test_cell_summary_text():
    check("no summary yields placeholder text",
          plots.cell_summary_text(None) == "No completed run yet.")
    summary = {"final_rwp": 8.4231, "cells": {"FeF3": {}, "MgO": {}}}
    text = plots.cell_summary_text(summary)
    check("rwp formatted to 2 decimals with percent sign", "8.42%" in text)
    check("phase count included", "2 phase(s) refined" in text)

    summary_no_rwp = {"cells": {"FeF3": {}}}
    check("missing final_rwp handled without raising",
          "?" in plots.cell_summary_text(summary_no_rwp))


def test_figures_save_to_png_without_error():
    with tempfile.TemporaryDirectory() as tmp:
        raw_fig = plots.make_raw_pattern_figure(_sample_raw(), tmin=11.0, tmax=13.0)
        raw_path = Path(tmp) / "raw.png"
        raw_fig.savefig(raw_path)
        check("raw pattern figure saves to PNG", raw_path.is_file() and raw_path.stat().st_size > 0)

        fit_fig = plots.make_fit_overlay_figure(_sample_fit())
        fit_path = Path(tmp) / "fit.png"
        fit_fig.savefig(fit_path)
        check("fit overlay figure saves to PNG", fit_path.is_file() and fit_path.stat().st_size > 0)


if __name__ == "__main__":
    test_raw_pattern_figure_empty_data_shows_placeholder()
    test_raw_pattern_figure_with_data()
    test_raw_pattern_figure_with_trim_range_shades_region()
    test_raw_pattern_figure_no_trim_range_no_shading()
    test_fit_overlay_figure_empty_data_shows_placeholder()
    test_fit_overlay_figure_with_full_data()
    test_fit_overlay_figure_missing_optional_columns_degrades_gracefully()
    test_cell_summary_text()
    test_figures_save_to_png_without_error()
    print("\nAll plot-building checks passed (headless Agg backend, no display required).")
