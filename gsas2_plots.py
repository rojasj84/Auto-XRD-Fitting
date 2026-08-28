#!/usr/bin/env python3
"""
gsas2_plots.py — pure matplotlib figure-builders for the GUI's Results tab.

Deliberately tkinter-free and GSAS-II-free, same separation-of-concerns
reasoning as gsas2_gui_logic.py: these functions take plain data (the
dicts read_xy_csv()/read_summary() hand back) and return matplotlib
Figure objects. That makes them testable headlessly with the Agg backend
(no display needed — see test_plots.py) and reusable outside Tkinter if
we ever want a report export.

gsas2_gui.py embeds the returned Figures with FigureCanvasTkAgg; it does
not build figures itself.
"""

import matplotlib.pyplot as plt

# Deliberately does NOT call matplotlib.use(...) here. This module only ever
# constructs Figures directly (plt.Figure(...)) and never touches pyplot's
# stateful API (plt.figure()/plt.show()), so it doesn't care which backend
# is active — that's the caller's decision. gsas2_gui.py selects "TkAgg"
# before importing this module; test_plots.py selects "Agg" for headless
# runs. Forcing a backend in here would fight whichever one the caller
# already picked.


def _empty_axes_message(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
             color="#888", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def make_raw_pattern_figure(raw: dict, tmin=None, tmax=None):
    """
    Builds the "before you run" plot: the raw imported pattern
    (two_theta vs y_obs), with the currently-selected trim range (if any)
    shaded so it's obvious what will and won't be fit. `raw` is whatever
    read_xy_csv("pattern_raw.csv") returned — {} is handled (no data
    imported yet) and just shows a placeholder message.
    """
    fig = plt.Figure(figsize=(6.4, 3.2), dpi=100)
    ax = fig.add_subplot(111)

    x = raw.get("two_theta")
    y = raw.get("y_obs")
    if not x or not y:
        _empty_axes_message(ax, "No pattern loaded yet")
        fig.tight_layout()
        return fig

    ax.plot(x, y, color="#1f6feb", linewidth=0.8)
    ax.set_xlabel("2θ (°)")
    ax.set_ylabel("Intensity")
    ax.set_title("Raw pattern")

    if tmin is not None and tmax is not None:
        ax.axvspan(tmin, tmax, color="#2ea043", alpha=0.15, label="Fit range")
        ax.axvline(tmin, color="#2ea043", linewidth=1, linestyle="--")
        ax.axvline(tmax, color="#2ea043", linewidth=1, linestyle="--")

    fig.tight_layout()
    return fig


def make_fit_overlay_figure(fit: dict):
    """
    Builds the "after you run" plot: observed vs calculated pattern with
    the difference curve, the standard Rietveld overlay. `fit` is
    read_xy_csv("fit_final.csv") — {} (no completed run yet) shows a
    placeholder message instead of raising or drawing empty axes.

    Layout: a taller top panel for obs/calc/background, a shorter bottom
    panel for the difference curve, sharing the 2-theta axis — mirrors
    how GSAS-II's own plot window is read.
    """
    fig = plt.Figure(figsize=(6.4, 4.4), dpi=100)

    x = fit.get("two_theta")
    y_obs = fit.get("y_obs")
    if not x or not y_obs:
        ax = fig.add_subplot(111)
        _empty_axes_message(ax, "No completed run yet")
        fig.tight_layout()
        return fig

    ax_fit = fig.add_subplot(211)
    ax_diff = fig.add_subplot(212, sharex=ax_fit)

    ax_fit.plot(x, y_obs, "o", color="#57606a", markersize=1.6, label="Observed")
    if fit.get("y_calc"):
        ax_fit.plot(x, fit["y_calc"], color="#cf222e", linewidth=1.0, label="Calculated")
    if fit.get("y_bkg"):
        ax_fit.plot(x, fit["y_bkg"], color="#9a6700", linewidth=0.8,
                    linestyle="--", label="Background")
    ax_fit.set_ylabel("Intensity")
    ax_fit.set_title("Fit overlay")
    ax_fit.legend(fontsize=7, loc="upper right")
    ax_fit.tick_params(labelbottom=False)

    ax_diff.axhline(0, color="#888", linewidth=0.6)
    if fit.get("y_diff"):
        ax_diff.plot(x, fit["y_diff"], color="#1f6feb", linewidth=0.7)
    ax_diff.set_xlabel("2θ (°)")
    ax_diff.set_ylabel("Obs − calc")

    fig.tight_layout()
    return fig


def cell_summary_text(summary) -> str:
    """
    A short human-readable line for the Results header, e.g.
    "Final Rwp: 8.42%  |  2 phase(s) refined". Returns a placeholder if
    summary is None (no completed run yet). Kept separate from
    format_cell_rows() (which drives the detail table) — this is just
    the one-line headline.
    """
    if not summary:
        return "No completed run yet."
    rwp = summary.get("final_rwp")
    n_phases = len((summary.get("cells") or {}))
    rwp_text = f"{rwp:.2f}%" if isinstance(rwp, (int, float)) else "?"
    return f"Final Rwp: {rwp_text}  |  {n_phases} phase(s) refined"
