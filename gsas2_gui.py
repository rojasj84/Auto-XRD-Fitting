#!/usr/bin/env python3
"""
gsas2_gui.py — desktop GUI front end for gsas2_auto_refine.py.

A thin presentation layer: all validation, command-building, config
persistence, and subprocess-output parsing live in gsas2_gui_logic.py
(tkinter-free, unit-tested in test_gui_logic.py). This module only wires
that logic to widgets and to a background thread that drives
gsas2_auto_refine.py as a subprocess and streams its --emit-events JSON
progress back into the UI.

Built on Tkinter (ships with a standard Python install — no extra
dependency to install alongside GSAS-II) rather than a heavier GUI
toolkit, so it runs the same way on both the office and home machines
without adding another package to keep in sync.

Workflow, top to bottom:
    Tab 1 — Data & Instrument : pick the pattern + instrument files
                                  (or one click to load a bundled example)
    Tab 2 — Phases (CIF)       : add one or more phase CIFs
    Tab 3 — Options             : GSAS-II install path, output folder,
                                  atom-refinement toggle, cell-drift bound
    Bottom panel (always visible): Run / Cancel, live per-stage progress
                                  table, scrolling log, and buttons to open
                                  the results once a run finishes.

Run with:  python3 gsas2_gui.py
"""

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont

import matplotlib
matplotlib.use("TkAgg")  # must happen before pyplot/backend imports, and before
                          # gsas2_plots (which defaults to Agg for headless use)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.widgets import SpanSelector  # noqa: E402

import gsas2_gui_logic as logic  # noqa: E402
import gsas2_plots as plots  # noqa: E402
import gsas2_swarm_gui_logic as swarm_logic  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFINE_SCRIPT = SCRIPT_DIR / "gsas2_auto_refine.py"
SWEEP_SCRIPT = SCRIPT_DIR / "gsas2_candidate_sweep.py"
SWARM_SCRIPT = SCRIPT_DIR / "gsas2_swarm_optimize.py"

STAGE_LABELS = {
    "background_scale": "Background + scale",
    "sample_displacement": "Sample displacement",
    "unit_cell": "Unit cell",
    "profile_instrument": "Profile (instrument)",
    "profile_microstrain_size": "Profile (microstrain/size)",
    "atoms": "Atom positions",
}


# Preferred UI font families, best first. Tk's font.families() on Linux
# sometimes only enumerates a handful of legacy X11 core bitmap fonts
# (e.g. "nimbus sans l", "fixed") even when fontconfig has real TrueType
# fonts like DejaVu Sans installed system-wide (confirmed via fc-list) —
# whether that's the case depends on how the Tcl/Tk build in use was
# compiled, not on what's actually on disk. Requesting a family that
# Tk can't resolve doesn't raise or warn; it silently falls back to the
# blocky bitmap "fixed" font, which looks worse than doing nothing —
# confirmed by rendering both to a screenshot and comparing. So this
# never sets a family blindly: it only applies the first name in each
# list below that's actually present in font.families().
_PREFERRED_SANS_FONTS = [
    "DejaVu Sans", "Noto Sans", "Liberation Sans", "Segoe UI",
    "San Francisco", "Helvetica", "Nimbus Sans L", "Arial",
]
_PREFERRED_MONO_FONTS = [
    "DejaVu Sans Mono", "Noto Sans Mono", "Liberation Mono", "Consolas",
    "Menlo", "Nimbus Mono L", "Courier New", "Courier",
]


def _pick_available_font(preferred: list, available: set) -> str | None:
    lowered = {name.lower(): name for name in available}
    for want in preferred:
        if want.lower() in lowered:
            return lowered[want.lower()]
    return None


def configure_fonts(root: tk.Misc, size: int = 10) -> tuple:
    """
    Points Tk's standard named fonts (which every ttk widget follows by
    default) at the best available anti-aliased font instead of whatever
    Tk's own default heuristic picked. Returns (sans_family, mono_family)
    — either may be None if nothing in the preferred lists was available,
    in which case that font family is left untouched rather than forced
    to something worse (see _PREFERRED_SANS_FONTS's comment).
    """
    available = set(tkfont.families(root))
    sans = _pick_available_font(_PREFERRED_SANS_FONTS, available)
    mono = _pick_available_font(_PREFERRED_MONO_FONTS, available)

    if sans:
        for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont",
                      "TkMenuFont", "TkCaptionFont", "TkSmallCaptionFont",
                      "TkIconFont", "TkTooltipFont"):
            try:
                tkfont.nametofont(name, root).configure(family=sans, size=size)
            except tk.TclError:
                pass  # not every named font exists on every platform
        ttk.Style(root).configure(".", font=(sans, size))

    if mono:
        try:
            tkfont.nametofont("TkFixedFont", root).configure(family=mono, size=size - 1)
        except tk.TclError:
            pass

    return sans, mono


def _no_window_kwargs() -> dict:
    """Extra subprocess.Popen kwargs to suppress the console window
    Windows otherwise briefly flashes open for every subprocess spawned
    from a windowed (no-console) Tkinter app — CREATE_NO_WINDOW only
    exists in the subprocess module on Windows, so this is a no-op
    dict everywhere else rather than an AttributeError."""
    if sys.platform.startswith("win"):
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _format_elapsed(seconds: float) -> str:
    """Formats a duration for the Swarm tab's elapsed-time readout —
    seconds with one decimal under a minute (so short test runs show
    meaningful precision), minutes:seconds beyond that."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


def open_path(path: str) -> None:
    """Opens a file or folder in the OS's default handler. Best-effort —
    a failure here (unsupported platform, no handler configured) is
    reported to the user rather than raised."""
    try:
        if sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform.startswith("win"):
            import os
            os.startfile(path)  # noqa: S606 — Windows-only, no shell injection risk here
        else:
            raise RuntimeError(f"Don't know how to open files on {sys.platform!r}")
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("Couldn't open", f"Couldn't open {path}:\n{exc}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GSAS-II Auto Refine")
        self.geometry("1080x900")
        self.minsize(860, 680)

        # Must happen before any widget is built: every ttk widget picks up
        # TkDefaultFont at creation time, not live afterward. See
        # configure_fonts()'s docstring for why this doesn't just hardcode
        # a family name.
        sans_family, _mono_family = configure_fonts(self)
        # Bold variant of the (now-corrected) default font, reused for the
        # status/results-summary labels below instead of the family=""
        # they used to use — an empty family string doesn't mean "inherit
        # the default," it silently resolves to Tk's blocky bitmap "fixed"
        # font (confirmed by rendering both to a screenshot and comparing
        # pixel output), which is likely a real contributor to "all text
        # looks horrible": those two labels are the run status and results
        # summary, some of the most-looked-at text in the app.
        self.bold_font = tkfont.Font(font=tkfont.nametofont("TkDefaultFont", self))
        self.bold_font.configure(weight="bold")

        self.config_data = logic.load_config()
        self.examples = logic.find_example_datasets(str(SCRIPT_DIR))

        self.cif_paths: list = []
        self.proc = None
        self.event_queue: "queue.Queue" = queue.Queue()
        self.stage_row_by_name: dict = {}
        self.last_outdir = None
        self.last_refined_cifs: list = []
        # True until the user explicitly types into or Browse's the output
        # folder field — while True, _on_run() generates a fresh timestamped
        # folder (see logic.auto_outdir) right before every run instead of
        # requiring the user to pick one. Flipped to False only by a genuine
        # user edit (bound on the Entry widget itself — see _build_options_tab
        # — not by our own programmatic .set() calls when auto-filling).
        self._outdir_auto = True

        # Results tab state — the plot canvases/figures/span-selector get
        # replaced wholesale on every refresh (see _mount_raw_canvas /
        # _mount_fit_canvas), so they start as None rather than being built
        # in _build_widgets.
        self.raw_fig = None
        self.raw_canvas = None
        self.raw_span_selector = None
        self.fit_fig = None
        self.fit_canvas = None
        self._trim_parse_error = None

        # Sweep tab state (gsas2_candidate_sweep.py) — kept separate from
        # the single-refinement run state above since candidates run in
        # parallel, not as a sequence of stages, and share nothing with a
        # single run's outdir/stage-tree bookkeeping.
        self.sweep_candidates: list = []  # [{"name":..., "instprm":..., "cifs": [...]}]
        self.sweep_proc = None
        self.sweep_event_queue: "queue.Queue" = queue.Queue()
        self.sweep_row_by_name: dict = {}
        self.sweep_outdir = None

        # Swarm tab state (gsas2_swarm_optimize.py) — kept separate for the
        # same reason as the Sweep tab above: its own subprocess, own
        # progress shape (one row per outer iteration, not per stage or
        # candidate), own outdir. self.swarm_start_time backs the elapsed-
        # time readout, the main point of this tab (see its build method).
        self.swarm_proc = None
        self.swarm_event_queue: "queue.Queue" = queue.Queue()
        self.swarm_outdir = None
        self.swarm_start_time = None
        self._swarm_outdir_auto = True

        self._build_widgets()
        self._prefill_from_config()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widgets(self):
        # A resizable vertical split between the config/results notebook
        # (which needs real room once the Results tab has two plots in it)
        # and the run panel (stage progress + log), so the user can drag
        # the sash instead of either view being permanently cramped.
        self.main_pane = ttk.Panedwindow(self, orient="vertical")
        self.main_pane.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        notebook = ttk.Notebook(self.main_pane)
        self._build_data_tab(notebook)
        self._build_phases_tab(notebook)
        self._build_options_tab(notebook)
        self._build_swarm_tab(notebook)
        # Results isn't a step in the 1-2-3-4 configure-and-run sequence
        # above — it's a "check what happened" destination for either
        # workflow, so it's deliberately unnumbered and placed last
        # rather than sitting in the middle of the numbered tabs.
        self._build_results_tab(notebook)
        # Sweep tab hidden (not deleted — _build_sweep_tab is untouched
        # and easy to re-enable) per the user's request; its own state/
        # methods are left as-is below.
        self.main_pane.add(notebook, weight=3)

        run_panel = ttk.Frame(self.main_pane)
        self._build_run_panel(run_panel)
        self.main_pane.add(run_panel, weight=2)

    def _build_data_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="1. Data & Instrument")
        tab.columnconfigure(1, weight=1)

        if self.examples:
            ttk.Label(tab, text="Load a bundled example:").grid(row=0, column=0, sticky="w")
            self.example_var = tk.StringVar(value="")
            example_box = ttk.Combobox(tab, textvariable=self.example_var,
                                        values=sorted(self.examples), state="readonly")
            example_box.grid(row=0, column=1, sticky="we", padx=(6, 6))
            ttk.Button(tab, text="Load", command=self._load_example).grid(row=0, column=2)
            ttk.Separator(tab).grid(row=1, column=0, columnspan=3, sticky="we", pady=10)

        ttk.Label(tab, text="Pattern file:").grid(row=2, column=0, sticky="w", pady=4)
        self.pattern_var = tk.StringVar()
        ttk.Entry(tab, textvariable=self.pattern_var).grid(row=2, column=1, sticky="we", padx=6)
        ttk.Button(tab, text="Browse...", command=self._browse_pattern).grid(row=2, column=2)

        ttk.Label(tab, text="Instrument parameter file:").grid(row=3, column=0, sticky="w", pady=4)
        self.instprm_var = tk.StringVar()
        ttk.Entry(tab, textvariable=self.instprm_var).grid(row=3, column=1, sticky="we", padx=6)
        ttk.Button(tab, text="Browse...", command=self._browse_instprm).grid(row=3, column=2)

        note = ("Pattern file: a pre-integrated 1D powder pattern (.xy / .dat / .fxye / .chi / .txt).\n"
                "Instrument file: the matching GSAS instrument parameter file (.prm / .instprm).")
        ttk.Label(tab, text=note, foreground="#555").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))

    def _build_phases_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="2. Phases (CIF)")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        ttk.Label(tab, text="Phase CIFs - add one per phase (multi-phase samples: add more than one).").grid(
            row=0, column=0, columnspan=2, sticky="w")

        self.cif_listbox = tk.Listbox(tab, height=8)
        self.cif_listbox.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

        btns = ttk.Frame(tab)
        btns.grid(row=1, column=1, sticky="n", padx=(8, 0), pady=(6, 0))
        ttk.Button(btns, text="Add...", command=self._add_cif).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove selected", command=self._remove_selected_cif).pack(fill="x", pady=2)

    def _build_options_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="3. Options")
        tab.columnconfigure(1, weight=1)

        ttk.Label(tab, text="GSAS-II install folder:").grid(row=0, column=0, sticky="w", pady=4)
        self.gsasii_var = tk.StringVar()
        ttk.Entry(tab, textvariable=self.gsasii_var).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(tab, text="Browse...", command=self._browse_gsasii).grid(row=0, column=2)
        ttk.Label(tab, text="Folder containing GSASIIscriptable.py. Not needed for a dry run.",
                  foreground="#555").grid(row=1, column=1, columnspan=2, sticky="w")

        ttk.Label(tab, text="Output folder:").grid(row=2, column=0, sticky="w", pady=4)
        self.outdir_var = tk.StringVar()
        outdir_entry = ttk.Entry(tab, textvariable=self.outdir_var)
        outdir_entry.grid(row=2, column=1, sticky="we", padx=6)
        # A real keystroke here means the user is taking over — stop
        # auto-generating a fresh folder on each run (see _outdir_auto in
        # __init__). Bound on the widget itself, not a trace on
        # outdir_var, so it only fires for the user actually typing —
        # not for our own .set() calls when auto-filling below.
        outdir_entry.bind("<Key>", lambda e: setattr(self, "_outdir_auto", False))
        ttk.Button(tab, text="Browse...", command=self._browse_outdir).grid(row=2, column=2)
        ttk.Label(tab, text="Left blank (the default): a fresh results_<timestamp> folder is "
                             "created next to the pattern file for every run.",
                  foreground="#555").grid(row=3, column=1, columnspan=2, sticky="w", pady=(2, 8))

        self.refine_atoms_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Refine atom positions / thermal parameters (optional, last stage)",
                         variable=self.refine_atoms_var).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=4)

        ttk.Label(tab, text="Max cell drift (fraction, e.g. 0.15 = 15%):").grid(
            row=5, column=0, sticky="w", pady=4)
        self.max_drift_var = tk.StringVar(value="0.15")
        ttk.Spinbox(tab, from_=0.01, to=0.90, increment=0.01, textvariable=self.max_drift_var,
                    width=8).grid(row=5, column=1, sticky="w", padx=6)

        self.dry_run_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Dry run (validate inputs and print the plan; no GSAS-II needed)",
                         variable=self.dry_run_var).grid(row=6, column=0, columnspan=3, sticky="w", pady=4)

    def _build_results_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text="Results")
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(1, weight=3)
        tab.rowconfigure(2, weight=1)

        self.results_summary_var = tk.StringVar(value="No completed run yet.")
        ttk.Label(tab, textvariable=self.results_summary_var, font=self.bold_font).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        self.raw_plot_frame = ttk.Frame(tab)
        self.raw_plot_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.fit_plot_frame = ttk.Frame(tab)
        self.fit_plot_frame.grid(row=1, column=1, sticky="nsew", padx=(4, 0))

        self._mount_raw_canvas(plots.make_raw_pattern_figure({}))
        self._mount_fit_canvas(plots.make_fit_overlay_figure({}))

        bottom = ttk.Frame(tab)
        bottom.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        bottom.columnconfigure(0, weight=2)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(0, weight=1)

        cell_columns = ("phase", "param", "value", "esd")
        self.cell_tree = ttk.Treeview(bottom, columns=cell_columns, show="headings", height=6)
        for col, label, width in [
            ("phase", "Phase", 110), ("param", "Param", 60),
            ("value", "Value", 110), ("esd", "Esd", 90),
        ]:
            self.cell_tree.heading(col, text=label)
            self.cell_tree.column(col, width=width, anchor="w")
        self.cell_tree.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        trim_box = ttk.Labelframe(bottom, text="Trim 2-theta range", padding=8)
        trim_box.grid(row=0, column=1, sticky="nsew")
        ttk.Label(trim_box, text="Drag on the raw pattern plot, or type bounds:",
                  wraplength=230).grid(row=0, column=0, columnspan=4, sticky="w")

        trim_row = ttk.Frame(trim_box)
        trim_row.grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 4))
        ttk.Label(trim_row, text="Low:").pack(side="left")
        self.trim_min_var = tk.StringVar()
        ttk.Entry(trim_row, textvariable=self.trim_min_var, width=8).pack(side="left", padx=(2, 10))
        ttk.Label(trim_row, text="High:").pack(side="left")
        self.trim_max_var = tk.StringVar()
        ttk.Entry(trim_row, textvariable=self.trim_max_var, width=8).pack(side="left", padx=(2, 0))

        btn_row = ttk.Frame(trim_box)
        btn_row.grid(row=2, column=0, columnspan=4, sticky="we", pady=(4, 0))
        ttk.Button(btn_row, text="Clear", command=self._clear_trim).pack(side="left")
        ttk.Button(btn_row, text="Trim & re-run", command=self._on_trim_rerun).pack(
            side="left", padx=(6, 0))

    # ------------------------------------------------------------------
    # Results tab — plot mounting, span selection, cell table
    # ------------------------------------------------------------------

    def _mount_raw_canvas(self, fig):
        """(Re)builds the raw-pattern canvas + its SpanSelector from
        scratch. Simpler and plenty fast for "redraw once per run" than
        trying to mutate an existing Figure's axes in place, and it sidesteps
        having to re-attach a SpanSelector to axes that changed anyway."""
        # Clear every child (canvas + toolbar from the previous mount, if
        # any) rather than tracking just the canvas widget, so repeated
        # re-runs don't stack up duplicate toolbars in the frame. No
        # explicit Figure cleanup needed: these were built via plots.py's
        # plt.Figure(...) constructor rather than pyplot's stateful
        # plt.figure(), so they were never registered with pyplot's figure
        # manager — dropping the reference is enough for normal GC.
        for child in self.raw_plot_frame.winfo_children():
            child.destroy()

        self.raw_fig = fig
        self.raw_canvas = FigureCanvasTkAgg(self.raw_fig, master=self.raw_plot_frame)
        toolbar_frame = ttk.Frame(self.raw_plot_frame)
        toolbar_frame.pack(side="top", fill="x")
        NavigationToolbar2Tk(self.raw_canvas, toolbar_frame).update()
        self.raw_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.raw_canvas.draw()

        self.raw_span_selector = self._make_span_selector(self.raw_fig.axes[0])

    def _mount_fit_canvas(self, fig):
        for child in self.fit_plot_frame.winfo_children():
            child.destroy()

        self.fit_fig = fig
        self.fit_canvas = FigureCanvasTkAgg(self.fit_fig, master=self.fit_plot_frame)
        toolbar_frame = ttk.Frame(self.fit_plot_frame)
        toolbar_frame.pack(side="top", fill="x")
        NavigationToolbar2Tk(self.fit_canvas, toolbar_frame).update()
        self.fit_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.fit_canvas.draw()

    def _make_span_selector(self, ax):
        """
        Builds the SpanSelector defensively across matplotlib versions —
        `props=` and `drag_from_anywhere=` are newer (3.5+) additions that
        replaced/joined the older `rectprops=` argument. Rather than pin an
        exact matplotlib version for this (GSAS-II's own dependency chain
        already puts a matplotlib version on each machine, and the office
        and home installs won't necessarily match), try newest-API-first
        and fall back gracefully — a missing `drag_from_anywhere` isn't
        worth losing the whole Results tab over.
        """
        try:
            return SpanSelector(
                ax, self._on_span_select, "horizontal", useblit=True,
                props=dict(alpha=0.2, facecolor="#2ea043"),
                interactive=True, drag_from_anywhere=True,
            )
        except TypeError:
            pass
        try:
            return SpanSelector(
                ax, self._on_span_select, "horizontal", useblit=True,
                rectprops=dict(alpha=0.2, facecolor="#2ea043"), interactive=True,
            )
        except TypeError:
            return SpanSelector(
                ax, self._on_span_select, "horizontal", useblit=True,
                rectprops=dict(alpha=0.2, facecolor="#2ea043"),
            )

    def _on_span_select(self, xmin, xmax):
        lo, hi = sorted((xmin, xmax))
        self.trim_min_var.set(f"{lo:.3f}")
        self.trim_max_var.set(f"{hi:.3f}")

    def _clear_trim(self):
        self.trim_min_var.set("")
        self.trim_max_var.set("")
        try:
            self.raw_span_selector.set_visible(False)
            self.raw_fig.canvas.draw_idle()
        except Exception:  # noqa: BLE001 — purely cosmetic, never block on it
            pass

    def _current_trim_floats(self):
        """Best-effort parse of the trim entry fields to (tmin, tmax) floats
        for redrawing the shaded region — returns (None, None) on anything
        unparsable rather than raising; validate_run_config() is what
        actually blocks a run on a bad value, this is just for the plot."""
        try:
            lo_text, hi_text = self.trim_min_var.get().strip(), self.trim_max_var.get().strip()
            if not lo_text or not hi_text:
                return None, None
            return float(lo_text), float(hi_text)
        except ValueError:
            return None, None

    def _populate_cell_tree(self, summary):
        self.cell_tree.delete(*self.cell_tree.get_children())
        for row in logic.format_cell_rows(summary):
            value = f"{row['value']:.5f}" if isinstance(row["value"], (int, float)) else str(row["value"])
            esd = f"{row['esd']:.5f}" if isinstance(row["esd"], (int, float)) else ""
            self.cell_tree.insert("", "end", values=(row["phase"], row["param"], value, esd))

    def _refresh_results(self):
        outdir = self.last_outdir
        if not outdir or not Path(outdir).is_dir():
            return
        raw = logic.read_xy_csv(str(Path(outdir) / "pattern_raw.csv"))
        fit = logic.read_xy_csv(str(Path(outdir) / "fit_final.csv"))
        summary = logic.read_summary(outdir)

        if summary and not self.trim_min_var.get().strip() and not self.trim_max_var.get().strip():
            applied = summary.get("limits_applied")
            if applied and len(applied) == 2:
                self.trim_min_var.set(f"{applied[0]:.3f}")
                self.trim_max_var.set(f"{applied[1]:.3f}")

        tmin, tmax = self._current_trim_floats()
        self._mount_raw_canvas(plots.make_raw_pattern_figure(raw, tmin=tmin, tmax=tmax))
        self._mount_fit_canvas(plots.make_fit_overlay_figure(fit))
        self.results_summary_var.set(plots.cell_summary_text(summary))
        self._populate_cell_tree(summary)

    def _on_trim_rerun(self):
        """The trim range is just another field in the same RunConfig, so
        re-running with a trimmed range is the same flow as the main Run
        button (_on_run) — this exists as its own labeled button on the
        Results tab because that's where the user is looking after picking
        a range, not because the underlying action differs."""
        self._on_run()

    # ------------------------------------------------------------------
    # Sweep tab — gsas2_candidate_sweep.py: run several (instprm, CIF)
    # candidates against the pattern from Tab 1 in parallel, ranked by
    # fit quality. Self-contained (its own candidate list, run/cancel,
    # progress table, and log) rather than sharing the single-refinement
    # run panel below, since a sweep runs several candidates at once, not
    # one sequence of stages.
    # ------------------------------------------------------------------

    def _build_sweep_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="5. Sweep")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        tab.rowconfigure(5, weight=1)

        ttk.Label(tab, text="Runs each candidate below as a full refinement against the "
                             "pattern file set on tab 1, in parallel, and ranks them by "
                             "fit quality (not just Rwp) - see gsas2_candidate_sweep.py.",
                  foreground="#555", wraplength=760).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        candidates_frame = ttk.Frame(tab)
        candidates_frame.grid(row=1, column=0, columnspan=2, sticky="we")
        candidates_frame.columnconfigure(0, weight=1)
        ttk.Label(candidates_frame, text="Candidates:").grid(row=0, column=0, sticky="w")
        btns = ttk.Frame(candidates_frame)
        btns.grid(row=0, column=1, sticky="e")
        ttk.Button(btns, text="Add...", command=self._open_add_candidate_dialog).pack(side="left")
        ttk.Button(btns, text="Remove selected",
                   command=self._remove_selected_sweep_candidate).pack(side="left", padx=(6, 0))

        candidate_columns = ("name", "instprm", "cifs")
        self.sweep_candidate_tree = ttk.Treeview(tab, columns=candidate_columns,
                                                  show="headings", height=5)
        for col, label, width in [("name", "Name", 140), ("instprm", "Instrument file", 260),
                                    ("cifs", "CIF(s)", 300)]:
            self.sweep_candidate_tree.heading(col, text=label)
            self.sweep_candidate_tree.column(col, width=width, anchor="w")
        self.sweep_candidate_tree.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(4, 8))

        run_row = ttk.Frame(tab)
        run_row.grid(row=3, column=0, columnspan=2, sticky="we")
        self.sweep_run_button = ttk.Button(run_row, text="Run sweep", command=self._on_run_sweep)
        self.sweep_run_button.pack(side="left")
        self.sweep_cancel_button = ttk.Button(run_row, text="Cancel", command=self._on_cancel_sweep,
                                               state="disabled")
        self.sweep_cancel_button.pack(side="left", padx=(6, 0))
        self.sweep_open_outdir_button = ttk.Button(run_row, text="Open sweep folder",
                                                    command=self._open_sweep_outdir, state="disabled")
        self.sweep_open_outdir_button.pack(side="left", padx=(16, 0))
        self.sweep_status_var = tk.StringVar(value="Idle")
        ttk.Label(run_row, textvariable=self.sweep_status_var, font=self.bold_font).pack(side="right")

        result_columns = ("name", "status", "correlation", "rwp")
        self.sweep_result_tree = ttk.Treeview(tab, columns=result_columns, show="headings", height=5)
        for col, label, width in [("name", "Name", 160), ("status", "Status", 140),
                                    ("correlation", "Correlation", 100), ("rwp", "Final Rwp", 100)]:
            self.sweep_result_tree.heading(col, text=label)
            self.sweep_result_tree.column(col, width=width, anchor="w")
        # The winning candidate (see _handle_sweep_event's "sweep_done"
        # handling) gets this tag so it's visually obvious which row to
        # trust without reading every number.
        self.sweep_result_tree.tag_configure("winner", background="#d7f5d7")
        self.sweep_result_tree.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(0, 8))

        sweep_log_frame = ttk.Frame(tab)
        sweep_log_frame.grid(row=5, column=0, columnspan=2, sticky="nsew")
        sweep_log_frame.columnconfigure(0, weight=1)
        sweep_log_frame.rowconfigure(0, weight=1)
        self.sweep_log_text = tk.Text(sweep_log_frame, height=8, state="disabled", wrap="word",
                                       font=tkfont.nametofont("TkFixedFont", self))
        self.sweep_log_text.grid(row=0, column=0, sticky="nsew")
        sweep_log_scroll = ttk.Scrollbar(sweep_log_frame, orient="vertical",
                                          command=self.sweep_log_text.yview)
        sweep_log_scroll.grid(row=0, column=1, sticky="ns")
        self.sweep_log_text.configure(yscrollcommand=sweep_log_scroll.set)

    def _open_add_candidate_dialog(self):
        dialog = tk.Toplevel(self)
        dialog.title("Add candidate")
        dialog.transient(self)
        dialog.columnconfigure(1, weight=1)
        dialog_cifs: list = []

        ttk.Label(dialog, text="Name:").grid(row=0, column=0, sticky="w", padx=8, pady=(10, 4))
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var).grid(
            row=0, column=1, columnspan=2, sticky="we", padx=(0, 8), pady=(10, 4))

        ttk.Label(dialog, text="Instrument file:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        instprm_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=instprm_var).grid(row=1, column=1, sticky="we", pady=4)

        def browse_instprm():
            path = filedialog.askopenfilename(title="Select instrument parameter file",
                                               filetypes=logic.INSTPRM_FILETYPES)
            if path:
                instprm_var.set(path)

        ttk.Button(dialog, text="Browse...", command=browse_instprm).grid(
            row=1, column=2, sticky="w", padx=(4, 8), pady=4)

        ttk.Label(dialog, text="CIF(s):").grid(row=2, column=0, sticky="nw", padx=8, pady=4)
        cif_listbox = tk.Listbox(dialog, height=4)
        cif_listbox.grid(row=2, column=1, sticky="nsew", pady=4)
        dialog.rowconfigure(2, weight=1)

        def add_cif():
            paths = filedialog.askopenfilenames(title="Select phase CIF(s)",
                                                 filetypes=logic.CIF_FILETYPES)
            for p in paths:
                if p not in dialog_cifs:
                    dialog_cifs.append(p)
                    cif_listbox.insert("end", p)

        def remove_selected_cif():
            for idx in reversed(cif_listbox.curselection()):
                del dialog_cifs[idx]
                cif_listbox.delete(idx)

        cif_btns = ttk.Frame(dialog)
        cif_btns.grid(row=2, column=2, sticky="n", padx=(4, 8), pady=4)
        ttk.Button(cif_btns, text="Add...", command=add_cif).pack(fill="x", pady=2)
        ttk.Button(cif_btns, text="Remove", command=remove_selected_cif).pack(fill="x", pady=2)

        def on_ok():
            name = name_var.get().strip() or f"candidate_{len(self.sweep_candidates) + 1}"
            instprm = instprm_var.get().strip()
            problems = []
            if any(c["name"] == name for c in self.sweep_candidates):
                problems.append(f"A candidate named {name!r} already exists.")
            if not instprm:
                problems.append("Instrument file is required.")
            if not dialog_cifs:
                problems.append("At least one CIF is required.")
            if problems:
                messagebox.showerror("Can't add candidate", "\n".join(f"- {p}" for p in problems),
                                      parent=dialog)
                return
            self.sweep_candidates.append({"name": name, "instprm": instprm, "cifs": list(dialog_cifs)})
            self._refresh_sweep_candidate_tree()
            dialog.destroy()

        btn_row = ttk.Frame(dialog)
        btn_row.grid(row=3, column=0, columnspan=3, sticky="e", padx=8, pady=(4, 10))
        ttk.Button(btn_row, text="Cancel", command=dialog.destroy).pack(side="right")
        ttk.Button(btn_row, text="Add", command=on_ok).pack(side="right", padx=(0, 6))

        dialog.grab_set()

    def _refresh_sweep_candidate_tree(self):
        self.sweep_candidate_tree.delete(*self.sweep_candidate_tree.get_children())
        for c in self.sweep_candidates:
            self.sweep_candidate_tree.insert(
                "", "end", values=(c["name"], c["instprm"], ", ".join(c["cifs"])))

    def _remove_selected_sweep_candidate(self):
        selected = self.sweep_candidate_tree.selection()
        selected_names = {self.sweep_candidate_tree.item(i, "values")[0] for i in selected}
        self.sweep_candidates = [c for c in self.sweep_candidates if c["name"] not in selected_names]
        self._refresh_sweep_candidate_tree()

    def _on_run_sweep(self):
        if self.sweep_proc is not None:
            return  # already running — button is disabled, but be defensive

        pattern = self.pattern_var.get().strip()
        gsasii_path = self.gsasii_var.get().strip()
        problems = []
        if not pattern:
            problems.append("No pattern file selected (set it on tab 1).")
        elif not Path(pattern).is_file():
            problems.append(f"Pattern file not found: {pattern}")
        if not gsasii_path:
            problems.append("GSAS-II install path is required (set it on tab 3).")
        elif not Path(gsasii_path).is_dir():
            problems.append(f"GSAS-II install path not found: {gsasii_path}")
        if not self.sweep_candidates:
            problems.append("Add at least one candidate.")
        if problems:
            messagebox.showerror("Can't start sweep yet", "\n".join(f"- {p}" for p in problems))
            return

        base_outdir = Path(logic.auto_outdir(pattern))
        outdir = base_outdir.parent / (base_outdir.name + "_sweep")
        outdir.mkdir(parents=True, exist_ok=True)
        manifest_path = outdir / "manifest.json"
        manifest = {
            "pattern": pattern,
            "gsasii_path": gsasii_path,
            "candidates": [
                {"name": c["name"], "instprm": c["instprm"], "cif": list(c["cifs"])}
                for c in self.sweep_candidates
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        cmd = [sys.executable, "-u", str(SWEEP_SCRIPT),
               "--manifest", str(manifest_path), "--outdir", str(outdir), "--emit-events"]

        self.sweep_result_tree.delete(*self.sweep_result_tree.get_children())
        self.sweep_row_by_name.clear()
        for c in self.sweep_candidates:
            row_id = self.sweep_result_tree.insert("", "end", values=(c["name"], "pending", "", ""))
            self.sweep_row_by_name[c["name"]] = row_id
        self.sweep_log_text.configure(state="normal")
        self.sweep_log_text.delete("1.0", "end")
        self.sweep_log_text.configure(state="disabled")

        self.sweep_status_var.set("Running...")
        self.sweep_run_button.configure(state="disabled")
        self.sweep_cancel_button.configure(state="normal")
        self.sweep_open_outdir_button.configure(state="disabled")
        self.sweep_outdir = str(outdir)

        thread = threading.Thread(target=self._run_sweep_subprocess, args=(cmd,), daemon=True)
        thread.start()
        self.after(100, self._poll_sweep_queue)

    def _on_cancel_sweep(self):
        if self.sweep_proc is not None:
            self.sweep_proc.terminate()
        self.sweep_status_var.set("Cancelling...")

    def _run_sweep_subprocess(self, cmd: list):
        try:
            # stdin=DEVNULL — see the identical comment on _run_subprocess;
            # confirmed as a real hang risk while building
            # gsas2_candidate_sweep.py, not just theoretical.
            self.sweep_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, bufsize=1,
                **_no_window_kwargs(),
            )
        except OSError as exc:
            self.sweep_event_queue.put({"event": "log", "text": f"Failed to start: {exc}"})
            self.sweep_event_queue.put({"event": "sweep_done", "ok": False, "winner": None})
            return

        for line in self.sweep_proc.stdout:
            self.sweep_event_queue.put(logic.parse_event_line(line))

        self.sweep_proc.wait()
        self.sweep_event_queue.put({"event": "process_exit", "returncode": self.sweep_proc.returncode})
        self.sweep_proc = None

    def _poll_sweep_queue(self):
        try:
            while True:
                event = self.sweep_event_queue.get_nowait()
                self._handle_sweep_event(event)
        except queue.Empty:
            pass

        if self.sweep_proc is not None or not self.sweep_event_queue.empty():
            self.after(100, self._poll_sweep_queue)

    def _handle_sweep_event(self, event: dict):
        kind = event.get("event")

        if kind == "log":
            self.sweep_log_text.configure(state="normal")
            self.sweep_log_text.insert("end", event.get("text", "") + "\n")
            self.sweep_log_text.see("end")
            self.sweep_log_text.configure(state="disabled")

        elif kind == "candidate_start":
            row_id = self.sweep_row_by_name.get(event.get("name"))
            if row_id:
                vals = list(self.sweep_result_tree.item(row_id, "values"))
                vals[1] = "running"
                self.sweep_result_tree.item(row_id, values=vals)

        elif kind == "candidate_done":
            row_id = self.sweep_row_by_name.get(event.get("name"))
            if row_id:
                summary = event.get("summary")
                if summary:
                    fq = summary.get("fit_quality") or {}
                    status = "needs review" if fq.get("needs_review", True) else "ok"
                    corr = fq.get("calc_obs_correlation")
                    rwp = summary.get("final_rwp")
                    corr_str = f"{corr:.4f}" if isinstance(corr, (int, float)) else ""
                    rwp_str = f"{rwp:.3f}" if isinstance(rwp, (int, float)) else ""
                else:
                    status, corr_str, rwp_str = "crashed", "", ""
                name = self.sweep_result_tree.item(row_id, "values")[0]
                self.sweep_result_tree.item(row_id, values=(name, status, corr_str, rwp_str))

        elif kind == "sweep_done":
            winner = event.get("winner")
            ranking = event.get("ranking") or []
            # Reorder rows best-first to match the ranking, and tag the
            # winner so it's visually obvious — see the "winner" tag
            # configured in _build_sweep_tab.
            for index, name in enumerate(ranking):
                row_id = self.sweep_row_by_name.get(name)
                if row_id:
                    self.sweep_result_tree.move(row_id, "", index)
                    self.sweep_result_tree.item(row_id, tags=("winner",) if name == winner else ())
            if winner:
                self.sweep_status_var.set(f"Done - best: {winner}")
            else:
                self.sweep_status_var.set("Done - no candidate passed the fit-quality check")
            self._finish_sweep()

        elif kind == "process_exit":
            if self.sweep_status_var.get() == "Running...":
                rc = event.get("returncode")
                self.sweep_status_var.set(f"Exited (code {rc})")
                self._finish_sweep()

    def _finish_sweep(self):
        self.sweep_run_button.configure(state="normal")
        self.sweep_cancel_button.configure(state="disabled")
        if self.sweep_outdir and Path(self.sweep_outdir).is_dir():
            self.sweep_open_outdir_button.configure(state="normal")

    def _open_sweep_outdir(self):
        if self.sweep_outdir:
            open_path(self.sweep_outdir)

    # ------------------------------------------------------------------
    # Swarm tab — gsas2_swarm_optimize.py: surrogate-assisted search over
    # Size/Mustrain starting points from a checkpoint .gpx. Built as a
    # speed-testing front end first: every knob that controls how much
    # work one run does (perturbations per generation, number of
    # generations, surrogate particle/generation counts, backend, worker
    # count) is exposed directly, plus an elapsed-time readout, so you can
    # see how fast a given setting actually runs before committing to a
    # long unattended one. Self-contained state (own subprocess, own
    # progress table shape — one row per outer iteration, not per stage or
    # candidate), same pattern as the Sweep tab above.
    # ------------------------------------------------------------------

    def _build_swarm_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="4. Swarm")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(9, weight=1)

        ttk.Label(tab, text="Searches many Size/Mustrain starting points from a checkpoint .gpx "
                             "(from a real gsas2_auto_refine.py run) and reports the best, "
                             "physically-sane result found — see gsas2_swarm_optimize.py.",
                  foreground="#555", wraplength=760).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(tab, text="Checkpoint (.gpx):").grid(row=1, column=0, sticky="w", pady=4)
        self.swarm_checkpoint_var = tk.StringVar()
        ttk.Entry(tab, textvariable=self.swarm_checkpoint_var).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(tab, text="Browse...", command=self._browse_swarm_checkpoint).grid(row=1, column=2)

        ttk.Label(tab, text="Output folder:").grid(row=2, column=0, sticky="w", pady=4)
        self.swarm_outdir_var = tk.StringVar()
        swarm_outdir_entry = ttk.Entry(tab, textvariable=self.swarm_outdir_var)
        swarm_outdir_entry.grid(row=2, column=1, sticky="we", padx=6)
        # Same "stop auto-generating the moment the user takes over" pattern
        # as the main Options tab's outdir field — see _outdir_auto's
        # docstring in __init__.
        swarm_outdir_entry.bind("<Key>", lambda e: setattr(self, "_swarm_outdir_auto", False))
        ttk.Button(tab, text="Browse...", command=self._browse_swarm_outdir).grid(row=2, column=2)
        ttk.Label(tab, text="Left blank (the default): a fresh swarm_<timestamp> folder is "
                             "created next to the checkpoint file for every run.",
                  foreground="#555").grid(row=3, column=1, columnspan=2, sticky="w", pady=(2, 8))

        knobs = ttk.Labelframe(tab, text="How much work one run does", padding=8)
        knobs.grid(row=4, column=0, columnspan=3, sticky="we", pady=(0, 8))
        for col in (1, 3):
            knobs.columnconfigure(col, weight=0)

        ttk.Label(knobs, text="Perturbations per generation:").grid(row=0, column=0, sticky="w", pady=3)
        self.swarm_perturbations_var = tk.IntVar(value=50)
        ttk.Spinbox(knobs, from_=1, to=2000, increment=1, textvariable=self.swarm_perturbations_var,
                    width=8).grid(row=0, column=1, sticky="w", padx=(4, 20))

        ttk.Label(knobs, text="Generations (outer iterations):").grid(row=0, column=2, sticky="w", pady=3)
        self.swarm_generations_var = tk.IntVar(value=20)
        ttk.Spinbox(knobs, from_=1, to=1000, increment=1, textvariable=self.swarm_generations_var,
                    width=8).grid(row=0, column=3, sticky="w", padx=(4, 0))

        ttk.Label(knobs, text="Surrogate particles:").grid(row=1, column=0, sticky="w", pady=3)
        self.swarm_surrogate_particles_var = tk.IntVar(value=200)
        ttk.Spinbox(knobs, from_=1, to=100000, increment=10, textvariable=self.swarm_surrogate_particles_var,
                    width=8).grid(row=1, column=1, sticky="w", padx=(4, 20))

        ttk.Label(knobs, text="Surrogate generations:").grid(row=1, column=2, sticky="w", pady=3)
        self.swarm_surrogate_generations_var = tk.IntVar(value=150)
        ttk.Spinbox(knobs, from_=1, to=100000, increment=10,
                    textvariable=self.swarm_surrogate_generations_var,
                    width=8).grid(row=1, column=3, sticky="w", padx=(4, 0))

        ttk.Label(knobs, text="Backend:").grid(row=2, column=0, sticky="w", pady=3)
        self.swarm_backend_var = tk.StringVar(value="auto")
        ttk.Combobox(knobs, textvariable=self.swarm_backend_var, values=("auto", "cpu", "gpu"),
                     state="readonly", width=6).grid(row=2, column=1, sticky="w", padx=(4, 20))

        ttk.Label(knobs, text="Max parallel workers (blank = no limit):").grid(
            row=2, column=2, sticky="w", pady=3)
        self.swarm_max_workers_var = tk.StringVar(value="")
        ttk.Entry(knobs, textvariable=self.swarm_max_workers_var, width=8).grid(
            row=2, column=3, sticky="w", padx=(4, 0))

        ttk.Label(knobs, text="Seed (blank = random each run):").grid(row=3, column=0, sticky="w", pady=3)
        self.swarm_seed_var = tk.StringVar(value="")
        ttk.Entry(knobs, textvariable=self.swarm_seed_var, width=8).grid(
            row=3, column=1, sticky="w", padx=(4, 20))
        ttk.Label(knobs, text="A fixed seed makes a run exactly reproducible (same result every time).",
                  foreground="#555").grid(row=3, column=2, columnspan=2, sticky="w")

        ttk.Label(knobs, text="Mustrain type:").grid(row=4, column=0, sticky="w", pady=3)
        self.swarm_mustrain_type_var = tk.StringVar(value="isotropic")
        ttk.Combobox(knobs, textvariable=self.swarm_mustrain_type_var,
                     values=("isotropic", "uniaxial"), state="readonly", width=9).grid(
            row=4, column=1, sticky="w", padx=(4, 20))
        ttk.Label(knobs, text="isotropic (recommended): uniaxial Mustrain + isotropic Size are "
                              "~98% correlated on real data, causing most 'insane' results.",
                  foreground="#555", wraplength=420).grid(row=4, column=2, columnspan=2, sticky="w")

        self.swarm_keep_evaluations_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(knobs, text="Keep every candidate's evaluation files",
                        variable=self.swarm_keep_evaluations_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Label(knobs, text="Off (recommended): only the winner is kept, as best.gpx — a real "
                              "run can otherwise leave hundreds of MB to multiple GB of "
                              "discarded candidates on disk.",
                  foreground="#555", wraplength=420).grid(row=5, column=2, columnspan=2, sticky="w")

        trim_box = ttk.Labelframe(tab, text="Fit-range trimming (optional)", padding=8)
        trim_box.grid(row=6, column=0, columnspan=3, sticky="we", pady=(0, 8))
        ttk.Label(trim_box, text="Also search how many degrees to discard from each end of the "
                                  "fit range (e.g. beamstop shadow at low angle, vanishing peak "
                                  "statistics at high angle). Rwp is NOT directly comparable "
                                  "across different cutoffs — trimming can mechanically lower "
                                  "it without the model actually improving. Keep bounds tight, "
                                  "to whatever's independently justified for your data. Leave a "
                                  "LO/HI pair blank to leave that side of the range untouched.",
                  foreground="#555", wraplength=740).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        ttk.Label(trim_box, text="Low-angle cutoff, LO/HI (deg):").grid(row=1, column=0, sticky="w", pady=3)
        low_angle_frame = ttk.Frame(trim_box)
        low_angle_frame.grid(row=1, column=1, sticky="w", padx=(4, 20))
        self.swarm_low_angle_lo_var = tk.StringVar(value="")
        self.swarm_low_angle_hi_var = tk.StringVar(value="")
        ttk.Entry(low_angle_frame, textvariable=self.swarm_low_angle_lo_var, width=6).pack(side="left")
        ttk.Label(low_angle_frame, text=" to ").pack(side="left")
        ttk.Entry(low_angle_frame, textvariable=self.swarm_low_angle_hi_var, width=6).pack(side="left")

        ttk.Label(trim_box, text="High-angle cutoff, LO/HI (deg):").grid(row=1, column=2, sticky="w", pady=3)
        high_angle_frame = ttk.Frame(trim_box)
        high_angle_frame.grid(row=1, column=3, sticky="w")
        self.swarm_high_angle_lo_var = tk.StringVar(value="")
        self.swarm_high_angle_hi_var = tk.StringVar(value="")
        ttk.Entry(high_angle_frame, textvariable=self.swarm_high_angle_lo_var, width=6).pack(side="left")
        ttk.Label(high_angle_frame, text=" to ").pack(side="left")
        ttk.Entry(high_angle_frame, textvariable=self.swarm_high_angle_hi_var, width=6).pack(side="left")

        run_row = ttk.Frame(tab)
        run_row.grid(row=7, column=0, columnspan=3, sticky="we")
        self.swarm_run_button = ttk.Button(run_row, text="Run swarm", command=self._on_run_swarm)
        self.swarm_run_button.pack(side="left")
        self.swarm_cancel_button = ttk.Button(run_row, text="Cancel", command=self._on_cancel_swarm,
                                               state="disabled")
        self.swarm_cancel_button.pack(side="left", padx=(6, 0))
        self.swarm_open_outdir_button = ttk.Button(run_row, text="Open output folder",
                                                     command=self._open_swarm_outdir, state="disabled")
        self.swarm_open_outdir_button.pack(side="left", padx=(16, 0))
        self.swarm_view_fit_button = ttk.Button(run_row, text="View best fit",
                                                  command=self._open_swarm_best_fit_view, state="disabled")
        self.swarm_view_fit_button.pack(side="left", padx=(6, 0))
        self.swarm_elapsed_var = tk.StringVar(value="")
        ttk.Label(run_row, textvariable=self.swarm_elapsed_var).pack(side="right", padx=(0, 12))
        self.swarm_status_var = tk.StringVar(value="Idle")
        ttk.Label(run_row, textvariable=self.swarm_status_var, font=self.bold_font).pack(side="right")

        progress_columns = ("iteration", "sane", "best_fitness")
        self.swarm_progress_tree = ttk.Treeview(tab, columns=progress_columns, show="headings", height=6)
        for col, label, width in [
            ("iteration", "Generation", 90), ("sane", "Sane / total", 100),
            ("best_fitness", "Best Rwp so far", 130),
        ]:
            self.swarm_progress_tree.heading(col, text=label)
            self.swarm_progress_tree.column(col, width=width, anchor="w")
        self.swarm_progress_tree.grid(row=8, column=0, columnspan=3, sticky="we", pady=(10, 6))

        swarm_log_frame = ttk.Frame(tab)
        swarm_log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew")
        swarm_log_frame.columnconfigure(0, weight=1)
        swarm_log_frame.rowconfigure(0, weight=1)
        self.swarm_log_text = tk.Text(swarm_log_frame, height=8, state="disabled", wrap="word",
                                       font=tkfont.nametofont("TkFixedFont", self))
        self.swarm_log_text.grid(row=0, column=0, sticky="nsew")
        swarm_log_scroll = ttk.Scrollbar(swarm_log_frame, orient="vertical",
                                          command=self.swarm_log_text.yview)
        swarm_log_scroll.grid(row=0, column=1, sticky="ns")
        self.swarm_log_text.configure(yscrollcommand=swarm_log_scroll.set)

    def _browse_swarm_checkpoint(self):
        path = filedialog.askopenfilename(title="Select checkpoint .gpx file",
                                           filetypes=swarm_logic.GPX_FILETYPES)
        if path:
            self.swarm_checkpoint_var.set(path)
            if self._swarm_outdir_auto:
                self.swarm_outdir_var.set(swarm_logic.auto_swarm_outdir(path))

    def _browse_swarm_outdir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.swarm_outdir_var.set(path)
            self._swarm_outdir_auto = False

    def _collect_swarm_config(self) -> swarm_logic.SwarmRunConfig:
        def _parse_optional_int(text: str):
            text = text.strip()
            if not text:
                return None
            try:
                return int(text)
            except ValueError:
                return -1  # deliberately invalid — validate_swarm_config() catches counts < 1

        def _parse_optional_bounds(lo_text: str, hi_text: str):
            lo_text, hi_text = lo_text.strip(), hi_text.strip()
            if not lo_text and not hi_text:
                return None
            try:
                return (float(lo_text), float(hi_text))
            except ValueError:
                return (-1.0, -1.0)  # deliberately invalid (also catches one field left blank)
                                      # — validate_swarm_config() reports the real problem

        return swarm_logic.SwarmRunConfig(
            checkpoint=self.swarm_checkpoint_var.get().strip(),
            gsasii_path=self.gsasii_var.get().strip(),
            outdir=self.swarm_outdir_var.get().strip(),
            outer_iterations=self.swarm_generations_var.get(),
            perturbations=self.swarm_perturbations_var.get(),
            surrogate_particles=self.swarm_surrogate_particles_var.get(),
            surrogate_generations=self.swarm_surrogate_generations_var.get(),
            backend=self.swarm_backend_var.get(),
            mustrain_type=self.swarm_mustrain_type_var.get(),
            keep_evaluations=self.swarm_keep_evaluations_var.get(),
            seed=_parse_optional_int(self.swarm_seed_var.get()),
            max_workers=_parse_optional_int(self.swarm_max_workers_var.get()),
            low_angle_cutoff_bounds=_parse_optional_bounds(
                self.swarm_low_angle_lo_var.get(), self.swarm_low_angle_hi_var.get()),
            high_angle_cutoff_bounds=_parse_optional_bounds(
                self.swarm_high_angle_lo_var.get(), self.swarm_high_angle_hi_var.get()),
        )

    def _on_run_swarm(self):
        if self.swarm_proc is not None:
            return  # already running — button is disabled, but be defensive

        if self._swarm_outdir_auto:
            self.swarm_outdir_var.set(swarm_logic.auto_swarm_outdir(self.swarm_checkpoint_var.get()))

        cfg = self._collect_swarm_config()
        problems = swarm_logic.validate_swarm_config(cfg)
        if problems:
            messagebox.showerror("Can't start swarm run yet", "\n".join(f"- {p}" for p in problems))
            return

        Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
        cmd = swarm_logic.build_swarm_command(cfg, script_path=str(SWARM_SCRIPT), python_exe=sys.executable)

        self.swarm_progress_tree.delete(*self.swarm_progress_tree.get_children())
        self.swarm_log_text.configure(state="normal")
        self.swarm_log_text.delete("1.0", "end")
        self.swarm_log_text.configure(state="disabled")

        self.swarm_status_var.set("Running...")
        self.swarm_run_button.configure(state="disabled")
        self.swarm_cancel_button.configure(state="normal")
        self.swarm_open_outdir_button.configure(state="disabled")
        self.swarm_view_fit_button.configure(state="disabled")
        self.swarm_outdir = cfg.outdir
        self.swarm_start_time = time.time()
        self.swarm_elapsed_var.set("0.0s elapsed")

        thread = threading.Thread(target=self._run_swarm_subprocess, args=(cmd,), daemon=True)
        thread.start()
        self.after(100, self._poll_swarm_queue)
        self._tick_swarm_elapsed()

    def _on_cancel_swarm(self):
        if self.swarm_proc is not None:
            self.swarm_proc.terminate()
        self.swarm_status_var.set("Cancelling...")

    def _run_swarm_subprocess(self, cmd: list):
        try:
            # stdin=DEVNULL — see the identical comment on _run_subprocess;
            # the same hang risk applies to every subprocess this GUI
            # launches, not just gsas2_auto_refine.py.
            self.swarm_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, bufsize=1,
                **_no_window_kwargs(),
            )
        except OSError as exc:
            self.swarm_event_queue.put({"event": "log", "text": f"Failed to start: {exc}"})
            self.swarm_event_queue.put({"event": "swarm_done", "ok": False})
            return

        for line in self.swarm_proc.stdout:
            self.swarm_event_queue.put(logic.parse_event_line(line))

        self.swarm_proc.wait()
        self.swarm_event_queue.put({"event": "process_exit", "returncode": self.swarm_proc.returncode})
        self.swarm_proc = None

    def _poll_swarm_queue(self):
        try:
            while True:
                event = self.swarm_event_queue.get_nowait()
                self._handle_swarm_event(event)
        except queue.Empty:
            pass

        if self.swarm_proc is not None or not self.swarm_event_queue.empty():
            self.after(100, self._poll_swarm_queue)

    def _tick_swarm_elapsed(self):
        """Updates the elapsed-time readout roughly every 200ms while a run
        is active — the whole point of this tab is seeing how fast a given
        set of knobs actually runs, so this needs to keep moving during the
        run, not just report a final total once it's done."""
        if self.swarm_start_time is None or self.swarm_proc is None:
            return
        elapsed = time.time() - self.swarm_start_time
        self.swarm_elapsed_var.set(f"{_format_elapsed(elapsed)} elapsed")
        self.after(200, self._tick_swarm_elapsed)

    def _handle_swarm_event(self, event: dict):
        kind = event.get("event")

        if kind == "log":
            self.swarm_log_text.configure(state="normal")
            self.swarm_log_text.insert("end", event.get("text", "") + "\n")
            self.swarm_log_text.see("end")
            self.swarm_log_text.configure(state="disabled")

        elif kind == "iteration_result":
            iteration = event.get("iteration")
            n_sane = event.get("n_sane")
            best_fitness = event.get("best_fitness")
            best_str = f"{best_fitness:.4f}" if isinstance(best_fitness, (int, float)) else ""
            sane_str = f"{n_sane}/{self.swarm_perturbations_var.get()}" if n_sane is not None else ""
            self.swarm_progress_tree.insert(
                "", "end", values=(iteration, sane_str, best_str))

        elif kind == "swarm_done":
            ok = event.get("ok", False)
            best_rwp = event.get("best_rwp")
            if self.swarm_start_time is not None:
                total = _format_elapsed(time.time() - self.swarm_start_time)
                self.swarm_elapsed_var.set(f"{total} total")
            if ok:
                self.swarm_status_var.set(f"Done - best Rwp={best_rwp:.4f}"
                                           if isinstance(best_rwp, (int, float)) else "Done")
            else:
                self.swarm_status_var.set("Finished - no sane result found")
            self._finish_swarm()

        elif kind == "process_exit":
            if self.swarm_status_var.get() == "Running...":
                rc = event.get("returncode")
                self.swarm_status_var.set(f"Exited (code {rc})")
                self._finish_swarm()

    def _finish_swarm(self):
        self.swarm_run_button.configure(state="normal")
        self.swarm_cancel_button.configure(state="disabled")
        self.swarm_start_time = None
        if self.swarm_outdir and Path(self.swarm_outdir).is_dir():
            self.swarm_open_outdir_button.configure(state="normal")
        summary = logic.read_summary(self.swarm_outdir, filename="swarm_summary.json") if self.swarm_outdir else None
        if summary and summary.get("fit_final_csv"):
            self.swarm_view_fit_button.configure(state="normal")

    def _open_swarm_outdir(self):
        if self.swarm_outdir:
            open_path(self.swarm_outdir)

    def _open_swarm_best_fit_view(self):
        """Opens a small standalone window plotting the swarm's best
        verified result (raw pattern + fit overlay), reusing the exact
        same figure-builders as the main Results tab (gsas2_plots.py) —
        deliberately a separate lightweight window rather than repointing
        the Results tab at the swarm's outdir, since that tab's trim/
        re-run controls are specific to the main refinement workflow and
        would re-run the wrong thing if clicked here."""
        summary = logic.read_summary(self.swarm_outdir, filename="swarm_summary.json") if self.swarm_outdir else None
        if not summary or not summary.get("fit_final_csv"):
            messagebox.showinfo("No fit to view", "No completed swarm run with an exportable "
                                                    "fit yet.")
            return

        win = tk.Toplevel(self)
        win.title(f"Swarm best fit — {plots.cell_summary_text(summary)}")
        win.geometry("900x760")

        raw = logic.read_xy_csv(summary.get("pattern_raw_csv", ""))
        fit = logic.read_xy_csv(summary.get("fit_final_csv", ""))

        raw_frame = ttk.Frame(win)
        raw_frame.pack(side="top", fill="both", expand=True)
        raw_canvas = FigureCanvasTkAgg(plots.make_raw_pattern_figure(raw), master=raw_frame)
        raw_toolbar_frame = ttk.Frame(raw_frame)
        raw_toolbar_frame.pack(side="top", fill="x")
        NavigationToolbar2Tk(raw_canvas, raw_toolbar_frame).update()
        raw_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        raw_canvas.draw()

        fit_frame = ttk.Frame(win)
        fit_frame.pack(side="top", fill="both", expand=True)
        fit_canvas = FigureCanvasTkAgg(plots.make_fit_overlay_figure(fit), master=fit_frame)
        fit_toolbar_frame = ttk.Frame(fit_frame)
        fit_toolbar_frame.pack(side="top", fill="x")
        NavigationToolbar2Tk(fit_canvas, fit_toolbar_frame).update()
        fit_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        fit_canvas.draw()

    def _build_run_panel(self, parent):
        panel = ttk.Frame(parent, padding=(10, 6))
        panel.pack(side="top", fill="both", expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(2, weight=1)

        btn_row = ttk.Frame(panel)
        btn_row.grid(row=0, column=0, sticky="we")
        self.run_button = ttk.Button(btn_row, text="Run refinement", command=self._on_run)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(btn_row, text="Cancel", command=self._on_cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=(6, 0))
        self.open_outdir_button = ttk.Button(btn_row, text="Open output folder",
                                              command=self._open_outdir, state="disabled")
        self.open_outdir_button.pack(side="left", padx=(16, 0))
        self.open_cif_button = ttk.Button(btn_row, text="Open refined CIF(s)",
                                           command=self._open_refined_cifs, state="disabled")
        self.open_cif_button.pack(side="left", padx=(6, 0))

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(btn_row, textvariable=self.status_var, font=self.bold_font).pack(side="right")

        columns = ("stage", "status", "rwp_before", "rwp_after")
        self.stage_tree = ttk.Treeview(panel, columns=columns, show="headings", height=7)
        for col, label, width in [
            ("stage", "Stage", 220), ("status", "Status", 120),
            ("rwp_before", "Rwp before", 100), ("rwp_after", "Rwp after", 100),
        ]:
            self.stage_tree.heading(col, text=label)
            self.stage_tree.column(col, width=width, anchor="w")
        self.stage_tree.grid(row=1, column=0, sticky="we", pady=(10, 6))

        log_frame = ttk.Frame(panel)
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=12, state="disabled", wrap="word",
                                 font=tkfont.nametofont("TkFixedFont", self))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    # ------------------------------------------------------------------
    # Config persistence / prefill
    # ------------------------------------------------------------------

    def _prefill_from_config(self):
        self.gsasii_var.set(self.config_data.get("gsasii_path", ""))

    def _current_config_to_save(self) -> dict:
        return {"gsasii_path": self.gsasii_var.get()}

    def _on_close(self):
        logic.save_config(self._current_config_to_save())
        self.destroy()

    # ------------------------------------------------------------------
    # File pickers
    # ------------------------------------------------------------------

    def _browse_pattern(self):
        path = filedialog.askopenfilename(title="Select pattern file",
                                           filetypes=logic.PATTERN_FILETYPES)
        if path:
            self.pattern_var.set(path)

    def _browse_instprm(self):
        path = filedialog.askopenfilename(title="Select instrument parameter file",
                                           filetypes=logic.INSTPRM_FILETYPES)
        if path:
            self.instprm_var.set(path)

    def _add_cif(self):
        paths = filedialog.askopenfilenames(title="Select phase CIF(s)",
                                             filetypes=logic.CIF_FILETYPES)
        for p in paths:
            if p not in self.cif_paths:
                self.cif_paths.append(p)
                self.cif_listbox.insert("end", p)

    def _remove_selected_cif(self):
        for idx in reversed(self.cif_listbox.curselection()):
            del self.cif_paths[idx]
            self.cif_listbox.delete(idx)

    def _browse_gsasii(self):
        path = filedialog.askdirectory(title="Select GSAS-II install folder")
        if path:
            self.gsasii_var.set(path)

    def _browse_outdir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.outdir_var.set(path)
            self._outdir_auto = False  # explicit choice — stop auto-generating

    def _load_example(self):
        name = self.example_var.get()
        example = self.examples.get(name)
        if not example:
            return
        self.pattern_var.set(example["pattern"])
        self.instprm_var.set(example["instprm"])
        self.cif_paths = list(example["cifs"])
        self.cif_listbox.delete(0, "end")
        for c in self.cif_paths:
            self.cif_listbox.insert("end", c)

    # ------------------------------------------------------------------
    # Run / cancel
    # ------------------------------------------------------------------

    def _collect_config(self) -> logic.RunConfig:
        try:
            drift = float(self.max_drift_var.get())
        except ValueError:
            drift = -1.0  # deliberately invalid — validate_run_config() will catch it

        self._trim_parse_error = None
        tmin = self._parse_trim_field(self.trim_min_var.get())
        tmax = self._parse_trim_field(self.trim_max_var.get())

        return logic.RunConfig(
            pattern=self.pattern_var.get().strip(),
            instprm=self.instprm_var.get().strip(),
            cifs=list(self.cif_paths),
            outdir=self.outdir_var.get().strip(),
            gsasii_path=self.gsasii_var.get().strip(),
            refine_atoms=self.refine_atoms_var.get(),
            max_cell_drift=drift,
            dry_run=self.dry_run_var.get(),
            tmin=tmin,
            tmax=tmax,
        )

    def _parse_trim_field(self, text: str):
        """Parses one trim Entry's text to a float or None (blank field).
        An unparsable non-blank value is reported via self._trim_parse_error
        (checked in _on_run) and treated as None here so it doesn't also
        trip validate_run_config's separate "set both bounds" check with a
        second, more confusing message."""
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            self._trim_parse_error = f"Trim bound {text!r} is not a number."
            return None

    def _on_run(self):
        if self.proc is not None:
            return  # already running — Run button is disabled, but be defensive

        if self._outdir_auto:
            # Fresh timestamped folder next to the pattern file for every
            # run — see logic.auto_outdir and _outdir_auto's docstring in
            # __init__. Skipped once the user has typed into or Browse'd
            # the field themselves, so a deliberate choice always sticks.
            self.outdir_var.set(logic.auto_outdir(self.pattern_var.get()))

        cfg = self._collect_config()
        problems = logic.validate_run_config(cfg)
        if self._trim_parse_error:
            problems = [self._trim_parse_error] + problems
        if problems:
            messagebox.showerror("Can't start refinement yet", "\n".join(f"- {p}" for p in problems))
            return

        Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
        cmd = logic.build_command(cfg, script_path=str(REFINE_SCRIPT), python_exe=sys.executable)

        self._reset_run_ui()
        self.status_var.set("Running...")
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.open_outdir_button.configure(state="disabled")
        self.open_cif_button.configure(state="disabled")
        self.last_outdir = cfg.outdir
        self.last_refined_cifs = []

        logic.save_config(self._current_config_to_save())

        thread = threading.Thread(target=self._run_subprocess, args=(cmd,), daemon=True)
        thread.start()
        self.after(100, self._poll_queue)

    def _on_cancel(self):
        if self.proc is not None:
            self.proc.terminate()
        self.status_var.set("Cancelling...")

    def _reset_run_ui(self):
        self.stage_tree.delete(*self.stage_tree.get_children())
        self.stage_row_by_name.clear()
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _run_subprocess(self, cmd: list):
        try:
            # stdin=DEVNULL: gsas2_auto_refine.py is fully non-interactive
            # and should never read from stdin, but without explicitly
            # closing it the child inherits whatever this process's own
            # stdin is — confirmed as a real hang risk (not just
            # theoretical) via gsas2_candidate_sweep.py, whose
            # subprocess.run() calls blocked for minutes with ~0 CPU usage
            # until stdin was closed the same way.
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, bufsize=1,
                **_no_window_kwargs(),
            )
        except OSError as exc:
            self.event_queue.put({"event": "log", "text": f"Failed to start: {exc}"})
            self.event_queue.put({"event": "done", "ok": False, "failed_stages": [],
                                   "refined_cifs": [], "outdir": self.last_outdir})
            return

        for line in self.proc.stdout:
            self.event_queue.put(logic.parse_event_line(line))

        self.proc.wait()
        # Belt-and-suspenders: if the child exited without ever emitting a
        # "done" event (crash before that point, or --dry-run which has no
        # results to report), still unblock the UI.
        self.event_queue.put({"event": "process_exit", "returncode": self.proc.returncode})
        self.proc = None

    def _poll_queue(self):
        try:
            while True:
                event = self.event_queue.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass

        if self.proc is not None or not self.event_queue.empty():
            self.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # Event handling — updates widgets from the parsed subprocess stream
    # ------------------------------------------------------------------

    def _handle_event(self, event: dict):
        kind = event.get("event")

        if kind == "log":
            self._append_log(event.get("text", ""))

        elif kind == "plan":
            for stage in event.get("stages", []):
                name = stage["name"]
                label = STAGE_LABELS.get(name, name)
                if stage.get("optional"):
                    label += " (optional)"
                row_id = self.stage_tree.insert("", "end", values=(label, "pending", "", ""))
                self.stage_row_by_name[name] = row_id

        elif kind == "stage_start":
            row_id = self.stage_row_by_name.get(event.get("name"))
            if row_id:
                vals = list(self.stage_tree.item(row_id, "values"))
                vals[1] = "running"
                self.stage_tree.item(row_id, values=vals)

        elif kind == "stage_result":
            row_id = self.stage_row_by_name.get(event.get("name"))
            if row_id:
                rwp_before = event.get("rwp_before")
                rwp_after = event.get("rwp_after")
                vals = [
                    self.stage_tree.item(row_id, "values")[0],
                    event.get("status", ""),
                    f"{rwp_before:.3f}" if isinstance(rwp_before, (int, float)) else "",
                    f"{rwp_after:.3f}" if isinstance(rwp_after, (int, float)) else "",
                ]
                self.stage_tree.item(row_id, values=vals)

        elif kind == "done":
            ok = event.get("ok", False)
            self.last_outdir = event.get("outdir") or self.last_outdir
            self.last_refined_cifs = event.get("refined_cifs", [])
            if ok:
                self.status_var.set("Done")
            else:
                failed = ", ".join(event.get("failed_stages", [])) or "unknown stage"
                self.status_var.set(f"Finished - not converged: {failed}")
            self._finish_run(success=ok)

        elif kind == "process_exit":
            # Only meaningful if we never saw a "done" event above (e.g. a
            # --dry-run, which prints a plan and exits with no results).
            if self.status_var.get() == "Running...":
                rc = event.get("returncode")
                self.status_var.set("Dry run complete" if rc == 0 else f"Exited (code {rc})")
                self._finish_run(success=(rc == 0))

    def _finish_run(self, success: bool):
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if self.last_outdir and Path(self.last_outdir).is_dir():
            self.open_outdir_button.configure(state="normal")
        if self.last_refined_cifs:
            self.open_cif_button.configure(state="normal")
        # Refresh the Results tab even on a failed/partial run: main() in
        # gsas2_auto_refine.py exports pattern_raw.csv right after the
        # histogram loads, before any stage runs, so there's often
        # something worth plotting even when the refinement itself didn't
        # converge.
        self._refresh_results()

    def _append_log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _open_outdir(self):
        if self.last_outdir:
            open_path(self.last_outdir)

    def _open_refined_cifs(self):
        for c in self.last_refined_cifs:
            open_path(c)


def main():
    if not REFINE_SCRIPT.is_file():
        print(f"ERROR: expected to find gsas2_auto_refine.py next to this GUI at "
              f"{REFINE_SCRIPT}, but it's not there.", file=sys.stderr)
        return 2
    if not SWEEP_SCRIPT.is_file():
        print(f"ERROR: expected to find gsas2_candidate_sweep.py next to this GUI at "
              f"{SWEEP_SCRIPT}, but it's not there.", file=sys.stderr)
        return 2
    if not SWARM_SCRIPT.is_file():
        print(f"ERROR: expected to find gsas2_swarm_optimize.py next to this GUI at "
              f"{SWARM_SCRIPT}, but it's not there.", file=sys.stderr)
        return 2
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
