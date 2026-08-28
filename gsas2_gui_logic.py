#!/usr/bin/env python3
"""
gsas2_gui_logic.py — pure, tkinter-free logic used by gsas2_gui.py.

Deliberately separated from the widget/event code in gsas2_gui.py so it can
be unit-tested in any environment (including one with no display and no
tkinter installed, like a headless CI box) — see test_gui_logic.py. Nothing
in this module imports tkinter, subprocess side effects aside from writing
the config file, or GSASIIscriptable.
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path.home() / ".gsas2_auto_refine_gui.json"

PATTERN_FILETYPES = [
    ("Powder patterns", "*.xy *.dat *.fxye *.chi *.txt"),
    ("All files", "*.*"),
]
INSTPRM_FILETYPES = [
    ("Instrument parameter files", "*.prm *.instprm"),
    ("All files", "*.*"),
]
CIF_FILETYPES = [
    ("CIF files", "*.cif"),
    ("All files", "*.*"),
]


@dataclass
class RunConfig:
    """Everything the GUI needs to launch one refinement run."""
    pattern: str = ""
    instprm: str = ""
    cifs: list = field(default_factory=list)
    outdir: str = ""
    gsasii_path: str = ""
    refine_atoms: bool = False
    max_cell_drift: float = 0.15
    dry_run: bool = False
    tmin: Optional[float] = None
    tmax: Optional[float] = None


def validate_run_config(cfg: RunConfig) -> list:
    """Returns a list of human-readable problems; empty list means ready to run."""
    problems = []
    if not cfg.pattern:
        problems.append("No pattern file selected.")
    elif not Path(cfg.pattern).is_file():
        problems.append(f"Pattern file not found: {cfg.pattern}")

    if not cfg.instprm:
        problems.append("No instrument parameter file selected.")
    elif not Path(cfg.instprm).is_file():
        problems.append(f"Instrument file not found: {cfg.instprm}")

    if not cfg.cifs:
        problems.append("No phase CIF added — add at least one.")
    else:
        for c in cfg.cifs:
            if not Path(c).is_file():
                problems.append(f"Phase CIF not found: {c}")

    if not cfg.outdir:
        problems.append("No output folder selected.")

    if not cfg.dry_run and not cfg.gsasii_path:
        problems.append("GSAS-II install path is required for a real run "
                         "(only a dry run can skip it).")
    if not cfg.dry_run and cfg.gsasii_path and not Path(cfg.gsasii_path).is_dir():
        problems.append(f"GSAS-II install path not found: {cfg.gsasii_path}")

    if not (0.0 < cfg.max_cell_drift < 1.0):
        problems.append("Max cell drift should be a fraction between 0 and 1 (e.g. 0.15 for 15%).")

    if (cfg.tmin is None) != (cfg.tmax is None):
        problems.append("Set both the low and high 2-theta trim bounds, or clear both.")
    elif cfg.tmin is not None and cfg.tmin >= cfg.tmax:
        problems.append(f"Trim low bound ({cfg.tmin}) must be less than the high bound ({cfg.tmax}).")

    return problems


def build_command(cfg: RunConfig, script_path: str, python_exe: str = "python3") -> list:
    """
    Builds the exact subprocess argv that runs gsas2_auto_refine.py for this
    config. `-u` forces unbuffered stdout so the GUI's event stream (see
    --emit-events) arrives line-by-line instead of batched by Python's
    default pipe buffering.
    """
    cmd = [python_exe, "-u", script_path,
           "--pattern", cfg.pattern,
           "--instprm", cfg.instprm]
    for c in cfg.cifs:
        cmd += ["--cif", c]
    cmd += ["--outdir", cfg.outdir]
    if cfg.gsasii_path:
        cmd += ["--gsasii-path", cfg.gsasii_path]
    if cfg.refine_atoms:
        cmd.append("--refine-atoms")
    cmd += ["--max-cell-drift", str(cfg.max_cell_drift)]
    if cfg.tmin is not None:
        cmd += ["--tmin", str(cfg.tmin), "--tmax", str(cfg.tmax)]
    if cfg.dry_run:
        cmd.append("--dry-run")
    cmd.append("--emit-events")
    return cmd


def parse_event_line(line: str):
    """
    Parses one line of the child process's stdout. Returns a dict with
    event="log" for a plain human-readable line (the common case — most of
    the output is the CLI's normal log text), or the decoded event dict for
    a --emit-events JSON line (event in {"plan", "stage_start",
    "stage_result", "done"}).
    """
    line = line.rstrip("\n")
    stripped = line.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            event = json.loads(stripped)
            if isinstance(event, dict) and "event" in event:
                return event
        except json.JSONDecodeError:
            pass
    return {"event": "log", "text": line}


def load_config() -> dict:
    """Loads persisted GUI defaults (last-used paths). Never raises — a
    missing/corrupt config file just means empty defaults."""
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_config(values: dict) -> None:
    """Persists GUI defaults for next launch. Best-effort — a failure to
    save (e.g. read-only home directory) should not interrupt the user."""
    try:
        CONFIG_PATH.write_text(json.dumps(values, indent=2))
    except OSError:
        pass


def auto_outdir(pattern_path: str, now: Optional[datetime] = None) -> str:
    """A fresh, timestamped default output folder next to the pattern file
    itself, e.g. Data/FeF3/r3c 20.txt -> Data/FeF3/results_20260828_143022.
    Every call with a distinct `now` gets its own folder, so re-running a
    refinement never overwrites a previous run's checkpoints/results the
    way a fixed name (e.g. one derived only from the pattern's filename)
    would. `now` is a parameter (defaulting to datetime.now()) purely so
    this stays testable without mocking the clock."""
    if not pattern_path:
        return ""
    now = now or datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    return str(Path(pattern_path).resolve().parent / f"results_{stamp}")


def find_example_datasets(script_dir: str) -> dict:
    """
    Looks for a Data/ folder next to the GUI script and returns any example
    datasets found, keyed by subfolder name — powers the "Load example"
    buttons. Returns {} if there's no Data/ folder (e.g. the GUI script was
    copied somewhere else) rather than raising.
    """
    data_dir = Path(script_dir) / "Data"
    if not data_dir.is_dir():
        return {}

    examples = {}
    for sub in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        patterns = [f for f in sub.iterdir()
                    if f.suffix.lower() in (".xy", ".dat", ".fxye", ".chi", ".txt")]
        instprms = [f for f in sub.iterdir() if f.suffix.lower() in (".prm", ".instprm")]
        cifs = [f for f in sub.iterdir() if f.suffix.lower() == ".cif"]
        if patterns and instprms and cifs:
            examples[sub.name] = {
                "pattern": str(patterns[0]),
                "instprm": str(instprms[0]),
                "cifs": [str(c) for c in cifs],
            }
    return examples


# ---------------------------------------------------------------------------
# Reading a run's output back — pattern_raw.csv / fit_final.csv / summary.json
# as written by gsas2_auto_refine.py. Kept here (not in gsas2_auto_refine.py)
# since this is GUI-side *consumption* of plain files, not GSAS-II access.
# ---------------------------------------------------------------------------

def read_xy_csv(path: str) -> dict:
    """
    Reads a CSV written by gsas2_auto_refine.py's export_histogram_csv
    (pattern_raw.csv or fit_final.csv) into {column_name: [floats...]}.
    Returns {} if the file doesn't exist yet (e.g. no run has completed
    that far) rather than raising — callers should treat that as "nothing
    to plot yet", not an error.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open(newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return {}
        columns = {name: [] for name in header}
        for row in reader:
            for name, value in zip(header, row):
                try:
                    columns[name].append(float(value))
                except ValueError:
                    columns[name].append(float("nan"))
    return columns


def read_summary(outdir: str) -> Optional[dict]:
    """Reads summary.json from a run's output folder. None if it doesn't
    exist yet or is unreadable — never raises."""
    p = Path(outdir) / "summary.json"
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


CELL_DISPLAY_FIELDS = [
    ("length_a", "a"), ("length_b", "b"), ("length_c", "c"),
    ("angle_alpha", "alpha"), ("angle_beta", "beta"), ("angle_gamma", "gamma"),
    ("volume", "volume"),
]


def format_cell_rows(summary: dict) -> list:
    """
    Turns summary['cells']/summary['cell_esds'] (as written by
    gsas2_auto_refine.py's get_phase_cells()) into a flat list of display
    rows: [{"phase": name, "param": "a", "value": 5.4668, "esd": 0.0003}, ...]
    ready for a Treeview. Returns [] if summary has no cell data (e.g. a
    dry run, or a run that failed before setup completed).
    """
    cells = (summary or {}).get("cells") or {}
    esds = (summary or {}).get("cell_esds") or {}
    rows = []
    for phase_name, cell in cells.items():
        phase_esds = esds.get(phase_name, {})
        for key, label in CELL_DISPLAY_FIELDS:
            if key in cell:
                rows.append({
                    "phase": phase_name,
                    "param": label,
                    "value": cell[key],
                    "esd": phase_esds.get(key),
                })
    return rows
