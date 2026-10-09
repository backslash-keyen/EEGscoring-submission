"""Shared helpers for the Task 2 ask scripts (one script per item of the brief, 2a-2e).

The printing helpers (header, section, say, table, check, save, finish, prediction_order, decision) are the Task 1
ones in task1/asks/_common.py, so both tasks answer in the same format. This file adds what is specific to Task 2:
where the Sleep-EDF files are, and loaders for the ledger runs that task2/ledger.py wrote.
"""
import json, os, sys
from pathlib import Path

T2 = Path(__file__).resolve().parents[1]
ROOT = T2.parent
sys.path.insert(0, str(ROOT / "task1" / "asks"))
from _common import *                       # noqa: E402,F401,F403  (shared printing helpers, np, pd, plt)
from _common import _git as git, _md_block as md_block   # noqa: E402  (underscored there, so * skips them)

sys.path.insert(0, str(T2))
# same default as main.py: the Sleep-EDF cache lives under data/, not in MNE's ~/mne_data
os.environ.setdefault("MNE_DATA", str(ROOT / "data" / "mne_sleep"))
STAGES = ["W", "N1", "N2", "N3", "REM"]
SEEDS = [42, 43, 44]
METRICS = ["accuracy", "macro_f1", "kappa"]


def has_sleep_data():
    """True when the 29 Sleep-EDF recordings are on disk. Without them the scripts print the answer from the
    committed outputs and skip the live part (fetching them takes hours at PhysioNet's throttled rate)."""
    d = Path(os.environ["MNE_DATA"]) / "physionet-sleep-data"
    return d.is_dir() and len(list(d.glob("SC4*-PSG.edf"))) >= 29


def no_sleep_data_note():
    print("  [live part skipped: Sleep-EDF files not found under $MNE_DATA (default data/mne_sleep). "
          "Run `python task2/prefetch.py` or `python main.py --only T2` once. The answer below comes from the committed outputs.]")


def ledger_runs():
    """Every ledger run as {(variant, seed): result dict}, from task2/ledger_results/*.json (written by task2/ledger.py)."""
    out = {}
    for p in sorted((T2 / "ledger_results").glob("*.json")):
        r = json.loads(p.read_text())
        out[(r["variant"], r["seed"])] = r
    return out


def text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def report_block(start, stop):
    """A passage of report/REPORT.md, quoted rather than rewritten so the ask script and the report cannot disagree."""
    return md_block("report/REPORT.md", start, stop)
