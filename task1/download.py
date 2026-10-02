"""Fetch EEGBCI motor-imagery runs for subjects 70-109 into a path outside git."""
from pathlib import Path
from mne.datasets import eegbci

# Data lives next to the repo but is git-ignored (see .gitignore); keeps the repo small.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SUBJECTS = list(range(70, 110))  # 70-109 inclusive, per the brief
# Runs 4, 8, 12 = imagery, left vs right fist. Source: mne.datasets.eegbci.load_data
# docstring (verified 2026-10-01). Runs 6/10/14 are hands-vs-feet imagery and must NOT be
# used here: their T1/T2 codes mean both-fists/both-feet, not left/right.
RUNS = [4, 8, 12]

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for s in SUBJECTS:
        eegbci.load_data(s, RUNS, path=DATA_DIR, update_path=False)
        print(f"subject {s} ok", flush=True)
