"""Shared helpers for the A3 walkthrough: paths, the project modules, and a figure saver. Run scripts from this folder."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
if os.environ.get("HEADLESS"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK1))
import data, confound as C            # the real pipeline: loader (data.py) and decoders (confound.py)

OUT = data.ROOT / "outputs"
FIG = OUT / "a3_walkthrough"
FIG.mkdir(parents=True, exist_ok=True)
SUBJECT = 72                           # the subject shown in the single-subject demos; change it to look at anyone else


def save(fig, name):
    fig.savefig(FIG / name, dpi=110, bbox_inches="tight")
    return fig
