"""Shared helpers for the one-script-per-ask files in this folder.

Each ask script answers one item of the brief. It recomputes what is cheap from the real data with the same pipeline
functions main.py uses (task1/*.py), and reads what is expensive (the 156 B2 training runs) from the files main.py wrote.
Nothing here re-implements an analysis, so an ask script can never disagree with the pipeline it explains.
"""
import importlib, os, re, subprocess, sys, textwrap, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib

# The Windows console code page cannot print every character in DECISIONS.md; replacing beats crashing mid-answer.
sys.stdout.reconfigure(errors="replace")
warnings.filterwarnings("ignore")   # library chatter (welch segment length, sklearn convergence) is not part of an answer
# Resolved before any pipeline module runs matplotlib.use("Agg"), so figures can still open in a window afterwards.
_BACKEND = matplotlib.get_backend()
import matplotlib.pyplot as plt

TASK1 = Path(__file__).resolve().parents[1]
ROOT = TASK1.parent
OUT = ROOT / "outputs"
FIG = OUT / "asks"
sys.path.insert(0, str(TASK1))
pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_colwidth", 80)


def pipeline(*names):
    """Import task1 modules (data, erd, confound, ...). Some of them force the Agg backend on import; undo that so
    plt.show() still opens windows when the script is run from VS Code."""
    mods = [importlib.import_module(n) for n in names]
    if matplotlib.get_backend() != _BACKEND:
        plt.switch_backend(_BACKEND)
    return mods[0] if len(mods) == 1 else mods


def has_data():
    """True when the cached epochs exist (python main.py --only A1 A2 creates them). Without data an ask script still
    prints the answer from the committed outputs and says it skipped the live recomputation."""
    c = ROOT / "data" / "cache_causal"
    return c.is_dir() and len(list(c.glob("S*.npz"))) == 40


def no_data_note():
    print("  [live recomputation skipped: data/cache_causal not found. Run `python main.py --only A1 A2 A3` once,"
          " then re-run this script. The answer below comes from the committed outputs.]")


def header(code, ask, how):
    bar = "=" * 100
    print(bar)
    print(f"{code}")
    print(textwrap.fill("ASK: " + ask, 100, subsequent_indent="     "))
    print(textwrap.fill("HOW: " + how, 100, subsequent_indent="     "))
    print(bar)


def section(title):
    print(f"\n--- {title} " + "-" * max(0, 95 - len(title)))


def say(text, indent=2):
    for para in str(text).strip().split("\n"):
        print(textwrap.fill(para, 100, initial_indent=" " * indent, subsequent_indent=" " * indent))


def table(df, floatfmt="{:.3f}"):
    print(df.to_string(index=False, float_format=lambda v: floatfmt.format(v)))


def csv(name):
    """A table main.py wrote: outputs/<name>, or the repo root for audit.csv (where the brief asks for it)."""
    return pd.read_csv(OUT / name if (OUT / name).exists() else ROOT / name)


def check(label, live, saved, tol=1e-6):
    """Prints a live number next to the number main.py saved, so a reader sees the committed output is reproduced."""
    ok = abs(float(live) - float(saved)) <= tol
    print(f"  check {label}: live {float(live):.4f} | saved {float(saved):.4f} | {'MATCH' if ok else 'DIFFERENT'}")
    return ok


def _md_block(path, start_pat, stop_pat):
    txt = (ROOT / path).read_text(encoding="utf-8")
    m = re.search(start_pat, txt, flags=re.M)
    if not m:
        return f"(section not found in {path})"
    rest = txt[m.start():]
    n = re.search(stop_pat, rest[1:], flags=re.M)
    return rest[: n.start() + 1 if n else len(rest)].strip()


def decision(dnum):
    """Text of one DECISIONS.md entry (e.g. 'D4'): the justification lives there once, ask scripts quote it."""
    return _md_block("DECISIONS.md", rf"^## {dnum} ", r"^## ")


def print_decision(dnum):
    section(f"DECISIONS.md {dnum}")
    say(decision(dnum))


def prediction(heading_regex):
    return _md_block("PREDICTIONS.md", heading_regex, r"^## ")


def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def prediction_order(pred_commit, result_paths):
    """The brief requires each prediction to be committed before the commit with its result. Checked from git history:
    the first commit that added any result file must have the prediction commit among its ancestors."""
    section("Was the prediction committed before the result? (checked from git history, not asserted)")
    code, pred = _git("log", "-1", "--format=%h %ad %s", "--date=format:%Y-%m-%d %H:%M", pred_commit)
    if code or not pred:
        print("  git history not available here (no .git or commit missing); see PREDICTIONS.md")
        return
    _, out = _git("log", "--diff-filter=A", "--reverse", "--format=%h %ad %s", "--date=format:%Y-%m-%d %H:%M", "--", *result_paths)
    first = out.splitlines()[0] if out else None
    print(f"  prediction: {pred}")
    print(f"  first result commit: {first}")
    if first:
        ok = _git("merge-base", "--is-ancestor", pred_commit, first.split()[0])[0] == 0
        print(f"  prediction commit is an ancestor of the first result commit: {'YES' if ok else 'NO'}")


def retrain(exp, model, fold, seed, n_train, threads=3):
    """Train one B2 run again with the grid's exact settings (task1/b2_run.py fit_kwargs) and compare its test accuracy
    with the saved run. Nothing is written to outputs/. Takes several minutes on a CPU. threads=3 is what the grid used
    (main.py --threads): float summation order depends on the thread count, so only the same count reproduces exactly."""
    import torch
    b2_run, train = pipeline("b2_run", "train")
    j = (exp, model, fold, seed, n_train)
    torch.set_num_threads(threads)
    section(f"Re-training {b2_run.job_id(j)} now (several minutes on a CPU) ...")
    _, _, res, _ = train.fit(model, **b2_run.fit_kwargs(j))
    R = csv("partb/b2_all_runs.csv").set_index("job")
    saved = R.loc[b2_run.job_id(j)]
    print(f"  best epoch {res['best_epoch']} (saved {saved.best_epoch}), {res['sec']:.0f} s")
    check("test accuracy", res["test_acc"], saved.test_acc)
    check("validation loss", res["val_loss"], saved.val_loss)
    return res


def show_png(name, title=None):
    """Open a figure that main.py already saved (outputs/...)."""
    p = OUT / name
    if not p.exists():
        print(f"  (figure {name} not found; run main.py)")
        return
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(plt.imread(p))
    ax.axis("off")
    ax.set_title(title or name, fontsize=9)
    print(f"  figure: outputs/{name}")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=110, bbox_inches="tight")
    print(f"  figure saved: outputs/asks/{name}")


def finish():
    """Open every figure in a window (VS Code / terminal). Under main.py MPLBACKEND=Agg, so this returns at once."""
    if plt.get_fignums() and matplotlib.get_backend().lower() != "agg":
        print("\n(close the figure windows to end the script)")
        plt.show()
