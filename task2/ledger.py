"""Task 2c impact ledger.

Takes the FIXED sleep_pipeline.py, re-introduces ONE defect (or a stated combination) by reversing that
defect's edit in the source text, runs it with several seeds, and reports the change against the fixed
pipeline on the SAME seed (a seed also fixes the subject split, so the comparison is paired).

    python ledger.py --all --jobs 4          # run every missing (variant, seed) then write the table
    python ledger.py --table                 # rebuild the table from ledger_results/*.json
    python ledger.py --run D3 42             # one run (used by --all in a subprocess)

Needs MNE_DATA to point at the Sleep-EDF cache if it is not in MNE's default location.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sleep_pipeline.py")
OUT = os.path.join(HERE, "ledger_results")
SEEDS = [42, 43, 44]

# --------------------------------------------------------------------------
# Each defect = the text of the fixed script -> the text of the original (defective) script.
# Reversing the edit in the source keeps every other line identical to the fixed pipeline.
# --------------------------------------------------------------------------
SWAPS = {
    # D1: recording-level instead of subject-level split (same 20/20/60 proportions, same three-way scheme)
    "D1": [("""    subj = sorted({subject_of(f) for f in files})
    random.Random(SEED).shuffle(subj)
    n_hold = max(1, int(0.2 * len(subj)))
    test_s, val_s, train_s = set(subj[:n_hold]), set(subj[n_hold:2 * n_hold]), set(subj[2 * n_hold:])
    test_files = [f for f in files if subject_of(f) in test_s]
    val_files = [f for f in files if subject_of(f) in val_s]
    train_files = [f for f in files if subject_of(f) in train_s]
""", """    random.Random(SEED).shuffle(files)
    n_hold = max(1, int(0.2 * len(files)))
    test_files, val_files, train_files = files[:n_hold], files[n_hold:2 * n_hold], files[2 * n_hold:]
    test_s, val_s, train_s = ({subject_of(f) for f in g} for g in (test_files, val_files, train_files))
""")],
    # D2: best epoch chosen on the test subjects
    "D2": [("y_true, y_pred = predict(model, val_ds)", "y_true, y_pred = predict(model, test_ds)")],
    # D3: attention across the batch
    "D3": [("dropout=0.1,\n                                           batch_first=True)", "dropout=0.1)")],
    # D4: positional encoding after the transformer
    "D4": [("z = self.transformer(self.pos(z))", "z = self.pos(self.transformer(z))")],
    # D5: label = last epoch while the head reads the centre
    "D5": [("i + SEQ_LEN // 2]", "i + SEQ_LEN - 1]")],
    # D6: threshold in uV compared against volts, all channels (nothing is ever rejected)
    "D6": [("keep = (np.ptp(X[:, eeg], axis=-1) * 1e6).max(axis=1) < REJECT_PTP",
            "keep = np.ptp(X, axis=-1).max(axis=1) < REJECT_PTP")],
    # D6N: the 'obvious' repair of D6 (convert units, keep all channels): rejects the EOG channel's normal swings
    "D6N": [("keep = (np.ptp(X[:, eeg], axis=-1) * 1e6).max(axis=1) < REJECT_PTP",
             "keep = (np.ptp(X, axis=-1) * 1e6).max(axis=1) < REJECT_PTP")],
    # D7: rejected epochs deleted from the sequence instead of masked
    "D7": [("    # rejected epochs stay in the array (so neighbours stay 30 s apart) and are masked out of the windows instead\n",
            "    X, y = X[keep], y[keep]\n    keep = np.ones(len(y), dtype=bool)\n")],
    # D8: per-epoch z-scoring
    "D8": [("""    mu = X[keep].mean(axis=(0, 2), keepdims=True)
    sd = X[keep].std(axis=(0, 2), keepdims=True)
""", """    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
""")],
    # D9: unscored / unknown epochs labelled Wake
    "D9": [("y = np.full(n_ep, -1, dtype=np.int64)", "y = np.zeros(n_ep, dtype=np.int64)"),
           ("STAGE_MAP.get(desc, -1)", "STAGE_MAP.get(desc, 0)")],
    # D10: no cropping of the lights-on Wake
    "D10": [("WAKE_MARGIN = 60 ", "WAKE_MARGIN = 10**9 ")],
}

# variant name -> (defects re-introduced, what it shows)
VARIANTS = {
    "FIXED": ([], "fully fixed pipeline (reference)"),
    "D1": (["D1"], "recording-level split"),
    "D2": (["D2"], "epoch selected on test"),
    "D3": (["D3"], "attention across the batch"),
    "D4": (["D4"], "positional encoding after attention"),
    "D5": (["D5"], "label = last epoch, head = centre"),
    "D6": (["D6"], "REJECT_PTP in uV vs volts (no rejection)"),
    "D7": (["D7"], "rejected epochs deleted, not masked"),
    "D8": (["D8"], "per-epoch z-scoring"),
    "D9": (["D9"], "unscored epochs labelled Wake"),
    "D10": (["D10"], "no 30-min crop of Wake"),
    # interactions
    "D3+D4": (["D3", "D4"], "interaction: D4 should vanish while D3 is present"),
    "D1+D2": (["D1", "D2"], "interaction: both inflate the test score"),
    "D9+D10": (["D9", "D10"], "interaction: unscored tail + uncropped Wake"),
    "D6N": (["D6N"], "naive unit fix of D6 (all channels), masking kept"),
    "D6N+D7": (["D6N", "D7"], "naive unit fix of D6 with the original deletion"),
    "ALL": ([f"D{k}" for k in range(1, 11)], "all ten defects (approximates the given script; val set kept)"),
}


def build_source(variant, seed):
    text = open(SRC).read()
    for key in VARIANTS[variant][0]:
        for old, new in SWAPS[key]:
            assert old in text, f"{key}: pattern not found in sleep_pipeline.py:\n{old}"
            text = text.replace(old, new)
    assert "SEED = 42" in text
    return text.replace("SEED = 42", f"SEED = {seed}", 1)


def run_one(variant, seed):
    import torch
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
    torch.set_num_threads(int(os.environ.get("LEDGER_THREADS", "3")))
    ns = {"__name__": "variant", "__file__": SRC}
    exec(compile(build_source(variant, seed), f"<{variant}:{seed}>", "exec"), ns)
    t0 = time.time()
    y_true, y_pred = ns["main"]()
    res = dict(variant=variant, seed=seed, n_test_windows=int(len(y_true)),
               accuracy=float(accuracy_score(y_true, y_pred)),
               macro_f1=float(f1_score(y_true, y_pred, average="macro")),
               kappa=float(cohen_kappa_score(y_true, y_pred)),
               per_class_f1=f1_score(y_true, y_pred, average=None, labels=range(5)).tolist(),
               confusion=confusion_matrix(y_true, y_pred, labels=range(5)).tolist(),
               minutes=round((time.time() - t0) / 60, 1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{variant}_{seed}.json"), "w") as f:
        json.dump(res, f)
    print(json.dumps({k: res[k] for k in ("variant", "seed", "accuracy", "macro_f1", "kappa")}))


def run_all(jobs, seeds, variants):
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    todo = [(v, s) for s in seeds for v in variants if not os.path.exists(os.path.join(OUT, f"{v}_{s}.json"))]
    print(f"{len(todo)} runs to do, {jobs} in parallel", flush=True)

    def go(job):
        v, s = job
        log = os.path.join(OUT, "logs", f"{v}_{s}.log")
        with open(log, "w") as lf:
            r = subprocess.run([sys.executable, __file__, "--run", v, str(s)], stdout=lf, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONUNBUFFERED": "1", "LEDGER_THREADS": str(max(1, 12 // jobs))})
        print(f"done {v} seed {s} (exit {r.returncode})", flush=True)

    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(go, todo))


# Predicted direction: copied from PREDICTIONS.md (Task 2 / 2c), which was committed before any ledger run.
PREDICTED = {
    "D1": "up +0.01..+0.05, sign unstable", "D2": "up +0.005..+0.02", "D3": "down -0.03..-0.10",
    "D4": "down 0..-0.03", "D5": "down -0.01..-0.04", "D6": "about 0", "D7": "about 0",
    "D8": "down -0.01..-0.04 (N3 most)", "D9": "down 0..-0.01", "D10": "acc up +0.08..+0.14, F1 about 0",
    "D3+D4": "= D3 (D4 vanishes)", "D1+D2": "up, more than either", "D9+D10": "D9 effect grows",
    "D6N": "down -0.03..-0.10", "D6N+D7": "worse than D6N", "ALL": "acc same/up, F1 -0.08..-0.20, kappa -0.05..-0.15",
}
# Mechanism, one line each; the evidence for each is in DEFECTS.md.
MECHANISM = {
    "D1": "test subjects' other night in training; also changes which subjects are tested",
    "D2": "reported score = max over 12 noisy test evaluations",
    "D3": "attention mixes windows of the batch; no epoch context",
    "D4": "attention is order-blind; neighbours are an unordered set",
    "D5": "target is the last epoch, the head reads the centre one",
    "D6": "no rejection at all; EEG rarely exceeds 500 uV, so ~nothing changes",
    "D7": "deleted epochs splice windows across time gaps; nothing deleted while D6 is present",
    "D8": "per-epoch scaling erases the >75 uV slow-wave amplitude",
    "D9": "940 unscored/movement epochs become Wake targets",
    "D10": "~16 h/recording of lights-on Wake; easy class dominates acc and kappa",
    "D3+D4": "with batch attention there is no epoch order to lose",
    "D1+D2": "two optimistic biases, not additive here",
    "D9+D10": "mislabelled epochs drown in the uncropped Wake",
    "D6N": "units fixed on all channels: EOG swings reject 54% of epochs; masking then drops most windows",
    "D6N+D7": "deletion keeps the data (spliced) where masking drops whole windows",
    "ALL": "given-script approximation: acc inflated by Wake, sleep stages worse",
}


def table():
    sys.stdout.reconfigure(encoding="utf-8")   # the table prints delta and plus-minus signs; a Windows pipe is cp1252
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    by = {(r["variant"], r["seed"]): r for r in rows}
    with open(os.path.join(HERE, "ledger_runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "accuracy", "macro_f1", "kappa", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM", "n_test_windows"])
        for r in sorted(rows, key=lambda r: (list(VARIANTS).index(r["variant"]), r["seed"])):
            w.writerow([r["variant"], r["seed"], *(f"{r[k]:.4f}" for k in ("accuracy", "macro_f1", "kappa")),
                        *(f"{x:.4f}" for x in r["per_class_f1"]), r["n_test_windows"]])
    seeds = sorted({r["seed"] for r in rows})
    ms = lambda a: f"{np.mean(a):+.3f} ± {np.std(a, ddof=1) if len(a) > 1 else float('nan'):.3f}"
    lines = ["| variant | what | n seeds | Δ accuracy | Δ macro-F1 | Δ kappa | Δ F1 N1 | Δ F1 N3 | Δ F1 REM | predicted (macro-F1 unless stated) | mechanism |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    ref = {s: by[("FIXED", s)] for s in seeds if ("FIXED", s) in by}
    if ref:
        a = lambda k: [ref[s][k] for s in ref]
        pc = lambda i: [ref[s]["per_class_f1"][i] for s in ref]
        lines.append(f"| FIXED (absolute) | reference | {len(ref)} | {np.mean(a('accuracy')):.3f} ± {np.std(a('accuracy'), ddof=1):.3f} | "
                     f"{np.mean(a('macro_f1')):.3f} ± {np.std(a('macro_f1'), ddof=1):.3f} | {np.mean(a('kappa')):.3f} ± {np.std(a('kappa'), ddof=1):.3f} | "
                     f"{np.mean(pc(1)):.3f} | {np.mean(pc(3)):.3f} | {np.mean(pc(4)):.3f} |")
    body = []
    for v, (_, what) in VARIANTS.items():
        if v == "FIXED":
            continue
        ss = [s for s in seeds if (v, s) in by and s in ref]
        if not ss:
            continue
        d = lambda k: [by[(v, s)][k] - ref[s][k] for s in ss]
        dp = lambda i: [by[(v, s)]["per_class_f1"][i] - ref[s]["per_class_f1"][i] for s in ss]
        body.append((abs(np.mean(d("macro_f1"))),
                     f"| {v} | {what} | {len(ss)} | {ms(d('accuracy'))} | {ms(d('macro_f1'))} | {ms(d('kappa'))} | "
                     f"{ms(dp(1))} | {ms(dp(3))} | {ms(dp(4))} | {PREDICTED.get(v, '')} | {MECHANISM.get(v, '')} |"))
    lines += [b for _, b in sorted(body, key=lambda x: -x[0])]
    open(os.path.join(HERE, "ledger_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, metavar=("VARIANT", "SEED"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    if a.run:
        run_one(a.run[0], int(a.run[1]))
    if a.all:
        run_all(a.jobs, a.seeds, a.variants)
    if a.all or a.table:
        table()
