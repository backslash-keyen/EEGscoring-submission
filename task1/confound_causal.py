"""A3 follow-up 2 (POST-HOC, DECISIONS D14): does the zero-phase 1-40 Hz filter leak the post-cue response into the pre-cue window?
Compares the pre-cue "null" twins and a bin-by-bin accuracy time course on the zero-phase cache vs a minimum-phase (causal) cache.

python task1/confound_causal.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import data, confound as C

OUT = data.ROOT / "outputs"
DEC = ["F1", "F1_pre", "F2", "O1", "O1_pre", "N2", "N2_pre", "M1", "M1_pre", "A1", "A1_pre"]
BINS = np.arange(-1.5, 1.4, 0.1)


def load(causal):
    subj = data.SUBJECTS
    Ds = {s: data.load_subject(s, causal) for s in subj}
    folds_of = {s: i % C.N_FOLDS for i, s in enumerate(np.random.default_rng(0).permutation(subj))}
    names = list(Ds[subj[0]]["ch_names"])
    specs = C.decoder_specs(names)
    Y = np.concatenate([Ds[s]["y"] for s in subj])
    fold = np.concatenate([np.full(len(Ds[s]["y"]), folds_of[s]) for s in subj])
    return Ds, names, specs, Y, fold


def xs_acc(F, Y, fold):
    pred = np.empty(len(Y), dtype=int)
    for f in range(C.N_FOLDS):
        te = fold == f
        pred[te] = C.fit_predict(F[~te], Y[~te], F[te])
    return (pred == Y).mean()


def main():
    rows, tc = [], []
    for causal in (False, True):
        Ds, names, specs, Y, fold = load(causal)
        Xall = {s: Ds[s]["X"].astype(np.float64) for s in Ds}
        for k in DEC:
            F = np.concatenate([specs[k][2](Xall[s]) for s in Ds])
            rows.append(dict(filter="minimum-phase" if causal else "zero-phase", decoder=k, pooled_xs_acc=xs_acc(F, Y, fold)))
            print(rows[-1], flush=True)
        for nm, chs in (("frontal", C.FRONT), ("occipital", C.OCC)):
            ch = C.idx(names, chs)
            for b in BINS:
                F = np.concatenate([(Xall[s][:, ch][:, :, (C.TIMES >= b) & (C.TIMES < b + 0.1)].mean(2)) * 1e6 for s in Ds])
                tc.append(dict(filter="minimum-phase" if causal else "zero-phase", set=nm, t=round(b + 0.05, 2), acc=xs_acc(F, Y, fold)))
    pd.DataFrame(rows).to_csv(OUT / "a3c_causal_twins.csv", index=False)
    tcd = pd.DataFrame(tc)
    tcd.to_csv(OUT / "a3c_timecourse.csv", index=False)
    thr = C.binom_thr(len(Y))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, nm in zip(axes, ("frontal", "occipital")):
        for flt, col in (("zero-phase", "tab:blue"), ("minimum-phase", "tab:orange")):
            d = tcd[(tcd.set == nm) & (tcd["filter"] == flt)]
            ax.plot(d.t, d.acc, color=col, label=flt)
        ax.axvline(0, color="k", lw=.7); ax.axhline(0.5, color="gray", lw=.7); ax.axhline(thr, color="gray", ls=":", lw=.7)
        ax.set_title(f"{nm} channels, pooled cross-subject, 100 ms bins"); ax.set_xlabel("time from cue (s)")
    axes[0].set_ylabel("accuracy (dotted = binomial 5% threshold)"); axes[0].legend()
    fig.savefig(OUT / "a3c_timecourse.png", dpi=110, bbox_inches="tight")
    print(pd.DataFrame(rows).pivot(index="decoder", columns="filter", values="pooled_xs_acc").round(3))


if __name__ == "__main__":
    main()
