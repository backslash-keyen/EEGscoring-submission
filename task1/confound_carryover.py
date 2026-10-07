"""A3 follow-up (POST-HOC, added after the first pass showed label order is not random; DECISIONS D14).

Separates "decoder reads the current cue" from "decoder reads carry-over of the previous trial and exploits that consecutive
labels tend to alternate". Carry-over decoders are right on alternating trials and WRONG on repeat trials; cue decoders are right on both.
Also trains with (current, previous) cell-balanced sample weights so the shortcut cannot be learned at all.

python task1/confound_carryover.py [--zero-phase]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
import data, confound as C

CAUSAL = "--zero-phase" not in sys.argv
TAG = "" if CAUSAL else "firstpass_zerophase_"
DEC = ["F1", "F1_pre", "F2", "O1", "O1_pre", "O2", "G1", "N1", "N2", "N2_pre", "M1", "A1"]


def prev_label(y, run):
    p = np.full(len(y), -1)
    for i in range(1, len(y)):
        if run[i] == run[i - 1]:
            p[i] = y[i - 1]
    return p   # -1 = first trial of a run


def cell_weights(y, prev):
    """Equal total weight for each (current, previous) cell among trials with a known previous label; 0 for the rest."""
    w = np.zeros(len(y))
    for c in (0, 1):
        for p in (0, 1):
            m = (y == c) & (prev == p)
            if m.any():
                w[m] = 1.0 / m.sum()
    return w


def fit_predict_w(Xtr, ytr, Xte, w=None):
    keep = np.ones(len(ytr), bool) if w is None else w > 0
    sc = StandardScaler().fit(Xtr[keep])
    clf = LogisticRegression(C=C.C_REG, max_iter=500).fit(sc.transform(Xtr[keep]), ytr[keep],
                                                           sample_weight=None if w is None else w[keep] * keep.sum() / w[keep].sum())
    return clf.predict(sc.transform(Xte))


def within_subject(s, causal):
    D = data.load_subject(s, causal)
    names, X, y, run = list(D["ch_names"]), D["X"].astype(np.float64), D["y"], D["run"]
    prev = prev_label(y, run)
    specs = C.decoder_specs(names)
    rows = []
    for k in DEC:
        F = specs[k][2](X)
        for scheme in ("plain", "cell_balanced", "readout_prev"):
            pred = np.full(len(y), -1)
            for r in np.unique(run):
                te = run == r
                if scheme == "plain":
                    pred[te] = fit_predict_w(F[~te], y[~te], F[te])
                elif scheme == "cell_balanced":
                    pred[te] = fit_predict_w(F[~te], y[~te], F[te], cell_weights(y[~te], prev[~te]))
                else:   # can the features tell what the PREVIOUS label was? (carry-over present at all?)
                    ok = prev[~te] >= 0
                    pred[te] = fit_predict_w(F[~te][ok], prev[~te][ok], F[te])
            tgt = prev if scheme == "readout_prev" else y
            rows.append(dict(subject=s, decoder=k, scheme=scheme, regime="within",
                             rep_n=int(((y == prev) & (prev >= 0)).sum()), alt_n=int(((y != prev) & (prev >= 0)).sum()),
                             rep_c=int(((pred == tgt) & (y == prev) & (prev >= 0)).sum()),
                             alt_c=int(((pred == tgt) & (y != prev) & (prev >= 0)).sum()),
                             ok_c=int(((pred == tgt) & (prev >= 0)).sum()), ok_n=int((prev >= 0).sum())))
    return rows


def pooled(causal):
    subj = data.SUBJECTS
    folds_of = {s: i % C.N_FOLDS for i, s in enumerate(np.random.default_rng(0).permutation(subj))}
    F = {k: [] for k in DEC}; Y, P, S = [], [], []
    for s in subj:
        D = data.load_subject(s, causal)
        names, X, y, run = list(D["ch_names"]), D["X"].astype(np.float64), D["y"], D["run"]
        specs = C.decoder_specs(names)
        for k in DEC:
            F[k].append(specs[k][2](X))
        Y.append(y); P.append(prev_label(y, run)); S.append(np.full(len(y), s))
    F = {k: np.concatenate(v) for k, v in F.items()}
    Y, P, S = np.concatenate(Y), np.concatenate(P), np.concatenate(S)
    fold = np.array([folds_of[s] for s in S])
    rows = []
    for k in DEC:
        for scheme in ("plain", "cell_balanced", "readout_prev"):
            pred = np.full(len(Y), -1)
            for f in range(C.N_FOLDS):
                te = fold == f
                if scheme == "plain":
                    pred[te] = fit_predict_w(F[k][~te], Y[~te], F[k][te])
                elif scheme == "cell_balanced":
                    pred[te] = fit_predict_w(F[k][~te], Y[~te], F[k][te], cell_weights(Y[~te], P[~te]))
                else:
                    ok = P[~te] >= 0
                    pred[te] = fit_predict_w(F[k][~te][ok], P[~te][ok], F[k][te])
            tgt = P if scheme == "readout_prev" else Y
            rows.append(dict(subject=-1, decoder=k, scheme=scheme, regime="pooled_xs",
                             rep_n=int(((Y == P) & (P >= 0)).sum()), alt_n=int(((Y != P) & (P >= 0)).sum()),
                             rep_c=int(((pred == tgt) & (Y == P) & (P >= 0)).sum()),
                             alt_c=int(((pred == tgt) & (Y != P) & (P >= 0)).sum()),
                             ok_c=int(((pred == tgt) & (P >= 0)).sum()), ok_n=int((P >= 0).sum())))
    return rows


def main():
    rows = [r for rr in Parallel(n_jobs=-1)(delayed(within_subject)(s, CAUSAL) for s in data.SUBJECTS) for r in rr]
    rows += pooled(CAUSAL)
    df = pd.DataFrame(rows)
    df.to_csv(data.ROOT / "outputs" / f"{TAG}a3b_carryover_raw.csv", index=False)
    g = df.groupby(["regime", "scheme", "decoder"])[["rep_n", "alt_n", "rep_c", "alt_c", "ok_c", "ok_n"]].sum().reset_index()
    g["acc_all"] = g.ok_c / g.ok_n
    g["acc_repeat"] = g.rep_c / g.rep_n
    g["acc_alternate"] = g.alt_c / g.alt_n
    g["acc_balanced"] = (g.acc_repeat + g.acc_alternate) / 2   # equals the accuracy of a (cur x prev)-balanced evaluation
    se = 0.5 * np.sqrt(0.25 / g.rep_n + 0.25 / g.alt_n)        # SE of that mean under the null of no information
    g["z_balanced"] = (g.acc_balanced - 0.5) / se
    g["p_balanced"] = norm.sf(g.z_balanced)
    g.to_csv(data.ROOT / "outputs" / f"{TAG}a3b_carryover.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["decoder", "acc_all", "acc_repeat", "acc_alternate", "acc_balanced", "z_balanced", "rep_n", "alt_n"]
    for reg in ("pooled_xs", "within"):
        for sc in ("plain", "cell_balanced", "readout_prev"):
            print(f"\n== {reg} | {sc} ==")
            print(g[(g.regime == reg) & (g.scheme == sc)][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
