"""A3: confound audit. Restricted-channel / -band / -time decoders for every non-motor route that could separate left from right
trials, each with an exact binomial chance threshold and a label-permutation check (DECISIONS D8-D13).

python task1/confound.py            real labels, causal filter (writes outputs/a3_*.csv, a3_*.png); add --zero-phase for the first pass
python task1/confound.py --smoke    2 subjects, labels shuffled within run, few permutations: tests the code without reading results
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import data

warnings.filterwarnings("ignore")
OUT = data.ROOT / "outputs"
ALPHA = 0.05
C_REG = 0.1            # fixed in advance, never tuned on test accuracy (D10)
N_PERM_SUBJ, N_PERM_POOL = 200, 100
N_FOLDS = 8
BANDS = {"mu": (8, 13), "beta": (13, 30), "g": (30, 40)}
TIMES = np.arange(data.N_T) / data.FS + data.TMIN

FRONT = ["Fp1", "Fpz", "Fp2", "Af7", "Af3", "Afz", "Af4", "Af8", "F5", "F6", "F7", "F8"]
OCC = ["Po7", "Po3", "Poz", "Po4", "Po8", "O1", "Oz", "O2", "Iz"]
TEMP = ["Ft7", "Ft8", "T7", "T8", "T9", "T10", "Tp7", "Tp8"]
SENSORIMOTOR = [c for c in ["Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                            "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6"]]


def idx(names, chans):
    low = [n.lower() for n in names]
    return [low.index(c.lower()) for c in chans]


def mirror_pairs(names, chans):
    """(left, right) index pairs inside `chans`: odd electrode number k pairs with k+1 of the same stem (C3-C4, Fp1-Fp2)."""
    low = {c.lower(): i for c, i in zip(chans, idx(names, chans))}
    pairs = []
    for c, i in low.items():
        stem, num = c.rstrip("0123456789"), c[len(c.rstrip("0123456789")):]
        if num and int(num) % 2 == 1 and f"{stem}{int(num) + 1}" in low:
            pairs.append((i, low[f"{stem}{int(num) + 1}"]))
    return pairs


def feat_td(X, ch, win, base, binw):
    """Per-trial baseline-subtracted bin means of raw channels (D9). X in volts -> microvolts."""
    b = X[:, ch][:, :, (TIMES >= base[0]) & (TIMES < base[1])].mean(2, keepdims=True)
    t0, t1 = win
    out = []
    for a in np.arange(t0, t1 - 1e-9, binw):
        m = (TIMES >= a) & (TIMES < a + binw)
        out.append((X[:, ch][:, :, m].mean(2) - b[:, :, 0]) * 1e6)
    return np.concatenate(out, 1)


def feat_bp(X, names, chans, win, bands):
    """Welch log10 band power, left minus right over mirror pairs (D9)."""
    pr = mirror_pairs(names, chans)
    L, R = [p[0] for p in pr], [p[1] for p in pr]
    m = (TIMES >= win[0]) & (TIMES < win[1])
    f, P = welch(X[:, :, m], fs=data.FS, nperseg=128, noverlap=64, axis=-1)
    out = []
    for b in bands:
        lo, hi = BANDS[b]
        bp = np.log10(P[:, :, (f >= lo) & (f <= hi)].mean(-1) + 1e-30)
        out.append(bp[:, L] - bp[:, R])
    return np.concatenate(out, 1)


def decoder_specs(names):
    """id -> (route, kind, feature function X -> array). Twins (suffix _pre) move the window before the cue (D8)."""
    nm = lambda cs: idx(names, cs)
    nonmotor = [c for c in names if c.lower() not in {s.lower() for s in SENSORIMOTOR}]
    S = {}
    td = lambda ch, win, base, bw: (lambda X: feat_td(X, nm(ch), win, base, bw))
    bp = lambda ch, win, bands: (lambda X: feat_bp(X, names, ch, win, bands))
    ACT, PRE_TD, PRE_BP = (0.5, 4.0), ((-0.5, 0.0), (-1.0, -0.5)), (-1.5, -0.1)
    S["F1"] = ("oculomotor", "headline: frontal TD 0-0.5 s", td(FRONT, (0.0, 0.5), (-0.5, 0.0), 0.1))
    S["F1_pre"] = ("pre-cue twin", "frontal TD pre-cue", td(FRONT, PRE_TD[0], PRE_TD[1], 0.1))
    S["F2"] = ("oculomotor", "frontal TD 0.5-4 s", td(FRONT, ACT, (-0.5, 0.0), 0.5))
    S["O1"] = ("visual", "headline: occipital TD 0-0.5 s", td(OCC, (0.0, 0.5), (-0.5, 0.0), 0.1))
    S["O1_pre"] = ("pre-cue twin", "occipital TD pre-cue", td(OCC, PRE_TD[0], PRE_TD[1], 0.1))
    S["O2"] = ("visual", "occipital mu+beta 0.5-4 s", bp(OCC, ACT, ["mu", "beta"]))
    S["O2_pre"] = ("pre-cue twin", "occipital mu+beta pre-cue", bp(OCC, PRE_BP, ["mu", "beta"]))
    S["G1"] = ("muscle", "temporal 30-40 Hz 0.5-4 s", bp(TEMP, ACT, ["g"]))
    S["G1_pre"] = ("pre-cue twin", "temporal 30-40 Hz pre-cue", bp(TEMP, PRE_BP, ["g"]))
    S["N1"] = ("non-motor (all)", "all but sensorimotor, mu+beta+30-40 0.5-4 s", bp(nonmotor, ACT, ["mu", "beta", "g"]))
    S["N1_pre"] = ("pre-cue twin", "all but sensorimotor pre-cue", bp(nonmotor, PRE_BP, ["mu", "beta", "g"]))
    S["N2"] = ("non-motor (all)", "all but sensorimotor TD 0-0.5 s", td(nonmotor, (0.0, 0.5), (-0.5, 0.0), 0.1))
    S["N2_pre"] = ("pre-cue twin", "all but sensorimotor TD pre-cue", td(nonmotor, PRE_TD[0], PRE_TD[1], 0.1))
    S["M1"] = ("motor (control)", "headline: sensorimotor mu+beta 0.5-4 s", bp(SENSORIMOTOR, ACT, ["mu", "beta"]))
    S["M1_pre"] = ("pre-cue twin", "sensorimotor mu+beta pre-cue", bp(SENSORIMOTOR, PRE_BP, ["mu", "beta"]))
    S["A1"] = ("all channels (control)", "all channels mu+beta 0.5-4 s", bp(list(names), ACT, ["mu", "beta"]))
    S["A1_pre"] = ("pre-cue twin", "all channels mu+beta pre-cue", bp(list(names), PRE_BP, ["mu", "beta"]))
    return S


def order_features(D):
    """R1 position in run + onset time; R2 previous 1-3 labels as +-1 (0 if absent) (D13)."""
    y, run, trial, onset = D["y"], D["run"], D["trial"], D["onset"]
    sgn = np.where(y == 0, -1.0, 1.0)
    prev = np.zeros((len(y), 3))
    for k in range(3):
        for i in range(len(y)):
            j = i - (k + 1)
            if j >= 0 and run[j] == run[i]:
                prev[i, k] = sgn[j]
    return {"R1": np.c_[trial, onset], "R2": prev, "R12": np.c_[trial, onset, prev]}


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C_REG, max_iter=500).fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def loro_acc(F, y, run):
    """Leave-one-run-out accuracy pooled over held-out runs (D10)."""
    correct = 0
    for r in np.unique(run):
        te = run == r
        correct += (fit_predict(F[~te], y[~te], F[te]) == y[te]).sum()
    return correct / len(y)


def perm_labels(y, run, rng):
    yp = y.copy()
    for r in np.unique(run):
        m = np.where(run == r)[0]
        yp[m] = y[rng.permutation(m)]
    return yp


def binom_thr(n, alpha=ALPHA):
    """Smallest accuracy k/n with P(X >= k | n, 0.5) <= alpha (D11)."""
    k = 0
    while binom.sf(k - 1, n, 0.5) > alpha:   # sf(k-1) = P(X >= k)
        k += 1
    return k / n


def subject_job(s, dec_ids, smoke, n_perm, causal):
    D = data.load_subject(s, causal)
    names = list(D["ch_names"])
    X = D["X"].astype(np.float64)
    y, run = D["y"], D["run"]
    if smoke:   # labels shuffled within run: any "result" is noise by construction
        y = perm_labels(y, run, np.random.default_rng(10_000 + s))
    specs = decoder_specs(names)
    feats = {k: specs[k][2](X) for k in dec_ids if k in specs}
    feats.update({k: v for k, v in order_features(D).items() if k in dec_ids})
    n = len(y)
    thr = binom_thr(n)
    rows = []
    for k, F in feats.items():
        acc = loro_acc(F, y, run)
        rng = np.random.default_rng(s * 1000 + dec_ids.index(k))
        null = np.array([loro_acc(F, perm_labels(y, run, rng), run) for _ in range(n_perm)])
        rows.append(dict(subject=s, decoder=k, n=n, acc=acc, thr_binom=thr, above_binom=acc >= thr,
                         p_binom=float(binom.sf(round(acc * n) - 1, n, 0.5)), p_perm=(1 + (null >= acc).sum()) / (1 + n_perm),
                         null_frac_above_thr=float((null >= thr).mean())))
    return rows


def cross_subject(dec_ids, subjects, smoke, n_perm, causal):
    """Pooled cross-subject accuracy with 8 subject-wise folds (D10); fixed fold assignment seed 0."""
    nf = min(N_FOLDS, len(subjects))   # only the smoke run (2 subjects) has fewer subjects than folds
    folds_of = {s: i % nf for i, s in enumerate(np.random.default_rng(0).permutation(subjects))}
    F, Y, R, S = {k: [] for k in dec_ids}, [], [], []
    for s in subjects:
        D = data.load_subject(s, causal)
        names, X, y, run = list(D["ch_names"]), D["X"].astype(np.float64), D["y"], D["run"]
        if smoke:
            y = perm_labels(y, run, np.random.default_rng(10_000 + s))
        specs = decoder_specs(names)
        orf = order_features(D)
        for k in dec_ids:
            F[k].append(specs[k][2](X) if k in specs else orf[k])
        Y.append(y); R.append(run); S.append(np.full(len(y), s))
    F = {k: np.concatenate(v) for k, v in F.items()}
    Y, R, S = np.concatenate(Y), np.concatenate(R), np.concatenate(S)
    fold = np.array([folds_of[s] for s in S])

    def cv_pred(Fk, y):
        pred = np.empty(len(y), dtype=int)
        for f in range(nf):
            te = fold == f
            pred[te] = fit_predict(Fk[~te], y[~te], Fk[te])
        return pred

    def perm_job(k, seed):
        rng = np.random.default_rng(seed)
        yp = Y.copy()
        for s in subjects:   # permute within subject and run so per-subject class counts and runs are preserved
            for r in np.unique(R[S == s]):
                m = np.where((S == s) & (R == r))[0]
                yp[m] = Y[rng.permutation(m)]
        return (cv_pred(F[k], yp) == yp).mean()

    rows, per_sub = [], []
    thr = binom_thr(len(Y))
    for j, k in enumerate(dec_ids):
        pred = cv_pred(F[k], Y)
        acc = (pred == Y).mean()
        null = np.array(Parallel(n_jobs=-1)(delayed(perm_job)(k, 7_000 + 97 * j + b) for b in range(n_perm)))
        rows.append(dict(decoder=k, n=len(Y), acc=acc, thr_binom=thr, p_binom=float(binom.sf(round(acc * len(Y)) - 1, len(Y), 0.5)),
                         p_perm=(1 + (null >= acc).sum()) / (1 + n_perm), null_95=float(np.quantile(null, .95)),
                         null_frac_above_thr=float((null >= thr).mean())))
        for s in subjects:
            m = S == s
            per_sub.append(dict(subject=s, decoder=k, n=int(m.sum()), xs_acc=(pred[m] == Y[m]).mean()))
        print("  cross-subject", k, f"acc={acc:.3f} thr={thr:.3f} p_perm={rows[-1]['p_perm']:.3f}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(per_sub)


def sequence_stats(subjects, smoke):
    """Lag-1 label agreement vs within-run permutation, pooled over subjects; run-majority oracle (D13)."""
    obs, exp, var, orc, n_tot = 0, 0, 0, 0, 0
    rows = []
    rng = np.random.default_rng(5)
    for s in subjects:
        D = data.load_subject(s)
        y, run = D["y"], D["run"]
        if smoke:
            y = perm_labels(y, run, np.random.default_rng(10_000 + s))
        same = sum(((y[run == r][1:] == y[run == r][:-1]).sum()) for r in np.unique(run))
        pairs = sum(max((run == r).sum() - 1, 0) for r in np.unique(run))
        null = np.array([sum(((p[1:] == p[:-1]).sum()) for p in [perm_labels(y, run, rng)[run == r] for r in np.unique(run)]) for _ in range(500)])
        orc_s = sum(max((y[run == r] == 0).sum(), (y[run == r] == 1).sum()) for r in np.unique(run))
        rows.append(dict(subject=s, lag1_same=int(same), pairs=int(pairs), null_mean=null.mean(), null_sd=null.std(), oracle_acc=orc_s / len(y), n=len(y)))
        obs += same; exp += null.mean(); var += null.var(); orc += orc_s; n_tot += len(y)
    df = pd.DataFrame(rows)
    summ = dict(lag1_observed=int(obs), lag1_expected=float(exp), z=float((obs - exp) / np.sqrt(var)), oracle_pooled=orc / n_tot,
                oracle_thr_note="analytic worst case for any run-identity route (D13)")
    return df, summ


def main():
    smoke = "--smoke" in sys.argv
    subjects = data.SUBJECTS[:2] if smoke else data.SUBJECTS
    n_perm, n_pool = (5, 4) if smoke else (N_PERM_SUBJ, N_PERM_POOL)
    names = list(data.load_subject(subjects[0])["ch_names"])
    dec_ids = list(decoder_specs(names)) + ["R1", "R2", "R12"]
    causal = "--zero-phase" not in sys.argv   # default = minimum-phase pre-filter (DECISIONS D14); --zero-phase reproduces the first pass
    tag = ("smoke_" if smoke else "") + ("" if causal else "firstpass_zerophase_")

    res = Parallel(n_jobs=-1)(delayed(subject_job)(s, dec_ids, smoke, n_perm, causal) for s in subjects)
    within = pd.DataFrame([r for rows in res for r in rows])
    within.to_csv(OUT / f"{tag}a3_within_subject.csv", index=False)

    xs, xs_sub = cross_subject(dec_ids, subjects, smoke, n_pool, causal)
    xs.to_csv(OUT / f"{tag}a3_cross_subject.csv", index=False)
    xs_sub.to_csv(OUT / f"{tag}a3_cross_subject_per_subject.csv", index=False)

    sq, summ = sequence_stats(subjects, smoke)
    sq.to_csv(OUT / f"{tag}a3_sequence.csv", index=False)

    # group-level summary per decoder: how many subjects pass vs the 5% expected, one-sided binomial excess test (D11)
    g = within.groupby("decoder").agg(mean_acc=("acc", "mean"), n_above=("above_binom", "sum"), n_subj=("subject", "count"),
                                      perm_false_pos=("null_frac_above_thr", "mean")).reset_index()
    g["expected_by_chance"] = g.n_subj * ALPHA
    g["p_excess"] = [binom.sf(a - 1, n, ALPHA) for a, n in zip(g.n_above, g.n_subj)]
    g = g.merge(xs[["decoder", "acc", "thr_binom", "p_perm"]].rename(columns={"acc": "pooled_xs_acc", "thr_binom": "pooled_thr", "p_perm": "pooled_p_perm"}), on="decoder")
    g.to_csv(OUT / f"{tag}a3_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(g.round(3).to_string(index=False))
    print("sequence:", summ)

    if not smoke:   # per-subject table with the A2 lateralisation index (assignment output)
        lat = pd.read_csv(OUT / "a2_lateralisation.csv")[["subject", "mu_LI", "mu_p", "beta_LI", "beta_p", "label"]]
        wide = within.pivot(index="subject", columns="decoder", values="acc").add_prefix("acc_")
        thr = within.groupby("subject").thr_binom.first().rename("thr_binom")
        pd.concat([lat.set_index("subject"), thr, wide], axis=1).reset_index().to_csv(OUT / f"{tag}a3_per_subject_table.csv", index=False)
        plot_summary(g, within, tag)


def plot_summary(g, within, tag):
    ids = [d for d in ["F1", "F2", "O1", "O2", "G1", "N1", "N2", "R1", "R2", "R12", "M1", "A1"]]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, d in enumerate(ids):
        a = within[within.decoder == d].acc.values
        ax.scatter(np.full(len(a), i) + np.random.default_rng(i).uniform(-.18, .18, len(a)), a, s=14, alpha=.6)
        ax.hlines(a.mean(), i - .3, i + .3, color="k")
        pre = within[within.decoder == d + "_pre"].acc.values
        if len(pre):
            ax.hlines(pre.mean(), i - .3, i + .3, color="r", ls=":")
    ax.axhline(0.5, color="gray", lw=.8)
    ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids)
    ax.set_ylabel("within-subject leave-one-run-out accuracy")
    ax.set_title("A3: one dot per subject, black = mean, red dotted = mean of pre-cue twin (should be ~0.5)")
    fig.savefig(OUT / f"{tag}a3_within_subject.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
