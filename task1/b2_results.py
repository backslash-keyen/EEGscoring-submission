"""B2 tables and figures from the saved runs: every seed reported (assignment rule), seed spread shown, chance thresholds."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))
import data, train, b2_run, spatial

OUT = train.OUT
A = data.ROOT / "outputs"


def binom_thr(n, alpha=0.05):
    """Smallest accuracy k/n with P(X >= k) <= alpha under X ~ Binomial(n, 0.5): same rule as A3 (D15)."""
    k = int(stats.binom.isf(alpha, n, 0.5)) + 1
    return k / n


def load_runs():
    R = pd.DataFrame([json.loads(p.read_text()) for p in b2_run.RUNS.glob("*.json")])
    T = pd.concat([pd.read_csv(p) for p in b2_run.RUNS.glob("*.csv")], ignore_index=True)
    T["correct"] = ((T.p_right > 0.5) == T.y).astype(float)
    return R, T


def seed_level(T):
    """Accuracy per (exp, model, n_train, seed) pooled over the 4 test folds = the same 20 test subjects every time."""
    g = T.groupby(["exp", "model", "n_train", "seed"])
    return g.correct.mean().rename("acc").to_frame().join(g.size().rename("n_trials")).reset_index()


def summary(S):
    g = S.groupby(["exp", "model", "n_train"])
    out = g.acc.agg(["mean", "std", "min", "max", "count"]).join(g.n_trials.first()).reset_index()
    out["thr_binom_5pct"] = [binom_thr(n) for n in out.n_trials]
    return out


def plot_scaling(S):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for (m, c) in (("eegnet", "tab:blue"), ("tf_time", "tab:orange")):
        d = S[(S.exp == "base") & (S.model == m)]
        if d.empty:
            continue
        mean = d.groupby("n_train").acc.mean()
        ax.plot(mean.index, mean.values, "-o", color=c, label=m)
        ax.scatter(d.n_train + (0.4 if m == "tf_time" else -0.4), d.acc, color=c, alpha=0.5, s=14)
        sd = d.groupby("n_train").acc.std()
        ax.fill_between(mean.index, mean - sd, mean + sd, color=c, alpha=0.15)
    n = S.n_trials.max()
    ax.axhline(binom_thr(n), color="grey", ls="--", lw=0.8, label=f"chance threshold ({n} trials)")
    ax.set(xlabel="training subjects", ylabel="test accuracy (20 held-out subjects)", xticks=b2_run.SIZES,
           title="Data scaling (dots = seeds, band = +-1 SD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "b2_scaling.png", dpi=120)


def link_to_part_a(T):
    """Per test subject: accuracy (mean over seeds, 30 training subjects) vs A2 LI and the A3 within-subject decoders."""
    P = pd.read_csv(A / "a3_per_subject_table.csv")
    P["LI_mean"] = P[["mu_LI", "beta_LI"]].mean(1)
    d = T[T.n_train == 30].groupby(["exp", "model", "subject", "seed"]).correct.mean().groupby(["exp", "model", "subject"]).mean()
    d = d.rename("net_acc").reset_index().merge(P, on="subject")
    rows = []
    for (exp, m), g in d.groupby(["exp", "model"]):
        for col in ["LI_mean", "mu_LI", "beta_LI", "acc_F1", "acc_F2", "acc_O1", "acc_M1", "acc_N2"]:
            rho, p = stats.spearmanr(g.net_acc, g[col])
            rows.append(dict(exp=exp, model=m, part_a_measure=col, spearman_rho=rho, p=p, n_subjects=len(g)))
    d.to_csv(OUT / "b2_link_per_subject.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "b2_link_correlations.csv", index=False)
    return d


def displacement():
    f = OUT / "b2_displacement_runs.csv"
    if not f.exists():
        return
    D = pd.read_csv(f)
    D["label"] = np.where(D.exp == "aug", "eegnet+aug", D.model)
    # accuracy per seed pooled over folds, then mean/SD over seeds; directions averaged except the lateral-only table
    seedacc = D.assign(c=D.acc * D.n).groupby(["label", "seed", "shift_mm", "direction"]).agg(c=("c", "sum"), n=("n", "sum"))
    seedacc = (seedacc.c / seedacc.n).rename("acc").reset_index()
    allDir = seedacc.groupby(["label", "seed", "shift_mm"]).acc.mean().reset_index()
    tab = allDir.groupby(["label", "shift_mm"]).acc.agg(["mean", "std"]).reset_index()
    tab.to_csv(OUT / "b2_displacement.csv", index=False)
    lat = seedacc[seedacc.direction.isin(["left", "right", "any"])].groupby(["label", "seed", "shift_mm"]).acc.mean()
    lat.groupby(["label", "shift_mm"]).agg(["mean", "std"]).reset_index().to_csv(OUT / "b2_displacement_lateral.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for lab, g in tab.groupby("label"):
        ax.errorbar(g.shift_mm, g["mean"], yerr=g["std"], marker="o", capsize=3, label=lab)
    ax.set(xlabel="cap slide at test time (mm, mean of 4 directions)", ylabel="test accuracy", title="Electrode displacement")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "b2_displacement.png", dpi=120)
    L = OUT / "b2_displacement_layers.csv"
    if L.exists():
        Ly = pd.read_csv(L)
        Ly["label"] = np.where(Ly.exp == "aug", "eegnet+aug", Ly.model)
        Ly.groupby(["label", "layer"], sort=False).rel_change.agg(["mean", "std"]).reset_index().to_csv(OUT / "b2_displacement_layer_summary.csv", index=False)


def eegnet_spatial_sensitivity():
    """||W (M - I)||_F / ||W||_F for the depthwise spatial filters W (16 x 64) at a 10 mm slide: how much of each learned
    spatial filter's output a slide changes, for plain and augmentation-trained EEGNet (intervention mechanism)."""
    names = list(data.load_subject(data.SUBJECTS[0], causal=True)["ch_names"])
    Ms = [spatial.displacement_matrix(names, 10, v) for v in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    rows = []
    for p in b2_run.RUNS.glob("*eegnet*_n30.pt"):
        ck = torch.load(p, weights_only=False)
        if ck["res"]["exp"] not in ("base", "aug"):
            continue
        W = ck["state"]["block1.2.weight"].numpy()[:, 0, :, 0]    # (16, 64)
        sens = np.mean([np.linalg.norm(W @ (M - np.eye(64))) / np.linalg.norm(W) for M in Ms])
        rows.append(dict(exp=ck["res"]["exp"], fold=ck["res"]["fold"], seed=ck["res"]["seed"], spatial_sensitivity_10mm=sens))
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "b2_eegnet_spatial_sensitivity.csv", index=False)


def main():
    R, T = load_runs()
    R.sort_values(["exp", "model", "n_train", "fold", "seed"]).to_csv(OUT / "b2_all_runs.csv", index=False)
    S = seed_level(T)
    S.to_csv(OUT / "b2_seed_level.csv", index=False)
    Sm = summary(S)
    Sm.to_csv(OUT / "b2_summary.csv", index=False)
    print(Sm.round(4).to_string(index=False))
    plot_scaling(S)
    link_to_part_a(T)
    displacement()
    eegnet_spatial_sensitivity()


if __name__ == "__main__":
    main()
