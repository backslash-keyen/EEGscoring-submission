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


def pm(m, sd):
    return f"{100 * m:.1f} +- {100 * sd:.1f}" if sd == sd else f"{100 * m:.1f}"


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def report(Sm):
    """Fill the tables of task1/b2_report_template.md from the CSVs; the prose around them is written by hand."""
    T = {}
    b = Sm[Sm.exp == "base"]
    rows = []
    for m in ["eegnet", "tf_time"]:
        g = b[b.model == m].set_index("n_train")
        rows.append([m] + [f"{pm(g.loc[n, 'mean'], g.loc[n, 'std'])} (seeds={int(g.loc[n, 'count'])})" if n in g.index else "-"
                           for n in b2_run.SIZES])
    T["SCALING_TABLE"] = md(pd.DataFrame(rows, columns=["model"] + [f"{n} subjects (%, mean +- SD)" for n in b2_run.SIZES]))
    g = b[b.n_train == 30].set_index("model")
    T["TOKEN_TABLE"] = md(pd.DataFrame(
        [[m, pm(g.loc[m, "mean"], g.loc[m, "std"]), f"{100 * g.loc[m, 'min']:.1f}-{100 * g.loc[m, 'max']:.1f}", int(g.loc[m, "count"])]
         for m in ["eegnet", "tf_time", "tf_chan_id", "tf_chan_noid"] if m in g.index],
        columns=["model", "accuracy % (mean +- SD)", "range over seeds", "seeds"]))
    D = pd.read_csv(OUT / "b2_displacement.csv")
    D["v"] = [pm(a, c) for a, c in zip(D["mean"], D["std"])]
    T["DISP_TABLE"] = md(D.pivot(index="label", columns="shift_mm", values="v").rename(columns=lambda c: f"{c:g} mm").reset_index())
    L = pd.read_csv(OUT / "b2_link_correlations.csv")
    L = L[L.part_a_measure.isin(["LI_mean", "acc_F1", "acc_M1", "acc_O1"])].copy()
    L["v"] = [f"{r:.2f} (p={p:.3f})" for r, p in zip(L.spearman_rho, L.p)]
    T["LINK_TABLE"] = md(L.pivot(index=["exp", "model"], columns="part_a_measure", values="v").reset_index()
                         .rename(columns={"LI_mean": "A2 LI (mean of mu, beta)", "acc_F1": "A3 F1 frontal 0-0.5 s",
                                          "acc_M1": "A3 M1 motor strip", "acc_O1": "A3 O1 occipital"}))
    g = Sm[Sm.n_train == 30].set_index(["exp", "model"])
    T["NOCONF_TABLE"] = md(pd.DataFrame(
        [[m, pm(g.loc[("base", m), "mean"], g.loc[("base", m), "std"]), pm(g.loc[("noconf", m), "mean"], g.loc[("noconf", m), "std"]),
          int(g.loc[("noconf", m), "count"])] for m in ["eegnet", "tf_time"] if ("noconf", m) in g.index],
        columns=["model", "full input %", "confound removed %", "seeds (removed)"]))
    Dd = D.set_index(["label", "shift_mm"])
    S = pd.read_csv(OUT / "b2_eegnet_spatial_sensitivity.csv").groupby("exp").spatial_sensitivity_10mm.mean()
    T["INTERV_TABLE"] = md(pd.DataFrame(
        [[lab] + [Dd.loc[(lab, s), "v"] for s in (0, 10, 20)] + [f"{S[e]:.3f}"] for lab, e in (("eegnet", "base"), ("eegnet+aug", "aug"))],
        columns=["model", "0 mm %", "10 mm %", "20 mm %", "spatial-filter sensitivity at 10 mm"]))
    A = pd.read_csv(OUT / "b2_attr_input.csv")
    r = A[A.kind == "region"].copy()
    r["share %"] = (100 * r.share).round(1)
    r["per electrode %"] = (100 * r.share / r.n_electrodes).round(2)
    t = A[A.kind == "time_0.1s"].copy()
    t["window"] = pd.cut(t.key.astype(float), [-0.01, 0.45, 1.45, 4.0], labels=["0-0.5 s (A3 eye)", "0.5-1.5 s", "1.5-4 s"])
    tw = (100 * t.groupby(["model", "window"], observed=True).share.sum()).round(1).unstack()
    top = A[A.kind == "electrode"].sort_values("share", ascending=False).groupby("model").key.apply(lambda k: ", ".join(k[:6]))
    F = pd.read_csv(OUT / "b2_faithfulness.csv")
    F["v"] = [pm(a, c) for a, c in zip(F["mean"], F["std"])]
    parts = ["Input-space grad x input by scalp region (`b2_attr_input.csv`, figure `b2_attribution_maps.png`):", "",
             md(r[["model", "key", "n_electrodes", "share %", "per electrode %"]].rename(columns={"key": "region"})), "",
             "By time window (% of attribution):", "", md(tw.reset_index()), "",
             "Top 6 electrodes: " + "; ".join(f"{m}: {v}" for m, v in top.items()), "",
             "Faithfulness (`b2_faithfulness.csv`, figure `b2_faithfulness.png`): accuracy after deleting each trial's top-ranked "
             "tokens (key-padding mask + zeroed embedding), mean +- SD over the 12 models (4 folds x 3 seeds):", "",
             md(F.pivot(index="method", columns="frac", values="v").rename(columns=lambda c: f"{100 * c:g}% deleted").reset_index())]
    T["ATTR_TEXT"] = "\n".join(parts)
    text = (Path(__file__).parent / "b2_report_template.md").read_text(encoding="utf-8")
    for k, v in T.items():
        text = text.replace(k, v)
    (OUT / "B2_RESULTS.md").write_text(text, encoding="utf-8")


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
    if (OUT / "b2_attr_input.csv").exists() and (OUT / "b2_displacement.csv").exists():
        report(Sm)


if __name__ == "__main__":
    main()
