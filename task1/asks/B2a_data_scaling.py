"""B2 (part 1) - Data scaling: 5, 10, 20 and all training subjects, 3 seeds, same test subjects; both models.

Brief: "Train on 5, 10, 20 and all training subjects, 3 seeds each, same test subjects. Plot both models with spread and
explain any crossover (or its absence) from what each architecture assumes about EEG." (Predict first, then run.)

The 96 scaling runs come from task1/b2_run.py (main.py step B2, ~12 h on a CPU). This script prints every run, the
prediction made before them, and plots both models with spread.
RETRAIN = True re-trains one of those runs live with the grid's exact settings (several minutes) and compares.
Run: python task1/asks/B2a_data_scaling.py
"""
from _common import *

RETRAIN = False                        # True: re-train one run live (e.g. for "change something and predict the outcome")
RUN = ("base", "eegnet", 0, 0, 30)     # (experiment, model, test fold 0-3, seed 0-2, training subjects 5/10/20/30); ~8 min

header("B2.1  Data scaling",
       "5/10/20/all training subjects, 3 seeds, same test subjects; plot both models with spread; explain any crossover.",
       "EEGNet and the time-patch transformer, test = the same 20 subjects (folds 0-3), seeds also draw the training "
       "subjects. Accuracy pooled over the 913 test trials per seed; chance threshold 52.8%.")

section("Prediction (PREDICTIONS.md, B2 items 1-2)")
say("\n".join(l for l in prediction(r"^## B2 - all six").splitlines() if l.startswith(("1.", "2.", "Setup"))))
prediction_order("f1ee430", ["outputs/partb/b2_all_runs.csv", "outputs/partb/b2_summary.csv"])

R = csv("partb/b2_all_runs.csv")
R = R[(R.exp == "base") & R.model.isin(["eegnet", "tf_time"])]
section(f"Every run ({len(R)}): test accuracy per test fold, and pooled over the 913 test trials")
R["correct"] = R.test_acc * R.n_test
pv = R.pivot_table(index=["model", "n_train", "seed"], columns="fold", values="test_acc")
pv["pooled"] = R.groupby(["model", "n_train", "seed"]).correct.sum() / R.groupby(["model", "n_train", "seed"]).n_test.sum()
print(pv.round(3).to_string())

S = csv("partb/b2_summary.csv")
S = S[(S.exp == "base") & S.model.isin(["eegnet", "tf_time"])]
section("Mean +- SD over 3 seeds (min-max)")
for _, r in S.iterrows():
    print(f"  {r.model:8s} {int(r.n_train):2d} subjects: {100 * r['mean']:.1f} +- {100 * r['std']:.1f}%  "
          f"({100 * r['min']:.1f}-{100 * r['max']:.1f}, {int(r['count'])} seeds)")
thr = S.thr_binom_5pct.iloc[0]

fig, ax = plt.subplots(figsize=(7, 4.5))
for k, (m, g) in enumerate(S.groupby("model")):
    g = g.sort_values("n_train")
    x = g.n_train * (1 + 0.03 * (k - .5))
    ax.plot(x, 100 * g["mean"], "-o", color=f"C{k}", label=m)
    ax.fill_between(x, 100 * (g["mean"] - g["std"]), 100 * (g["mean"] + g["std"]), color=f"C{k}", alpha=.15)
    seeds = pv.loc[m].reset_index()
    ax.scatter(seeds.n_train * (1 + 0.03 * (k - .5)), 100 * seeds.pooled, s=12, color=f"C{k}", alpha=.6)
ax.axhline(100 * thr, color="0.5", ls=":", label=f"chance threshold {100 * thr:.1f}%")
ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 30]); ax.set_xticklabels(["5", "10", "20", "30 (all)"])
ax.set_xlabel("training subjects"); ax.set_ylabel("test accuracy, 20 test subjects (%)")
ax.set_title("B2.1  dots = seeds, line = mean, band = +-1 SD", fontsize=9); ax.legend(fontsize=8)
save(fig, "B2a_scaling.png")

section("Crossover?")
gap = S.pivot(index="n_train", columns="model", values="mean")
for n, r in gap.iterrows():
    print(f"  {int(n):2d} subjects: EEGNet - transformer = {100 * (r.eegnet - r.tf_time):+.1f} points")
say("No crossover: EEGNet leads at every size and the gap shrinks. EEGNet hard-codes what EEG decoding needs (one "
    "temporal filter bank shared by all electrodes, one spatial filter per band, pooled power), so ~2900 weights in that "
    "fixed form pin down the spatially broad frontal eye pattern from 5 subjects. The transformer has the same front end "
    "but must learn from data how 20 time segments combine (attention + learned positions); that freedom costs data. "
    "A crossover would need a signal richer than EEGNet's form can express; a broad eye field is not one. "
    "Prediction was 'crossover': wrong.")

if RETRAIN:
    if has_data():
        retrain(*RUN)
    else:
        no_data_note()
finish()
