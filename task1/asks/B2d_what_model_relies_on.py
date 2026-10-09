"""B2 (part 4) - What the best transformer relies on: attention, a gradient attribution, faithfulness by deletion.

Brief: "For the best transformer, extract attention and one gradient-based attribution; test faithfulness by deleting
most-attended vs most-attributed vs random tokens. Map what it relies on to your Part A findings: sensorimotor mu/beta
ERD, or an A3 confound." (Predict first.)

Attention = last-layer CLS attention averaged over heads; gradient attribution = gradient x input (on tokens, and on the
raw input for the electrode x time map). Faithfulness: accuracy after deleting each trial's top-ranked 5-50% of tokens
(key-padding mask + zeroed embedding), 12 models (4 folds x 3 seeds); random = 5 draws each. task1/b2_attribution.py.
Run: python task1/asks/B2d_what_model_relies_on.py
"""
from _common import *

header("B2.4  What the model relies on",
       "attention + one gradient attribution for the best transformer; faithfulness by deleting most-attended vs "
       "most-attributed vs random tokens; map to A2 (mu/beta ERD) or an A3 confound.",
       "Best transformer chosen by a rule fixed before results: highest mean test accuracy at 30 training subjects.")

section("Prediction (PREDICTIONS.md, B2 items 5-6)")
say("\n".join(l for l in prediction(r"^## B2 - all six").splitlines() if l.startswith(("5.", "6."))))

R = csv("partb/b2_all_runs.csv")
best = R[(R.exp == "base") & (R.n_train == 30) & R.model.str.startswith("tf_")].groupby("model").test_acc.mean()
section("Which transformer is 'best' (mean over 4 folds x 3 seeds, 30 training subjects)")
print(best.round(4).to_string())
print(f"  -> {best.idxmax()}")

section("Faithfulness: accuracy (%) after deleting each trial's top tokens, mean +- SD over 12 models")
F = csv("partb/b2_faithfulness.csv")
F["cell"] = [f"{100 * m:.1f} +- {100 * s:.1f}" for m, s in zip(F["mean"], F["std"])]
print(F.pivot(index="method", columns="frac", values="cell").to_string())
FR = csv("partb/b2_faithfulness_runs.csv")
print(f"  no deletion: {100 * FR[FR.method == 'none'].acc.mean():.1f}%;  every model x method x fraction: "
      f"outputs/partb/b2_faithfulness_runs.csv ({len(FR)} rows)")
say("Deleting the most-attended tokens hurts most; gradient x input is barely better than random. Gradient x input is a "
    "first-order estimate at the intact input, but deleting a token also renormalises the softmax over the others, which "
    "last-layer CLS attention captures directly. Prediction 'attention > gradient > random': right.")

section("Which time tokens (0.2 s each) carry attention and attribution (mean over the 12 models)")
T = csv("partb/b2_attr_tokens.csv").groupby("token", sort=False)[["attention", "gradxinput"]].mean()
print((100 * T).round(1).T.to_string())

A = csv("partb/b2_attr_input.csv")
section("Input-space gradient x input by scalp region (share of total, and per electrode)")
reg = A[A.kind == "region"].assign(per_electrode=lambda d: d.share / d.n_electrodes)
table(reg[["model", "key", "n_electrodes", "share", "per_electrode"]].rename(columns={"key": "region"}), "{:.3f}")
section("Top 6 electrodes and share by time window")
for m, g in A.groupby("model", sort=False):
    el = g[g.kind == "electrode"].sort_values("share", ascending=False)
    t = g[g.kind == "time_0.1s"].assign(t=lambda d: d.key.astype(float))
    w = [t[(t.t >= a) & (t.t < b)].share.sum() for a, b in ((0, .5), (.5, 1.5), (1.5, 4))]
    print(f"  {m:8s} top electrodes {', '.join(el.key.head(6))};  0-0.5 s {100 * w[0]:.1f}%, 0.5-1.5 s {100 * w[1]:.1f}%, "
          f"1.5-4 s {100 * w[2]:.1f}%;  peak 0.1 s bin at {t.loc[t.share.idxmax(), 't']:.1f} s")

section("Mapping to Part A")
say("Both networks give the most attribution per electrode to F7, F8, Ft7, Ft8, Af8, T7/T9: the horizontal EOG positions "
    "of the A3 eye route. Sensorimotor electrodes have the largest TOTAL share only because there are 21 of them; per "
    "electrode they are the least used region. In time, the transformer peaks at 0.2-0.4 s, the A3 saccade window (A3 "
    "frontal decoder peaks at 0.3-0.4 s); EEGNet peaks at 0.6-1.1 s, the A3 return-saccade peak (0.7-0.9 s). The ~40% "
    "spread over 1.5-4 s could be motor ERD or sustained eye position; attribution cannot separate them, the confound-"
    "removed retrain (B2.5) does. Prediction 'both eye and motor': partly right (mainly eye).")
show_png("partb/b2_attribution_maps.png", "B2.4  gradient x input, electrodes x time")
show_png("partb/b2_faithfulness.png", "B2.4  faithfulness by deletion")
finish()
