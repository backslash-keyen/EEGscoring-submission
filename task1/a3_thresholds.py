"""Chance thresholds for the A3 follow-up accuracies that had only a z-score or a dotted line (DECISIONS D11).
Every number is the exact one-sided binomial 5% threshold for that trial count, p0 = 0.5: the smallest k/n with P(X >= k) <= 0.05.
For the repeat/alternate 'balanced' accuracy (mean of two independent proportions) there is no exact binomial; the threshold is the
normal approximation 0.5 + 1.645 * SE, SE = 0.5 * sqrt(0.25/n_rep + 0.25/n_alt).   python task1/a3_thresholds.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import data, confound as C

OUT = data.ROOT / "outputs"
t = pd.read_csv(OUT / "a3b_carryover.csv")
t["n_all"] = t.ok_n
t["thr_all"] = [C.binom_thr(n) for n in t.n_all]
t["thr_repeat"] = [C.binom_thr(n) for n in t.rep_n]
t["thr_alternate"] = [C.binom_thr(n) for n in t.alt_n]
t["thr_balanced_normal_approx"] = 0.5 + 1.645 * 0.5 * np.sqrt(0.25 / t.rep_n + 0.25 / t.alt_n)
t["balanced_above_thr"] = t.acc_balanced >= t.thr_balanced_normal_approx
t.to_csv(OUT / "a3_followup_carryover_with_thresholds.csv", index=False)

n_pool = int(pd.read_csv(OUT / "a3_cross_subject.csv").n.iloc[0])
tc = pd.read_csv(OUT / "a3c_timecourse.csv")
tc["thr_binom"] = C.binom_thr(n_pool)
tc["above_thr"] = tc.acc >= tc.thr_binom
tc.to_csv(OUT / "a3c_timecourse_with_thresholds.csv", index=False)
tw = pd.read_csv(OUT / "a3c_causal_twins.csv")
tw["thr_binom"] = C.binom_thr(n_pool)
tw["above_thr"] = tw.pooled_xs_acc >= tw.thr_binom
tw.to_csv(OUT / "a3c_causal_twins_with_thresholds.csv", index=False)
print("pooled n =", n_pool, "thr =", round(C.binom_thr(n_pool), 4), "| repeat n=424 thr =", round(C.binom_thr(424), 4),
      "| alternate n=1249 thr =", round(C.binom_thr(1249), 4), "| all n=1673 thr =", round(C.binom_thr(1673), 4))
print("\ncausal pooled, plain scheme (acc vs its own threshold):")
x = t[(t.regime == "pooled_xs") & (t.scheme == "plain")][["decoder", "acc_all", "thr_all", "acc_repeat", "thr_repeat", "acc_alternate", "thr_alternate", "acc_balanced", "thr_balanced_normal_approx"]]
print(x.round(3).to_string(index=False))
print("\ntwins (zero- vs minimum-phase), pooled:")
print(tw.pivot(index="decoder", columns="filter", values="pooled_xs_acc").round(3).assign(thr=round(C.binom_thr(n_pool), 3)))
