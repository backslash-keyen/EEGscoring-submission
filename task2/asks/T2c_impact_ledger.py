"""2c - Impact ledger: re-introduce each defect alone into the fixed pipeline and measure it.

Brief: "Starting from your fully fixed pipeline, re-introduce each defect alone and measure accuracy, macro-F1 and kappa
on subject-independent held-out data, at least 3 seeds each. One table: defect -> delta per metric (mean +- spread) ->
predicted direction -> mechanism. Rank defects by how much they would have misled the report, and flag any whose effect
vanishes or flips once another defect is fixed."

Recomputes every delta from the saved runs (task2/ledger_results/*.json) and checks it against the committed table,
checks from git that the predictions came first, then quotes the ranking and flags from the report.
Set RUN = ("D3", 42) to train one variant now (about 15 minutes) and compare it with its saved run.
Run: python task2/asks/T2c_impact_ledger.py
"""
import re
from _t2 import *

RUN = None          # e.g. ("D3", 42): re-train that variant and seed now; nothing is written to ledger_results/

import ledger       # task2/ledger.py: VARIANTS, SWAPS, build_source, the predicted / mechanism columns

header("2c  Impact ledger",
       "re-introduce each defect alone into the fixed pipeline, >= 3 seeds, delta per metric with spread, predicted "
       "direction and mechanism; rank by how much each would have misled; flag effects that vanish or flip.",
       "task2/ledger.py reverses ONE fix in the fixed source text (so every other line stays fixed), runs seeds 42-44 "
       "(a seed also draws the subject split), and compares with the fixed pipeline on the same seed.")

section("How a defect is re-introduced: the text swap for D3 (from task2/ledger.py)")
for old, new in ledger.SWAPS["D3"]:
    print(f"  fixed    : {old.strip()}\n  defective: {new.strip()}")

section("Were the predictions committed before any ledger result? (from git)")
prediction_order("f7b566c", ["task2/ledger_results/*.json"])

runs = ledger_runs()
ref = {s: runs[("FIXED", s)] for s in SEEDS if ("FIXED", s) in runs}
table_md = text("task2/ledger_table.md")
rows, ok = [], True
for v, (_, what) in ledger.VARIANTS.items():
    if v == "FIXED":
        continue
    ss = [s for s in SEEDS if (v, s) in runs and s in ref]
    d = {k: np.array([runs[(v, s)][k] - ref[s][k] for s in ss]) for k in METRICS}
    rows.append(dict(variant=v, seeds=len(ss), **{f"d_{k}": d[k].mean() for k in METRICS},
                     sd_f1=d["macro_f1"].std(ddof=1), predicted=ledger.PREDICTED.get(v, ""), mechanism=ledger.MECHANISM.get(v, "")))
    line = next(l for l in table_md.splitlines() if l.startswith(f"| {v} |"))
    ok &= f"{d['macro_f1'].mean():+.3f}" in line and f"{d['accuracy'].mean():+.3f}" in line and f"{d['kappa'].mean():+.3f}" in line
T = pd.DataFrame(rows).sort_values("d_macro_f1", key=abs, ascending=False)
f = np.array([[ref[s][k] for k in METRICS] for s in ref])
section(f"Fixed pipeline (reference), mean +- SD over seeds {list(ref)}")
print("  " + ", ".join(f"{k} {m:.3f} +- {sd:.3f}" for k, m, sd in zip(METRICS, f.mean(0), f.std(0, ddof=1))))
section("Delta = defective - fixed on the same seed (recomputed from the saved runs)")
table(T)
print(f"\n  every recomputed mean equals the committed task2/ledger_table.md: {'MATCH' if ok else 'DIFFERENT'}")

section("Ranking and flags (quoted from report/REPORT.md)")
for start in (r"^\*\*Noise floor", r"^\*\*Ranked", r"^\*\*Flags"):
    say(report_block(start, r"^\*\*|^## ").replace("**", ""))
    print()

fig, ax = plt.subplots(figsize=(8, 5))
order = list(T.variant)[::-1]
for i, v in enumerate(order):
    ds = [runs[(v, s)]["macro_f1"] - ref[s]["macro_f1"] for s in SEEDS if (v, s) in runs]
    ax.barh(i, np.mean(ds), color="#c0504d" if np.mean(ds) < 0 else "#4f81bd", alpha=.6)
    ax.plot(ds, [i] * len(ds), "k.", ms=5)
ax.axvline(0, color="k", lw=.8)
ax.axvspan(-0.02, 0.02, color="grey", alpha=.15, label="about +-0.02: not resolved")
ax.set_yticks(range(len(order)), order)
ax.set_xlabel("delta macro-F1 vs fixed pipeline (bar = mean, dots = seeds)")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Impact ledger: each defect re-introduced alone (and stated combinations)", fontsize=9)
save(fig, "T2c_ledger_macro_f1.png")

if RUN:
    v, s = RUN
    if not has_sleep_data():
        no_sleep_data_note()
    else:
        import torch
        from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
        torch.set_num_threads(3)         # the thread count the ledger used; another count changes float sums slightly
        section(f"Training {v} seed {s} now (about 15 minutes) ...")
        ns = {"__name__": "variant", "__file__": ledger.SRC}
        exec(compile(ledger.build_source(v, s), f"<{v}:{s}>", "exec"), ns)
        yt, yp = ns["main"]()
        live = dict(accuracy=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro"), kappa=cohen_kappa_score(yt, yp))
        for k in METRICS:
            check(k, live[k], runs[(v, s)][k])
finish()
