"""Task 1 write-up (b) - Mechanistic explanation of the B2 results, including at least one result that contradicted the prediction.

Brief: "... (b) mechanistic explanation of the B2 results, including at least one result that contradicted your
prediction and why."

Prints the B2 prediction scorecard with the measured numbers next to each prediction, the mechanism behind each result,
and picks out the contradicted ones. Each line points to the ask script that shows the evidence in full.
Run: python task1/asks/T1_out_writeup_b_mechanisms.py
"""
from _common import *

header("Task 1 write-up (b)  Mechanisms of the B2 results",
       "mechanistic explanation of the B2 results, including at least one result that contradicted the prediction and why.",
       "Prediction (PREDICTIONS.md, committed f1ee430 before any B2 result) vs measured, with the mechanism.")

B = csv("partb/b2_summary.csv")
b = lambda exp, m, n=30: B[(B.exp == exp) & (B.model == m) & (B.n_train == n)].iloc[0]["mean"] * 100
D = csv("partb/b2_displacement.csv").set_index(["label", "shift_mm"])["mean"] * 100
F = csv("partb/b2_faithfulness.csv").set_index(["method", "frac"])["mean"] * 100
Lc = csv("partb/b2_link_correlations.csv").set_index(["exp", "model", "part_a_measure"]).spearman_rho
I = csv("partb/b2_invariance_runs.csv").groupby("model").same_pred_mirror.mean()

items = [
    ("1 scaling", "crossover", f"no crossover: EEGNet {b('base', 'eegnet', 5):.1f} -> {b('base', 'eegnet'):.1f}%, transformer "
     f"{b('base', 'tf_time', 5):.1f} -> {b('base', 'tf_time'):.1f}%", "WRONG",
     "EEGNet's fixed form (shared temporal filters, one spatial filter per band, pooled power) fits a broad eye field from "
     "5 subjects; the transformer must learn how segments combine, which costs data.", "B2a"),
    ("2 EEGNet at 30 subjects", "> 72%", f"{b('base', 'eegnet'):.1f}%", "right", "eye + motor signal combined beats any single A3 decoder.", "B2a"),
    ("3 tokenisation", "time > chan+id = chan-noid", f"time {b('base', 'tf_time'):.1f}, chan+id {b('base', 'tf_chan_id'):.1f}, "
     f"chan-noid {b('base', 'tf_chan_noid'):.1f}%", "PARTLY WRONG",
     f"without identity the model is exactly mirror-invariant (same answer on {100 * I['tf_chan_noid']:.0f}% of mirrored trials), "
     "but the label is a mirror variable.", "B2b"),
    ("4 displacement 10 mm", "time-patch drops most", f"drop at 20 mm: time {D['tf_time', 0] - D['tf_time', 20]:.1f}, EEGNet "
     f"{D['eegnet', 0] - D['eegnet', 20]:.1f} points; < 1 point at 10 mm for all", "right (tiny)",
     "time-patch spatial filters feed every token and softmax amplifies a coherent change; the fields used are smooth "
     "over 10 mm so all drops are small.", "B2c"),
    ("5 reliance", "both eye and motor", "mainly eye (F7/F8/Ft7/Ft8, transformer peak 0.2-0.4 s)", "partly right",
     "attribution per electrode and time matches the A3 saccade route; motor part not separable.", "B2d"),
    ("6 faithfulness", "attention > gradient > random", f"50% deleted: attention {F['attention', 0.5]:.1f}, gradient "
     f"{F['gradxinput', 0.5]:.1f}, random {F['random', 0.5]:.1f}%", "right",
     "deleting tokens renormalises the softmax, which CLS attention measures and a first-order gradient does not.", "B2d"),
    ("7 link + confound removed", "tracks A2 LI; small drop", f"EEGNet rho with LI {Lc['base', 'eegnet', 'LI_mean']:+.2f}; removal -> "
     f"{b('noconf', 'eegnet'):.1f}% / {b('noconf', 'tf_time'):.1f}% (chance)", "HALF WRONG",
     "the motor pattern is subject-specific in strength and topography; the eye pattern was the only one shared across "
     "subjects, so without it nothing transfers (training loss falls, validation-subject loss rises: D27).", "B2e"),
    ("8 intervention", "no effect", f"0 mm {b('base', 'eegnet'):.1f} vs {b('aug', 'eegnet'):.1f}%; 20 mm {D['eegnet', 20]:.1f} vs "
     f"{D['eegnet+aug', 20]:.1f}%", "right",
     "the augmentation penalty (sigma^2/2) E[(grad L . J x)^2] is ~0 when the loss does not depend on exact positions.", "B2f"),
]
section("Scorecard with numbers recomputed from the saved results")
for it, pred, found, verdict, mech, script in items:
    print(f"  {it}")
    print(f"      predicted: {pred}")
    print(f"      found:     {found}   -> {verdict}")
    say(f"mechanism: {mech}  (evidence: task1/asks/{script}_*.py)", indent=6)

section("Results that contradicted the prediction, and why")
for it, pred, found, verdict, mech, _ in items:
    if verdict.isupper():
        print(f"  {it}: predicted '{pred}', found {found}")
        say(f"why: {mech}", indent=6)
finish()
