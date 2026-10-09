"""Task 1 write-up (a) - If a model scores 70% on these subjects, how much of that could be explained without motor imagery?

Brief: "<= 1.5-page write-up: (a) if a model scores 70% on these subjects, how much of that could be explained without
motor imagery; ..."

Collects every number the answer rests on from the saved results, so the paragraph in the report can be checked line
by line. Nothing is computed that main.py did not already compute.
Run: python task1/asks/T1_out_writeup_a_seventy_percent.py
"""
from _common import *

header("Task 1 write-up (a)  How much of 70% needs motor imagery?",
       "if a model scores 70% on these subjects, how much of that could be explained without motor imagery?",
       "Line up the non-motor decoders (A3) and the networks with and without the eye route (B2) against 70%.")

S = csv("a3_summary.csv").set_index("decoder")
thr = S.pooled_thr.iloc[0]
section(f"Cross-subject (pooled) accuracies, 1793 trials, chance threshold {100 * thr:.1f}%")
rows = [("F1 frontal 0-0.5 s (saccade window, before ERD)", "eyes", "F1"), ("F2 frontal 0.5-4 s", "eyes", "F2"),
        ("N2 every site except sensorimotor, 0-0.5 s", "non-motor", "N2"), ("R2 previous labels only, no EEG", "trial order", "R2"),
        ("O1 occipital 0-0.5 s", "visual", "O1"), ("M1 sensorimotor strip mu+beta 0.5-4 s", "MOTOR", "M1"),
        ("F1_pre frontal before the cue (method check)", "none", "F1_pre")]
for lab, route, d in rows:
    acc = S.loc[d, "pooled_xs_acc"]
    print(f"  {lab:50s} {route:12s} {100 * acc:5.1f}%  {'>= 70%' if acc >= .70 else ''}")

B = csv("partb/b2_summary.csv")
b = lambda exp, m: B[(B.exp == exp) & (B.model == m) & (B.n_train == 30)].iloc[0]
section("Networks, 20 test subjects, 913 trials (chance threshold 52.8%)")
for m in ("eegnet", "tf_time"):
    full, rem = b("base", m), b("noconf", m)
    print(f"  {m:8s} all 64 channels {100 * full['mean']:.1f} +- {100 * full['std']:.1f}%   eye route removed "
          f"{100 * rem['mean']:.1f} +- {100 * rem['std']:.1f}%")

A = csv("partb/b2_attr_input.csv")
top = {m: ", ".join(g[g.kind == "electrode"].sort_values("share", ascending=False).key.head(5)) for m, g in A.groupby("model")}
section("Supporting evidence")
sq = csv("a3_sequence.csv")
ml = csv("a3_matlab_pooled.csv").set_index("decoder")
print(f"  subjects with the frontal decoder above their own threshold: {int(S.loc['F1', 'n_above'])}/40; motor strip: {int(S.loc['M1', 'n_above'])}/40")
print(f"  lateralised ERD present (A2): {(csv('a2_lateralisation.csv').label == 'present').sum()}/40")
print(f"  label order: {sq.lag1_same.sum()} same-label neighbours vs {sq.null_mean.sum():.0f} expected (alternating)")
print(f"  frontal decoder with a causal Butterworth instead of MNE's FIR (MATLAB): {100 * ml.loc['F1', 'acc']:.1f}% (effect size depends on the filter, conclusion does not)")
print(f"  most-attributed electrodes: EEGNet {top['eegnet']}; transformer {top['tf_time']} (horizontal EOG sites)")

section("Answer")
say(f"All of it can be. Eye movements toward the cue alone give {100 * S.loc['F1', 'pooled_xs_acc']:.0f}-"
    f"{100 * S.loc['F2', 'pooled_xs_acc']:.0f}% across subjects and the trial order alone {100 * S.loc['R2', 'pooled_xs_acc']:.0f}%, "
    f"while the motor strip gives {100 * S.loc['M1', 'pooled_xs_acc']:.1f}%. The networks reach 72-77% with the eye route "
    f"present and fall to chance ({100 * b('noconf', 'eegnet')['mean']:.1f}%, {100 * b('noconf', 'tf_time')['mean']:.1f}%) without it. "
    "A 70% score on these subjects is therefore no evidence of motor imagery decoding; the number to report is the "
    "accuracy after the eye route is removed, which here is chance.")
finish()
