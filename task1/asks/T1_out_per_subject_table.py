"""Task 1 outputs - The per-subject table.

Brief (Task 1 outputs): "audit.csv, ERD figures, per-subject table (lateralisation index mu/beta, p-value, label,
confound-decoder accuracies with thresholds)."

Prints outputs/a3_per_subject_table.csv (written by task1/confound.py from the A2 and A3 results) in readable form, marks
every accuracy at or above that subject's chance threshold, and lists where the other Task 1 output files are.
Run: python task1/asks/T1_out_per_subject_table.py
"""
from _common import *

DECODERS = ["F1", "F2", "O1", "O2", "G1", "N2", "R2", "M1", "A1", "F1_pre"]   # headline decoders; any acc_* column works

header("Task 1 outputs  Per-subject table",
       "per subject: lateralisation index mu/beta, p-value, label, confound-decoder accuracies with thresholds.",
       "A2 columns from a2_lateralisation.csv (Laplacian); A3 accuracies within subject, leave-one-run-out, causal filter; "
       "threshold = exact binomial 5% for that subject's trial count. '*' = at or above threshold.")

T = csv("a3_per_subject_table.csv")
show = T[["subject", "mu_LI", "mu_p", "beta_LI", "beta_p", "label", "thr_binom"]].copy()
for d in DECODERS:
    show[d] = [f"{a:.2f}{'*' if a >= t else ' '}" for a, t in zip(T[f"acc_{d}"], T.thr_binom)]
section("Per-subject table")
table(show, "{:.3f}")
section("Column totals: subjects at or above threshold (about 2 of 40 expected by luck)")
print("  " + "  ".join(f"{d} {int((T[f'acc_{d}'] >= T.thr_binom).sum())}" for d in DECODERS))
print(f"  lateralised ERD present: {(T.label == 'present').sum()} / 40")
say("Decoders: F1/F2 frontal (eyes) 0-0.5 s / 0.5-4 s; O1/O2 occipital (visual); G1 temporal 30-40 Hz (muscle); N2 all "
    "non-sensorimotor sites 0-0.5 s; R2 previous labels, no EEG; M1 sensorimotor mu+beta (motor control); A1 all "
    "channels mu+beta; F1_pre frontal before the cue (method check). Full column set: outputs/a3_per_subject_table.csv.")

fig, ax = plt.subplots(figsize=(9, 9))
M = np.column_stack([T[f"acc_{d}"] - T.thr_binom for d in DECODERS])
im = ax.imshow(100 * M, cmap="RdBu_r", vmin=-25, vmax=25, aspect="auto")
ax.set_xticks(range(len(DECODERS))); ax.set_xticklabels(DECODERS)
ax.set_yticks(range(len(T))); ax.set_yticklabels([f"S{s}{' (ERD)' if l == 'present' else ''}" for s, l in zip(T.subject, T.label)], fontsize=6)
fig.colorbar(im, label="accuracy minus that subject's chance threshold (points); red = above")
ax.set_title("Task 1 per-subject table: decoders vs chance threshold", fontsize=9)
save(fig, "T1_per_subject_table.png")

section("Where the other Task 1 output files are")
for f, what in [("audit.csv", "A1 audit, one row per subject"), ("outputs/erd/S070.png ... S109.png", "A2 ERD figure per subject"),
                ("outputs/a2_examples_3present_3absent.png", "A2 3 present + 3 absent"),
                ("outputs/a3_per_subject_table.csv", "this table"), ("outputs/A3_RESULTS.md", "A3 results and scorecard"),
                ("outputs/partb/b2_all_runs.csv", "every B2 run (156), every seed"), ("outputs/partb/B2_RESULTS.md", "B2 results and scorecard")]:
    print(f"  {f:45s} {what}")
finish()
