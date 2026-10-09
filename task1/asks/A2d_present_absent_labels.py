"""A2 (part 4) - Label every subject lateralised ERD present or absent; show 3 subjects of each.

Brief: "Label every subject lateralised ERD present or absent; show 3 subjects of each."

The 3 + 3 shown are picked by a rule fixed in task1/erd.py, not by eye: present = the 3 with the most negative
contralateral ERD among their significant bands; absent = the 3 with the highest smallest p (clearest absence).
Live part: re-derives the label of every subject from the saved LI/p/ERD columns with the stated rule.
Run: python task1/asks/A2d_present_absent_labels.py
"""
from _common import *

header("A2.4  Present / absent label for every subject, 3 shown of each",
       "label every subject lateralised ERD present or absent; show 3 subjects of each.",
       "present = mu or beta permutation p < 0.025 (Bonferroni over 2 bands) AND contralateral ERD < 0 dB (a real "
       "desynchronisation, not ipsilateral synchronisation). Laplacian reference.")

L = csv("a2_lateralisation.csv")
ALPHA = 0.05 / 2
rule = lambda r: any(r[f"{b}_p"] < ALPHA and r[f"{b}_contra_db"] < 0 for b in ("mu", "beta"))
L["label_rule"] = np.where(L.apply(rule, axis=1), "present", "absent")
section("Label of every subject")
table(L[["subject", "mu_LI", "mu_p", "beta_LI", "beta_p", "label"]])
print(f"\n  present: {(L.label == 'present').sum()} / 40   absent: {(L.label == 'absent').sum()} / 40")
print(f"  rule re-applied here agrees with the pipeline label for {(L.label == L.label_rule).sum()} / 40 subjects")
print("  present:", L[L.label == "present"].subject.tolist())

section("The 3 + 3 shown, chosen by the fixed rule")
pp = L[L.label == "present"].copy()
pp["clean_erd"] = pp.apply(lambda r: min(r[f"{b}_contra_db"] for b in ("mu", "beta") if r[f"{b}_p"] < ALPHA), axis=1)
pres = pp.sort_values("clean_erd").subject.head(3).tolist()
absn = L[L.label == "absent"].assign(pm=lambda d: d[["mu_p", "beta_p"]].min(axis=1)).sort_values("pm", ascending=False).subject.head(3).tolist()
print("  present (most negative contralateral ERD):", pres)
print("  absent  (largest min p):                  ", absn)
say("Each row of the figure is one subject: left-hand imagery at C3 and C4, then right-hand imagery at C3 and C4 "
    "(ERD%, blue = power drop). Present: blue in the hemisphere opposite the hand. Absent: no consistent contralateral drop.")
show_png("a2_examples_3present_3absent.png", f"A2.4  present {pres} (rows 1-3), absent {absn} (rows 4-6)")

section("Robustness of the labels")
C = csv("a2_lateralisation_causal.csv")
m = L.merge(C, on="subject", suffixes=("", "_causal"))
print(f"  minimum-phase (causal) filter instead of zero-phase: labels identical for {(m.label == m.label_causal).sum()} / 40")
finish()
