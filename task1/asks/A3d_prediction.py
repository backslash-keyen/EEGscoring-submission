"""A3 (part 4) - Predict which route, if any, will be decodable before running A3.

Brief: "Predict which route, if any, will be decodable before running A3."

Prints the prediction as committed, proves from git history that it was committed before the first A3 result, and
scores it against the primary results.
Run: python task1/asks/A3d_prediction.py
"""
from _common import *

header("A3.4  Prediction before running A3",
       "predict which route, if any, will be decodable before running A3.",
       "PREDICTIONS.md entry committed in 3bdff2a; outcome appended later below it, prediction text unchanged.")

section("The prediction, as committed")
txt = prediction(r"^## A3 - which")
say(txt)
prediction_order("3bdff2a", ["outputs/*a3_cross_subject.csv", "outputs/*a3_within_subject.csv"])

S = csv("a3_summary.csv").set_index("decoder")
section("Prediction vs result (numbers read from outputs/a3_summary.csv)")
rows = [
    ("1 decodable routes", "eyes AND visual", "eyes strong, visual weak, PLUS trial history (not predicted)", "partly right"),
    ("2 frontal: subjects above chance", "5-10 of 40", f"{int(S.loc['F1', 'n_above'])} of 40", "wrong"),
    ("3 frontal pooled", "50-52%", f"{100 * S.loc['F1', 'pooled_xs_acc']:.1f}%", "wrong"),
    ("4 occipital pooled", "53-57%", f"{100 * S.loc['O1', 'pooled_xs_acc']:.1f}%", "slightly above"),
    ("5 pre-cue and trial order at chance", "both chance",
     f"pre-cue {100 * S.loc['F1_pre', 'pooled_xs_acc']:.1f}% (after the filter fix); previous label {100 * S.loc['R2', 'pooled_xs_acc']:.1f}%", "wrong"),
    ("6 motor strip pooled (control)", "56-62%", f"{100 * S.loc['M1', 'pooled_xs_acc']:.1f}%", "slightly below"),
]
table(pd.DataFrame(rows, columns=["item", "predicted", "found", "verdict"]))
say("Why 2, 3 and 5 missed: the prediction assumed random trial order and an eye signal whose sign varies across "
    "subjects (so it would cancel when pooled). The sign is consistent across subjects, and the order alternates.")
finish()
