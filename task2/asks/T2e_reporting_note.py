"""2e - Reporting: a 150-word note to a non-technical reader on why the lower number is the one to report.

Brief: "Your fixed pipeline will likely score lower than the original on at least one headline metric. Write a
150-word note to a non-technical reader explaining why the lower number is the one to report."

Prints the note, counts its words, and checks every number in it against the output it comes from.
Run: python task2/asks/T2e_reporting_note.py
"""
import re
from _t2 import *

header("2e  Note for a non-technical reader",
       "a ~150-word note explaining why the fixed pipeline's lower headline number is the one to report.",
       "task2/REPORTING_NOTE.md; each number below is traced to the file that produced it.")

note = text("task2/REPORTING_NOTE.md").split("\n\n", 1)[1].split("\n\n<!--")[0]
section("The note")
say(note)
print(f"\n  words: {len(note.split())} (target about 150)")

runs = ledger_runs()
base = text("task2/baseline/2a_baseline_seed42.txt")
num = lambda pat, s: float(re.search(pat, s).group(1))
given_acc, given_f1 = num(r"accuracy\s+([0-9.]+)", base), num(r"macro-F1\s+([0-9.]+)", base.split("=== held-out")[1])
fixed_acc = np.mean([runs[("FIXED", s)]["accuracy"] for s in SEEDS])
fixed_f1 = np.mean([runs[("FIXED", s)]["macro_f1"] for s in SEEDS])
always_wake = num(r"accuracy ([0-9.]+)%", text("task2/evidence/d10_wake_padding_result.txt"))
wake_share = num(r"Wake \d+ = ([0-9.]+)%", text("task2/evidence/d10_wake_padding_result.txt"))

section("Every number in the note, traced")
rows = [("85% accuracy (original)", "85%", f"{given_acc:.1%}", "task2/baseline/2a_baseline_seed42.txt"),
        ("about 83% (corrected)", "83%", f"{fixed_acc:.1%}", "FIXED runs, seeds 42-44 (task2/ledger_results)"),
        ("'always awake' scores 69%", "69%", f"{always_wake:.1f}%", "task2/evidence/d10_wake_padding_result.txt"),
        ("two-thirds of each recording awake", "2/3", f"{wake_share:.1f}%", "task2/evidence/d10_wake_padding_result.txt"),
        ("balanced score 0.63 -> 0.79", "0.63, 0.79", f"{given_f1:.3f}, {fixed_f1:.3f}", "baseline output; FIXED runs")]
table(pd.DataFrame(rows, columns=["claim in the note", "as written", "source value", "source"]))
consistent = (round(given_acc * 100) == 85 and round(fixed_acc * 100) == 83 and round(always_wake) == 69
              and round(given_f1, 2) == 0.63 and round(fixed_f1, 2) == 0.79 and 60 < wake_share < 72)
print(f"\n  every number in the note rounds from its source: {'MATCH' if consistent else 'DIFFERENT'}")
finish()
