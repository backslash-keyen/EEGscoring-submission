"""2b - Find and fix: for each defect, what is wrong, the evidence, how it distorts the result, and the fix.

Brief: "For each defect give: what is wrong, the evidence you used (printout, plot, file inspection - not only reading
the code), how it distorts the result, and your fix. One defect per commit, minimal diffs."

For each defect: the DEFECTS.md entry, the fix commit's diff (from git, so it is the real minimal diff), and the
evidence printout. Where the evidence script takes seconds it is run again now and compared with the saved printout.
Set DEFECT to one number to see only that defect.
Run: python task2/asks/T2b_defects.py
"""
import re, subprocess
from _t2 import *

DEFECT = None       # e.g. 3 for one defect; None for all ten

FIX = {1: "39775aa", 2: "6136415", 3: "c6d6a49", 4: "c3523bb", 5: "5dd1693", 6: "d38cf1b", 7: "00846ed", 8: "9cac159",
       9: "2731f10", 10: "6d07e53"}
# evidence script -> its saved printout; LIVE marks the ones fast enough to re-run here (seconds, not minutes)
EVIDENCE = {1: [("d1_split_overlap.py", "d1_split_overlap_result.txt", "data")],
            2: [(None, "../baseline/d1_subject_split.txt", None)],
            3: [("d3_batch_attention.py", "d3_batch_attention_result.txt", "nodata")],
            4: [("d4_order_test.py", "d4_order_result.txt", "nodata")],
            5: [("d5_label_position.py", "d5_label_position_result.txt", "data")],
            6: [("d6_reject_units.py", "d6_reject_units_result.txt", None), ("d6_per_channel.py", "d6_per_channel_result.txt", None),
                (None, "d6_after_fix.txt", None)],
            7: [(None, "d7_after_fix.txt", None)],
            8: [("d8_amplitude_by_stage.py", "d8_amplitude_by_stage_result.txt", None), (None, "d8_after_fix.txt", None)],
            9: [("d9_label_defaults.py", "d9_label_defaults_result.txt", "data"), (None, "d9_after_fix.txt", None)],
            10: [("d10_wake_padding.py", "d10_wake_padding_result.txt", None), (None, "d10_after_fix.txt", None)]}
EV = T2 / "evidence"

header("2b  Find and fix",
       "for each defect: what is wrong, the evidence (not only code reading), how it distorts the result, the fix; "
       "one defect per commit with minimal diffs; judgement calls justified in DECISIONS.md.",
       "DEFECTS.md holds the write-up; each fix is its own commit on task2/sleep_pipeline.py; each piece of evidence "
       "is a script in task2/evidence/ with its printout saved next to it.")

section("One defect per commit? (every commit that changed task2/sleep_pipeline.py, from git)")
code, log = git("log", "--reverse", "--format=%h %s", "fb4a5e7..HEAD", "--", "task2/sleep_pipeline.py")
print("  git history not available here" if code else "  " + "\n  ".join(log.splitlines()))

defects = text("DEFECTS.md")
for k in ([DEFECT] if DEFECT else range(1, 11)):
    m = re.search(rf"^## {k}\. .*?(?=^## |\Z)", defects, flags=re.M | re.S)
    print("\n" + "=" * 100)
    print(m.group(0).splitlines()[0].lstrip("# ").upper())
    print("=" * 100)
    say("\n".join(l for l in m.group(0).splitlines()[1:] if l.strip()))
    section(f"The fix: commit {FIX[k]}, diff of task2/sleep_pipeline.py")
    code, diff = git("show", "--format=%h %s", FIX[k], "--", "task2/sleep_pipeline.py")
    print("  git history not available here" if code else "\n".join("  " + l for l in diff.splitlines()
                                                                         if not l.startswith(("diff --git", "index ", "--- ", "+++ "))))
    for script, saved_name, live in EVIDENCE[k]:
        saved = (EV / saved_name).read_text(encoding="utf-8").strip()
        section(f"Evidence: task2/evidence/{saved_name}" + (f" (printout of {script})" if script else ""))
        print("  " + "\n  ".join(saved.splitlines()[:14]) + ("\n  ..." if saved.count("\n") > 13 else ""))
        if live == "nodata" or (live == "data" and has_sleep_data()):
            r = subprocess.run([sys.executable, "-W", "ignore", script], cwd=EV, capture_output=True, text=True, env=dict(os.environ))
            same = r.stdout.strip().replace("\r", "") == saved.replace("\r", "")
            print(f"  re-run of {script} now: {'MATCH (identical printout)' if same else 'DIFFERENT'}")
            if not same:
                print("  " + "\n  ".join(r.stdout.strip().splitlines()[:10]))
        elif live == "data":
            no_sleep_data_note()
finish()
