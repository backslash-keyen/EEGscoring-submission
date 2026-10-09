"""2a - Baseline: run the given script unchanged with the given seed and record its output.

Brief: "Run the script unchanged with the given seed and record its output."

Proves from git that task2/baseline/sleep_pipeline_given.py is the given script byte for byte, prints the recorded
output, and draws its confusion matrix. Set RUN = True to run the given script again now (about 20 minutes on a CPU).
Run: python task2/asks/T2a_baseline.py
"""
import re, subprocess
from _t2 import *

RUN = False         # True: re-run the given script now and compare its numbers with the recorded ones

header("2a  Baseline",
       "run the given script unchanged with the given seed (42) and record its output.",
       "the given file was committed untouched (fb4a5e7) before any fix; a byte-identical copy is kept as "
       "task2/baseline/sleep_pipeline_given.py so main.py can re-run it after the fixes; its output is saved as text.")

section("Is the kept copy the given script, unchanged? (checked from git, not asserted)")
code, given = git("rev-parse", "fb4a5e7:task2/sleep_pipeline.py")
_, msg = git("log", "-1", "--format=%h %s", "fb4a5e7")
if code:
    print("  git history not available here; see commit fb4a5e7")
else:
    # git hashes the exact bytes of a file, so equal object ids = byte-identical files
    _, kept = git("hash-object", "task2/baseline/sleep_pipeline_given.py")
    print(f"  commit {msg}")
    print(f"  given file object id {given[:12]}, kept copy {kept[:12]}: {'IDENTICAL' if kept == given else 'DIFFERENT'}")

out = text("task2/baseline/2a_baseline_seed42.txt")
section("Recorded output (task2/baseline/2a_baseline_seed42.txt), held-out part")
print("  " + "\n  ".join(l for l in out[out.index("=== held-out"):].splitlines() if l.strip()))
num = lambda k: float(re.search(rf"{k}\s+([0-9.]+)", out).group(1))
saved = {"accuracy": num("accuracy"), "macro_f1": num("macro-F1"), "kappa": num("kappa")}
cm = np.array([[int(v) for v in re.findall(r"\d+", l)] for l in out.splitlines() if l.strip().startswith("[")][-5:])

section("What the numbers already show")
say(f"{cm[0].sum()} of {cm.sum()} test windows ({cm[0].sum() / cm.sum():.1%}) are Wake, so accuracy {saved['accuracy']:.3f} "
    f"mostly measures Wake; macro-F1 {saved['macro_f1']:.3f} and N1 F1 0.207 show the sleep stages are far worse. "
    "Every reason is a defect in 2b (T2b_defects.py).")

fig, ax = plt.subplots(figsize=(5, 4.2))
rn = cm / cm.sum(1, keepdims=True)
ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{cm[i, j]}\n{rn[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if rn[i, j] > .5 else "black")
ax.set_xticks(range(5), STAGES); ax.set_yticks(range(5), STAGES)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title("Given script, seed 42: confusion (count, row share)", fontsize=9)
save(fig, "T2a_baseline_confusion.png")

if RUN:
    if not has_sleep_data():
        no_sleep_data_note()
    else:
        section("Re-running the given script now (about 20 minutes) ...")
        r = subprocess.run([sys.executable, "-W", "ignore", str(T2 / "baseline" / "sleep_pipeline_given.py")],
                           capture_output=True, text=True, cwd=ROOT, env=dict(os.environ))
        live = {"accuracy": float(re.search(r"accuracy\s+([0-9.]+)", r.stdout).group(1)),
                "macro_f1": float(re.search(r"macro-F1\s+([0-9.]+)", r.stdout.split("=== held-out")[1]).group(1)),
                "kappa": float(re.search(r"kappa\s+([0-9.]+)", r.stdout).group(1))}
        for k in METRICS:
            check(k, live[k], saved[k], tol=0.0005)
        say("A small difference is possible: the given script does not fix the CPU thread count, and float summation "
            "order (and so training) depends on it (DECISIONS T2-9).")
finish()
