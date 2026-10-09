"""Reproduce every number and figure: `python main.py` runs all steps in order, no manual steps.

Each step is a script run as its own process (same interpreter), so a step sees exactly what it would see when run alone.
Steps are idempotent: caches (data/cache*, outputs/partb/runs) are reused when present, so a rerun after an
interruption continues where it stopped. Delete data/cache*, data/cache_causal and outputs/partb/runs for a cold run.

  python main.py                 everything (download ~0.3 GB, then ~12-14 h on an 8-core CPU, most of it the B2 grid)
  python main.py --list          show the steps
  python main.py --only A2 A3    run selected parts (A1 A2 A3 B1 B2 ASK T2 R)
  python main.py --from B2       run from a part onwards
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = [sys.executable, "-W", "ignore"]
# Sleep-EDF goes under data/ like the EEGBCI files; without this, MNE's default is ~/mne_data, outside the repo, so a
# second checkout downloads the ~1.4 GB again at PhysioNet's throttled rate. Task 1 passes explicit paths and is unaffected.
os.environ.setdefault("MNE_DATA", str(ROOT / "data" / "mne_sleep"))

# (part, script and arguments, what it produces). Order matters: later steps read earlier outputs.
STEPS = [
    ("A1", ["task1/download.py"], "EEGBCI runs 4/8/12, subjects 70-109 -> data/ (skips files already present)"),
    ("A1", ["task1/audit.py"], "audit.csv"),
    ("A2", ["task1/erd.py"], "outputs/erd/*.png, a2_lateralisation.csv, a2_all_references.csv, 3 present / 3 absent figure"),
    ("A2", ["task1/a2_causal_check.py"], "a2_lateralisation_causal.csv (A2 on the minimum-phase filter, D18)"),
    # the zero-phase first pass is kept as evidence of the filter leak (D18); the default run is the primary result
    ("A3", ["task1/confound.py", "--zero-phase"], "firstpass_zerophase_a3_* (first pass, superseded)"),
    ("A3", ["task1/confound.py"], "a3_* confound decoders, thresholds, per-subject table (primary)"),
    ("A3", ["task1/confound_carryover.py", "--zero-phase"], "firstpass_zerophase_a3b_*"),
    ("A3", ["task1/confound_carryover.py"], "a3b_carryover*.csv"),
    ("A3", ["task1/confound_causal.py"], "a3c_causal_twins.csv, a3c_timecourse.*"),
    ("A3", ["task1/a3_thresholds.py"], "*_with_thresholds.csv"),
    ("A3", ["task1/artifact_scan.py"], "a3d_artifact_scan.csv"),
    ("B1", ["task1/models.py"], "parameter counts of the four models (printed)"),
    ("B2", ["task1/b2_run.py", "--stage", "all"], "outputs/partb/runs: 156 trained models + per-trial predictions (resumable)"),
    ("B2", ["task1/b2_noconf_check.py"], "b2_noconf_check.csv (post-hoc diagnostic, D27)"),
    ("B2", ["task1/b2_displacement.py"], "b2_displacement_runs.csv, b2_displacement_layers.csv"),
    ("B2", ["task1/b2_invariance.py"], "b2_invariance_runs.csv"),
    ("B2", ["task1/b2_attribution.py"], "b2_attr_*.csv, b2_faithfulness*.csv/png, b2_attribution_maps.png"),
    ("B2", ["task1/b2_results.py"], "b2_summary.csv, b2_scaling.png, b2_displacement.png, link tables, B2_RESULTS.md"),
    # one script per ask of the brief (task1/asks/README.md); they read the outputs above, so they run after B2
    ("ASK", ["task1/asks/run_all.py"], "every Task 1 ask answered and checked against the outputs; figures in outputs/asks/"),
    # 2a: the given script, byte-identical to commit fb4a5e7, so the baseline can be re-run after the fixes
    ("T2", ["task2/baseline/sleep_pipeline_given.py"], "2a: given pipeline, seed 42 (printed; saved copy task2/baseline/2a_baseline_seed42.txt)"),
    ("T2", ["task2/evidence/run_all.py"], "2b: task2/evidence/*_result.txt, the evidence cited in DEFECTS.md"),
    ("T2", ["task2/sleep_pipeline.py"], "fixed pipeline, seed 42 (printed)"),
    ("T2", ["task2/ledger.py", "--all", "--jobs", "4"], "2c: task2/ledger_results/*.json, ledger_runs.csv, ledger_table.md (resumable)"),
    ("T2", ["task2/physiology.py"], "2d: task2/physiology/ (dataset facts, per-epoch uV features, per-window test predictions, report)"),
    # one script per ask of the brief (task2/asks/README.md); they read the outputs above, so they run last in T2
    ("T2", ["task2/asks/run_all.py"], "every Task 2 ask answered and checked against the outputs; figures in outputs/asks/"),
    # last, so the PDF typesets the outputs just produced; skipped with a message if pandoc/xelatex are missing
    ("R", ["report/build_report.py"], "REPORT.pdf from report/REPORT.md + task1/WRITEUP.md"),
]

PARTS = ["A1", "A2", "A3", "B1", "B2", "ASK", "T2", "R"]


def run(cmd, label):
    t0 = time.time()
    print(f"\n=== [{label}] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"step failed ({r.returncode}): {' '.join(cmd)}")
    print(f"=== done in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=PARTS)
    ap.add_argument("--from", dest="start", choices=PARTS)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="parallel training jobs in B2 (each uses --threads threads)")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    parts = a.only or (PARTS[PARTS.index(a.start):] if a.start else PARTS)
    steps = [(p, s, d) for p, s, d in STEPS if p in parts]
    if a.list:
        for p, s, d in steps:
            print(f"{p:3s} {' '.join(s):45s} {d}")
        return
    t0 = time.time()
    for p, s, d in steps:
        if not (ROOT / s[0]).exists():
            print(f"\n=== [{p}] SKIPPED, not in this checkout: {s[0]} ({d})", flush=True)
            continue
        if s[0] == "task1/b2_run.py":
            s = s + ["--workers", str(a.workers), "--threads", str(a.threads)]
        run(PY + s, p)
    print(f"\nall steps finished in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
