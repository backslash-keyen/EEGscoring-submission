# EEG × Deep Learning research assignment

Two tasks on public PhysioNet data:

- **Task 1:** motor imagery (EEGBCI, subjects 70-109). What is in the signal, and what EEGNet and a transformer learn from it.
- **Task 2:** sleep staging (Sleep-EDF Expanded, Sleep Cassette, subjects 0-14). A pipeline that looks like it works, its defects, and what each one did to the reported numbers.

Every number and figure comes from code in this repository run on the real data, and `python main.py` reproduces all of it with no manual steps. The write-up is **[REPORT.pdf](REPORT.pdf)** (6 pages, both tasks).

## Deliverables

| The brief asks for | Where |
|---|---|
| `main.py`, pinned `requirements.txt` | [main.py](main.py), [requirements.txt](requirements.txt) |
| Predictions committed before their results | [PREDICTIONS.md](PREDICTIONS.md): A3, B2 and Task 2c, each with its outcome appended afterwards. Git history shows the order. |
| Every non-trivial choice | [DECISIONS.md](DECISIONS.md): D1-D27 for Task 1, T2-1 to T2-10 for Task 2 |
| Task 2 defects (2b) | [DEFECTS.md](DEFECTS.md) |
| Report, at most 6 pages | [REPORT.pdf](REPORT.pdf), built from [report/REPORT.md](report/REPORT.md) |
| Task 1: audit.csv, ERD figures, per-subject table | [audit.csv](audit.csv), `outputs/erd/`, [outputs/a3_per_subject_table.csv](outputs/a3_per_subject_table.csv) |
| Task 1: all B2 results with every seed | [outputs/partb/b2_all_runs.csv](outputs/partb/b2_all_runs.csv), [outputs/partb/B2_RESULTS.md](outputs/partb/B2_RESULTS.md) |
| Task 1: write-up (a) and (b) | [task1/WRITEUP.md](task1/WRITEUP.md), also inside REPORT.pdf |
| Task 2: fixed script, one commit per defect | [task2/sleep_pipeline.py](task2/sleep_pipeline.py); the given file unchanged: [task2/baseline/sleep_pipeline_given.py](task2/baseline/sleep_pipeline_given.py) |
| Task 2: impact-ledger table and the script that produces it | [task2/ledger_table.md](task2/ledger_table.md), [task2/ledger.py](task2/ledger.py) |
| Task 2: answers to 2d and 2e | [task2/PHYSIOLOGY.md](task2/PHYSIOLOGY.md), [task2/REPORTING_NOTE.md](task2/REPORTING_NOTE.md) |

## One script per ask

Every item of the brief has a script that prints the ask, the method and the answer, and checks its numbers against the saved outputs. Each has a switch at the top for re-running part of it live.

- **Task 1:** [task1/asks/](task1/asks/README.md), 21 scripts (A1-A3, Part B folds, B1, B2a-f, per-subject table, write-up a and b)
- **Task 2:** [task2/asks/](task2/asks/README.md), 5 scripts (2a-2e)

```
python task1/asks/A2c_lateralisation_index.py
python task2/asks/T2c_impact_ledger.py
```

## Running it

Setup (Python 3.14, CPU only; versions pinned):

```
python -m venv .venv
.venv\Scripts\activate            (Windows; source .venv/bin/activate elsewhere)
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Then:

```
python main.py --list             every step, in order, with what it writes
python main.py                    everything
python main.py --only A1 A2 A3    Task 1 Part A
python main.py --only B2          Task 1 Part B experiments (resumable)
python main.py --only T2          Task 2
python main.py --only ASK         the Task 1 ask scripts
```

Steps run in order (A1, A2, A3, B1, B2, ASK, T2, R). Each step reuses what is already on disk, so an interrupted run continues where it stopped. Data downloads through MNE: EEGBCI into `data/`, Sleep-EDF into `data/mne_sleep/`.

Timing on an 8-core CPU: Task 1 takes about 12-14 hours end to end, most of it the B2 grid (156 trained models). Task 2's ledger is 47 trained models of about 15-20 minutes each, run 4 at a time. The Sleep-EDF download (about 1.4 GB) is slow because PhysioNet throttles each connection; `task2/prefetch.py` downloads it in parallel.

`report/build_report.py` rebuilds REPORT.pdf (needs pandoc and xelatex; without them it keeps the committed PDF). `python main.py --matlab` also runs the MATLAB twins of A1-A3; no Python number depends on them.

**Seeds and reproducibility.** All seeds are fixed and listed in the scripts (B2: seeds 0-2; Task 2: 42-44). Runs are bit-identical at the same seed and CPU thread count. A different thread count changes floating-point summation order, so training results can shift slightly (DECISIONS T2-9; B2 runs use `--threads 3`).

## Layout

```
main.py, requirements.txt         run everything; pinned versions
PREDICTIONS.md, DECISIONS.md, DEFECTS.md, REPORT.pdf
audit.csv                         Task 1 A1, one row per subject
task1/                            data.py (loading, filtering, epochs), erd.py (A2), confound*.py (A3), models.py (EEGNet,
                                  transformer), train.py (cross-subject harness), b2_*.py (B2 experiments), *.m/*.mlx (MATLAB twins)
task1/asks/                       one script per ask
task2/                            sleep_pipeline.py (fixed), ledger.py (2c), physiology.py (2d), evidence/ (2b), baseline/ (2a)
task2/asks/                       one script per ask
outputs/                          every table and figure (Task 1 at the top level and in partb/, ask figures in asks/)
report/                           REPORT.md source and its build script
```
