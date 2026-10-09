# EEG × Deep Learning research assignment

Two tasks, both on public PhysioNet data.

- Task 1 is motor imagery (EEGBCI, subjects 70-109). It looks at what is actually in the signal, and then at what EEGNet and a transformer learn from it.
- Task 2 is sleep staging (Sleep-EDF Expanded, Sleep Cassette, subjects 0-14). The given pipeline looks like it works. I find its defects and measure what each one did to the reported numbers.

The write-up is [REPORT.pdf](REPORT.pdf) (6 pages, both tasks). All of it comes from code in this repo run on the real data, and `python main.py` regenerates everything without manual steps.

## Deliverables

| The brief asks for | Where |
|---|---|
| `main.py`, pinned `requirements.txt` | [main.py](main.py), [requirements.txt](requirements.txt) |
| Predictions committed before their results | [PREDICTIONS.md](PREDICTIONS.md) has A3, B2 and Task 2c, with each outcome added afterwards. Git history shows the order. |
| Every non-trivial choice | [DECISIONS.md](DECISIONS.md), D1-D27 for Task 1 and T2-1 to T2-10 for Task 2 |
| Task 2 defects (2b) | [DEFECTS.md](DEFECTS.md) |
| Report, at most 6 pages | [REPORT.pdf](REPORT.pdf), built from [report/REPORT.md](report/REPORT.md) |
| Task 1 audit.csv, ERD figures, per-subject table | [audit.csv](audit.csv), `outputs/erd/`, [outputs/a3_per_subject_table.csv](outputs/a3_per_subject_table.csv) |
| Task 1 B2 results with every seed | [outputs/partb/b2_all_runs.csv](outputs/partb/b2_all_runs.csv), [outputs/partb/B2_RESULTS.md](outputs/partb/B2_RESULTS.md) |
| Task 1 write-up (a) and (b) | [task1/WRITEUP.md](task1/WRITEUP.md), also inside REPORT.pdf |
| Task 2 fixed script, one commit per defect | [task2/sleep_pipeline.py](task2/sleep_pipeline.py). The given file, unchanged, is [task2/baseline/sleep_pipeline_given.py](task2/baseline/sleep_pipeline_given.py) |
| Task 2 impact-ledger table and the script behind it | [task2/ledger_table.md](task2/ledger_table.md), [task2/ledger.py](task2/ledger.py) |
| Task 2 answers to 2d and 2e | [task2/PHYSIOLOGY.md](task2/PHYSIOLOGY.md), [task2/REPORTING_NOTE.md](task2/REPORTING_NOTE.md) |

## One script per ask

Each item in the brief also has its own script. It prints the ask, the method and the answer, and checks its numbers against the saved outputs. There is a switch at the top of each one if you want to re-run part of it live.

- [task1/asks/](task1/asks/README.md) has 21 scripts for Task 1 (A1-A3, Part B folds, B1, B2a-f, per-subject table, write-up a and b).
- [task2/asks/](task2/asks/README.md) has 5 scripts for Task 2 (2a-2e).

```
python task1/asks/A2c_lateralisation_index.py
python task2/asks/T2c_impact_ledger.py
```

## Running it

It runs on Python 3.14, CPU only, with pinned versions.

```
python -m venv .venv
.venv\Scripts\activate            (Windows; source .venv/bin/activate elsewhere)
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Then run whichever part you need.

```
python main.py --list             every step, in order, with what it writes
python main.py                    everything
python main.py --only A1 A2 A3    Task 1 Part A
python main.py --only B2          Task 1 Part B experiments (resumable)
python main.py --only T2          Task 2
python main.py --only ASK         the Task 1 ask scripts
```

Steps run in the order A1, A2, A3, B1, B2, ASK, T2, R. A step reuses whatever is already on disk, so an interrupted run picks up where it stopped. MNE handles the downloads (EEGBCI goes into `data/` and Sleep-EDF into `data/mne_sleep/`).

On an 8-core CPU, Task 1 takes about 12-14 hours end to end, and most of that is the B2 grid (156 trained models). Task 2's ledger is 47 trained models at about 15-20 minutes each, run 4 at a time. PhysioNet throttles each connection, so the Sleep-EDF download (about 1.4 GB) is slow. `task2/prefetch.py` downloads it in parallel.

`report/build_report.py` rebuilds REPORT.pdf. It needs pandoc and xelatex, and without them it leaves the committed PDF as it is.

All seeds are fixed and listed in the scripts (B2 uses seeds 0-2 and Task 2 uses 42-44). At the same seed and CPU thread count, runs are bit-identical. A different thread count changes the floating-point summation order, so training results can shift slightly (DECISIONS T2-9). The B2 runs use `--threads 3`.

## Layout

```
main.py, requirements.txt         run everything; pinned versions
PREDICTIONS.md, DECISIONS.md, DEFECTS.md, REPORT.pdf
audit.csv                         Task 1 A1, one row per subject
task1/                            data.py (loading, filtering, epochs), erd.py (A2), confound*.py (A3), models.py (EEGNet,
                                  transformer), train.py (cross-subject harness), b2_*.py (B2 experiments)
task1/asks/                       one script per ask
task2/                            sleep_pipeline.py (fixed), ledger.py (2c), physiology.py (2d), evidence/ (2b), baseline/ (2a)
task2/asks/                       one script per ask
outputs/                          every table and figure (Task 1 at the top level and in partb/, ask figures in asks/)
report/                           REPORT.md source and its build script
```
