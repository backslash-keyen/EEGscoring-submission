# One script per ask (Task 2)

Each item of the brief has its own script, in the same format as `task1/asks/`. Open it in VS Code and press Run (or `python task2/asks/<name>.py`). It prints the ask, the method and the answer with its numbers, then opens its figures. Close the figure windows to end it.

How the scripts work. Training one sleep-staging model takes about 15-20 minutes on a CPU, and the ledger is 47 of them, so the scripts read the saved runs (`task2/ledger_results/*.json`) and recompute every table from them. Each prints `MATCH` where it checks a recomputed number against the committed output. Evidence scripts that take seconds are re-run and their printout compared with the saved one. Each script has a switch at the top for a live run (`RUN`, `DEFECT`, `FEATURE`).

The scripts need the Sleep-EDF files for their live parts (`$MNE_DATA`, default `data/mne_sleep`; `python main.py --only T2` downloads them). Without them they still print the answer from the committed outputs and say they skipped the live part. `python task2/asks/run_all.py` runs them all without opening windows and saves figures to `outputs/asks/`.

| Ask in the brief | Script | Live part |
|---|---|---|
| 2a run the given script unchanged with the given seed, record its output | `T2a_baseline.py` | git proves the kept copy is the given file byte for byte; `RUN = True` re-runs it (~20 min) |
| 2b each defect: what, evidence, distortion, fix; one defect per commit | `T2b_defects.py` | lists the fix commits from git, shows each real diff, re-runs the fast evidence scripts (defects 1, 3, 4, 5, 9) |
| 2c impact ledger: delta per metric over 3 seeds, predicted direction, mechanism, ranking, flags | `T2c_impact_ledger.py` | recomputes every delta; checks the predictions were committed first; `RUN = ("D3", 42)` re-trains one variant (~15 min) |
| 2d worst stage from scoring rules and recorded signals; one expected, one unexpected confusion | `T2d_physiology.py` | channels and rates read from an EDF header; per-stage F1, confusion and uV features recomputed |
| 2e 150-word note for a non-technical reader | `T2e_reporting_note.py` | word count; every number traced to its source |
