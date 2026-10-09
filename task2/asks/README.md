# One script per ask (Task 2)

Each item of the brief has its own script, set up the same way as the ones in `task1/asks/`. Open one in VS Code and press Run, or call `python task2/asks/<name>.py`. It prints the ask, the method and the answer with its numbers, then opens its figures. Closing the figure windows ends it.

Training one sleep-staging model takes about 15-20 minutes on a CPU and the ledger has 47 of them. The scripts therefore read the saved runs (`task2/ledger_results/*.json`) and recompute every table from those. Wherever a script checks a recomputed number against the committed output, it prints `MATCH`. Evidence scripts that finish in seconds get re-run, and their printout is compared with the saved one. A switch at the top of each script turns on a live run (`RUN`, `DEFECT`, `FEATURE`).

For their live parts the scripts need the Sleep-EDF files (`$MNE_DATA`, default `data/mne_sleep`), which `python main.py --only T2` downloads. Without the files they still print the answer from the committed outputs and say they skipped the live part. `python task2/asks/run_all.py` runs them all without opening windows and saves the figures to `outputs/asks/`.

| Ask in the brief | Script | Live part |
|---|---|---|
| 2a run the given script unchanged with the given seed, record its output | `T2a_baseline.py` | git confirms the kept copy is the given file byte for byte. `RUN = True` re-runs it (~20 min) |
| 2b each defect (what, evidence, distortion, fix), one defect per commit | `T2b_defects.py` | lists the fix commits from git, shows each real diff, re-runs the fast evidence scripts (defects 1, 3, 4, 5, 9) |
| 2c impact ledger with delta per metric over 3 seeds, predicted direction, mechanism, ranking and flags | `T2c_impact_ledger.py` | recomputes every delta and checks the predictions were committed first. `RUN = ("D3", 42)` re-trains one variant (~15 min) |
| 2d worst stage from scoring rules and recorded signals, plus one expected and one unexpected confusion | `T2d_physiology.py` | channels and rates read from an EDF header, then per-stage F1, confusion and uV features recomputed |
| 2e 150-word note for a non-technical reader | `T2e_reporting_note.py` | word count, and every number traced to its source |
