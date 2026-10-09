# One script per ask (Task 1)

Each item of the brief has its own script. Open it in VS Code and press Run (or `python task1/asks/<name>.py`). It prints the ask, the method, and the answer with its numbers, then opens its figures. Close the figure windows to end it.

How the scripts work. Wherever recomputing takes seconds, a script recomputes from the real data with the same pipeline functions `main.py` uses (`task1/*.py`). It then checks the result against the saved output and prints `MATCH`. The 156 B2 training runs take about 12 hours, so the B2 scripts read their saved results instead. `B2a` can retrain any one run live (`RETRAIN = True`, about 8 minutes), and that run reproduces the saved accuracy exactly. Each script has a variable at the top to change before re-running (`SUBJECT`, `REF`, `N`, `SHIFT_MM`, ...).

The scripts need `data/` (created by `python main.py --only A1 A2 A3`). Without it they still print the answer from the committed outputs and say they skipped the live part. `python main.py --only ASK` (or `python task1/asks/run_all.py`) runs them all without opening windows and saves their figures to `outputs/asks/`.

| Ask in the brief | Script | Live part |
|---|---|---|
| A1 runs chosen and verified, audit.csv, every anomaly handled | `A1_data_audit.py` | re-audits one subject from the raw EDF files |
| A2 time-frequency power at C3, C4 and neighbours vs a justified baseline | `A2a_time_frequency_power.py` | Morlet maps at 10 sites for one subject |
| A2 unknown reference: what it means, which spatial filter | `A2b_reference_choice.py` | re-references to 5 electrodes: raw index moves, Laplacian index stays the same |
| A2 lateralisation index mu/beta + single-subject test | `A2c_lateralisation_index.py` | permutation null rebuilt; p equals the saved p |
| A2 present/absent for every subject, 3 shown of each | `A2d_present_absent_labels.py` | label rule re-applied to all 40 subjects |
| A3 list every non-motor route | `A3a_non_motor_routes.py` | decoder definitions read from `confound.py` |
| A3 a test for each route | `A3b_route_tests.py` | all decoders for one subject + pooled F1/M1/R2 over 40 |
| A3 chance threshold for every accuracy, and its derivation | `A3c_chance_thresholds.py` | exact binomial, step by step, + simulation |
| A3 prediction before running | `A3d_prediction.py` | checks in git that the prediction commit precedes the results |
| Part B rule: cross-subject only, fixed subject-wise folds | `B0_cross_subject_folds.py` | rebuilds folds, checks all 48 splits are disjoint |
| B1 EEGNet, every hyperparameter changed for 160 Hz | `B1a_eegnet.py` | layer shapes and parameter counts |
| B1 transformer within 3x EEGNet | `B1b_transformer.py` | three variants, budget check |
| B2 data scaling | `B2a_data_scaling.py` | every run printed; optional live retrain |
| B2 tokenisation | `B2b_tokenisation.py` | shuffle/mirror test: the no-identity model gives identical outputs |
| B2 electrode displacement | `B2c_electrode_displacement.py` | electrode spacing, interpolation matrix, input change |
| B2 what the model relies on | `B2d_what_model_relies_on.py` | (reads attribution and faithfulness results) |
| B2 link to Part A + confound removed | `B2e_link_to_part_a.py` | Spearman recomputed; why the removal keeps focal ERD |
| B2 one intervention | `B2f_intervention.py` | augmentation bank drawn and applied |
| Outputs: per-subject table | `T1_out_per_subject_table.py` | |
| Write-up (a): how much of 70% needs motor imagery | `T1_out_writeup_a_seventy_percent.py` | |
| Write-up (b): B2 mechanisms, contradicted predictions | `T1_out_writeup_b_mechanisms.py` | |
