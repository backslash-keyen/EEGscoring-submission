---
title: "EEG x Deep Learning research assignment"
author: "Vishnu KN"
date: "October 2026"
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{tbp}
---

All numbers come from `python main.py` (fixed seeds). Every run is in the files named in brackets. Choices are in `DECISIONS.md` (D-numbers), predictions and their scores in `PREDICTIONS.md`.

# Task 1: Motor imagery (EEGBCI, subjects 70-109)

## A1 Data audit (`audit.csv`)

| | |
|--|----------|
| Runs | 4, 8, 12: imagined left vs right fist (T1 = left, T2 = right). Verified: MNE `eegbci` docstring, PhysioNet page (D1) |
| Anomalies | S88, S92, S100 have 128 Hz headers. Header is correct (no 0.8x alpha shift), so kept and resampled to 160 Hz (D2) |
| Dropped | 20 trials whose -1.5 to 4.0 s window runs past the recording, listed per subject (D3) |
| Kept | 897 left, 896 right trials |

## A2 Physiological ground truth (`a2_lateralisation.csv`, `outputs/erd/`)

| | |
|--|----------|
| Reference | Undocumented, so a lateral reference would bias C3 vs C4. Local Laplacian (C3/C4 minus mean of 4 nearest electrodes) cancels it (D4, D10) |
| Power | Morlet, baseline -1.0 to -0.1 s, window 0.5-4.0 s (D5) |
| Index | LI = (ipsi - contra)/(\|ipsi\| + \|contra\|), mu and beta |
| Test | One-sided label permutation, 10 000 permutations, Bonferroni over 2 bands. Present only if contralateral power drops (D6) |
| Result | **Lateralised ERD present in 14 of 40 subjects.** Mean LI 0.26 (mu), 0.32 (beta). CAR 0.19/0.29, raw 0.15/0.21 |

![ERD at C3 and C4 relative to baseline (blue = desynchronisation). Top three: present (S72, S102, S93). Bottom three: absent (S99, S87, S101).](outputs/a2_examples_3present_3absent.png){width=32%}

## A3 Confound audit (`a3_summary.csv`, `a3_per_subject_table.csv`)

One restricted decoder per non-motor route, each with a pre-cue twin (D12-D17). Threshold: exact one-sided binomial at 5% (52.0% pooled over 1793 trials, 63-67% within subject).

| Decoder (route) | Within subject | Above threshold | Pooled | Pre-cue |
|------|----|----|----|----|
| F1 frontal 0-0.5 s (eye) | 66.4% | 25/40 | **71.9%** | 48.2% |
| F2 frontal 0.5-4 s | 63.5% | 18/40 | **75.3%** | - |
| O1 occipital 0-0.5 s (visual) | 56.5% | 14/40 | 59.3% | 49.7% |
| G1 temporal 30-40 Hz (muscle) | 50.8% | 1/40 | 49.6% | 49.3% |
| M1 sensorimotor mu+beta (motor) | 54.9% | 8/40 | 54.6% | 52.2%\* |
| R2 previous 1-3 labels (no EEG) | 70.8% | 31/40 | **72.6%** | - |

\*Marginal (permutation p 0.02-0.03 among 10 uncorrected twins). Label order alternates (z = -17.6). A zero-phase filter leaked into the pre-cue window, so all results use a causal filter (D18). Prediction (motor 56-62%, order at chance): wrong.

## B1 Models and evaluation

| Model | Parameters | Details |
|-------|--|-----------|
| EEGNet-8,2 | 2962 | 0.5 s kernels rescaled 128 to 160 Hz: 64 to 80 and 16 to 20 samples (D20) |
| Transformer, time-patch | 6834 | d_model 16, 2 layers, 2 heads, EEGNet temporal filter (D21) |
| Transformer, channel + identity | 7250 | learned electrode identity |
| Transformer, channel, no identity | 6226 | |

Cross-subject only: 8 fixed subject-wise folds of 5, fold f+1 for early stopping (D22). B2 tests on folds 0-3 (20 subjects, 913 trials, threshold 52.8%), seeds 0-2 (D24).

## B2 Experiments (`outputs/partb/b2_all_runs.csv`, `B2_RESULTS.md`)

| Accuracy %, mean ± SD (3 seeds) | 5 subj. | 10 | 20 | 30 | 30, eye route removed |
|--------|---|---|---|---|----|
| EEGNet | 72.1 ± 0.5 | 75.0 ± 1.3 | 76.2 ± 1.3 | 76.9 ± 1.0 | 50.5 ± 2.9 |
| Transformer, time-patch | 54.8 ± 3.1 | 64.3 ± 3.0 | 66.8 ± 4.1 | 71.6 ± 3.2 | 49.1 ± 1.3 |
| Transformer, channel + identity | | | | 70.2 ± 1.2 | |
| Transformer, channel, no identity | | | | 57.5 ± 1.6 | |

| Experiment | Result |
|------|------------|
| Displacement (test time, spline interpolation) | < 1 point lost at 10 mm, ≤ 1.5 at 20 mm. Time-patch transformer drops most |
| Faithfulness (delete half the tokens) | Most-attended 71.4% to 57.7%, gradient x input 64.1%, random 65.6% |
| What it relies on | F7, F8, Ft7, Ft8, Af8 (horizontal EOG sites), the A3 eye route |
| Link to Part A (Spearman, per test subject) | EEGNet vs A2 LI 0.72, vs A3 frontal decoder 0.53. LI vs frontal -0.02 |
| Confound removed (Laplacian, 21 Fc/C/Cp channels, D25) | EEGNet 50.5%, transformer 49.1%: chance |
| Intervention (random 0-15 mm slides in training, D26) | No effect: 77.7% vs 76.9% |

<!-- include: task1/WRITEUP.md -->


\newpage

# Task 2: Sleep staging (Sleep-EDF, subjects 0-14)

## 2a Baseline (`task2/baseline/2a_baseline_seed42.txt`)
Given script, unchanged, seed 42: **accuracy 0.847, macro-F1 0.628, kappa 0.693**. N1 F1 0.207. 70% of test windows are Wake.

## 2b Defects (`DEFECTS.md`, `task2/evidence/`, one commit each)

| # | Defect | Evidence | Fix |
|-|------|------------|------|
| 1 | split by recording | 5/5 test subjects' other night in training | split by subject, 9/3/3 |
| 2 | epoch chosen on test | test F1 0.58-0.67 over 12 epochs, max reported | select on validation |
| 3 | `batch_first=False` | own neighbour changes output by 0, other window by 9e-4 | `batch_first=True` |
| 4 | position after attention | shuffled neighbours change output by 3e-7 | encode before |
| 5 | label = last epoch | differs from centre in 8.7% of windows, 71% of N1 | label centre epoch |
| 6 | 500 "uV" on volts | 0 of 80070 rejected, units-only fix rejects 54% (EOG) | uV, EEG only |
| 7 | rejected epochs deleted | windows span time gaps | mask, skip windows |
| 8 | per-epoch z-score | 99.6% of N3 > 75 uV, all std = 1 after | per-recording z |
| 9 | unscored as Wake | 905 "?" + 35 movement epochs | label -1 |
| 10 | ~23 h recordings | 68% lights-on Wake, "always Wake" = 68.7% | crop to sleep ± 30 min |

Fixed pipeline, seeds 42-44: **accuracy 0.827 ± 0.012, macro-F1 0.788 ± 0.015, kappa 0.769 ± 0.020**. Judgement calls: T2-1 to T2-10.

## 2c Impact ledger (`task2/ledger.py`, `task2/ledger_table.md`)
Each defect re-introduced alone into the fixed script. Δ = defective - fixed, same seed, mean ± SD over seeds 42-44 († seeds 42-43 only).

| Defect | Δ accuracy | Δ macro-F1 | Δ kappa | Predicted (F1) | Outcome |
|------|----|----|----|------|-----|
| 10 no crop | **+0.098 ± 0.017** | +0.013 ± 0.018 | **+0.089 ± 0.028** | acc up, F1 ~0 | right |
| 3 batch attention | -0.026 ± 0.022 | **-0.034 ± 0.020** | -0.035 ± 0.027 | down | right |
| 8 per-epoch z | -0.012 ± 0.010 | -0.025 ± 0.019 | -0.020 ± 0.013 | down, N3 most | N1 most |
| 9 unscored as W | -0.015 ± 0.007 | -0.017 ± 0.007 | -0.020 ± 0.006 | ~0 | larger |
| 2 select on test | +0.010 ± 0.009 | +0.016 ± 0.014 | +0.015 ± 0.013 | up | right |
| 1 recording split | -0.007 ± 0.023 | -0.012 ± 0.008 | -0.014 ± 0.036 | up | wrong sign |
| 4 position after | -0.003 ± 0.021 | -0.006 ± 0.023 | -0.003 ± 0.027 | small down | noise |
| 5 last-epoch label | +0.009 ± 0.011 | +0.007 ± 0.015 | +0.013 ± 0.014 | down | noise |
| 6 units | +0.006 ± 0.010 | +0.008 ± 0.014 | +0.009 ± 0.016 | ~0 | right (noise) |
| 7 deletion | +0.011 ± 0.019 | +0.008 ± 0.015 | +0.014 ± 0.024 | ~0 | right (noise) |
| 6N naive units fix † | -0.028 ± 0.004 | -0.049 ± 0.002 | -0.069 ± 0.000 | down | right |
| 6N + 7 † | -0.016 ± 0.015 | -0.021 ± 0.026 | -0.029 ± 0.024 | worse than 6N | flips, less bad |
| 3 + 4 | -0.037 ± 0.016 | -0.046 ± 0.031 | -0.048 ± 0.024 | = 3 | right |
| 1 + 2 | +0.015 ± 0.033 | +0.012 ± 0.026 | +0.014 ± 0.053 | more than either | not additive |
| 9 + 10 † | +0.097 ± 0.012 | +0.005 ± 0.017 | +0.081 ± 0.021 | 9 grows | 9 vanishes |
| all ten † | +0.027 ± 0.031 | -0.147 ± 0.032 | -0.059 ± 0.054 | acc up, F1 -0.08..-0.20 | right |

**Noise floor.** Runs are bit-identical at the same seed and thread count. Changing 11 of ~17,000 training windows moves macro-F1 by 0.024, so effects under ±0.02 are not resolved.

**Ranked by how much each would have misled the report:** 10 (inflates accuracy and kappa through lights-on Wake), 3 (the context model never ran), 8 (removes the N3 amplitude rule), then 9 and 2. 1, 4, 5, 6, 7 are within noise.

**Flags.** 4 vanishes under 3. 7 vanishes under 6 and flips with the naive units fix (masking keeps 11k training windows, deletion 17.3k). 9 vanishes once Wake is uncropped. 1 has the wrong sign because the seed also draws a harder test set.

## 2d Worst stage (`task2/PHYSIOLOGY.md`)

| | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| F1 (3 seeds) | 0.889 | **0.511** | 0.851 | 0.862 | 0.827 |
| Chin EMG median (uV) | 3.23 | 2.68 | 1.98 | 1.48 | 0.69 |

Signals: EEG Fpz-Cz, Pz-Oz and EOG at 100 Hz. Chin EMG only as a 1 Hz envelope, not used by the pipeline.

- Worst: **N1** (precision 0.44, recall 0.61). On EEG and EOG it matches REM (Fpz-Cz amplitude 78 vs 79 uV). Only muscle tone separates them.
- Expected confusion: N3 scored as N2 (17.5%). The 20% slow-wave rule cuts a continuum.
- Unexpected confusion: W and REM (5.1%, 3.3%). Chin EMG separates them, but it is not in the input.

## 2e Note for a non-technical reader
Our first sleep-staging program reported 85% accuracy. The corrected version scores about 83%, and the lower figure is the honest one.

The first test was too easy. Each recording runs for almost a whole day, and for two-thirds of it the person is awake. A program that always answered "awake" would score 69%. The corrected test keeps only the night, plus half an hour either side.

The old test had two other flaws. It included people the program had already studied, since their other night was in its training material. And its best version was picked using the test itself. Real patients will be new to it.

A fairer score gives every sleep stage equal weight, so easy wakefulness earns nothing extra. On that score the corrected program improved from 0.63 to 0.79 (out of 1). The headline dropped because the test got harder, while the program got better at telling sleep stages apart.
