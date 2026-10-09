---
title: "EEG x Deep Learning: what is in the signal, and what the networks learn from it"
author: "Vishnu KN (Keyen)"
date: "October 2026"
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{tbp}
---

All numbers come from `python main.py` (fixed seeds, runs in the CSVs named in brackets). Choices are justified in `DECISIONS.md` (D-numbers) and predictions are scored in `PREDICTIONS.md`.

# Task 1: Motor imagery, subjects 70-109

## A1 Data audit
Runs 4, 8 and 12 are imagined left vs right fist (T1 = left, T2 = right), verified in the `mne.datasets.eegbci` docstring and on the PhysioNet page (D1). Subjects 88, 92 and 100 carry a 128 Hz header that is correct (no 0.8x alpha shift, consistent protocol), so they are kept and resampled to 160 Hz (D2). Twenty trials whose -1.5 to 4.0 s window overruns the recording are dropped (D3), leaving 897 left and 896 right trials (`audit.csv`).

## A2 Physiological ground truth
The reference is undocumented and a lateral one would bias C3 and C4 unequally, so I use a local Laplacian (C3 or C4 minus the mean of its 4 nearest electrodes, D4, D10), whose zero-sum weights cancel it. Morlet power relative to a pre-cue baseline (-1.0 to -0.1 s, D5) is averaged over 0.5-4.0 s, and LI = (ipsi - contra)/(|ipsi| + |contra|) per band is tested with a one-sided label permutation (10 000 permutations, Bonferroni over mu and beta), counting only a contralateral desynchronisation (D6). **14 of 40 subjects show lateralised ERD** (`a2_lateralisation.csv`, mean LI 0.26 mu, 0.32 beta), and CAR and raw referencing give smaller LIs (0.19/0.29 and 0.15/0.21).

![Laplacian-referenced time-frequency power relative to baseline at C3 and C4, for left and right imagery (blue = desynchronisation). Top three rows: lateralised ERD present (S72, S102, S93). Bottom three: absent (S99, S87, S101).](outputs/a2_examples_3present_3absent.png){width=42%}

## A3 Confound audit
The cue is a target on one side of the screen, so each non-motor route got a restricted decoder (D12-D17) with a pre-cue twin as a leak check. The chance threshold is an exact one-sided binomial at 5% (63-67% within subject, 52.0% pooled over 1793 trials).

| decoder | within-subject mean | subjects above threshold (of 40) | pooled cross-subject | pre-cue twin |
|----------|-----|-----|-----|-----|
| F1 frontal, 0-0.5 s (eye) | 66.4% | 25 | **71.9%** | 48.2% |
| F2 frontal, 0.5-4 s | 63.5% | 18 | **75.3%** | - |
| O1 occipital, 0-0.5 s (visual) | 56.5% | 14 | 59.3% | 49.7% |
| G1 temporal 30-40 Hz (muscle) | 50.8% | 1 | 49.6% | 49.3% |
| M1 sensorimotor mu+beta (motor) | 54.9% | 8 | 54.6% | 52.2%\* |
| R2 previous 1-3 labels (no EEG) | 70.8% | 31 | **72.6%** | - |

\*Marginal (permutation p 0.02-0.03 among 10 uncorrected pre-cue twins), not interpreted.

The prediction (motor 56-62%, order at chance) was wrong. The eye route dominates, labels alternate (z = -17.6), and a zero-phase filter leaked into the pre-cue window until a causal filter removed it (D18). Per-subject results are in `a3_per_subject_table.csv`.

## B1 Models and evaluation
Evaluation is cross-subject only (8 fixed subject-wise folds of 5, fold f+1 for early stopping, D22). B2 tests on folds 0-3 (20 subjects, 913 trials, threshold 52.8%), seeds 0-2 (D24). EEGNet-8,2 (2962 parameters) has both 0.5 s kernels rescaled from 128 to 160 Hz (64 to 80 and 16 to 20 samples, D20). The transformer (d_model 16, 2 layers, 2 heads) shares EEGNet's temporal filter and uses time-patch tokens (6834 parameters) or channel tokens with (7250) or without (6226) a learned electrode identity (D21).

| model | 5 subj. | 10 | 20 | 30 | 30, no eye route |
|----------|----|----|----|----|------|
| EEGNet | 72.1 +- 0.5 | 75.0 +- 1.3 | 76.2 +- 1.3 | 76.9 +- 1.0 | 50.5 +- 2.9 |
| Transformer, time-patch | 54.8 +- 3.1 | 64.3 +- 3.0 | 66.8 +- 4.1 | 71.6 +- 3.2 | 49.1 +- 1.3 |
| Transformer, channel + id | | | | 70.2 +- 1.2 | |
| Transformer, channel, no id | | | | 57.5 +- 1.6 | |

Test accuracy %, mean +- SD over 3 seeds (`outputs/partb/b2_all_runs.csv`). The other B2 experiments are in `outputs/partb/B2_RESULTS.md`.

<!-- include: task1/WRITEUP.md -->

# Task 2: Sleep staging pipeline

## 2a Baseline
The given script, unchanged with seed 42 (`task2/baseline/2a_baseline_seed42.txt`), scores **accuracy 0.847, macro-F1 0.628, kappa 0.693** (N1 F1 0.207), with 70% of test windows Wake.

## 2b Defects
Ten defects, one commit each (`DEFECTS.md`, printouts in `task2/evidence/`, decisions T2-1 to T2-10).

| # | defect | evidence | fix |
|-|------|------------|------|
| 1 | split by recording | 5/5 test subjects' other night in training (seed 42) | split by subject, 9/3/3 |
| 2 | epoch chosen on test | test F1 0.58-0.67 over 12 epochs, max reported | select on validation |
| 3 | `batch_first=False` | own neighbour changes output by 0, other window 9e-4 | `batch_first=True` |
| 4 | position after attention | shuffled neighbours change output by 3e-7 | encode before |
| 5 | label = last epoch | labels differ in 8.7% of windows, 71% of N1 | label centre epoch |
| 6 | 500 "uV" on volts | 0 of 80070 epochs rejected, units-only fix rejects 54% (EOG) | uV, EEG only |
| 7 | rejected epochs dropped | windows span time gaps | mask, skip windows |
| 8 | per-epoch z-score | 99.6% of N3 epochs >75 uV, all std = 1 after | per-recording z |
| 9 | unscored as Wake | 905 "?" + 35 movement epochs became Wake | label -1, context only |
| 10 | ~23 h recordings | 68% of epochs lights-on Wake, "always Wake" = 68.7% | crop to sleep ± 30 min |

Fixed pipeline, mean ± SD over seeds 42-44: **accuracy 0.827 ± 0.012, macro-F1 0.788 ± 0.015, kappa 0.769 ± 0.020**.

## 2c Impact ledger
Each defect is re-introduced alone into the fixed script (`task2/ledger.py`) on seeds 42-44, and Δ is defective minus fixed on the same seed (mean ± SD, `task2/ledger_table.md`, † = seeds 42 and 43 only).

| defect | Δ accuracy | Δ macro-F1 | Δ kappa | predicted (F1) | outcome |
|------|----|----|----|------|-----|
| 10 no crop | **+0.098 ± 0.017** | +0.013 ± 0.018 | **+0.089 ± 0.028** | acc up, F1 ~0 | right |
| 3 batch attention | -0.026 ± 0.022 | **-0.034 ± 0.020** | -0.035 ± 0.027 | down | right (N1 -0.089) |
| 8 per-epoch z | -0.012 ± 0.010 | -0.025 ± 0.019 | -0.020 ± 0.013 | down, N3 most | N1 most |
| 9 unscored as W | -0.015 ± 0.007 | -0.017 ± 0.007 | -0.020 ± 0.006 | ~0 | larger |
| 2 select on test | +0.010 ± 0.009 | +0.016 ± 0.014 | +0.015 ± 0.013 | up | right |
| 1 recording split | -0.007 ± 0.023 | -0.012 ± 0.008 | -0.014 ± 0.036 | up | **wrong sign** |
| 4 position after | -0.003 ± 0.021 | -0.006 ± 0.023 | -0.003 ± 0.027 | small down | noise |
| 5 last-epoch label | +0.009 ± 0.011 | +0.007 ± 0.015 | +0.013 ± 0.014 | down | **noise** |
| 6 units (no rejection) | +0.006 ± 0.010 | +0.008 ± 0.014 | +0.009 ± 0.016 | ~0 | right (noise) |
| 7 deletion | +0.011 ± 0.019 | +0.008 ± 0.015 | +0.014 ± 0.024 | ~0 | right (noise) |
| 6N naive units fix † | -0.028 ± 0.004 | -0.049 ± 0.002 | -0.069 ± 0.000 | down | right |
| 6N + 7 † | -0.016 ± 0.015 | -0.021 ± 0.026 | -0.029 ± 0.024 | worse than 6N | **flips, less bad** |
| 3 + 4 | -0.037 ± 0.016 | -0.046 ± 0.031 | -0.048 ± 0.024 | = 3 | right |
| 1 + 2 | +0.015 ± 0.033 | +0.012 ± 0.026 | +0.014 ± 0.053 | more than either | not additive |
| 9 + 10 † | +0.097 ± 0.012 | +0.005 ± 0.017 | +0.081 ± 0.021 | 9 grows | **9 vanishes** |
| all ten † | +0.027 ± 0.031 | -0.147 ± 0.032 | -0.059 ± 0.054 | acc up, F1 -0.08..-0.20 | right |

**Noise floor.** Runs are bit-identical for the same code, seed and thread count, yet changing 11 of ~17,000 training windows (one recording on seed 44) moves macro-F1 by +0.024, so 3 seeds cannot resolve effects under about ±0.02 macro-F1.

**Ranked by how much each would have misled the report**, defect 10 is first, adding ~10 points of accuracy and 9 of kappa from lights-on Wake. Defect 3 is second because the context model never ran yet the CNN still scores well, and 8 third because it removes the amplitude criterion of N3 and N1. Defects 9 and 2 are consistent but small, and 1, 4, 5, 6 and 7 sit in the noise floor.

**Flags.** Defect 4 vanishes under 3, and 7 vanishes under 6 but *flips* with the naive units fix, because EOG rejection of waking eye movements leaves masking 11k training windows against 17.3k for deletion (Wake 2694 to 381). Defect 9 vanishes once Wake is uncropped (10). Defect 1 has the wrong sign because the seed also draws a harder set of test subjects, and with 3 test subjects who is tested matters more than leakage.

## 2d Physiology: the worst stage
N1 is the worst stage (F1 0.511 ± 0.017, others 0.83-0.89), with recall 0.61 but precision 0.44, because the model uses it for epochs it cannot place. On EEG and EOG, N1 overlaps REM (median Fpz-Cz amplitude 78 vs 79 uV), and only muscle tone separates them (EMG 2.68 vs 0.69 uV), which the pipeline never uses. The expected confusion is N3 scored as N2 (17.5% of N3), because the 20% slow-wave rule thresholds a continuum. The unexpected one is W with REM (5.1% and 3.3%), which human scorers separate by chin atonia (EMG 3.23 vs 0.69 uV), a channel the pipeline drops. The full analysis is in `task2/PHYSIOLOGY.md`.

## 2e Note for a non-technical reader
Our first sleep-staging program reported 85% accuracy. The corrected version scores about 83%, and the lower figure is the honest one.

The first test was too easy. Each recording runs for almost a whole day, and for two-thirds of it the person is awake. A program that always answered "awake" would score 69%. The corrected test keeps only the night, plus half an hour either side.

The old test had two other flaws. It included people the program had already studied, since their other night was in its training material. And its best version was picked using the test itself. Real patients will be new to it.

A fairer score gives every sleep stage equal weight, so easy wakefulness earns nothing extra. On that score the corrected program improved from 0.63 to 0.79 (out of 1). The headline dropped because the test got harder, while the program got better at telling sleep stages apart.
