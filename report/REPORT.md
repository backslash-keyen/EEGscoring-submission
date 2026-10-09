---
title: "EEG x Deep Learning: what is in the signal, and what the networks learn from it"
author: "Vishnu KN (Keyen)"
date: "October 2026"
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{tbp}
---

Every number below is produced by `python main.py` (seeds fixed, every run reported in the CSVs named in brackets). Rationale for each choice is in `DECISIONS.md` (D-numbers), and predictions with their scorecards in `PREDICTIONS.md`.

# Task 1: Motor imagery, subjects 70-109

## A1 Data audit
Runs 4, 8 and 12 are imagined left vs right fist (T1 = left, T2 = right). Runs 3/7/11 are execution and 6/10/14 imagine fists vs feet, so the event codes mean different things per run (D1, verified in the `mne.datasets.eegbci` docstring and on the PhysioNet page). `audit.csv` has one row per subject. Three subjects (88, 92, 100) carry a 128 Hz header. They were kept and resampled to 160 Hz because their alpha peak does not show the 0.8x shift a mislabelled 160 Hz file would give, and their protocol differs consistently (19 trials per run, 5.12 s cues), so the header is right (D2). Twenty trials whose -1.5 to 4.0 s window runs past the end of the recording are dropped and listed per subject (D3), leaving 897 left and 896 right trials.

## A2 Physiological ground truth
The recording reference is undocumented. Every channel then carries the same unknown reference signal, and a lateral reference (e.g. a mastoid) would bias C3 and C4 unequally, so a raw C3-vs-C4 difference partly measures the reference. A local Laplacian (C3 or C4 minus the mean of its 4 nearest electrodes, D4, D10) has weights summing to zero, so any component common to the neighbourhood (the reference included) cancels, and the comparison becomes local. Power is Morlet time-frequency power at C3, C4 and their neighbours, relative to a pooled pre-cue baseline (-1.0 to -0.1 s, D5), averaged over 0.5-4.0 s. The lateralisation index LI = (ipsi - contra)/(|ipsi| + |contra|) per band, tested with a one-sided label-permutation test (10 000 permutations, Bonferroni over mu and beta) and labelled present only if the contralateral change is a desynchronisation (D6). **14 of 40 subjects show lateralised ERD** (`a2_lateralisation.csv`; mean LI 0.26 mu, 0.32 beta). CAR and raw referencing give smaller LIs (0.19/0.29 and 0.15/0.21, `a2_all_references.csv`).

![Laplacian-referenced time-frequency power relative to baseline at C3 and C4, for left and right imagery (blue = desynchronisation). Top three rows: lateralised ERD present (S72, S102, S93). Bottom three: absent (S99, S87, S101).](outputs/a2_examples_3present_3absent.png){width=42%}

## A3 Confound audit
The cue is a target on one side of the screen, shown until the subject stops imagining, with rest between trials. Each non-motor route got its own restricted decoder (D12-D17): eye movement (frontal channels, 0-0.5 s and 0.5-4 s), visual response (occipital), muscle (temporal 30-40 Hz), trial order and history (no EEG), and pre-cue twins of every EEG decoder as a leak check. Threshold: exact one-sided binomial at 5% (63-67% within subject, 52.0% pooled over 1793 trials). A permutation null gave 2.7-8.3% false positives against the nominal 5% (`outputs/A3_RESULTS.md`).

| decoder | within-subject mean | subjects above threshold (of 40) | pooled cross-subject | pre-cue twin |
|---|---|---|---|---|
| F1 frontal, 0-0.5 s (eye) | 66.4% | 25 | **71.9%** | 48.2% |
| F2 frontal, 0.5-4 s | 63.5% | 18 | **75.3%** | - |
| O1 occipital, 0-0.5 s (visual) | 56.5% | 14 | 59.3% | 49.7% |
| G1 temporal 30-40 Hz (muscle) | 50.8% | 1 | 49.6% | 49.3% |
| M1 sensorimotor mu+beta (motor) | 54.9% | 8 | 54.6% | 52.2%\* |
| R2 previous 1-3 labels (no EEG) | 70.8% | 31 | **72.6%** | - |

\*Marginal (permutation p 0.02-0.03 among 10 uncorrected pre-cue twins), not interpreted.

The prediction (eyes and visual decodable but weak, motor 56-62%, order at chance) was wrong on size and on order: the eye route dominates, labels alternate (z = -17.6), and the zero-phase filter first made the pre-cue window look decodable until a causal filter removed the leak (D18). The per-subject table with LI, p, label and every decoder accuracy with its threshold is `a3_per_subject_table.csv`.

## B1 Models and evaluation
Cross-subject only: 8 fixed subject-wise folds of 5, test = fold f, validation (early stopping only) = fold f+1, normalisation from training subjects only (D22). B2 uses folds 0-3 as test (20 subjects, 913 trials, threshold 52.8%), seeds 0-2 everywhere (D24). EEGNet-8,2 (2962 parameters) is the 128 Hz design with both 0.5 s kernels rescaled to 160 Hz (64 to 80 and 16 to 20 samples), so the lowest shapeable frequency stays at 2 Hz (D20). The transformer (d_model 16, 2 layers, 2 heads) shares EEGNet's 0.5 s temporal filter and changes only the token axis: time-patch tokens (6834 parameters), channel tokens with (7250) or without (6226) a learned electrode identity, all within 3x EEGNet (D21).

| model | 5 subj. | 10 | 20 | 30 | 30, eye route removed |
|---|---|---|---|---|---|
| EEGNet | 72.1 +- 0.5 | 75.0 +- 1.3 | 76.2 +- 1.3 | 76.9 +- 1.0 | 50.5 +- 2.9 |
| Transformer, time-patch | 54.8 +- 3.1 | 64.3 +- 3.0 | 66.8 +- 4.1 | 71.6 +- 3.2 | 49.1 +- 1.3 |
| Transformer, channel + identity | | | | 70.2 +- 1.2 | |
| Transformer, channel, no identity | | | | 57.5 +- 1.6 | |

Test accuracy %, mean +- SD over 3 seeds. Every fold x seed run is in `outputs/partb/b2_all_runs.csv`. Displacement, attribution, faithfulness and the intervention are in `outputs/partb/B2_RESULTS.md` and figures `b2_*.png`.

<!-- include: task1/WRITEUP.md -->

# Task 2: Sleep staging pipeline

*Status: in progress. This section is filled from `DEFECTS.md` and the impact ledger once Task 2 is complete.*

## 2a Baseline
Run unchanged with seed 42 (`task2/baseline/2a_baseline_seed42.txt`): accuracy 0.847, macro-F1 0.628, kappa 0.693, with per-class F1 W 0.945, N1 0.207, N2 0.703, N3 0.642, REM 0.641.

## 2b Defects
TODO: one row per defect (what, evidence, distortion, fix), from `DEFECTS.md`.

## 2c Impact ledger
TODO: defect, delta per metric (mean +- spread over at least 3 seeds), predicted direction, mechanism, rank, and interactions.

## 2d Physiology: worst stage
TODO.

## 2e Note for a non-technical reader
TODO (150 words).
