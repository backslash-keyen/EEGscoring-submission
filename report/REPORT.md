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

## 2a Baseline
Given script, unchanged, seed 42 (`task2/baseline/2a_baseline_seed42.txt`): **accuracy 0.847, macro-F1 0.628, kappa 0.693** (per-class F1 W 0.945, N1 0.207, N2 0.703, N3 0.642, REM 0.641). 70% of its test windows are Wake.

## 2b Defects
Ten defects, one commit each; what, evidence, distortion and fix are in `DEFECTS.md`, evidence printouts in `task2/evidence/`, judgement calls in `DECISIONS.md` (T2-1 to T2-10).

| # | defect | evidence (not only code reading) | fix |
|---|---|---|---|
| 1 | split by recording | 5/5 test subjects' other night in training (seed 42) | split by subject, 9/3/3 |
| 2 | epoch chosen on test | test F1 0.58-0.67 over 12 epochs, max reported | choose on validation subjects |
| 3 | `batch_first=False` | own neighbour moves output by 0, another window by 9e-4 | `batch_first=True` |
| 4 | position added after attention | shuffling neighbours changes output by 3e-7 | encode before attention |
| 5 | label = last epoch, head = centre | labels differ in 8.7% of windows, 71% of N1 | label the centre epoch |
| 6 | 500 "uV" vs data in volts | 0 of 80070 epochs rejected; units-only fix rejects 54% (EOG) | uV, EEG channels only |
| 7 | rejected epochs deleted | windows span time gaps | mask, skip such windows |
| 8 | z-score per epoch | 99.6% of N3 epochs >75 uV; after scaling all std = 1 | z-score per recording |
| 9 | unknown labels -> Wake | 905 "?" + 35 movement epochs became Wake | label -1, context only |
| 10 | ~23 h recordings | 68% of epochs lights-on Wake; "always Wake" = 68.7% | crop to sleep ± 30 min |

Fixed pipeline, mean ± SD over seeds 42-44: **accuracy 0.827 ± 0.012, macro-F1 0.788 ± 0.015, kappa 0.769 ± 0.020**.

## 2c Impact ledger
Each defect re-introduced alone into the fixed script by reversing its edit (`task2/ledger.py`), seeds 42-44 (a seed also draws the subject split); Δ = defective - fixed on the same seed, mean ± SD (`task2/ledger_table.md`, per run `task2/ledger_runs.csv`). Predictions were committed before any run (`PREDICTIONS.md`). Every single defect has 3 seeds; the four combination rows marked † have seeds 42 and 43 only (their seed-44 runs were cut for time; the ALL run ran out of memory).

| defect | Δ accuracy | Δ macro-F1 | Δ kappa | predicted (F1) | outcome |
|---|---|---|---|---|---|
| 10 no crop | **+0.098 ± 0.017** | +0.013 ± 0.018 | **+0.089 ± 0.028** | acc up, F1 ~0 | right |
| 3 batch attention | -0.026 ± 0.022 | **-0.034 ± 0.020** | -0.035 ± 0.027 | down | right (N1 -0.089) |
| 8 per-epoch z | -0.012 ± 0.010 | -0.025 ± 0.019 | -0.020 ± 0.013 | down, N3 most | down, but N1 most |
| 9 unscored -> W | -0.015 ± 0.007 | -0.017 ± 0.007 | -0.020 ± 0.006 | ~0 | larger than predicted |
| 2 select on test | +0.010 ± 0.009 | +0.016 ± 0.014 | +0.015 ± 0.013 | up | right |
| 1 recording split | -0.007 ± 0.023 | -0.012 ± 0.008 | -0.014 ± 0.036 | up | **wrong sign** |
| 4 position after | -0.003 ± 0.021 | -0.006 ± 0.023 | -0.003 ± 0.027 | small down | within noise |
| 5 last-epoch label | +0.009 ± 0.011 | +0.007 ± 0.015 | +0.013 ± 0.014 | down | **within noise** |
| 6 units (no rejection) | +0.006 ± 0.010 | +0.008 ± 0.014 | +0.009 ± 0.016 | ~0 | right (noise floor) |
| 7 deletion | +0.011 ± 0.019 | +0.008 ± 0.015 | +0.014 ± 0.024 | ~0 | right (noise floor) |
| 6N naive units fix † | -0.028 ± 0.004 | -0.049 ± 0.002 | -0.069 ± 0.000 | down | right |
| 6N + 7 † | -0.016 ± 0.015 | -0.021 ± 0.026 | -0.029 ± 0.024 | worse than 6N | **flips: less bad** |
| 3 + 4 | -0.037 ± 0.016 | -0.046 ± 0.031 | -0.048 ± 0.024 | = 3 | right (4 vanishes) |
| 1 + 2 | +0.015 ± 0.033 | +0.012 ± 0.026 | +0.014 ± 0.053 | more than either | not additive |
| 9 + 10 † | +0.097 ± 0.012 | +0.005 ± 0.017 | +0.081 ± 0.021 | 9 grows | **9 vanishes** |
| all ten † | +0.027 ± 0.031 | -0.147 ± 0.032 | -0.059 ± 0.054 | acc up, F1 -0.08..-0.20 | right |

**Noise floor.** Runs are deterministic: same code, seed and thread count give bit-identical results (defects 6 and 7 equal the fixed run exactly on seeds 42 and 43, where the one epoch they touch lies in a validation subject). On seed 44 that recording is in training, and changing its 11 of ~17,000 training windows moves macro-F1 by +0.024. Training is that sensitive to small data changes, so effects under about ±0.02 macro-F1 are not resolved by 3 seeds.

**Ranked by how much each would have misled the report.** (1) **10**: it alone adds ~10 points of accuracy and 9 of kappa from lights-on Wake, while N1 gets worse; the headline would have been inflated by the easiest class. (2) **3**: the context model the architecture advertises never ran; the CNN alone still scores well, so nothing looks wrong. (3) **8**: removes the amplitude criterion of N3 and N1. (4) **9** and (5) **2**: consistent across all seeds but small. Defects 1, 4, 5, 6 and 7 are inside the noise floor on their own.

**Flags.** Defect 4 vanishes while 3 is present (3+4 ≈ 3). Defect 7 vanishes while 6 is present (nothing is rejected, so nothing is deleted), and with the naive units fix it *flips*: deleting keeps 17.3k training windows where masking keeps only 11k (an 11-epoch window needs 11 kept epochs; Wake training windows fall from 2694 to 381, as waking eye movements trip the EOG threshold), so the original deletion hurts less than the "correct" masking. Defect 9's cost on accuracy vanishes once Wake is uncropped (10), the opposite of the prediction: the mislabelled epochs drown in the extra Wake. Defect 1 went the wrong way: subject leakage did not inflate the score here, because the same seed also draws a different, harder set of test subjects, and the test set changes with it (D1 test windows 4750-5143 vs 5415-6653); with 3 test subjects, who is tested matters more than whether their other night was seen.

## 2d Physiology: the worst stage
Full analysis in `task2/PHYSIOLOGY.md`. The files hold EEG Fpz-Cz and Pz-Oz and horizontal EOG at 100 Hz; chin EMG, respiration and temperature only as 1 Hz envelopes; labels are R&K stages (3+4 merged into N3). The pipeline uses the EEG and EOG, not the EMG.

**Worst stage: N1** (F1 0.511 ± 0.017, others 0.83-0.89). Recall 0.61 but precision 0.44: the model uses N1 as the place for epochs it cannot place (over 3 seeds, 486 N2, 276 REM and 262 W windows called N1, against 832 true N1). N1 is defined by transitions (alpha falling below half the epoch, low-amplitude 2-7 Hz activity, slow eye movements, ended by the first spindle or K-complex), and on these channels it overlaps its neighbours: median Fpz-Cz amplitude 78 uV (REM 79), Pz-Oz alpha 0.084 (REM 0.088), slow-EOG power 284 uV² (REM 271). What tells N1 from REM is muscle tone (EMG 2.68 vs 0.69 uV), which the model never sees; what tells it from N2 is a single transient that a 30-s epoch dilutes. It is also the rarest stage (7% of scored epochs) and the one human scorers agree on least.

**Expected confusion: N3 -> N2** (17.5% of N3). Stage 3 starts at 20% of the epoch with >75 uV, 0.5-2 Hz waves: a threshold on a continuum (median N3 coverage 27%), so borderline epochs fall to N2; the reverse is rare (3.2%).
**Unexpected confusion: W <-> REM** (5.1% and 3.3%, 304 windows). Human scorers rarely confuse them, because REM requires chin atonia; the file records it (EMG 3.23 vs 0.69 uV, the largest separation of any feature), but the pipeline drops the channel, and on EEG + EOG REM looks like relaxed Wake with eye movements.

## 2e Note for a non-technical reader
The first version of our sleep-staging program reported 85% accuracy; the corrected version scores about 83%. The lower figure is the honest one.

The first test was too easy. Each recording runs for almost a whole day, and two-thirds of it is the person awake with the lights on. A program that always answered "awake" would already score 69%. The corrected test keeps only the night, plus half an hour either side.

The first test also included people the program had already studied (their other night was in its training material), and it picked its best version by peeking at the test. Real patients will be new to it.

On a balanced score that counts every sleep stage equally, rather than rewarding easy wakefulness, the corrected program improved from 0.63 to 0.79 (out of 1). It looks weaker only where the old test flattered it.
