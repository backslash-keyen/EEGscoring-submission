# A3 confound audit - results
Primary = minimum-phase (causal) pre-filter, `python task1/confound.py --causal` (+ `confound_carryover.py --causal`). First pass = zero-phase filter, the files without `causal_` prefix (kept as evidence of the filter leak, DECISIONS D14).
Thresholds: exact one-sided binomial at 5%, p0 = 0.5. Within subject, n = 36-45 trials gives 63.0-66.7% (see `thr_binom` per subject). Pooled over 40 subjects, n = 1813 gives 52.0%. Permutation null (labels permuted within run, whole CV repeated) gave 2.7-8.3% false positives against the 5% nominal, so the per-subject calls are slightly liberal; group counts are far beyond that.
Within-subject = leave-one-run-out; pooled = 8 subject-wise folds, nothing from a test subject touches fitting or scaling.

| decoder (what it sees) | within-subject mean acc | subjects above threshold (of 40; ~2 by luck) | pooled cross-subject acc | pre-cue twin (pooled) |
|---|---|---|---|---|
| F1 frontal, 0-0.5 s (eye movement) | 66.4% | 25 | 71.9% | 48.2% |
| F2 frontal, 0.5-4 s | 63.5% | 18 | 75.3% | - |
| O1 occipital, 0-0.5 s (visual evoked) | 56.5% | 14 | 59.3% | 49.7% |
| O2 occipital mu+beta, 0.5-4 s | 57.4% | 9 | 52.6% | 50.4% |
| G1 temporal 30-40 Hz (muscle, weak test) | 50.8% | 1 | 49.6% | 49.3% |
| N2 everything except sensorimotor, 0-0.5 s | 63.1% | 20 | 70.6% | 48.8% |
| N1 everything except sensorimotor, power | 57.8% | 11 | 54.1% | 49.3% |
| M1 sensorimotor strip, mu+beta (positive control) | 54.9% | 8 | 54.6% | 52.2%* |
| A1 all channels, mu+beta | 58.9% | 14 | 55.4% | 52.4%* |
| R1 trial index / onset time | 49.5% | 0 | 50.8% | - |
| R2 previous 1-3 labels (no EEG at all) | 70.8% | 31 | 72.6% | - |
| run-majority oracle (analytic) | - | - | 52.9% | - |
*marginal (permutation p 0.02-0.03, 10 pre-cue twins tested, uncorrected); not interpreted.

## Findings
1. The side of the cue can be read without motor cortex, strongly: frontal channels in the first 0.5 s reach 72% pooled, 25/40 subjects individually. The time course (`a3c_timecourse.png`) shows chance until ~0.2 s, then ~72% at 0.3-0.4 s and again at 0.7-0.9 s: a saccade toward the target and its return. This is the current cue, not carry-over: accuracy is as high on repeat trials (78%) as on alternating trials (69%); cell-balanced training (current x previous) leaves it at 73%; the previous label cannot be read from the features (46-51% balanced).
2. The motor strip, the signal the task is about, gives 54.6% pooled and 8/40 subjects. A model scoring 70% on these subjects is inside what frontal eye signal (72-75%) or trial history (72.6%) gives with no motor imagery.
3. Trial order is not random. Consecutive labels alternate: lag-1 agreement 424 observed vs 780 expected (z = -17.6, `a3_sequence.csv`); previous label alone predicts the current at 70.8% within subject. Position in run (R1) carries nothing (50.8%); run identity can give at most 52.9% (analytic).
4. Visual route: weak but real (O1 59% pooled, 14/40, peak ~57% at 0.25 s). Muscle route: not detectable here, but the 40 Hz low-pass makes this a weak test.
5. Method failure caught: with the zero-phase filter the pre-cue twins looked decodable (F1_pre 61%) because the filter spreads the post-cue eye response up to 1.25 s backwards. Minimum-phase filtering removes it (F1_pre 48%, N2_pre 49%, O1_pre 50%).
6. A2 link: per-subject M1 accuracy vs mean(mu_LI, beta_LI): Spearman 0.34 (p = 0.03). F1 accuracy vs M1 accuracy: 0.06 (p = 0.73). 14 subjects have lateralised ERD (A2 label), 8 of them have F1 above threshold.

## Prediction scorecard (PREDICTIONS.md, committed 3bdff2a before any result)
| # | Predicted | Found (primary) | Verdict |
|---|---|---|---|
| 1 | eyes and visual response decodable | eyes strongly, visual weakly; plus an unpredicted route (trial history) | partly right |
| 2 | frontal: 5-10 of 40 subjects | 25 of 40 | wrong |
| 3 | frontal pooled 50-52% | 71.9% | wrong |
| 4 | occipital pooled 53-57% | 59.3% | slightly above |
| 5 | pre-cue and trial-order at chance | pre-cue at chance only after fixing my filter; previous-label 72.6% | wrong |
| 6 | motor pooled 56-62% | 54.6% | slightly below |
Cause of #3 and #5: I assumed random order and a consistent-sign eye signal would be diluted across subjects; the sign is consistent, and the order is anti-persistent.
