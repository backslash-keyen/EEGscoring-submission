# A3 confound audit - results
Primary = minimum-phase (causal) pre-filter, `python task1/confound.py` (+ `confound_carryover.py`); files `a3_*`, `a3b_*`. First pass = zero-phase filter (`--zero-phase`), files `firstpass_zerophase_*` (kept as evidence of the filter leak, DECISIONS D18).
Thresholds: exact one-sided binomial at 5%, p0 = 0.5. Within subject, n = 36-45 trials gives 63.0-66.7% (see `thr_binom` per subject). Pooled over 40 subjects, n = 1793 gives 52.0%. Permutation null (labels permuted within run, whole CV repeated) gave 2.7-8.3% false positives against the 5% nominal, so the per-subject calls are slightly liberal; group counts are far beyond that.
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

## Closing checks (added after the main results; DECISIONS D19)
**A2 on the causal filter** (`a2_causal_check.py`, `a2_lateralisation_causal.csv`): the lateralisation labels are identical for all 40 subjects (14 present either way); correlation of LI 0.998 (mu) and 0.999 (beta); contralateral ERD in dB correlates 1.000. A2 is unaffected.
**Thresholds for the follow-up accuracies** (`a3_thresholds.py`): pooled n = 1793 -> 52.0%; repeat trials n = 424 -> 54.3%; alternating n = 1249 -> 52.4%; all with a known previous label n = 1673 -> 52.1%; the repeat/alternate balanced accuracy uses 0.5 + 1.645 SE (normal approximation, 52.3%). Files: `a3_followup_carryover_with_thresholds.csv`, `a3c_*_with_thresholds.csv`. The pooled trial count is 1793, not 1813: 20 trials whose window runs past the recording end are dropped (D3).
**MATLAB re-implementation** (`task1/a3_confound.m` / `.mlx`, causal Butterworth 1-40 Hz instead of MNE's minimum-phase FIR; `a3_matlab_vs_python.csv`). Same trial counts for every subject. Decoders that use band power or no EEG reproduce Python almost exactly (M1 per-subject r = 0.999, pooled 54.7% vs 54.6%; R2 identical, pooled 72.6%). The time-domain eye/visual decoders are weaker with the Butterworth: F1 within-subject 59.2% vs 66.4% (17 vs 25 subjects above threshold), pooled 65.3% vs 71.9%; O1 54.3% vs 56.5% (9 vs 14 subjects), pooled 57.8% vs 59.3%. Per-subject correlations 0.56 (F1) and 0.65 (O1). Conclusions that survive both: the frontal decoder is far above chance, its pre-cue twin is at chance (pooled 51.1%, 1/40 subjects), the motor strip is weak, previous label gives 72.6%. What does not survive: the exact size of the frontal effect, which depends on the pre-filter's sub-1 Hz behaviour (agent diagnostic: post-cue shifts are ~2x larger with MNE's FIR than with the Butterworth). Flag: MATLAB O1_pre has 6/40 subjects above threshold (2 expected) although its mean accuracy is 51.4%; Python gives 2/40.

## File index (outputs/)
- `a3_within_subject.csv`, `a3_cross_subject.csv`, `a3_summary.csv`, `a3_per_subject_table.csv` (per-subject LI, p, label from A2 + decoder accuracies + thresholds), `a3_sequence.csv`, `a3_within_subject.png`: primary run.
- `a3b_carryover*.csv`: repeat vs alternating split and cell-balanced training; `a3c_*`: causal vs zero-phase twins and the 100 ms time course.
- `firstpass_zerophase_*`: the first pass, unchanged.
- `a3_matlab_*`: MATLAB run and comparison. `a3_walkthrough/`: figures used in `task1/a3_walkthrough/a3_walkthrough.ipynb`.
- `a3d_artifact_scan.csv`: label-free artefact scan (not required for A3; its 100 uV rule saturates, see the chat note).
