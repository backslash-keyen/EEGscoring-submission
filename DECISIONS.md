# DECISIONS
Format: chosen / alternative / why / what the alternative would change. "Verified" = where the fact was checked.

## D1 Runs 4, 8, 12 (imagery left/right fist)
Alternative: 3/7/11 (execution) or 6/10/14 (imagery fists vs feet). Why: the task is imagined left vs right fist; T1/T2 mean different things per run type. Verified: `mne.datasets.eegbci.load_data` docstring and PhysioNet page (wiki/dataset.md). Alternative would change labels entirely (execution gives stronger, cleaner ERD).

## D2 128 Hz-header subjects (88, 92, 100): resample to 160 Hz, keep
Alternative: exclude them, or relabel as 160 Hz. Why: my alpha-peak check (audit.csv) did not show the 0.8x shift a mislabelled 160 Hz file would produce, and their protocol differs (cue 5.12 s, 19 trials/run), so the header looks right. Excluding would cost 3/40 subjects. Expect: negligible change in group results; their per-subject accuracy may differ.

## D3 Windows past the end of a recording are dropped (20 trials) 
Epoch window is -1.5..4.0 s around the cue. Alternative: shorter windows (3 s) to keep every trial. Why: 4.0 s covers the shortest regular cue; padding would invent data. Affected: S72,73,74,76,88,92,102,104 (listed per subject in audit.csv, nothing silent). Expect: <1.2% fewer trials.

## D4 Spatial reference: local Laplacian (C minus mean of 4 nearest electrodes), CAR and raw as sensitivity
The recording reference is undocumented (wiki/dataset.md). With an unknown reference electrode R, every channel contains the same -R(t) term, so raw C3-C4 is a difference of two references-contaminated signals only if R is not common (it is common to both, but R's own activity and distance to each site still leak differently, and a lateral reference such as a mastoid would bias hemispheres asymmetrically). Laplacian weights sum to zero, so any component common to the neighbourhood, including the reference, cancels, and it is spatially local (sharpens C3 vs C4). CAR also cancels common terms but spreads frontal/ocular artefact over all channels. Result: Laplacian gave the largest mean LI (mu .26, beta .32) vs CAR (.19/.29) and raw (.15/.21); present counts 14/16/12 (outputs/a2_all_references.csv). Expect: raw would label fewer lateralised.

## D5 Baseline and window for ERD
Baseline -1.0..-0.1 s (inside the preceding rest, pooled over trials per subject and band, label-blind). Active window 0.5..4.0 s. Alternatives: per-trial baseline (noisier), post-cue baseline (contaminated by imagery), -1.5..0 (includes wavelet edge and the previous cue's offset ERD rebound). Skipping the first 0.5 s removes the cue-evoked visual response and ERD onset lag. Expect: per-trial baseline would shrink ERD magnitudes and raise p-values.

## D6 Lateralisation statistic and label rule
Per trial dB ERD per band at C3, C4. T = mean(C3-C4 | left) - mean(C3-C4 | right) = sum over classes of (ipsi - contra); one-sided label-permutation p (10000 perms, seed = subject id) because the physiological direction is predicted. LI = (ipsi - contra)/(|ipsi|+|contra|), in [-1,1]. Label "present" if either band p < 0.025 (Bonferroni over 2 bands) AND contralateral ERD is a desynchronisation (<0 dB). Alternative: two-sided test (would also count wrong-direction asymmetries as lateralisation).

## D7 No artefact rejection (so far)
Alternative: amplitude-threshold or ICA rejection. Why not yet: nothing may be dropped silently and per-subject trial counts are small. Known cost: heavy trials dominate (e.g. S95 C3). Expect: rejection would raise ERD clarity for noisy subjects.

# Task 2 (sleep staging) - judgement calls. IDs are T2-n to stay apart from the Task 1 D-numbers.
Evidence files are in task2/evidence/; commit hashes are in DEFECTS.md.

## T2-1 Split: train 9 / validation 3 / test 3 subjects (20% held out twice), disjoint, both nights of a subject together
Alternative: 2-way split with a larger test fraction, or leave-subject-out CV. Why: the given script put a subject's two nights on both sides (5/5 test subjects were also in training); a third disjoint group is what lets the epoch be chosen without touching test. 3 test subjects is a small, noisy referee, so every ledger row uses 3 seeds that each change the split. Expect: leave-one-subject-out would give a tighter estimate at ~15x the cost.

## T2-2 Epoch selection by validation macro-F1, one final test evaluation
Alternative: a fixed training budget with no selection, or select on validation loss. Why: macro-F1 is the headline metric and the classes are imbalanced. Known weakness: validation macro-F1 swung 0.70-0.82 between epochs on 3 subjects, so the chosen epoch is itself noisy (seed 42 chose epoch 1 of 12 in one run). Expect: a fixed budget would remove that noise but could stop before or after the best point.

## T2-3 Window label = centre epoch, not last epoch
Alternative: keep the last-epoch label and read the head at position -1 (causal staging). Why: the script documents centre-epoch staging; the label differs from the centre in 8.2% of windows (71% for N1-centred ones, task2/evidence via hypnograms). Cost: the first and last 5 epochs of a recording are never targets (about 1%, mostly Wake). The causal alternative would use every epoch but remove look-ahead.

## T2-4 Artefact rejection: 500 uV peak-to-peak on the two EEG channels only, in uV
Alternative: all channels with a higher threshold, or no rejection. Why: MNE data is in volts; the EOG channel's median peak-to-peak is 534 uV so converting units on all channels rejects 54% of epochs (d6_per_channel_result.txt); the EEG channels exceed 500 uV in one epoch of 80070, so rejection is now correct but nearly inert. This is a finding about the data, not a claim that the pipeline is artefact-free. Expect: a lower threshold (250 uV) would reject 18.6% of epochs.

## T2-5 Rejected epochs are masked, windows containing one are skipped
Alternative: split the recording into separate segments at each gap. Why: same windows survive with less bookkeeping. Affects 11 windows of one recording on this data, so the practical effect is below noise.

## T2-6 Unscored ("Sleep stage ?", 905 epochs) and "Movement time" (35) epochs are labelled -1: usable as context, never as a target
Alternative: map movement time to the previous stage, or drop whole windows containing them. Why: movement time is an epoch that cannot be staged by definition, and calling it Wake invents a label. Expect: dropping every window that contains one would lose a few hundred more windows for no gain.

## T2-7 Crop each recording to the first/last sleep epoch +- 30 min
Alternative: 15 or 60 min margins, or no crop. Why: recordings are ~23 h, 97% of all Wake lies outside the sleep period and 'always Wake' scores 68.7% accuracy (d10_wake_padding_result.txt). 30 min follows the usual convention but is a convention, not a derived value. Expect: accuracy, kappa and Wake F1 depend on the margin; macro-F1 depends on it much less. The class weights are inverse frequency of the cropped training windows (verified weight*count equal for every class).

## T2-8 Standardise per recording and channel (non-rejected epochs), not per epoch
Alternative: keep per-epoch z-scoring, or feed the epoch's log-amplitude as an extra input. Why: per-epoch z-scoring set every epoch std to 1 and deleted the amplitude that defines N3 (99.6% of N3 epochs exceed 75 uV in the 0.5-2 Hz band vs 25-28% of N1/REM). Per-recording scaling still removes between-person scale. Cost: a single noisy channel in one recording changes that recording's scale; test recordings are scaled with their own statistics (no training statistics are used, so no leakage, but it is a transductive step).

## T2-9 Left unchanged on purpose
- Class-weighted loss + uniform sampling with replacement: after the crop the weights match the training windows exactly, so this is a metric choice (balanced loss favours macro-F1 over accuracy), not a defect.
- Printed training loss is the last batch only (reporting detail). Seeding: torch/numpy/random are seeded and identical seeds reproduced identical loss curves (checked on seed 42), so no determinism flags were added.
- The EMG channel (1 Hz envelope) is not used. Relevant to 2d: REM atonia is the scoring rule the model cannot see.

## T2-10 batch_first=True instead of permuting dimensions
Alternative: transpose (B, L, d) to (L, B, d) before and after the encoder. Why: one argument, same maths, no shape bugs. Verified by perturbing one epoch of a window: neighbouring epochs now change the output (0.004-0.009), other windows in the batch do not (0.000).
