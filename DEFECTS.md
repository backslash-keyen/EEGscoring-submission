# DEFECTS (Task 2b)

The given script is `task2/sleep_pipeline.py` at commit `fb4a5e7` (an unchanged copy), and the fixed script is the same file at HEAD.
Line numbers below refer to the given script. Each defect has one commit with a minimal diff, and judgement calls are argued in DECISIONS.md (T2-1 to T2-10).
Evidence scripts live in `task2/evidence/` (run them from that folder with `MNE_DATA` pointing at the Sleep-EDF cache), and each `*_result.txt` next to a script is its printout.
How much each defect moves the metrics (3 seeds, interactions measured) is the impact ledger (2c). This file gives what is wrong, the evidence, the mechanism and the fix.

## Headline numbers along the way (seed 42, test subjects)

| state of the script | accuracy | macro-F1 | kappa | output |
|---|---|---|---|---|
| given, unchanged (2a) | 0.847 | 0.628 | 0.693 | `task2/baseline/2a_baseline_seed42.txt` |
| defect 1 fixed | 0.853 | 0.667 | 0.722 | `task2/baseline/d1_subject_split.txt` |
| defects 1-2 fixed | 0.853 | 0.667 | 0.722 | `task2/baseline/d2_val_selection.txt` |
| defects 1-7, 9, 10 fixed | 0.821 | 0.778 | 0.769 | `task2/baseline/d10_all_fixed_so_far.txt` |
| all 10 fixed | 0.845 | 0.810 | 0.799 | `task2/baseline/d8_all_fixed_so_far.txt` |

The test set changes size and content as defects are fixed (subject split, cropping), so these rows are a log of the work. The controlled comparison is 2c.

## Summary

| # | defect | where (given script) | fix commit | evidence | what it does to the result |
|---|---|---|---|---|---|
| 1 | split by recording instead of subject | l. 170-172 | `39775aa` | `d1_split_overlap_result.txt` | test people are in training (5/5 subjects, seed 42) |
| 2 | best epoch chosen on the test set | l. 190-206 | `6136415` | `baseline/d1_subject_split.txt` | reported score is a maximum over 12 test evaluations |
| 3 | `batch_first=False` with (B, L, d) input | l. 140 | `c6d6a49` | `d3_batch_attention_result.txt` | attention runs across the windows of a batch instead of the 11 epochs |
| 4 | positional encoding added after the transformer | l. 148-149 | `c3523bb` | `d4_order_result.txt` | the sequence model cannot see epoch order |
| 5 | window labelled with its last epoch, head reads the centre | l. 103, 178 | `5dd1693` | `d5_label_position_result.txt` | wrong target in 8.7% of windows (71% of N1-centred ones) |
| 6 | artefact threshold in uV compared with data in volts | l. 41, 81 | `d38cf1b` | `d6_reject_units_result.txt`, `d6_per_channel_result.txt` | rejection never fires, and a units-only fix would delete 54% of epochs |
| 7 | rejected epochs deleted, so windows span time gaps | l. 82, 95 | `00846ed` | `d7_after_fix.txt` | "11 consecutive epochs" are not consecutive |
| 8 | z-scoring per epoch | l. 85 | `9cac159` | `d8_amplitude_by_stage_result.txt`, `d8_after_fix.txt` | deletes the amplitude that defines N3 |
| 9 | unknown annotations default to Wake | l. 75, 78 | `2731f10` | `d9_label_defaults_result.txt` | 940 unscored / movement epochs trained and scored as Wake |
| 10 | ~23 h recordings used whole | (absent crop, l. 70-86) | `6d07e53` | `d10_wake_padding_result.txt`, `d10_after_fix.txt` | 68% of epochs are lights-on Wake, and "always Wake" scores 68.7% |

Defect 8 was found and fixed last, so its commit comes after defect 10 in the history.

---

## 1. Train/test split by recording instead of by subject
`random.Random(SEED).shuffle(files)` shuffles the 29 recordings (most subjects have two nights) and takes the first 20% as test, so a subject's two nights can land on opposite sides.

`evidence/d1_split_overlap.py` re-runs the given rule. With seed 42 all 5 test recordings come from subjects whose other night is in training (`05, 07, 09, 10, 11`), seed 43 gives 4 of 5 and seed 44 gives 5 of 5. The fixed rule shares 0 subjects on every seed.

Electrode placement, skull thickness, individual alpha frequency and sleep architecture are shared across a person's two nights. The model is therefore scored on people it has trained on, and the test score stops being the subject-independent number the report would claim. The expected inflation did not appear on seed 42, though. Fixing the split *raised* macro-F1 (0.628 to 0.667) because it also changes which test subjects are drawn and how hard they are. The defect is in what the number means, whatever its size, and 2c measures the size over 3 seeds.

The fix splits the 15 subjects (instead of recordings) into train 9 / validation 3 / test 3, keeping both nights of a subject in the same group (`subject_of()` reads the subject from the file name, DECISIONS T2-1).

## 2. Model selection on the test set
Every training epoch is scored on `test_ds`, the best macro-F1 state is kept, and the same `test_ds` is then reported (l. 198-206).

The evidence is `baseline/d1_subject_split.txt` (defect 1 fixed, defect 2 still present). It prints the test macro-F1 after each of the 12 epochs, which ranges from 0.582 to 0.667, and the reported 0.667 is the maximum of that list.

A maximum over 12 noisy evaluations is an optimistically biased estimate, and the bias grows with the epoch-to-epoch noise (large here, with only 3 test subjects). On seed 42 the validation curve happened to pick the same epoch (`baseline/d2_val_selection.txt` gives the identical 0.853 / 0.667 / 0.722), so the bias on that seed is zero. Across seeds it is not (2c).

The fix chooses the epoch on the validation subjects and scores the test subjects once with that state (DECISIONS T2-2).

## 3. Transformer attends across the batch instead of the 11 epochs
`nn.TransformerEncoderLayer` defaults to `batch_first=False` and therefore expects (L, B, d). The model passes (B, L, d), so PyTorch treats the 32 windows of a batch as the sequence and the 11 epochs as the batch.

`evidence/d3_batch_attention.py` runs the same weights in both layouts. With the given layout, replacing a neighbouring epoch of window 0 changes window 0's output by exactly 0, while replacing a *different window* in the batch changes it by 9.0e-4. With `batch_first=True` the pattern reverses (2.4e-3 from its own neighbour and 0 from other windows).

The architecture exists for sequence context (N1, REM and stage transitions are scored from neighbouring epochs), but the given model uses none. Each prediction instead depends on whichever unrelated windows share its batch, so the test output also depends on batch composition and order. It still runs and still scores well because the CNN epoch encoder alone carries most of the information, which is exactly why nobody notices.

The fix is `batch_first=True` (DECISIONS T2-10).

## 4. Positional encoding added after the transformer
The forward pass runs `z = self.transformer(z); z = self.pos(z)`. Self-attention without position information is permutation-equivariant, and adding the encoding afterwards only adds the constant vector `pe[5]` to the centre token read by the head (a fixed bias).

`evidence/d4_order_test.py` keeps the centre epoch fixed and shuffles the other 10. The output changes by 3e-7 (float noise) with the given order of operations and by 3e-3 to 5e-3 with the encoding added before attention (`d4_order_result.txt`).

The model sees the neighbourhood as an unordered set and cannot tell "N2 before, REM after" from the reverse, so transition rules (e.g. REM continuation, N1 after Wake) are unavailable. While defect 3 is present, attention never runs over epochs, so defect 4 has nothing to break and its effect only appears once defect 3 is fixed (2c tests D3 alone against D3+D4).

The fix is `z = self.transformer(self.pos(z))`.

## 5. Window labelled with its last epoch while the head reads the centre
The docstring says the model "predicts the stage of the CENTRE epoch", and the head reads `z[:, L // 2]`, but `__getitem__` returns `y[i + SEQ_LEN - 1]` (l. 103). The class weights are computed from the same shifted labels (l. 178).

`evidence/d5_label_position.py` counts, from the hypnograms, how often the last-epoch label differs from the centre label. Over the whole recordings it differs in 8.7% of all windows, 71.3% of N1-centred ones, ~20% of N2/N3/REM-centred and 1.5% of Wake-centred ones. In the cropped recordings of the fixed pipeline the mismatch is 23.0%, because the long stable Wake tails are gone (an interaction with defect 10).

Since the target sits 2.5 min ahead of the token the head reads, the model learns to predict a future stage from the present one. The labels are noisy exactly where staging is hard (transitions, N1). Part of the damage is recoverable once defects 3 and 4 are fixed, because the transformer can then attend to position 10, so its size depends on the other fixes (2c).

The fix sets the label to `y[i + SEQ_LEN // 2]` in both places (DECISIONS T2-3).

## 6. Artefact threshold in microvolts compared with data in volts
`REJECT_PTP = 500` is commented "uV", but `raw.get_data()` returns volts, so the test is `ptp_in_volts < 500`.

Per `evidence/d6_reject_units.py`, the largest peak-to-peak in any epoch is 1.53e-3 V, so 0 of 80070 epochs are rejected. Converting units alone would reject 54.1% of epochs, because the horizontal EOG channel has a median peak-to-peak of 534 uV (`d6_per_channel_result.txt`). The EEG channels have medians of 175 and 85 uV and maxima of 509 and 414 uV.

As given, the pipeline does no artefact rejection although the code and comment say it does, and a reader would trust that electrode pops were removed. On this data the EEG channels almost never exceed 500 uV, so the score effect is small. The naive fix is the larger risk, since it deletes half the epochs, mostly those with eye movements (Wake and REM). It also hides defect 7, since nothing rejected means nothing deleted, so defect 7 stays invisible.

The fix multiplies by 1e6 and tests only the two EEG channels, because the EOG channel's normal range overlaps the threshold. It removes 1 of 80070 epochs (`d6_after_fix.txt`, DECISIONS T2-4).

## 7. Rejected epochs deleted, so windows span time gaps
`X, y = X[keep], y[keep]` removes epochs from the array, and windows are cut from the shortened array, so epochs either side of a removed one become neighbours.

After defect 6 is fixed, recording SC4061E0 has one rejected epoch (index 1659), and 11 windows would contain the splice. `d7_after_fix.txt` shows those 11 windows skipped and that no kept window contains the rejected epoch.

A window that claims to be 5.5 min of consecutive sleep then contains a jump in time, so the sequence context (once defects 3-4 are fixed) is fed false neighbours. On this data only 11 windows are affected. With the naive units fix for defect 6 (54% deleted) most windows would be, and 2c measures that interaction as D6N+D7.

The fix keeps every epoch in place with a `keep` mask and skips windows that contain a rejected epoch (DECISIONS T2-5).

## 8. Per-epoch z-scoring removes the amplitude that defines N3
Each 30-s epoch and channel is scaled to zero mean and unit std (l. 85), "so the network is insensitive to amplitude drift".

`evidence/d8_amplitude_by_stage.py` shows on raw Fpz-Cz (0.5-2 Hz) that 99.6% of N3 epochs exceed 75 uV peak-to-peak (median 159 uV), against 28% of N1 and 25% of REM. After the given normalisation every epoch has std 0.999, whatever its stage. After the fix (per recording) the stage contrast survives, with median epoch std N3 1.45, N2 0.76, REM 0.52 and N1 0.51 (`d8_after_fix.txt`).

AASM defines N3 by the amount of high-amplitude (>75 uV) slow activity. Without amplitude the model must infer it from waveform shape alone, so N2/N3 separation gets harder, and so does N1/REM against Wake, which also differ in amplitude. With this fix the headline seed-42 macro-F1 rose from 0.778 to 0.810 (N3 F1 0.893 to 0.887, N2 0.828 to 0.839). This is one seed, and 2c gives the controlled change.

The fix standardises per recording and channel, using the non-rejected epochs, so between-person scale is still removed (DECISIONS T2-8).

## 9. Unscored and movement epochs labelled as Wake
Labels start as `np.zeros` (= Wake), and unknown descriptions map through `STAGE_MAP.get(desc, 0)` (also Wake).

`evidence/d9_label_defaults.py` lists every annotation description. "Sleep stage ?" covers 905 epochs and "Movement time" 35, and since neither is in `STAGE_MAP`, both become Wake. Every epoch is annotated, so the zero initialisation is harmless on this data and the `.get(desc, 0)` fallback is the actual defect. Some unscored stretches are long (802 epochs in SC4092E).

So 940 epochs with no valid stage are trained on and scored as Wake, which invents labels (movement time is by definition not stageable). On its own the effect is small, but it grows when the uncropped recordings are kept (defect 10), since the unscored stretches sit mostly at the ends.

The fix initialises with -1 and maps unknown descriptions to -1. Such epochs may appear as context but are never a window's target (`d9_after_fix.txt`, DECISIONS T2-6).

## 10. Whole ~23-hour recordings, in which 68% of all epochs are lights-on Wake
Sleep Cassette files run about 22-23 h, of which the night is ~8 h, and the script uses every epoch.

`evidence/d10_wake_padding.py` finds Wake in 54348 of 80070 epochs (67.9%), and 97% of it lies before the first or after the last sleep epoch (e.g. SC4051E, 10.7 h before and 7.3 h after). A classifier that always says Wake scores 68.7% accuracy with kappa 0. In the given baseline's confusion matrix, 9666 of 13714 test windows (70.5%) are Wake.

Daytime Wake with eyes open is trivially separable from sleep, so accuracy and kappa are dominated by an easy and irrelevant class. The headline accuracy is inflated, and it hides poor sleep-stage performance (given N1 F1 0.207). The fixed pipeline's accuracy is therefore *lower* than the given one while its macro-F1 is higher. Class weights are inverse frequency of the training windows, so they adjust automatically after the crop (`d10_after_fix.txt` shows weight x count equal for all classes).

The fix crops each recording to the first/last sleep epoch +- 30 min, which drops Wake to 17.3% of epochs. The 30-min margin is a convention (DECISIONS T2-7).

---

## Interactions (fixing one changes what another does)
Defect 3 masks defect 4. With attention over the batch, epoch order has no effect anyway, so defect 4 only shows once 3 is fixed.

Defect 6 masks defect 7, and the obvious fix for 6 amplifies 7. As given, nothing is rejected, so nothing is deleted. Fixing only the units deletes 54% of epochs, and deletion then splices most windows.

Defect 10 changes defects 5 and 9. Cropping raises the share of windows whose last and centre labels differ from 8.7% to 23.0%, so defect 5 matters more in the fixed pipeline. Without the crop, the unscored-as-Wake labels of defect 9 sit among a sea of real Wake, so the cost of defect 9 depends on whether the crop is present.

Defect 10 also interacts with the metric. The crop lowers accuracy and raises macro-F1, so which way the "headline" moves depends on which metric is reported (2e).

## Checked and left unchanged
Class-weighted loss with uniform sampling, the last-batch training loss in the printout, the unused EMG channel (relevant to 2d) and the seeding are discussed in DECISIONS T2-9.
