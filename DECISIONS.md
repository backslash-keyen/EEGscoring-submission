# DECISIONS
Format: choice, alternative, why, expected change. "Verified" = where a fact was checked.

## D1 Runs 4, 8, 12 (imagery left/right fist)
Alternative: 3/7/11 (execution) or 6/10/14 (fists vs feet). Why: the task is imagined left vs right fist, and T1/T2 mean different things per run type. Verified: eegbci.load_data docstring, PhysioNet page. Expect: different labels, and execution gives stronger ERD.

## D2 128 Hz-header subjects (88, 92, 100): resample to 160 Hz, keep
Alternative: exclude them, or relabel as 160 Hz. Why: no 0.8x alpha-peak shift (verified: own check, audit.csv) and their protocol differs, so the header looks right. Expect: excluding costs 3/40 subjects with negligible group change.

## D3 Windows past the end of a recording are dropped (20 trials)
Epoch -1.5..4.0 s. Alternative: 3 s windows that keep every trial. Why: 4.0 s covers the shortest regular cue and padding invents data. Drops listed per subject in audit.csv. Expect: <1.2% fewer trials.

## D4 Spatial reference: Laplacian (4 nearest), CAR and raw as sensitivity
The recording reference is undocumented (verified: wiki/dataset.md). Why: Laplacian weights sum to zero, so the common reference cancels, and it is spatially local. Alternative: CAR (spreads ocular artefact) or raw. Laplacian gave the largest mean LI (mu .26, beta .32). Expect: raw would label fewer lateralised.

## D5 Baseline and window for ERD
Baseline -1.0..-0.1 s, pooled per subject and band, label-blind. Active window 0.5..4.0 s, skipping the cue-evoked response. Alternative: per-trial or post-cue baseline (contains imagery). Expect: per-trial baseline would shrink ERD and raise p-values.

## D6 Lateralisation statistic and label rule
One-sided label-permutation test of C3-C4 ERD difference (10000 perms). "Present" if either band p < 0.025 (Bonferroni, 2 bands) and contralateral ERD < 0 dB. Alternative: two-sided test. Why: the direction is predicted. Expect: two-sided would count wrong-direction asymmetries.

## D7 No artefact rejection
Alternative: amplitude-threshold or ICA rejection. Why: nothing may be dropped silently and trial counts are small. Expect: rejection would sharpen ERD for noisy subjects (e.g. S95).

## D10 Neighbours of C3 and C4: the 4 nearest electrodes by 3-D distance
Cp3, Fc3, C5, C1 and Cp4, Fc4, C6, C2. Why: the same set as the Laplacian (D4). Alternative: 8 neighbours or a fixed radius. Verified: erd.neighbours() on standard_1005 (own run). Expect: 8 neighbours give a wider, smoother reference.

## D12 A3 routes and decoders (fixed before any labelled decoder was run)
Routes: oculomotor (frontal, 0-0.5 s headline F1, 0.5-4 s F2), visual (occipital O1/O2), muscle (30-40 Hz G1), trial order (R1-R3), everything-except-sensorimotor N1/N2, controls M1/A1, and a pre-cue twin of each as a leak check. Alternative: pick the best window per route afterwards. Why: PREDICTIONS.md names no window. Expect: that would inflate every number.

## D13 A3 features
Fixed before results. TD: baseline-subtracted bin means per channel. BP: Welch log power (mu, beta, 30-40 Hz) as left-minus-right mirror-pair differences. Why: differences remove subject amplitude and the common reference. Alternative: Laplacian or CAR first. Expect: weaker frontal and occipital effects.

## D14 A3 classifier and evaluation scheme
Fixed before results. L2 logistic regression, C = 0.1 untuned, leave-one-run-out within subject, 8 folds of 5 subjects across subjects, scaling fitted on training data only. Alternative: random stratified k-fold. Why: neighbouring trials share drift. Expect: a few percent higher within-subject accuracy.

## D15 A3 chance thresholds
Fixed before results. Threshold = smallest k/n with exact one-sided Binomial(n, 0.5) P <= 0.05 (about 52% pooled), checked by a label-permutation null. Per-subject calls are uncorrected, the claim is the excess test P(count >= observed | Binomial(40, 0.05)). Alternative: Bonferroni per subject. Expect: it would hide weak real leaks in a minority of subjects.

## D16 Known limits of the A3 tests
Fixed before results. The 1-40 Hz cache removes sustained gaze offset and EMG above 40 Hz, so oculomotor and EMG results are lower bounds. Alternative: reload at 0.1-100 Hz for these routes. Why not: only needed if the oculomotor result were ambiguous. Expect: higher frontal accuracy and a real EMG test.

## D17 A3 order and protocol routes
Fixed before results. Order features: trial index, onset, previous 1-3 labels. Run identity is bounded analytically (run-majority oracle). Alternative: an empirical run-identity decoder. Why: leave-one-run-out cannot learn it. Expect: it would show nothing and hide the worst case. Protocol groups have equal class ratios (verified: audit.csv).

## D18 POST-HOC additions to A3 (added after first-pass results were seen, not pre-registered)
First pass showed near-alternating order (previous label ~71%), leaking pre-cue twins (F1_pre 61%) and frontal 74%. Added: a carry-over split (decoders read the current cue) and a minimum-phase refilter, since the zero-phase FIR moved post-cue signal 1.25 s earlier (F1_pre 61%->48%). The causal rerun is PRIMARY. Alternative: causal filter from the start. Expect: A2 mostly unaffected. "Order randomised" was false (verified: outputs/a3_sequence.csv).

## D19 A3 closing steps
(1) Causal results carry plain `a3_*` names and `--zero-phase` reproduces the first pass. Alternative: keep `--causal`, rejected so the default gives the primary result. (2) A2 labels identical on the causal cache for 40/40. (3) A3 entries renumbered D8-D15 to D12-D19 at merge.

## D20 EEGNet hyperparameters changed for 160 Hz (task1/models.py)
Kernels rescaled to stay 0.5 s: first 64 -> 80 samples, separable 16 -> 20. Other EEGNet-8,2 settings kept, max 150 epochs with early stopping. Alternative: keep 64/16 samples (0.4 s). Why: keep the lowest shaped frequency at 2 Hz. Expect: almost no change for mu/beta, only below ~4 Hz.

## D21 Transformer design (task1/models.py, EEGTransformer)
Budget 3 x 2962 = 8886 parameters forces d_model = 16. All variants share EEGNet's 8-filter 0.5 s temporal front end, so only the token axis changes (time patches vs channel tokens, with or without electrode embedding). Alternative: larger d_model on fixed band-power features. Expect: more attention capacity but no learned frequency bands, the property B2 compares.

## D22 Subject-wise folds for Part B (task1/train.py)
The 8 A3 folds of 5 subjects, so network and A3 accuracy compare on the same subjects. Test fold f, validation fold f+1 for early stopping only, 30 training subjects, normalisation fitted on them. Alternative: leave-one-subject-out. Expect: slightly higher accuracy (38 training subjects) at 5x compute.

## D23 Network input: causal cache, 0-4.0 s after the cue, all 64 channels, microvolts
Minimum-phase cache (D18), so nothing leaks before the cue. 0-4.0 s is the full imagery period (D3). The 0-0.5 s eye route is kept on purpose (B2 asks what networks use) and removed only in the confound-removed retrain. Alternative: 0.5-4.0 s. Expect: a smaller but present confound (A3 F2 75% pooled).

## D24 B2 compute plan: 4 of the 8 folds, step-matched training (task1/b2_run.py)
Planned before B2 was run. About 12 h CPU, so test folds 0-3 (20 subjects) for every experiment, seeds 0-2, gradient steps matched across training-set sizes. Alternative: all 8 folds (~2x compute). Expect: same means, tighter spread, 40 instead of 20 points in the Part A correlation.

## D25 "Confound removed" input = Laplacian over 64 channels, then the 21 Fc/C/Cp electrodes (task1/confound_free.py)
Pre-registered, fixed geometry, nothing fitted. Why: the eye field reaches non-frontal sites (A3 N2 70.6%), so dropping frontal channels is not enough. The Laplacian cancels near-flat far-field EOG and keeps focal mu/beta ERD. Alternative: regress out a frontal EOG proxy fitted on training subjects. Expect: more eye signal left, since the proxy also holds brain activity.

## D26 Displacement simulation and the intervention (task1/spatial.py, task1/augment.py)
Cap slide = rigid rotation on the fitted sphere, signal by spherical-spline interpolation with alpha = None (MNE's 1e-5 alters data at 0 mm). Sizes 0-20 mm in 4 directions, headline 10 mm (about a quarter of electrode spacing), fixed before any displacement result. Intervention: train on random 0-15 mm slides (a Tikhonov smoothness penalty). Alternative: channel dropout. Expect: less robustness, since dropout does not model smooth spatial mixing.

## D27 POST-HOC: diagnostic for the confound-free runs (task1/b2_noconf_check.py, added after seeing seed-0 results)
Seed 0 of the pre-registered retrain (D25) kept epoch 0 in all 4 folds and scored 48.4-49.1%. Diagnostic: fold 0 with patience 40 and per-epoch losses. Why: separate a too-weak motor signal (A3 M1 54.6%) from early stopping. Alternative: no diagnostic. Expect: the two readings stay unseparated. Headline numbers stay pre-registered.

# Task 2 (sleep staging): IDs T2-n. Evidence in task2/evidence/, commit hashes in DEFECTS.md.

## T2-1 Split: train 9 / validation 3 / test 3 subjects, disjoint, both nights of a subject together
Alternative: 2-way split or leave-subject-out CV. Why: the given script leaked 5/5 test subjects into training, and a disjoint validation group picks the epoch without touching test. 3 seeds each change the split. Expect: leave-one-subject-out gives a tighter estimate at ~15x the cost.

## T2-2 Epoch selection by validation macro-F1, one final test evaluation
Alternative: a fixed budget, or validation loss. Why: macro-F1 is the headline metric on imbalanced classes. Expect: a fixed budget removes selection noise (validation macro-F1 swung 0.70-0.82) but may miss the best epoch.

## T2-3 Window label = centre epoch, not last epoch
Alternative: last-epoch label (causal staging). Why: the script documents centre staging, and labels differ in 8.7% of windows (verified: task2/evidence/d5_label_position_result.txt). Expect: causal staging uses every epoch (about 1% more, mostly Wake) but loses look-ahead.

## T2-4 Artefact rejection: 500 uV peak-to-peak on the two EEG channels only, in uV
Alternative: all channels, or none. Why: EOG median peak-to-peak is 534 uV, so all-channel rejection drops 54% of epochs (verified: d6_per_channel_result.txt). On EEG it is nearly inert (1 of 80070). Expect: 250 uV would reject 18.6%.

## T2-5 Rejected epochs are masked, windows containing one are skipped
Alternative: split the recording at each gap. Why: same windows survive with less bookkeeping. Expect: no visible change, only 11 windows of one recording are affected.

## T2-6 Unscored (905) and Movement time (35) epochs labelled -1: context only, never a target
Alternative: map movement time to the previous stage, or drop windows containing them. Why: such epochs cannot be staged, so any label is invented. Expect: dropping loses a few hundred more windows for no gain.

## T2-7 Crop each recording to the first/last sleep epoch +- 30 min
Alternative: 15 or 60 min, or no crop. Why: recordings are ~23 h and 'always Wake' scores 68.7% (verified: d10_wake_padding_result.txt). 30 min is a convention. Expect: accuracy, kappa and Wake F1 shift with the margin, macro-F1 much less.

## T2-8 Standardise per recording and channel (non-rejected epochs), not per epoch
Alternative: per-epoch z-scoring. Why: it deletes the amplitude that defines N3 (99.6% of N3 epochs exceed 75 uV at 0.5-2 Hz). Expect: per-epoch scaling would blur N3. Cost: a transductive per-recording step, with no leakage.

## T2-9 Left unchanged on purpose
Class-weighted loss with uniform sampling: weights match the cropped windows, a metric choice favouring macro-F1. No determinism flags: identical seeds gave identical curves (seed 42), bit-identical only at equal thread count (likely cause of 0.810 vs 0.803, not verified). EMG unused. Alternatives: balanced sampling, determinism flags, adding EMG. Expect: EMG could help REM, whose atonia the model cannot see.

## T2-10 batch_first=True instead of permuting dimensions
Alternative: transpose before and after the encoder. Why: same maths, no shape bugs. Verified: perturbing one epoch changes its neighbours (0.004-0.009), not other windows (0.000). Expect: no change.
