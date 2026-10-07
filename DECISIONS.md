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

## D8 A3 routes and the decoder that tests each (fixed before any labelled decoder was run)
Cue = target on the left/right of the screen, imagined until it disappears, rest between trials, order random within run. Routes and tests (all on the cached 1-40 Hz, raw-reference data of D2/D3/D7):
- Oculomotor (gaze toward target): frontal channels Fp*, Af*, F5-F8, time-domain 0-0.5 s = headline decoder F1; same channels 0.5-4 s = F2.
- Visual (lateralised evoked response / attention alpha): occipital Po*, O*, Iz time-domain 0-0.5 s = headline decoder O1; occipital mu/beta power 0.5-4 s = O2.
- Muscle / posture: temporal and edge channels Ft7/8, T7-10, Tp7/8, 30-40 Hz power 0.5-4 s = G1.
- Trial order: position in run and onset time (R1); previous 1-3 labels (R2); run-majority oracle (analytic, R3); lag-1 label agreement vs within-run permutation.
- Pre-cue baseline: a "twin" of every decoder with the window moved before the cue (TD: -0.5..0 with baseline -1.0..-0.5; power: -1.5..-0.1). No label information can exist there, so any accuracy is a method leak.
- Added beyond the draft (all fixed before running): previous-trial label (R2), everything-except-sensorimotor decoders N1 (power) and N2 (time-domain 0-0.5 s), 30-40 Hz EMG band (G1). Positive controls: motor strip M1 (Fc/C/Cp rows, mu+beta power, 0.5-4 s) and all channels A1.
Headline definitions matter because PREDICTIONS.md says "frontal", "occipital", "motor" without a window. Alternative: pick the best window per route after the fact, which would inflate every number. Expect: F2 and O2 (long windows) to be weaker than F1/O1 if the leak is the saccade/evoked response, stronger if it is sustained gaze or attention.

## D9 A3 features
Time-domain (TD): mean in bins (0.1 s for 0-0.5 s windows, 0.5 s for 0.5-4 s) of each channel in the set, baseline-subtracted per trial (mean of -0.5..0 s), microvolts. Power (BP): Welch log10 power (128-sample Hann segments, 50% overlap), averaged in mu 8-13, beta 13-30, 30-40 Hz, then left-minus-right difference over mirror channel pairs (C3-C4, Fp1-Fp2, ...). Why differences: the label is a left/right variable, differences remove each subject's overall amplitude (so cross-subject pooling does not need per-subject normalisation, which Part B forbids) and cancel anything common including the unknown recording reference (D4). Midline channels carry no pair and are dropped from BP. TD keeps raw channels because eye movements are antisymmetric (F7 vs F8) and a decoder can form the difference itself. Alternative: Laplacian or CAR before features (D4 sensitivity); would reduce the frontal and occipital effects through spatial averaging.

## D10 A3 classifier and evaluation scheme
L2 logistic regression, C = 0.1 fixed in advance (not tuned: tuning on test accuracy would leak), features standardised with training data only. Within subject: leave-one-run-out over the 3 imagery runs, accuracy pooled over the three held-out runs, so no trial of the test run touches fitting and slow drift between runs is not exploitable. Cross-subject: 8 folds of 5 subjects (assignment by np.random.default_rng(0).permutation), scaler and classifier fitted on the training subjects only. Alternative: random stratified k-fold within subject, which lets neighbouring trials share drift and inflates scores (Li et al. 2020 style leakage). Expect: random k-fold would raise every within-subject number by a few percent.

## D11 A3 chance thresholds
Accuracy over n trials, null "no information": X ~ Binomial(n, 0.5). Threshold = smallest accuracy k/n with P(X >= k) <= 0.05 (exact, one-sided, scipy.stats.binom). The same formula for the pooled number of trials (about 1800 gives about 52%). Because CV accuracies are not exactly binomial and classes are 22/23, I validate with a label-permutation null (labels permuted within run, the whole CV repeated, 200 times per subject and decoder, 100 for pooled): report the fraction of null accuracies above the binomial threshold (should be about 5%) and the permutation p. With 40 subjects, about 2 subjects per decoder pass by luck, so each decoder also gets an excess test: P(count >= observed | Binomial(40, 0.05)). Per-subject calls are uncorrected on purpose; the excess test is the claim. Alternative: Bonferroni per subject, which would hide weak real leaks in a minority of subjects.

## D12 Known limits of the A3 tests
The cache is 1-40 Hz and nothing is rejected (D7). The 1 Hz high-pass removes sustained gaze offset (leaving the saccade transient), and EMG above 40 Hz is invisible, so oculomotor and EMG decodability are lower bounds. Alternative: reload EDFs at 0.1-100 Hz for these two routes only. Expect: higher frontal accuracy, and a real EMG test. Not done unless the first pass leaves the oculomotor result ambiguous; any such sensitivity run will be reported next to this one, not instead of it.

## D13 A3 order and protocol routes
Order features: trial index in run (0-14), onset (s) in run, previous 1-3 labels (+1/-1, 0 when absent). Run identity cannot be learned under leave-one-run-out by construction, so its worst case is computed analytically: predicting each run's majority class gives sum over runs of max(L, R) / n. Protocol differences between subjects (128 Hz, cue length) are the same for both classes within a subject, so they cannot create a label route in per-subject decoders; they can only matter in pooled decoders if a protocol group has a different class ratio, which the audit counts show it does not (audit.csv).
