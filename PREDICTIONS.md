# PREDICTIONS
Rule: each entry is committed before the commit that contains its result.

## A3 - which non-motor route is decodable? (2026-10-07, before any confound decoder was run)
Context: in this experiment a target appears on the left or right of the screen and the subject imagines until it disappears; trials are separated by rest periods; order is randomised within run.
Routes: (1) oculomotor: target side drives gaze shift / EOG (frontal Fp/AF/F7/F8); (2) visual: lateralised visual-evoked response and attention-related alpha (occipital); (3) EMG/posture or blink artefacts; (4) trial order / time-in-run drift / previous-trial label; (5) pre-cue baseline (no label information should exist); (6) run identity / cue-length protocol differences (128 Hz subjects).
Pooled = cross-subject, train on some subjects and test on unseen ones; chance threshold from the binomial/permutation null (about 52% for ~1800 trials).

1. Decodable routes: eye movements AND the visual response (routes 1 and 2).
2. Frontal-only decoder, subjects individually above chance: 5-10 of 40 (luck alone gives ~2).
3. Frontal-only, pooled cross-subject accuracy: 50-52% (chance).
4. Occipital-only, pooled cross-subject accuracy: 53-57%.
5. Pre-cue-window decoder and trial-order / previous-label decoders: both at chance.
6. Positive control, motor strip only (C3/C4 neighbourhood, mu+beta, 0.5-4 s), pooled cross-subject accuracy: 56-62%.

Reasoning (drafted by Claude from Keyen's answers; Keyen to edit): the cue is lateralised in space, so gaze shift and a lateralised visual response are unavoidable routes; a randomised order leaves no label information in time or baseline. Items 2 and 3 together imply the eye signal is strong in a few subjects but not consistent in polarity or timing across subjects, so it cancels when pooled; the visual response is expected to transfer across subjects better than the eye signal. Pre-cue and order decoders are the method check: if they beat chance, the pipeline leaks, not the physiology.

History: the earlier draft of this entry (frontal-polar only, 10-15 subjects, frontal pooled 55-60%, occipital ~50%) is in commit fcf1214 and was superseded by the answers above before any result existed.

## A3 outcome (appended after results; the prediction text above is unchanged)
Scorecard and numbers: outputs/A3_RESULTS.md. Short version: eyes decodable far above prediction (25/40 subjects, 72% pooled vs predicted 5-10 and 50-52%); occipital slightly above (59% vs 53-57%); motor control slightly below (54.6% vs 56-62%); pre-cue and order were NOT at chance: label order alternates (z = -17.6), previous label alone gives 72.6%, and my zero-phase filter made the pre-cue window look decodable until replaced by a causal filter (DECISIONS D18).

## B2 - all six experiments (2026-10-08, multiple choice, option text quoted as chosen)
Setup fixed before answering: test subjects = folds 0-3 (20 subjects), 3 seeds; EEGNet and time-patch transformer for scaling; confound removal = local Laplacian then 21 sensorimotor channels; intervention = EEGNet trained with random simulated cap slides of 0-15 mm (DECISIONS D24-D26).
Timing note: the training grid (task1/b2_run.py --stage base) was started before this commit to save time on the deadline; its log prints no accuracy and no result file was opened before this commit.

1. Data scaling: "Crossover" - EEGNet ahead at 5-10 subjects (convolutional priors: shared temporal filter + one spatial filter), transformer catches up or passes by 30.
2. EEGNet test accuracy at 30 training subjects: "> 72%" - combines eye and motor signal, beats every single A3 decoder.
3. Tokenisation: "time > chan+id = chan-noid" - channel tokens struggle whether or not they have identity.
4. Electrode displacement (10 mm): "Time-patch drops most" - its spatial filter feeds every token, so the error spreads to all of them.
5. What the best transformer relies on: "Both" - eye signal early, motor later.
6. Faithfulness: "Attention > gradient > random" - attention to CLS points to the decisive tokens.
7. Link to Part A: "A2 LI; small drop" - networks mostly use real lateralised ERD; per-subject accuracy correlates most with the A2 lateralisation index, and removing the eye confound costs little.
8. Intervention (spatial augmentation): "No effect" - if the model relies on the spatially broad eye signal, it was never displacement-sensitive.

## B2 outcome (appended after results; the prediction text above is unchanged)
Full write-up and numbers: outputs/partb/B2_RESULTS.md. Scorecard:
| # | Predicted (Keyen) | Found | Verdict |
|---|---|---|---|
| 1 | Scaling: crossover | EEGNet ahead at every size; gap shrinks from ~17 to ~5 points | wrong |
| 2 | EEGNet at 30 subjects > 72% | ~77% | right |
| 3 | time > chan+id = chan-noid | time ~ chan+id >> chan-noid (identity worth ~13 points; no-identity model exactly mirror-invariant) | partly wrong |
| 4 | Displacement: time-patch drops most | yes at 15-20 mm, but every model loses < 1 point at 10 mm | right (ranking), effect tiny |
| 5 | Reliance: both eye and motor | mainly the eye route (F7/F8/Ft7/Ft8; transformer peaks 0.2-0.4 s, EEGNet 0.6-1.1 s); motor not separable and does not survive removal | partly right |
| 6 | Faithfulness: attention > gradient > random | attention > gradient ~ random | right |
| 7 | Link: A2 LI, small drop when confound removed | EEGNet tracks LI most (and F1); removal drops both models to chance | half right |
| 8 | Intervention: no effect | no effect on accuracy, curve or spatial-filter sensitivity | right |

## Task 2 / 2c - direction of each defect's effect (written by Claude BEFORE the ledger was run; Keyen may edit before the results commit)
Setup: fixed pipeline (task2/sleep_pipeline.py, defects 1-10 fixed) vs the same pipeline with ONE defect re-introduced; seeds 42, 43, 44, each seed also changes the subject split; delta = defective - fixed on the same seed; test metrics are accuracy / macro-F1 / kappa. Seed 42 of the fixed pipeline has been run before (acc .845, macro-F1 .810, kappa .799), so that row is a reproducibility check, not a prediction.
Expected sign and size (macro-F1 unless stated). "Seed spread" will be large: the test set is only 3 subjects.
- D1 recording-level split: UP, +0.01 to +0.05 on all three metrics (person-specific EEG leaks); the single seed-42 pair already run went the other way (+.04 for the FIXED split) so I expect the sign to be unstable across seeds.
- D2 selection on test: UP, +0.005 to +0.02 (maximum over 12 noisy evaluations); largest when validation and test curves disagree.
- D3 attention across the batch: DOWN, -0.03 to -0.10 macro-F1 and kappa; mostly N1 and REM, whose scoring needs neighbouring epochs. Accuracy moves least.
- D4 positional encoding after attention (D3 fixed): DOWN, small, 0 to -0.03. With D3 present D4 should have NO effect (VANISHES): batch attention has no epoch order to lose. D3+D4 should equal D3 within the spread.
- D5 last-epoch label: DOWN, -0.01 to -0.04, N1 most (71% of N1-centred windows get a different label); partly absorbed because the model can attend to position 10.
- D6 units bug (nothing rejected): about 0 (1 epoch of 80070 differs). Naive unit fix on all channels (D6N): DOWN, -0.03 to -0.10, REM and Wake lose most (EOG channel rejects 54% of epochs). D6N+D7 worse than D6N because windows then span large gaps.
- D7 deletion instead of masking: about 0, within noise (10 windows in one recording). It vanishes entirely when D6 is also present (nothing is ever deleted) and only becomes large combined with D6N.
- D8 per-epoch z-scoring: DOWN, -0.01 to -0.04, N3 F1 most (amplitude cue removed), N2/N3 confusion up.
- D9 unscored -> Wake: DOWN, tiny, 0 to -0.01 (940 of 80070 epochs).
- D10 no cropping: accuracy UP by +0.08 to +0.14 (free Wake: 'always Wake' scores 68.7%), kappa UP by +0.02 to +0.08, macro-F1 ambiguous (-0.03 to +0.03) because class weights shrink Wake and N1 can only lose from the extra Wake false positives. This is the defect most likely to mislead the headline accuracy.
- D1+D2: UP, more than either alone (both inflate the test number).
- D9+D10: the D9 effect should grow with D10 (many more unscored/Wake-labelled tail epochs); flag if the sign flips.
- ALL defects together (an approximation of the given script, with a validation set added): accuracy about the same or higher than the fixed pipeline, macro-F1 lower by 0.08 to 0.20 and kappa lower by 0.05 to 0.15 (the seed-42 original gave .847/.628/.693 against .845/.810/.799).
Reasoning for ranking before seeing data: D10 and D3 would have misled the report most (D10 inflates accuracy, D3 silently disables the context model that the architecture claims), then D1/D2 (inflation), D8, D5, D4, with D6/D7/D9 negligible on this data.
