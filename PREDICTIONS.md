# PREDICTIONS
Rule: each entry is committed before the commit that contains its result.

## A3 (DRAFT by Claude for Keyen to edit/approve before commit) - which non-motor route is decodable?
Context: in this experiment a target appears on the left or right of the screen and the subject imagines until it disappears; trials are separated by rest periods; order is randomised within run.
Routes: (1) visual/oculomotor: target side drives gaze shift / EOG and visual-evoked response (frontal Fp/AF, occipital); (2) EMG/posture or blink artefacts; (3) trial order / time-in-run drift; (4) pre-cue baseline (no label information should exist); (5) run identity / cue-length protocol differences (128 Hz subjects).
Prediction: route (1) is the decodable one, mainly via frontal-polar channels in the first 0.5 s after the cue, in a minority of subjects (~10-15 of 40), with pooled cross-subject accuracy of 55-60% from frontal channels only and ~50% from occipital-only; the pre-cue window will be at chance (within the binomial threshold) and trial-order decoders at chance. Reasoning: the cue is lateralised in space, an unavoidable eye-movement route exists, while a randomised order leaves no label information in time or baseline.

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
