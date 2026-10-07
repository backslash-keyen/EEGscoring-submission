# PREDICTIONS
Rule: each entry is committed before the commit that contains its result.

## A3 - which non-motor route is decodable? (answers by Keyen, 2026-10-07, before any confound decoder was run)
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
