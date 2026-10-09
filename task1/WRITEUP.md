# Task 1 write-up (draft for Keyen to edit)
Numbers come from `outputs/A3_RESULTS.md`, `outputs/partb/B2_RESULTS.md` and `PREDICTIONS.md`. Chance thresholds: exact one-sided binomial at 5% (D15). A3 pooled: 1793 trials, 52.0%. B2 test set: 20 subjects, 913 trials, 52.8%.

## (a) If a model scores 70% on these subjects, how much of that needs motor imagery?
Almost none. The subject looks at the cue target, and a frontal decoder on the first 0.5 s after the cue (the saccade window, before ERD develops) reaches 71.9% across subjects and passes threshold in 25 of 40, while its pre-cue twin is at chance (48.2%). The sensorimotor decoder reaches only 54.6% and 8/40, although A2 finds lateralised ERD in 14/40, so the motor signal is real but inconsistent across subjects while the eye signal has the same sign in everyone.

The networks follow the eye route. EEGNet (76.9%) and the time-patch transformer (71.6%) rely most on F7, F8, Ft7, Ft8 and Af8 (horizontal EOG positions), and both fall to chance (50.5%, 49.1%) once that route is removed (Laplacian, then the 21 Fc/C/Cp electrodes, D25). The problem is not that they cannot fit motor imagery, it is that what they fit does not transfer, as validation-subject loss rises from the first epoch (D27). A 70% score is therefore fully explainable without motor imagery, with the eye route alone at 72% and the motor route under 55%.

## (b) Mechanisms behind the B2 results
Scaling contradicted the predicted crossover. EEGNet leads at every size (72.1% at 5 subjects, 76.9% at 30) and the transformer closes the gap from ~17 to ~5 points without crossing. EEGNet hard-codes the problem as a filter bank with per-band spatial filters, so ~2900 weights pin down the frontal eye pattern from five subjects, while the transformer must learn how 20 time segments combine and that freedom costs data. A crossover needs a signal richer than EEGNet's form, and a spatially broad eye field is not one.

Tokenisation was partly wrong. Time-patch (71.6%) and channel tokens with identity (70.2%) tie, but without identity accuracy falls to 57.5%, because attention with a CLS read-out is then permutation-invariant over electrodes (predictions 100% unchanged under shuffling and left/right mirroring) and ERD under C3 and under C4 look the same.

Displacement ranked as predicted (time-patch drops most), but costs under one point at 10 mm and at most 1.5 at 20 mm. EEGNet's temporal pooling averages the spatial-layer perturbation back from 17% to 13%, while the transformer's grows to 17% because its spatial filters feed every token and the softmax amplifies a coherent change.

Faithfulness ranked as predicted. Deleting the most-attended half of the tokens drops accuracy from 71.4% to 57.7%, against 64.1% for gradient x input and 65.6% for random, because deletion renormalises the softmax, which last-layer CLS attention captures and a first-order gradient estimate does not.

The link to Part A was half right and holds the largest contradiction. Per test subject, EEGNet accuracy correlates with both the A2 LI (Spearman 0.72) and the A3 frontal decoder (0.53), which are unrelated (-0.02), so the network uses both routes in different subjects. I predicted a small drop without the eye route, but it falls to chance because the motor pattern is subject-specific and only the eye pattern is shared.

The intervention (random 0-15 mm cap slides in training) had no effect, as predicted, because to first order it penalises the loss change along a slide, which displacement showed is negligible (accuracy 77.7% vs 76.9%, filter sensitivity 0.40 vs 0.41).

A cross-subject network here learns motor imagery only if the shared non-motor route is removed and the motor pattern transfers across subjects, and on these 40 subjects the second condition fails.
