# Task 1 write-up
Numbers come from `outputs/A3_RESULTS.md`, `outputs/partb/B2_RESULTS.md` and `PREDICTIONS.md`. Thresholds: A3 pooled 52.0% (1793 trials), B2 test set 52.8% (913 trials).

## (a) If a model scores 70% on these subjects, how much needs motor imagery?
Almost none.

- Eye route alone (frontal, 0-0.5 s after the cue): 71.9% pooled, above threshold in 25/40. Its pre-cue twin is at chance (48.2%).
- Motor route (sensorimotor mu+beta): 54.6% pooled, 8/40, although A2 finds lateralised ERD in 14/40. The motor signal is real but differs between subjects. The eye signal has the same sign in everyone.
- Both networks rely on F7, F8, Ft7, Ft8, Af8 (horizontal EOG sites) and fall to chance without the eye route (EEGNet 50.5%, transformer 49.1%).

## (b) B2 mechanisms
| Result | Mechanism |
|------|------------|
| No crossover with data (predicted a crossover: wrong) | EEGNet hard-codes a filter bank with per-band spatial filters, so ~2900 weights fit the broad frontal eye pattern from 5 subjects. The transformer must learn how 20 time segments combine, which costs data |
| Channel tokens without identity: 57.5% | Attention with a CLS read-out is permutation-invariant over electrodes, so C3 and C4 look the same |
| Displacement costs < 1 point at 10 mm | EEGNet's temporal pooling averages the spatial perturbation down (17% to 13%). The transformer's spatial filters feed every token, so it drops most |
| Attention beats gradient in faithfulness | Deleting tokens renormalises the softmax, which last-layer attention captures and a first-order gradient does not |
| Confound removed: chance (predicted a small drop: wrong) | The motor pattern is subject-specific, so it does not transfer. Only the eye pattern is shared across subjects |
| Intervention: no effect (predicted) | Training on slides penalises the loss change along a slide, which displacement showed is already negligible |
