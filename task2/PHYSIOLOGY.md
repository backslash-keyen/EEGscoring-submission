# Task 2d: the worst stage in the fixed model, and why

Model: the fixed `task2/sleep_pipeline.py`, seeds 42-44 (each seed is also a different subject split, 3 test subjects).
Metrics from `task2/ledger_results/FIXED_*.json`; signal facts from the files themselves (`task2/physiology/dataset_facts.txt`);
per-epoch physiology in microvolts on the same cropped epochs (`task2/physiology/epoch_features.csv`). All produced by `task2/physiology.py` and `task2/ledger.py`.

## What this dataset records
Sleep Cassette, 29 nights (subjects 0-14; subject 13 has one night). Each file: EEG Fpz-Cz and Pz-Oz at 100 Hz, horizontal EOG at 100 Hz,
and submental EMG, oro-nasal respiration and rectal temperature only as **1 Hz** envelopes (`dataset_facts.txt`). The pipeline uses Fpz-Cz, Pz-Oz and EOG; the EMG is in the file but unused.
Labels are Rechtschaffen & Kales (R&K) stages per 30-s epoch, mapped to AASM by merging stages 3 and 4 into N3. There are no central (C3/C4) or occipital (O1/O2) AASM derivations: Fpz-Cz stands in for the central channel and Pz-Oz for the occipital one.

## Worst stage: N1
| stage | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| F1, mean ± SD (3 seeds) | 0.889 ± 0.041 | **0.511 ± 0.017** | 0.851 ± 0.033 | 0.862 ± 0.022 | 0.827 ± 0.032 |

Row-normalised confusion, P(predicted | true), mean of 3 seeds:

| true \ pred | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| W | **0.871** | 0.070 | 0.005 | 0.002 | 0.051 |
| N1 | 0.125 | **0.622** | 0.173 | 0.001 | 0.079 |
| N2 | 0.005 | 0.075 | **0.852** | 0.032 | 0.036 |
| N3 | 0.001 | 0.008 | 0.175 | **0.815** | 0.001 |
| REM | 0.033 | 0.078 | 0.073 | 0.001 | **0.816** |

N1 recall is 0.61, but precision only 0.44: N1 is where the model puts epochs it cannot place. Of the windows it called N1 (summed over seeds), 486 were N2, 276 REM and 262 Wake, against 832 that were N1.

**Why, from the scoring rules and these signals.** N1 is defined by what is *absent* or *transitional*: alpha falls to under half the epoch (from Wake), low-amplitude mixed-frequency 2-7 Hz activity appears, with slow rolling eye movements and vertex sharp waves; it ends at the first spindle or K-complex (N2) or when REM signs appear. Every one of its features is shared with a neighbour (median per-epoch values, all epochs):

| median | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| Fpz-Cz peak-to-peak (uV) | 169 | 78 | 119 | 178 | 79 |
| Pz-Oz relative alpha (8-12 Hz) | 0.113 | 0.084 | 0.042 | 0.013 | 0.088 |
| Fpz-Cz relative theta (4-8 Hz) | 0.066 | 0.118 | 0.102 | 0.042 | 0.150 |
| Fpz-Cz relative sigma (12-15 Hz) | 0.009 | 0.016 | 0.028 | 0.006 | 0.013 |
| EOG 0.3-1 Hz power (uV²), slow eye movements | 1757 | 284 | 38 | 119 | 271 |
| chin EMG (uV, 1 Hz envelope) | 3.23 | 2.68 | 1.98 | 1.48 | **0.69** |

On amplitude, alpha, theta and slow eye movements, N1 sits almost on top of REM (78 vs 79 uV, alpha 0.084 vs 0.088, EOG 284 vs 271), and between Wake and N2 on the rest. What separates it from REM in the rules is muscle tone (N1 2.68 uV vs REM 0.69 uV), which the pipeline never sees. What separates it from N2 is a single transient event (spindle or K-complex) that a 30-s spectrum dilutes. It is also the rarest stage (2184 of 29978 scored epochs, 7%) and the one human scorers agree on least (Rosenberg & Van Hout 2013, J Clin Sleep Med: N1 had the lowest inter-scorer agreement of all stages, about 63%), so its labels are the noisiest targets.

## One expected confusion: N3 → N2 (17.5% of true N3)
R&K stage 3 starts when 20% of the epoch contains >75 uV, 0.5-2 Hz waves. That is a threshold on a continuum: the median N3 epoch has 27% coverage by our 1-s approximation (`delta_cover`), close to the boundary, and N2 epochs carry K-complexes and some slow waves. Epochs near the 20% line are labelled by a count a scorer makes by eye, so N3 → N2 (517 windows over the 3 seeds) is the confusion the rules themselves predict. It goes almost only one way (N2 → N3 is 3.2%), which fits a frontal Fpz-Cz derivation that sees the slow waves well: what is lost is the borderline, not the clear N3.

## One unexpected confusion: Wake ↔ REM (5.1% of Wake → REM, 3.3% of REM → Wake; 304 windows)
For a human scorer these two are rarely confused: REM requires atonia of the chin muscle, and Wake has high tone. This recording has the cleanest separator of any feature in the table (chin EMG median 3.23 uV in Wake vs 0.69 in REM, a 4.7-fold difference), but the pipeline drops it (and its 0.3 Hz high-pass would remove most of a slowly varying 1 Hz envelope's level if it were added as is). On the channels it does use, REM looks like relaxed Wake: low-amplitude mixed EEG, alpha bursts on Pz-Oz (0.088 vs 0.113), and rapid eye movements on the EOG. The error is unexpected from the physiology and fully explained by the channel choice. Adding the EMG envelope as a per-epoch scalar input would be the targeted fix.

