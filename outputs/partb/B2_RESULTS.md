# Part B results (B1 models, B2 experiments)

Reproduce: `python task1/b2_run.py --stage all` (training grid, resumable), then `b2_results.py`, `b2_displacement.py`, `b2_attribution.py`, `b2_invariance.py`, `b2_noconf_check.py` (post hoc).
Evaluation: cross-subject only. Test subjects = A3 folds 0-3 (20 subjects, 913 trials) for every number below. Early stopping on a separate validation fold, normalisation from training subjects only (DECISIONS D22-D24). Chance threshold for 913 trials: 52.8% (exact one-sided binomial, 5%, same rule as A3 D15).
Every run (fold x seed) is in `b2_all_runs.csv`; seed-level accuracies (pooled over the 4 test folds) in `b2_seed_level.csv`; `b2_summary.csv` gives mean, SD, min and max over seeds. Predictions were committed before any result was read (PREDICTIONS.md, commit f1ee430).

## B1 models
| model | parameters | x EEGNet |
|---|---|---|
| EEGNet-8,2 at 160 Hz (0.5 s kernels rescaled: 64->80, 16->20 samples; D20) | 2962 | 1.00 |
| transformer, time-patch tokens (20 tokens of 0.2 s, all electrodes) | 6834 | 2.31 |
| transformer, channel tokens + learned electrode identity (64 tokens) | 7250 | 2.45 |
| transformer, channel tokens, no identity | 6226 | 2.10 |

All transformers share EEGNet's first 0.5 s temporal filter layer, so only the token axis differs (D21).

## 1. Data scaling (3 seeds; seeds also draw the training subjects)
| model | 5 subjects (%, mean +- SD) | 10 subjects (%, mean +- SD) | 20 subjects (%, mean +- SD) | 30 subjects (%, mean +- SD) |
|---|---|---|---|---|
| eegnet | 72.1 +- 0.5 (seeds=3) | 74.2 +- 0.2 (seeds=2) | 76.6 +- 1.6 (seeds=2) | 76.9 +- 1.0 (seeds=3) |
| tf_time | 55.0 +- 2.9 (seeds=3) | 64.4 +- 4.2 (seeds=2) | 67.9 +- 5.1 (seeds=2) | 71.6 +- 3.2 (seeds=3) |

Figure: `b2_scaling.png`. **No crossover**: EEGNet is ahead at every size; the transformer gains most per added subject (the gap shrinks from ~17 points at 5 subjects to ~5 at 30; table above). Prediction was "crossover": **wrong**.
Why, from the architectures: EEGNet hard-codes what EEG decoding needs: one temporal filter bank shared by all electrodes, then one spatial filter per frequency band (a CSP-like projection), then pooled power-like features. With ~2900 weights in that fixed form, five subjects (~220 trials) already pin down the frontal eye-movement pattern and the lateralised spatial pattern, and the curve is nearly flat (+5 points from 5 to 30 subjects). The time-patch transformer has the same front end but must learn from data how segments combine (attention plus a learned position embedding) instead of fixed average pooling; that extra freedom costs data, and with 5 subjects it overfits (near chance). At 30 subjects it approaches EEGNet but the data are too few (and the decodable signal too simple, see section 4) for the extra flexibility to pay off.

## 2. Tokenisation (30 training subjects, 3 seeds)
| model | accuracy % (mean +- SD) | range over seeds | seeds |
|---|---|---|---|
| eegnet | 76.9 +- 1.0 | 76.0-78.0 | 3 |
| tf_time | 71.6 +- 3.2 | 67.9-74.0 | 3 |
| tf_chan_id | 70.2 +- 1.2 | 69.2-71.5 | 3 |
| tf_chan_noid | 57.5 +- 1.6 | 55.8-58.8 | 3 |

Prediction "time > chan+id = chan-noid": **partly wrong**: time-patch and channel tokens with identity are close (difference within the seed spread), and identity matters a lot (~13 points).
What channel tokens without identity cannot represent: without an identity (or position) embedding the encoder sees an unordered set of electrode tokens. Self-attention and the CLS read-out are permutation-equivariant/invariant, so the output is the same for every reordering of the electrodes. It therefore cannot represent *where* a signal is: "ERD under C3" and "ERD under C4" are the same input set. Tested directly (`b2_invariance.py`, `b2_invariance_runs.csv`): with no identity, predictions are 100% unchanged under a random electrode shuffle and under the left/right mirror (C3<->C4, F7<->F8, ...); on mirrored test trials with flipped labels its accuracy is exactly 1 - accuracy. EEGNet and the time-patch transformer instead flip their answer on mirrored trials and stay as accurate on them (`b2_invariance_runs.csv`), i.e. they learned a genuinely lateralised (mirror-antisymmetric) decision. It matters for this task because the label *is* a left/right mirror variable: both the motor route (contralateral ERD) and the eye route (saccade towards the target: F7 vs F8 polarity) are mirror patterns. The above-chance accuracy the no-identity model still reaches can only come from what is not mirror-symmetric in the recordings (e.g. an off-centre recording reference, unequal left/right saccade sizes).

## 3. Electrode displacement (test time only; 4 directions averaged; 3 seeds)
| label | 0 mm | 2.5 mm | 5 mm | 10 mm | 15 mm | 20 mm |
|---|---|---|---|---|---|---|
| eegnet | 76.1 +- 2.3 | 76.3 +- 2.2 | 76.4 +- 2.2 | 76.3 +- 2.1 | 75.5 +- 2.6 | 75.1 +- 2.7 |
| eegnet+aug | 77.3 +- 1.3 | 77.3 +- 1.3 | 77.1 +- 1.4 | 76.8 +- 1.2 | 76.4 +- 0.8 | 76.0 +- 1.2 |
| tf_chan_id | 70.4 +- 1.6 | 70.4 +- 1.3 | 70.3 +- 1.4 | 70.1 +- 1.8 | 70.1 +- 2.0 | 70.0 +- 2.0 |
| tf_chan_noid | 58.4 +- 0.6 | 58.3 +- 0.6 | 58.2 +- 0.5 | 58.3 +- 0.7 | 58.2 +- 0.8 | 58.2 +- 0.7 |
| tf_time | 73.4 +- 0.9 | 73.3 +- 1.1 | 73.2 +- 1.3 | 73.0 +- 1.0 | 72.3 +- 0.5 | 71.6 +- 0.2 |

Figure: `b2_displacement.png`; lateral-only shifts in `b2_displacement_lateral.csv`. 10 mm = a quarter of the 35-39 mm electrode spacing (D26).
Degradation is small for every model: at most about half a point at 10 mm and 1-2 points at 20 mm (table). Largest at 20 mm: time-patch transformer, then EEGNet; channel tokens with identity lose a few tenths, without identity nothing. Prediction "time-patch drops most": **right in ranking**, but the effect is far smaller than the question implies.
Responsible layer (`b2_displacement_layer_summary.csv`, relative change of each layer's output for a 10 mm slide): the slide changes the input by ~13% in every model. In EEGNet the depthwise spatial filter raises this to 17% and block 2 to 21%, but the final dense layer brings it back to 13% at the logits (average pooling over 20 time steps of an eye/ERD pattern that is spatially smooth on the scale of 10 mm). In the time-patch transformer the change grows at every stage (spatial 13% -> tokens 15% -> logits 18%): its 16 spatial filters feed every token, and attention/softmax re-weighting turns a small, coherent change in all tokens into a larger change of the read-out. Channel tokens + identity: token change 12%, logits 15% (the identity embedding stays fixed while the signal under it moves). Channel tokens without identity: tokens change 13% but the CLS output only 3%: an unordered set of 64 tokens whose members each move a little still has nearly the same summary, which is why it is the most displacement-robust model (and the least accurate).
Why so small overall: what the models rely on (section 4) is the eye-movement field at F7/F8/Ft7/Ft8 and broad lateralised patterns; a 10 mm slide moves electrodes within fields that are much wider than 10 mm. The EEGNet spatial filters themselves are sensitive (||W(M-I)||/||W|| = 0.41 at 10 mm, `b2_eegnet_spatial_sensitivity.csv`), but the signal component they pick up is smooth, so the projection barely changes.

## 4. What the best transformer relies on
Best transformer by the pre-set rule (highest mean accuracy at 30 subjects): time-patch. Attribution: gradient x input on the tokens and on the raw input; attention = last-layer CLS attention averaged over heads (`b2_attribution.py`).
Input-space grad x input by scalp region (`b2_attr_input.csv`, figure `b2_attribution_maps.png`):

| model | region | n_electrodes | share % | per electrode % |
|---|---|---|---|---|
| tf_time | frontal (A3 eye) | 12.0 | 20.1 | 1.67 |
| tf_time | occipital (A3 visual) | 9.0 | 14.4 | 1.6 |
| tf_time | temporal | 8.0 | 15.6 | 1.96 |
| tf_time | sensorimotor (A2) | 21.0 | 30.6 | 1.46 |
| eegnet | frontal (A3 eye) | 12.0 | 20.9 | 1.74 |
| eegnet | occipital (A3 visual) | 9.0 | 12.7 | 1.41 |
| eegnet | temporal | 8.0 | 18.4 | 2.3 |
| eegnet | sensorimotor (A2) | 21.0 | 28.9 | 1.38 |

By time window (% of attribution):

| model | 0-0.5 s (A3 eye) | 0.5-1.5 s | 1.5-4 s |
|---|---|---|---|
| eegnet | 13.5 | 42.7 | 43.7 |
| tf_time | 26.1 | 31.9 | 42.0 |

Top 6 electrodes: eegnet: F7, Ft7, Ft8, Af8, T7, Po8; tf_time: Ft7, F7, F8, Ft8, T9, Af8

Faithfulness (`b2_faithfulness.csv`, figure `b2_faithfulness.png`): accuracy after deleting each trial's top-ranked tokens (key-padding mask + zeroed embedding), mean +- SD over the 12 models (4 folds x 3 seeds):

| method | 5% deleted | 10% deleted | 20% deleted | 30% deleted | 50% deleted |
|---|---|---|---|---|---|
| attention | 69.5 +- 8.8 | 66.3 +- 8.0 | 62.7 +- 8.9 | 60.9 +- 9.6 | 57.7 +- 10.7 |
| gradxinput | 70.1 +- 10.0 | 67.8 +- 9.8 | 66.9 +- 9.9 | 66.2 +- 9.7 | 64.1 +- 9.6 |
| random | 70.9 +- 9.2 | 70.4 +- 8.8 | 69.4 +- 9.1 | 68.3 +- 8.2 | 65.6 +- 7.0 |

Faithfulness: deleting the most-attended tokens hurts most (no deletion 71.4%; half of the tokens deleted: 57.7% by attention, 64.1% by gradient x input, 65.6% random). Prediction "attention > gradient > random": **right**. Gradient x input is a first-order estimate at the intact input; deleting a token also renormalises the softmax over the others, so a token whose own gradient is small can still be the one the CLS read-out depends on. Last-layer CLS attention measures that dependence directly. Gradient x input is barely better than random here, so it is not a faithful ranking for this model.
Mapping to Part A: both networks give the most attribution per electrode to the lateral frontal and temporal sites F7, F8, Ft7, Ft8, Af8, T7/T9: the horizontal electro-oculogram positions of the A3 eye route. Sensorimotor electrodes have the largest total share (31%) only because there are 21 of them; per electrode they are the least used region (1.46% vs 1.67% frontal and 1.96% temporal). In time, the transformer puts 26% of its attribution in the first 0.5 s (2x a uniform share), peaking at 0.2-0.4 s: the A3 saccade window. EEGNet peaks at 0.6-1.1 s, which matches the second frontal peak in A3 at 0.7-0.9 s (return saccade). The remaining ~40% spread over 1.5-4 s could be motor ERD or sustained eye position, and attribution cannot separate the two. Section 5 does: without the eye route, nothing transfers across subjects. Prediction "Both": **partly right**. The models rely mainly on the eye signal; any motor contribution cannot be separated by attribution and does not survive confound removal.

## 5. Link to Part A and confound removal
Per test subject (20 subjects), accuracy (mean over seeds, 30 training subjects) vs Part A measures (`b2_link_correlations.csv`, Spearman):
| exp | model | A2 LI (mean of mu, beta) | A3 F1 frontal 0-0.5 s | A3 M1 motor strip | A3 O1 occipital |
|---|---|---|---|---|---|
| aug | eegnet | 0.76 (p=0.000) | 0.54 (p=0.013) | 0.36 (p=0.115) | -0.06 (p=0.789) |
| base | eegnet | 0.72 (p=0.000) | 0.53 (p=0.016) | 0.34 (p=0.145) | -0.04 (p=0.861) |
| base | tf_chan_id | 0.52 (p=0.020) | 0.51 (p=0.020) | 0.30 (p=0.200) | 0.11 (p=0.630) |
| base | tf_chan_noid | 0.26 (p=0.272) | -0.05 (p=0.829) | 0.01 (p=0.979) | -0.15 (p=0.516) |
| base | tf_time | 0.28 (p=0.227) | 0.54 (p=0.013) | 0.24 (p=0.310) | 0.04 (p=0.868) |
| noconf | eegnet | -0.15 (p=0.529) | 0.54 (p=0.015) | 0.30 (p=0.206) | 0.02 (p=0.947) |
| noconf | tf_time | -0.01 (p=0.956) | 0.41 (p=0.069) | 0.13 (p=0.591) | -0.36 (p=0.122) |

Across all 40 subjects, A2 LI and A3 F1 (frontal eye decoder) are unrelated (rho = -0.02), so a model's accuracy tracking both means it draws on both, in different subjects.
Confound removed (Laplacian over 64 electrodes, then the 21 Fc/C/Cp electrodes; D25):
| model | full input % | confound removed % | seeds (removed) |
|---|---|---|---|
| eegnet | 76.9 +- 1.0 | 51.3 +- 3.6 | 2 |
| tf_time | 71.6 +- 3.2 | 49.7 +- 0.8 | 2 |

Both models fall to chance. Diagnostic (post hoc, D27, `b2_noconf_check.csv`): with patience 40, training loss falls steadily (0.70 -> 0.57) while validation-subject loss rises from the first epoch (0.71 -> 0.86) and validation accuracy stays below 50% for 40 epochs. The network does fit the training subjects' motor data, but what it learns does not transfer to new subjects. Prediction "A2 LI; small drop": **correlation part right** (EEGNet tracks LI most), **drop part wrong** (to chance).
Per-subject accuracy of the confound-removed EEGNet still correlates with the A3 frontal eye decoder (table above), so some eye signal survives the Laplacian in a few subjects, but not enough for a pooled effect.
What remains: essentially nothing cross-subject. The motor signal exists (A2: 14/40 subjects with significant lateralised ERD; A3 M1 within-subject 8/40 above threshold), but it is subject-specific in strength and spatial pattern; a decoder that cannot borrow the eye-movement signal, which is consistent across subjects, has no common signal to learn from 30 subjects.

## 6. Intervention: training with simulated cap slides (EEGNet; D26)
| model | 0 mm % | 10 mm % | 20 mm % | spatial-filter sensitivity at 10 mm |
|---|---|---|---|---|
| eegnet | 76.1 +- 2.3 | 76.3 +- 2.1 | 75.1 +- 2.7 | 0.413 |
| eegnet+aug | 77.3 +- 1.3 | 76.8 +- 1.2 | 76.0 +- 1.2 | 0.411 |

Prediction "No effect": **right**. Clean accuracy unchanged (within seed spread), the displacement curve is not flatter, and the learned spatial filters are as displacement-sensitive as before (0.41 vs 0.41). Mechanism: augmentation adds the penalty (sigma^2/2) E[(grad_x L . J x)^2], the loss change along a slide. That penalty is only large if the loss depends on the exact electrode positions. Section 3 shows it does not (the eye and lateralised fields are smooth over 10-15 mm), so the penalty is near zero and the optimum barely moves. The intervention targeted a mechanism (spatial-filter position sensitivity) that the data showed was not the limiting one.

## Results that contradicted the predictions
1. Scaling: no crossover (predicted crossover).
2. Confound removal: chance, not a small drop. The networks' cross-subject accuracy depends on the eye-movement route even though per-subject accuracy also tracks lateralised ERD.
3. Channel tokens: identity matters (~13 points); the no-identity model is not near chance but is exactly mirror-invariant (predicted chan+id = chan-noid).
