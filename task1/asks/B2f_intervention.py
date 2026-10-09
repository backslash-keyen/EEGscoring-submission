"""B2 (part 6) - One intervention aimed at the identified mechanism, explained mathematically, run, compared with the prediction.

Brief: "Propose one change targeting the mechanism you identified (not a hyperparameter sweep), explain it mathematically,
run it, and compare the outcome with your prediction."

Intervention: train EEGNet on randomly cap-slid copies of each trial (0-15 mm, task1/augment.py), so the spatial filters
are pushed toward patterns that are smooth over the scalp. Live part: draws the augmentation bank and shows what one
training batch looks like after augmentation.
Run: python task1/asks/B2f_intervention.py
"""
from _common import *

header("B2.6  Intervention: training with simulated cap slides",
       "one change targeting the identified mechanism, explained mathematically, run, compared with the prediction.",
       "Mechanism targeted: position sensitivity of the depthwise spatial filters (B2.3). Change: random 0-15 mm slides of "
       "each training trial; validation and test unaugmented. EEGNet, 30 training subjects, 3 seeds.")

section("Prediction (PREDICTIONS.md, B2 item 8)")
say("\n".join(l for l in prediction(r"^## B2 - all six").splitlines() if l.startswith("8.")))

section("The mathematics (DECISIONS.md D26, last part)")
say("For a small slide M = I + d J (J = spatial derivative of the spline interpolation along the slide), the loss "
    "averaged over random slides of variance sigma^2 is, to second order, E[L(Mx)] ~ L(x) + (sigma^2 / 2) E[(grad_x L . J x)^2]. "
    "Training with slides therefore adds a Tikhonov penalty on the loss change along a slide (Bishop 1995: training "
    "with noise = Tikhonov regularisation). It is large only if the loss depends on exact electrode positions; it pushes "
    "the depthwise spatial weights towards patterns that are smooth over the scalp.")

if has_data():
    augment, data = pipeline("augment", "data")
    A = augment.Augmenter(seed=0)
    dev = np.abs(A.bank - np.eye(64)).max((1, 2))
    print(f"\n  augmentation bank: {len(A.bank)} slide matrices; max |M - I| from {dev.min():.3f} to {dev.max():.3f}")
    D = data.load_subject(72, causal=True)
    X = (D["X"][:16, :, 240:880] * 1e6).astype(np.float32)
    Xa = A(X)
    rel = np.linalg.norm(Xa - X, axis=(1, 2)) / np.linalg.norm(X, axis=(1, 2))
    print(f"  one batch of 16 S072 trials: augmentation changes each by {100 * rel.min():.1f}-{100 * rel.max():.1f}% (relative L2)")
else:
    no_data_note()

section("Result: base EEGNet vs EEGNet + slide augmentation (every seed)")
L = csv("partb/b2_seed_level.csv")
L = L[(L.model == "eegnet") & (L.n_train == 30) & L.exp.isin(["base", "aug"])]
print((100 * L.pivot(index="exp", columns="seed", values="acc")).round(1).to_string())
Dp = csv("partb/b2_displacement.csv")
Dp = Dp[Dp.label.isin(["eegnet", "eegnet+aug"]) & Dp.shift_mm.isin([0, 10, 20])]
Dp["cell"] = [f"{100 * m:.1f} +- {100 * s:.1f}" for m, s in zip(Dp["mean"], Dp["std"])]
print("\n  accuracy (%) under test-time slides:")
print(Dp.pivot(index="label", columns="shift_mm", values="cell").to_string())
Ssens = csv("partb/b2_eegnet_spatial_sensitivity.csv").groupby("exp").spatial_sensitivity_10mm.agg(["mean", "std", "count"])
print("\n  spatial-filter sensitivity ||W(M - I)|| / ||W|| at 10 mm (12 models each):")
print(Ssens.round(3).to_string())

section("Outcome vs prediction")
say("No effect: clean accuracy within seed spread, the displacement curve is not flatter, and the learned spatial filters "
    "are as displacement-sensitive as before. The penalty is near zero because the loss barely depends on exact "
    "positions in the first place (B2.3: what the models use, the eye field and broad lateralised patterns, is smooth "
    "over 10-15 mm). The intervention targeted a mechanism the data showed was not the limiting one. Prediction "
    "'no effect': right.")
finish()
