"""B2 (part 3) - Electrode displacement at test time, by spatial interpolation; degradation curves; responsible layer.

Brief: "At test time only, simulate a physically plausible cap misplacement: justify its size (mm or degrees) and
implement it by spatial interpolation from the true electrode positions, not by shuffling channels. Report degradation
curves for EEGNet and the transformer variants and explain the difference down to the responsible layer or weights."

Live part: (1) the size justification, computed from the electrode positions; (2) the interpolation matrix of
task1/spatial.py for a slide of SHIFT_MM, checked (identity at 0 mm, how far each electrode moves); (3) how much that
slide changes real trials. Curves and the layer breakdown come from the 30-subject models (task1/b2_displacement.py).
Run: python task1/asks/B2c_electrode_displacement.py
"""
from _common import *

SHIFT_MM = 10.0                 # the headline size; try 5 or 20
DIRECTION = (1, 0)              # (+x towards the right ear, +y towards the nose)

header("B2.3  Electrode displacement",
       "test-time cap misplacement of justified size, by spatial interpolation from the true positions; degradation "
       "curves for EEGNet and the transformers; the responsible layer.",
       "Rigid cap slide = rotation of all 64 positions about the head centre; signal at each displaced site = spherical-"
       "spline interpolation of the recorded field (MNE, exact at the electrodes). 0-20 mm, 4 directions, 3 seeds.")
print_decision("D26")

spatial, data = pipeline("spatial", "data")
names = list(csv("electrode_positions_2d.csv").name)
p, r = spatial.positions(names)
section("Why 10 mm: electrode spacing on this montage (computed now from standard_1005 positions)")
d = np.linalg.norm(p[:, None] - p[None], axis=2) * 1000
np.fill_diagonal(d, np.inf)
for a, b in [("C3", "C1"), ("C3", "Cp3"), ("C3", "Fc3"), ("C3", "C5"), ("C4", "C2")]:
    print(f"  {a}-{b}: {d[names.index(a), names.index(b)]:.1f} mm")
nn_ = d.min(1)
print(f"  nearest-neighbour spacing over all 64 electrodes: median {np.median(nn_):.1f} mm (range {nn_.min():.1f}-{nn_.max():.1f})")
print(f"  head radius (sphere fitted through the electrodes): {1000 * r:.1f} mm -> 10 mm = {np.degrees(0.010 / r):.1f} degrees of rotation")
say("10 mm is about a quarter of the spacing between neighbouring electrodes: a cap that is visibly but not grossly "
    "misplaced. 20 mm (half the spacing) is the worst case shown.")

section(f"The interpolation matrix for a {SHIFT_MM:g} mm slide towards {DIRECTION}")
M0 = spatial.displacement_matrix(names, 0, DIRECTION)
M = spatial.displacement_matrix(names, SHIFT_MM, DIRECTION)
print(f"  0 mm: max |M - I| = {np.abs(M0 - np.eye(64)).max():.2e} (identity: exact spline, nothing changes without a slide)")
print(f"  {SHIFT_MM:g} mm: max |M - I| = {np.abs(M - np.eye(64)).max():.3f}; rows sum to {M.sum(1).min():.3f}-{M.sum(1).max():.3f}")
axis = np.cross([0, 0, 1.0], [DIRECTION[0], DIRECTION[1], 0])
R = spatial.rotation(axis, SHIFT_MM / 1000 / r)
moved = np.linalg.norm(p @ R.T - p, axis=1) * 1000
print(f"  distance each electrode moves: {moved.min():.1f}-{moved.max():.1f} mm (less near the rotation axis)")
i = names.index("C3")
top = np.argsort(-np.abs(M[i]))[:4]
print("  new C3 signal = " + " ".join(f"{M[i, j]:+.2f} x {names[j]}" for j in top) + " ...")

if has_data():
    D = data.load_subject(72, causal=True)
    X = D["X"][:, :, 240:880].astype(np.float64)              # 0-4 s after the cue, the network input window
    Xs = np.einsum("kc,nct->nkt", M, X)
    rel = np.linalg.norm(Xs - X, axis=(1, 2)) / np.linalg.norm(X, axis=(1, 2))
    print(f"  on S072's raw trials the slide changes the input by {100 * rel.mean():.1f}% (relative L2, mean over trials;"
          " the layer table below measures it after the first temporal filter, over the 20 test subjects)")
else:
    no_data_note()

def proj(q):   # azimuthal projection of the scalp, nose up
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    th, ph = np.arccos(np.clip(q[:, 2], -1, 1)), np.arctan2(q[:, 1], q[:, 0])
    return th * np.cos(ph), th * np.sin(ph)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
(x0, y0), (x1, y1) = proj(p), proj(p @ R.T)
ax[0].scatter(x0, y0, s=25, label="true positions"); ax[0].scatter(x1, y1, s=25, marker="x", label=f"after {SHIFT_MM:g} mm slide")
for k in range(64):
    ax[0].annotate(names[k], (x0[k], y0[k]), fontsize=5)
ax[0].set_aspect("equal"); ax[0].axis("off"); ax[0].legend(fontsize=7); ax[0].set_title("B2.3  simulated cap slide", fontsize=9)

C = csv("partb/b2_displacement.csv")
for k, (lab, g) in enumerate(C.groupby("label")):
    ax[1].errorbar(g.shift_mm + .15 * k, 100 * g["mean"], 100 * g["std"], marker="o", ms=4, capsize=2, label=lab)
ax[1].set_xlabel("cap slide at test time (mm, 4 directions averaged)"); ax[1].set_ylabel("test accuracy (%), mean +- SD over seeds")
ax[1].set_title("degradation curves (task1/b2_displacement.py)", fontsize=9); ax[1].legend(fontsize=7)
save(fig, "B2c_displacement.png")

section("Degradation curves: accuracy (%) by slide size, mean +- SD over 3 seeds")
C["cell"] = [f"{100 * m:.1f} +- {100 * s:.1f}" for m, s in zip(C["mean"], C["std"])]
print(C.pivot(index="label", columns="shift_mm", values="cell").to_string())
drop = C.pivot(index="label", columns="shift_mm", values="mean")
print("\n  loss vs 0 mm (points): " + "; ".join(f"{lab} {100 * (r[0.0] - r[10.0]):+.1f} @10, {100 * (r[0.0] - r[20.0]):+.1f} @20"
                                          for lab, r in drop.iterrows()))
runs = csv("partb/b2_displacement_runs.csv")
print(f"  every run x size x direction: outputs/partb/b2_displacement_runs.csv ({len(runs)} rows)")

section("Responsible layer: relative change of each layer's output for a 10 mm slide (b2_displacement_layer_summary.csv)")
Ls = csv("partb/b2_displacement_layer_summary.csv").pivot(index="label", columns="layer", values="mean")
print(Ls[[c for c in ["temporal", "spatial (depthwise)", "block2", "tokens", "cls output", "logits"] if c in Ls]].round(3).to_string())
say("The slide changes the first (temporal) layer's output by ~13% in every model. EEGNet: the depthwise spatial filter raises it to 17% and "
    "block 2 to 21%, but average pooling over 20 time steps and the dense layer bring it back to 13% at the logits (the "
    "eye/ERD fields are smooth on a 10 mm scale). Time-patch transformer: the change grows at every stage (13% -> 14% "
    "tokens -> 17% logits) because its 16 spatial filters feed every token and the softmax re-weighting amplifies a small "
    "coherent change in all tokens. Channel tokens + identity: the identity embedding stays put while the signal under it "
    "moves (logits 15%). Channel tokens without identity: an unordered set whose members each move a little keeps nearly "
    "the same summary (CLS 3%): most robust, least accurate. All effects are small (< 1 point at 10 mm) because what the "
    "models use is broad. Prediction 'time-patch drops most': right in ranking, effect tiny.")
finish()
