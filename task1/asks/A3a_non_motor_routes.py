"""A3 (part 1) - List every non-motor route by which left and right trials can differ.

Brief: "Left and right trials can differ in the recording for reasons other than motor imagery. From how trials were cued
in this experiment, list every such non-motor route, then design and run a test for each."

This script is the list (A3b runs the tests). The channel groups and decoders are read from task1/confound.py, so the
list printed here is exactly what was tested, not a description of it.
Run: python task1/asks/A3a_non_motor_routes.py
"""
from _common import *

header("A3.1  Non-motor routes from how trials were cued",
       "from how trials were cued, list every route other than motor imagery by which left and right trials can differ.",
       "Cue = a target on the LEFT or RIGHT of the screen; the subject imagines that fist until it disappears; rest between "
       "trials (PhysioNet EEGMMIDB description). Every route below follows from that protocol.")

ROUTES = [  # (route, why it differs between left and right trials, decoder ids in confound.py that test it)
    ("Oculomotor", "the target appears left or right, so gaze shifts toward it (saccade, EOG field at frontal and "
                   "temporal sites, opposite polarity at F7 vs F8)", ["F1", "F2"]),
    ("Visual", "a stimulus in one visual hemifield gives a lateralised visual evoked response and lateralised "
               "attention alpha over occipital cortex", ["O1", "O2"]),
    ("Muscle / posture", "jaw, neck or covert hand tension can differ by side; EMG shows as 30+ Hz power at temporal "
                         "edge sites", ["G1"]),
    ("Any non-motor site", "anything above, plus volume-conducted fields, measured with the sensorimotor strip excluded",
     ["N1", "N2"]),
    ("Trial order / time in run", "position in run, elapsed time (drift, fatigue) and the previous labels can predict "
                                  "the current label if the order is not random", ["R1", "R2", "R12"]),
    ("Run identity / protocol", "if a run has more left than right trials, anything run-specific (impedance, 128 vs "
                                "160 Hz protocol) predicts the label; bounded analytically (run-majority oracle)", ["oracle"]),
    ("Pre-cue baseline (method check)", "no label information can exist before the cue; any accuracy there is a leak "
                                        "in the method (filtering, carry-over)", ["*_pre twins"]),
]
CONTROLS = [("Motor (positive control)", "the route the task is about: lateralised mu/beta ERD", ["M1", "A1"])]

section("Routes, why each could separate left from right, and the decoders that test it")
for name, why, dec in ROUTES + CONTROLS:
    print(f"  {name}")
    say(f"why: {why}", indent=6)
    print(f"      tested by: {', '.join(dec)}")

confound = pipeline("confound")
names = list(csv("electrode_positions_2d.csv").name)
specs = confound.decoder_specs(names)
section("Exact decoder definitions (task1/confound.py decoder_specs; '_pre' = same features before the cue)")
for k, (route, desc, _) in specs.items():
    print(f"  {k:7s} {route:24s} {desc}")
print("  R1      trial order              position in run + onset time")
print("  R2      trial order              previous 1-3 labels (no EEG at all)")
print("  R12     trial order              R1 + R2")

section("Channel groups (confound.py)")
for g in ("FRONT", "OCC", "TEMP", "SENSORIMOTOR"):
    print(f"  {g:13s} {', '.join(getattr(confound, g))}")

print_decision("D12")
print_decision("D17")
section("Route found only after the first run (post hoc, DECISIONS D18)")
say("Labels are not in random order: runs are near-alternating (lag-1 agreement 424 observed vs 780 expected, "
    "z = -17.6), so the previous label is itself a route (R2). See A3b for its test and A3d for the prediction that missed it.")

pos = csv("electrode_positions_2d.csv")
fig, ax = plt.subplots(figsize=(6, 6))
groups = {"frontal (oculomotor)": confound.FRONT, "occipital (visual)": confound.OCC, "temporal (muscle)": confound.TEMP,
          "sensorimotor (motor control)": confound.SENSORIMOTOR}
ax.scatter(pos.x, pos.y, s=30, color="0.85")
for (lab, chs), col in zip(groups.items(), ["C0", "C2", "C1", "C3"]):
    m = pos.name.str.lower().isin([c.lower() for c in chs])
    ax.scatter(pos.x[m], pos.y[m], s=60, color=col, label=lab)
for _, r in pos.iterrows():
    ax.annotate(r["name"], (r.x, r.y), fontsize=6, ha="center", va="bottom")
ax.set_aspect("equal"); ax.axis("off"); ax.legend(fontsize=7, loc="lower right")
ax.set_title("A3.1  electrode groups used by the route decoders (nose up)", fontsize=9)
save(fig, "A3a_route_channel_groups.png")
finish()
