"""2d - Physiology: the worst stage of the fixed model, explained from the scoring rules and the recorded signals.

Brief: "In your fixed model, identify the worst stage and explain it from the scoring rules and the signals this dataset
actually records (which channels, at what rate, what defines each stage). Show one physiologically expected confusion
and one unexpected one."

Reads the channels and rates from the files themselves (when present), recomputes per-stage F1 and the confusion matrix
from the three fixed-pipeline runs, and the per-stage physiology in microvolts from task2/physiology/epoch_features.csv.
Change FEATURE to plot another measure by stage.
Run: python task2/asks/T2d_physiology.py
"""
import re
from _t2 import *

FEATURE = "emg_rms_uv"      # any column of epoch_features.csv, e.g. pz_alpha_rel, fpz_theta_rel, delta_cover, eog_slow_uv2

header("2d  Physiology of the worst stage",
       "identify the worst stage of the fixed model and explain it from the scoring rules and the signals this dataset "
       "records; show one expected and one unexpected confusion.",
       "per-stage F1 and confusion from the fixed pipeline on seeds 42-44; signal facts read from the EDF headers; R&K "
       "criteria measured in uV on every cropped epoch (task2/physiology.py). Write-up: task2/PHYSIOLOGY.md.")

section("What the files record (task2/physiology/dataset_facts.txt)")
if has_sleep_data():
    import mne, physiology
    mne.set_log_level("error")
    psg = physiology.files()[0][0]
    raw = mne.io.read_raw_edf(psg, preload=False)
    ex = raw._raw_extras[0]
    print(f"  live, {Path(psg).name}: " + ", ".join(f"{c} @ {ex['n_samps'][i] / ex['record_length'][0]:g} Hz" for i, c in enumerate(raw.ch_names)))
else:
    no_sleep_data_note()
say(text("task2/physiology/dataset_facts.txt"))

runs = ledger_runs()
cms = np.stack([np.array(runs[("FIXED", s)]["confusion"]) for s in SEEDS])
f1 = np.array([runs[("FIXED", s)]["per_class_f1"] for s in SEEDS])
section("Per-stage F1 of the fixed model, mean +- SD over seeds 42-44")
table(pd.DataFrame({"stage": STAGES, "F1": f1.mean(0), "SD": f1.std(0, ddof=1)}))
worst = STAGES[int(f1.mean(0).argmin())]
tot = cms.sum(0)
i = STAGES.index(worst)
print(f"\n  worst stage: {worst}; recall {tot[i, i] / tot[i].sum():.2f}, precision {tot[i, i] / tot[:, i].sum():.2f}; "
      f"windows predicted {worst} by true stage: " + ", ".join(f"{s} {tot[j, i]}" for j, s in enumerate(STAGES)))

rn = cms / cms.sum(2, keepdims=True)
section("P(predicted | true), mean of 3 seeds")
print(pd.DataFrame(rn.mean(0), index=[f"true {s}" for s in STAGES], columns=STAGES).round(3).to_string())

feats = pd.read_csv(T2 / "physiology" / "epoch_features.csv")
feats = feats[feats.stage >= 0]
cols = ["fpz_ptp_uv", "delta_cover", "pz_alpha_rel", "fpz_theta_rel", "fpz_sigma_rel", "eog_slow_uv2", "emg_rms_uv"]
section("Median per-epoch physiology by true stage (uV, relative band power, all 29 nights)")
print(feats.groupby("stage")[cols].median().rename(index=dict(enumerate(STAGES))).round(3).to_string())

section("The explanation (task2/PHYSIOLOGY.md)")
phys = text("task2/PHYSIOLOGY.md")
for head in (r"^\*\*Why, from the scoring rules", r"^## One expected confusion", r"^## One unexpected confusion"):
    say(re.sub(r"\*\*|^## ", "", md_block("task2/PHYSIOLOGY.md", head, r"^## |^\*\*Why|\Z")))
    print()

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
axs[0].imshow(rn.mean(0), cmap="Blues", vmin=0, vmax=1)
for a in range(5):
    for b in range(5):
        axs[0].text(b, a, f"{rn.mean(0)[a, b]:.2f}", ha="center", va="center", fontsize=8, color="white" if rn.mean(0)[a, b] > .5 else "black")
axs[0].set_xticks(range(5), STAGES); axs[0].set_yticks(range(5), STAGES)
axs[0].set_xlabel("predicted"); axs[0].set_ylabel("true"); axs[0].set_title("Fixed model, P(pred | true), 3 seeds", fontsize=9)
data = [feats.loc[feats.stage == k, FEATURE].clip(upper=feats[FEATURE].quantile(.99)) for k in range(5)]
axs[1].boxplot(data, showfliers=False)
axs[1].set_xticks(range(1, 6), STAGES)
axs[1].set_title(f"{FEATURE} by true stage (all cropped epochs)", fontsize=9)
save(fig, "T2d_confusion_and_feature.png")
finish()
