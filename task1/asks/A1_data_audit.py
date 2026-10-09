"""A1 - Data audit.

Brief: "Choose the runs for this task yourself and cite where you verified the choice. Produce audit.csv, one row per
subject: runs loaded, sampling rate, duration, left/right trial counts, event-code meaning applied, anomalies. State what
you did with each anomalous subject and why; nothing is dropped or kept silently."

Live part: re-audits SUBJECT from its raw EDF files with task1/audit.py and compares with audit.csv.
Change SUBJECT (70-109) and re-run. 88, 92, 100 have 128 Hz headers; 104 has a truncated last cue.
Run: python task1/asks/A1_data_audit.py   (or the Run button in VS Code)
"""
from _common import *

SUBJECT = 88

header("A1  Data audit",
       "choose the runs and cite where verified; audit.csv with one row per subject; say what was done with every anomaly.",
       "task1/audit.py reads every EDF header and annotation (runs 4, 8, 12). Live: one subject re-audited from raw files.")

section("Runs chosen and where it was verified")
print_decision("D1")

aud = csv("audit.csv")
section(f"Live re-audit of S{SUBJECT:03d} from the raw EDF files")
edf = ROOT / f"data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{SUBJECT:03d}/S{SUBJECT:03d}R04.edf"
if edf.exists():
    audit = pipeline("audit")
    live = audit.audit_subject(SUBJECT)
    for k in ("runs_loaded", "sfreq_hz", "duration_s", "n_left", "n_right", "cue_s", "alpha_peak_hz", "anomalies", "action"):
        print(f"  {k:14s} {live[k]}")
    saved = aud[aud.subject == SUBJECT].iloc[0]
    for k in ("sfreq_hz", "duration_s", "n_left", "n_right", "alpha_peak_hz"):
        check(f"{k:13s}", live[k], saved[k], tol=1e-6)
else:
    no_data_note()

section("audit.csv: one row per subject (the file the brief asks for)")
cols = ["subject", "runs_loaded", "sfreq_hz", "duration_s", "n_left", "n_right", "n_left_used", "n_right_used", "n_dropped", "anomalies"]
table(aud[cols].assign(anomalies=aud.anomalies.str.slice(0, 45)), "{:.1f}")

section("Event-code meaning applied (same for every subject)")
for v in aud.event_code_meaning.unique():
    print(" ", v, f"({(aud.event_code_meaning == v).sum()} subjects)")

section("Every subject that is not 'kept as is', with what was done and why")
flag = aud[(aud.anomalies != "none") | (aud.n_dropped > 0)]
for _, r in flag.iterrows():
    print(f"  S{r.subject:03d}  anomalies: {r.anomalies}")
    say(f"action: {r.action}", indent=8)
print(f"\n  {len(flag)} of {len(aud)} subjects flagged; all 40 are kept. Trials analysed: "
      f"{aud.n_left_used.sum()} left + {aud.n_right_used.sum()} right of {aud.n_left.sum()} + {aud.n_right.sum()} annotated "
      f"({aud.n_dropped.sum()} dropped, each listed above).")

section("Why the 128 Hz subjects are kept and how trials past the recording end are handled")
print_decision("D2")
print_decision("D3")

# Alpha-peak test of the 128 Hz header: a 160 Hz file mislabelled as 128 Hz would show its alpha peak at 0.8x the true value.
fig, ax = plt.subplots(figsize=(7, 3.5))
is128 = aud.sfreq_hz != 160
ax.scatter(aud.subject[~is128], aud.alpha_peak_hz[~is128], label="160 Hz header", s=18)
ax.scatter(aud.subject[is128], aud.alpha_peak_hz[is128], label="128 Hz header", s=40, color="C3")
ax.set_xlabel("subject"); ax.set_ylabel("rest alpha peak at header rate (Hz)"); ax.legend(fontsize=7)
ax.set_title("A1: alpha peak per subject (header-rate check)", fontsize=9)
save(fig, "A1_alpha_peak.png")
finish()
