# DECISIONS
Format: chosen / alternative / why / what the alternative would change. "Verified" = where the fact was checked.

## D1 Runs 4, 8, 12 (imagery left/right fist)
Alternative: 3/7/11 (execution) or 6/10/14 (imagery fists vs feet). Why: the task is imagined left vs right fist; T1/T2 mean different things per run type. Verified: `mne.datasets.eegbci.load_data` docstring and PhysioNet page (wiki/dataset.md). Alternative would change labels entirely (execution gives stronger, cleaner ERD).

## D2 128 Hz-header subjects (88, 92, 100): resample to 160 Hz, keep
Alternative: exclude them, or relabel as 160 Hz. Why: my alpha-peak check (audit.csv) did not show the 0.8x shift a mislabelled 160 Hz file would produce, and their protocol differs (cue 5.12 s, 19 trials/run), so the header looks right. Excluding would cost 3/40 subjects. Expect: negligible change in group results; their per-subject accuracy may differ.

## D3 Windows past the end of a recording are dropped (20 trials) 
Epoch window is -1.5..4.0 s around the cue. Alternative: shorter windows (3 s) to keep every trial. Why: 4.0 s covers the shortest regular cue; padding would invent data. Affected: S72,73,74,76,88,92,102,104 (listed per subject in audit.csv, nothing silent). Expect: <1.2% fewer trials.

## D4 Spatial reference: local Laplacian (C minus mean of 4 nearest electrodes), CAR and raw as sensitivity
The recording reference is undocumented (wiki/dataset.md). With an unknown reference electrode R, every channel contains the same -R(t) term, so raw C3-C4 is a difference of two references-contaminated signals only if R is not common (it is common to both, but R's own activity and distance to each site still leak differently, and a lateral reference such as a mastoid would bias hemispheres asymmetrically). Laplacian weights sum to zero, so any component common to the neighbourhood, including the reference, cancels, and it is spatially local (sharpens C3 vs C4). CAR also cancels common terms but spreads frontal/ocular artefact over all channels. Result: Laplacian gave the largest mean LI (mu .26, beta .32) vs CAR (.19/.29) and raw (.15/.21); present counts 14/16/12 (outputs/a2_all_references.csv). Expect: raw would label fewer lateralised.

## D5 Baseline and window for ERD
Baseline -1.0..-0.1 s (inside the preceding rest, pooled over trials per subject and band, label-blind). Active window 0.5..4.0 s. Alternatives: per-trial baseline (noisier), post-cue baseline (contaminated by imagery), -1.5..0 (includes wavelet edge and the previous cue's offset ERD rebound). Skipping the first 0.5 s removes the cue-evoked visual response and ERD onset lag. Expect: per-trial baseline would shrink ERD magnitudes and raise p-values.

## D6 Lateralisation statistic and label rule
Per trial dB ERD per band at C3, C4. T = mean(C3-C4 | left) - mean(C3-C4 | right) = sum over classes of (ipsi - contra); one-sided label-permutation p (10000 perms, seed = subject id) because the physiological direction is predicted. LI = (ipsi - contra)/(|ipsi|+|contra|), in [-1,1]. Label "present" if either band p < 0.025 (Bonferroni over 2 bands) AND contralateral ERD is a desynchronisation (<0 dB). Alternative: two-sided test (would also count wrong-direction asymmetries as lateralisation).

## D7 No artefact rejection (so far)
Alternative: amplitude-threshold or ICA rejection. Why not yet: nothing may be dropped silently and per-subject trial counts are small. Known cost: heavy trials dominate (e.g. S95 C3). Expect: rejection would raise ERD clarity for noisy subjects.

## D8 MATLAB twin (task1/a2_tfr.m): wavelet power on the continuous recording, then epoch
Alternative: epoch first and transform each 5.5 s epoch, as task1/erd.py does. Why: a wavelet run on an epoch zero-pads both ends; its support is 5 sigma_t = 0.4 s (MNE morlet source), so the end of the 0.5-4.0 s active window could be biased. Verified: S072 and S088 against erd.py on the same trials and onsets: median power ratio MATLAB/Python 1.002 and 1.004 (log10 +0.0009 / +0.0016) in the interior, 95% of samples within 2% (|log10| < 0.009 / 0.007), log-power correlation >= 0.9996; in the last 0.25 s MATLAB is only ~0.4-0.6% higher (log10 +0.002). Band ERD over 0.5-4.0 s differs by <= 0.05 dB in the S072 means at C3/C4. So the edge effect is real but negligible; kept as a precaution. Expect: porting back to Python epoch-first changes nothing visible.

## D9 MATLAB band-pass: bandpass(x,[1 40],fs,'ImpulseResponse','fir')
Alternative: MNE's firwin design (what data.py uses) or an IIR filter with filtfilt. Why: both are zero-phase 1-40 Hz FIRs on the continuous data, so epochs have no filter edge; the designs differ in transition width. Verified only through the power comparison in D8 (agreement above). Expect: an IIR/filtfilt design would change the 6-8 Hz edge of the maps most.

## D10 Neighbours of C3 and C4: the 4 nearest electrodes by 3-D distance
C3: Cp3, Fc3, C5, C1. C4: Cp4, Fc4, C6, C2. Alternative: 8 neighbours, or a fixed radius. Why 4: these are the electrodes the Laplacian (D4) subtracts, so the same maps serve the reference step. MATLAB has no montage built in, so the list is fixed in the script; verified against erd.neighbours() on the standard_1005 montage (own run, 2026-10-07). Expect: 8 neighbours would add the diagonal electrodes (Fc1, Cp1, ...) and shift the Laplacian towards a wider, smoother reference.

## D11 Morlet wavelet written by hand to match MNE (n_cycles = f/2, zero mean, norm sqrt(.5)*||W||)
Alternative: MATLAB cwt (Wavelet Toolbox), whose scaling and time support differ. Why: identical numbers in both languages. Note: n_cycles = f/2 gives sigma_t = 1/(4 pi) = 0.080 s and sigma_f = 2 Hz at every frequency, i.e. a constant absolute bandwidth; the comment in task1/erd.py (power()) calls it "constant relative bandwidth", which is wrong. Verified: MNE morlet source; MATLAB sine test (10 Hz sine peaks at 10 Hz, 20 Hz power 1e-11 of it). Expect: a true constant-Q wavelet (n_cycles proportional to f) would give finer frequency resolution in mu than beta.
