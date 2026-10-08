%% A2 walkthrough: from the raw recording to ERD, one stage at a time
% One subject (S072), channels C3 and C4, every stage drawn as a time series and as spectra (PSD curves).
% It re-derives, step by step, what task1/a2_tfr.m computes in bulk. Needs base MATLAB + Signal Processing Toolbox.
% Change `subject` to look at anyone else. Run from anywhere: matlab -batch "run('task1/a2_walkthrough.m')"

%% Setup
repo = fileparts(fileparts(mfilename('fullpath')));
dataRoot = fullfile(repo, 'data');
if ~isfolder(dataRoot), dataRoot = getenv('EEG_DATA_ROOT'); end    % git worktrees do not contain the git-ignored data
assert(isfolder(dataRoot), 'Data not found: put it in <repo>/data or set EEG_DATA_ROOT to the folder holding MNE-eegbci-data');
edfDir = fullfile(dataRoot, 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0');
cacheDir = fullfile(dataRoot, 'cache_matlab');
figDir = fullfile(repo, 'outputs', 'walkthrough');
if ~isfolder(figDir), mkdir(figDir); end

subject = 72;
chans = ["C3" "C4"];                       % channel 1 = left motor cortex, channel 2 = right motor cortex
cfg.FS = 160; cfg.TMIN = -1.5; cfg.TMAX = 4.0; cfg.N_T = floor((cfg.TMAX - cfg.TMIN) * cfg.FS) + 1;
cfg.BAND = [1 40];                         % general clean-up band (DECISIONS D5/D16)
cfg.MU = [8 13]; cfg.BETA = [13 30];
T_BASE = [-1.0 -0.1];                      % baseline: inside the rest before the cue (DECISIONS D5)
ACTIVE = [0.5 4.0];                        % active window: skips the first 0.5 s after the cue (DECISIONS D5)
times = cfg.TMIN + (0:cfg.N_T-1) / cfg.FS;
inBase = times >= T_BASE(1) & times <= T_BASE(2);
inAct = times >= ACTIVE(1) & times <= ACTIVE(2);
cC3 = [0.12 0.47 0.71]; cC4 = [0.85 0.45 0.10];          % line colours: C3 blue, C4 orange
cL = [0.12 0.47 0.71]; cR = [0.80 0.15 0.15];            % cue colours: left-fist blue, right-fist red
cBase = [0.80 0.80 0.80]; cAct = [1.00 0.92 0.55];       % baseline grey, active window yellow
chanCol = [cC3; cC4];
cueName = ["left" "right"]; cueLetter = ["L" "R"]; contraTag = ["" " (contralateral)"];   % lookup tables (MATLAB cannot index a literal)

D = loadSubject(subject, edfDir, chans, cfg);
left = D.y == 0; right = D.y == 1;
fprintf('S%03d: %d left and %d right trials, %d samples per epoch\n', subject, sum(left), sum(right), cfg.N_T);

%% Where the electrodes are
% The head seen from above, nose up, left of the page = left of the head. The 64 grey dots are all the electrodes in the recording
% (standard 10-10 positions, projected flat from the 3-D montage; outputs/electrode_positions_2d.csv). C3 (blue) lies over the left
% sensorimotor cortex and C4 (orange) over the right; each hand is controlled mostly by the opposite hemisphere, so imagining the
% LEFT fist should change C4 and imagining the RIGHT fist should change C3. The ring of electrodes around each is its 4 nearest
% neighbours (a2_tfr.m, DECISIONS D17).
pos = readtable(fullfile(repo, 'outputs', 'electrode_positions_2d.csv'));
nbrNames = ["Cp3" "Fc3" "C5" "C1" "Cp4" "Fc4" "C6" "C2"];
fig = figure('Position', [100 100 560 560], 'Color', 'w'); hold on; axis equal off;
a = linspace(0, 2*pi, 200); HR = 1.15;                                       % head radius in the flat map
plot(HR*cos(a), HR*sin(a), 'k', 'LineWidth', 1.5);
plot([-0.14 0 0.14], [HR*0.99 HR+0.2 HR*0.99], 'k', 'LineWidth', 1.5);       % nose
ea = linspace(pi/2, 3*pi/2, 40); plot(-HR + 0.10*cos(ea), 0.25*sin(ea), 'k', 'LineWidth', 1.5);   % left ear
ea = linspace(-pi/2, pi/2, 40); plot(HR + 0.10*cos(ea), 0.25*sin(ea), 'k', 'LineWidth', 1.5);     % right ear
scatter(pos.x, pos.y, 22, [0.78 0.78 0.78], 'filled');
other = ~ismember(lower(pos.name), lower([nbrNames chans]));                  % grey labels only where nothing is highlighted
text(pos.x(other), pos.y(other) - 0.06, pos.name(other), 'FontSize', 6.5, 'Color', [0.55 0.55 0.55], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
for nm = nbrNames
    j = find(strcmpi(pos.name, nm)); col = cC3; if pos.x(j) > 0, col = cC4; end
    scatter(pos.x(j), pos.y(j), 55, 'w', 'filled', 'MarkerEdgeColor', col, 'LineWidth', 2);
    text(pos.x(j), pos.y(j) - 0.07, pos.name(j), 'FontSize', 9, 'FontWeight', 'bold', 'Color', col, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
end
for k = 1:2
    j = find(strcmpi(pos.name, chans(k)));
    scatter(pos.x(j), pos.y(j), 130, chanCol(k, :), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
    text(pos.x(j), pos.y(j) - 0.09, chans(k), 'FontSize', 13, 'FontWeight', 'bold', 'Color', chanCol(k, :), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
end
text(-HR - 0.5, 0, 'L', 'FontSize', 16, 'FontWeight', 'bold', 'HorizontalAlignment', 'center'); text(HR + 0.5, 0, 'R', 'FontSize', 16, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(0, HR + 0.32, 'front', 'HorizontalAlignment', 'center'); xlim([-1.85 1.85]); ylim([-1.35 1.5]);
title('Electrode positions seen from above');
saveFig(fig, fullfile(figDir, 'electrodes.png'));

%% Stage 1: the raw recording
% First minute of run 4 at C3 and C4, exactly as stored in the file (microvolts). The coloured bands are the cues the
% subject saw: blue = imagine the left fist (T1), red = imagine the right fist (T2). White gaps are rest (T0).
% Nothing in the voltage tells left from right: it is ongoing activity of 10-50 uV, with occasional large transients (near 17 s,
% both channels at once, probably eye blinks or movement). They are not removed (DECISIONS D7).
r = D.runs(1); t1 = (0:size(r.raw, 1) - 1) / r.fs;
fig = figure('Position', [100 100 900 440], 'Color', 'w'); tiledlayout(2, 1, 'TileSpacing', 'compact');
for ci = 1:2
    nexttile; hold on;
    for e = find(r.code == "T1" | r.code == "T2")'
        col = cL; if r.code(e) == "T2", col = cR; end
        xregion(r.onset(e), r.onset(e) + r.dur(e), 'FaceColor', col, 'FaceAlpha', 0.15);
    end
    plot(t1, r.raw(:, ci), 'Color', [0.15 0.15 0.15], 'LineWidth', 0.6);
    xlim([0 60]); ylabel(chans(ci) + " (\muV)");
    if ci == 1
        for e = find(r.code == "T1" | r.code == "T2")'
            if r.onset(e) < 58, text(r.onset(e) + r.dur(e)/2, max(ylim) * 0.92, cueLetter(1 + (r.code(e) == "T2")), 'HorizontalAlignment', 'center', 'FontWeight', 'bold'); end
        end
    end
end
xlabel('time in run 4 (s)'); figTitle(sprintf('Stage 1: raw recording, S%03d', subject));
saveFig(fig, fullfile(figDir, 'stage1_raw.png'));

%% Stage 2: filtering, 1-40 Hz
% A zero-phase band-pass on the continuous recording. Top: ten seconds of C3 before and after; the slow wander of the grey trace is
% gone. Bottom: the power spectral density (PSD) of the whole run, before and after; PSD says how much power each frequency carries.
% The raw PSD climbs steeply towards 0 Hz (slow drift) and carries broadband power above 40 Hz (muscle, electrical noise). The filter
% removes both ends and leaves 1-40 Hz untouched (the curves overlap). The bump near 11 Hz is the mu rhythm: it is already visible.
fs = r.fs; win = 2 * fs;
[pRaw, f] = pwelch(r.raw(:, 1), hann(win), win / 2, 4 * fs, fs);
pFil = pwelch(r.filt(:, 1), hann(win), win / 2, 4 * fs, fs);
fig = figure('Position', [100 100 900 440], 'Color', 'w'); tiledlayout(1, 2, 'TileSpacing', 'compact');
nexttile; hold on; seg = (20 * fs + 1):(30 * fs);
plot(t1(seg), r.raw(seg, 1), 'Color', [0.6 0.6 0.6]); plot(t1(seg), r.filt(seg, 1), 'Color', cC3, 'LineWidth', 1);
legend('raw', 'filtered 1-40 Hz'); xlabel('time in run 4 (s)'); ylabel('C3 (\muV)'); title('10 s of C3, rest');
nexttile; semilogy(f, pRaw, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2); hold on; semilogy(f, pFil, 'Color', cC3, 'LineWidth', 1.2);
xregion(cfg.MU(1), cfg.MU(2), 'FaceColor', 'k', 'FaceAlpha', 0.08); xregion(cfg.BETA(1), cfg.BETA(2), 'FaceColor', 'k', 'FaceAlpha', 0.04);
xlim([0 80]); xlabel('frequency (Hz)'); ylabel('PSD (\muV^2/Hz)'); legend('raw', 'filtered', 'Location', 'northeast');
title('PSD of the whole run (mu and beta shaded)');
figTitle('Stage 2: filtering'); saveFig(fig, fullfile(figDir, 'stage2_filter.png'));

%% Stage 3: cutting epochs around the cue, and where baseline and active window sit
% Every cue becomes a 5.5 s epoch, -1.5 to 4.0 s, time 0 = cue onset. Grey = baseline window (-1.0 to -0.1 s, still resting);
% yellow = active window (0.5 to 4.0 s, subject imagining). First, one left-fist trial at C3 and C4 (C3 shifted up so they do not overlap).
k0 = find(left, 1);
fig = figure('Position', [100 100 900 400], 'Color', 'w'); hold on; xregion(T_BASE(1), T_BASE(2), 'FaceColor', cBase, 'FaceAlpha', 0.6); xregion(ACTIVE(1), ACTIVE(2), 'FaceColor', cAct, 'FaceAlpha', 0.5);
for ci = 1:2, plot(times, squeeze(D.Ef(:, ci, k0)) + (2 - ci) * 80, 'Color', chanCol(ci, :), 'LineWidth', 0.8); end
xline(0, 'k', 'cue'); xlabel('time from cue (s)'); ylabel('\muV (C3 shifted up by 80)'); legend('baseline', 'active', 'C3', 'C4', 'Location', 'northeast');
title(sprintf('Stage 3: one left-fist trial (trial %d)', k0)); saveFig(fig, fullfile(figDir, 'stage3_epochs.png'));
% Then every trial of each cue class, at C3 and at C4: single trials in grey, their average in colour. The average voltage stays within a
% few uV of zero in all four panels, apart from small bumps in the first 0.5 s after the cue, clearest for left-fist cues (a response to the cue itself; the reason
% that interval is skipped, DECISIONS D5). ERD is not a change in the average signal: it is a change in the size of an oscillation whose
% phase differs from trial to trial, so it averages away.
fig = figure('Position', [100 100 900 600], 'Color', 'w'); tl = tiledlayout(2, 2, 'TileSpacing', 'compact');
for cl = 1:2
    sel = (cl == 1) * left + (cl == 2) * right; sel = sel > 0;
    for ci = 1:2
        nexttile; hold on; xregion(T_BASE(1), T_BASE(2), 'FaceColor', cBase, 'FaceAlpha', 0.6); xregion(ACTIVE(1), ACTIVE(2), 'FaceColor', cAct, 'FaceAlpha', 0.5);
        plot(times, squeeze(D.Ef(:, ci, sel)), 'Color', [0.7 0.7 0.7 0.6], 'LineWidth', 0.4); plot(times, mean(D.Ef(:, ci, sel), 3), 'Color', chanCol(ci, :), 'LineWidth', 2);
        xline(0, 'k'); ylim([-40 60]); xlabel('time from cue (s)'); ylabel(chans(ci) + " (\muV)");
        title(sprintf('%s-fist, %s: %d trials and their mean', cueName(cl), chans(ci), sum(sel)));
    end
end
title(tl, 'Stage 3b: all trials, both channels'); saveFig(fig, fullfile(figDir, 'stage3b_trials.png'));

%% Stage 4: isolate the mu rhythm and follow its size over time
% Band-pass 8-13 Hz (mu), then take the envelope (Hilbert transform: the instantaneous size of the oscillation). Squared, the
% envelope is mu power over time. Top row: the same left-fist trial at C3 and C4 (thin = mu-filtered voltage, thick = envelope).
% Contralateral to the left hand is C4. In this single trial the C4 rhythm is not obviously smaller after the cue: one trial is noisy.
% Bottom: power averaged over trials, smoothed over 0.25 s for display, with the baseline level (mean of the grey window, over all
% trials) as a dashed line. Here it shows: after about 0.3 s, left-fist imagery pulls C4 far below its baseline and C3 less; right-fist
% imagery pulls C3 far below and leaves C4 close to its baseline.
P = D.Eenv .^ 2;                                           % mu power over time [time x chan x trial], \muV^2
base = reshape(mean(P(inBase, :, :), [1 3]), 1, []);             % baseline per channel, pooled over ALL trials, label-blind (D5)
sm = round(0.25 * cfg.FS);
fig = figure('Position', [100 100 900 620], 'Color', 'w'); tl = tiledlayout(2, 2, 'TileSpacing', 'compact');
for ci = 1:2
    nexttile; hold on; xregion(T_BASE(1), T_BASE(2), 'FaceColor', cBase, 'FaceAlpha', 0.6); xregion(ACTIVE(1), ACTIVE(2), 'FaceColor', cAct, 'FaceAlpha', 0.5);
    plot(times, squeeze(D.Emu(:, ci, k0)), 'Color', [chanCol(ci, :) 0.6], 'LineWidth', 0.6); plot(times, squeeze(D.Eenv(:, ci, k0)), 'Color', chanCol(ci, :), 'LineWidth', 2);
    xline(0, 'k'); ylabel('\muV'); xlabel('time from cue (s)'); title(sprintf('trial %d (left fist), %s mu 8-13 Hz', k0, chans(ci)));
end
for cl = 1:2
    sel = (cl == 1) * left + (cl == 2) * right; sel = sel > 0;
    nexttile; hold on; xregion(T_BASE(1), T_BASE(2), 'FaceColor', cBase, 'FaceAlpha', 0.6); xregion(ACTIVE(1), ACTIVE(2), 'FaceColor', cAct, 'FaceAlpha', 0.5);
    for ci = 1:2
        plot(times, movmean(mean(P(:, ci, sel), 3), sm), 'Color', chanCol(ci, :), 'LineWidth', 2);
        yline(base(ci), '--', 'Color', chanCol(ci, :));
    end
    xline(0, 'k'); xlabel('time from cue (s)'); ylabel('mu power (\muV^2)'); legend('baseline', 'active', 'C3', '', 'C4', '', 'Location', 'northeast');
    title(sprintf('mean mu power, %s-fist imagery (%d trials)', cueName(cl), sum(sel)));
end
figTitle('Stage 4: the mu rhythm over time'); saveFig(fig, fullfile(figDir, 'stage4_mu_power.png'));

%% Stage 5: baseline to ERD
% ERD is the mean power in the active window relative to the baseline power, in percent over time and in decibels for one number:
% ERD% = P(t) / P_baseline - 1;   ERD(dB) = 10 log10( mean over the active window of P / P_baseline ).
% Negative = the rhythm got smaller (desynchronised). The dashed zero line is "same as baseline". The table gives the mean
% over trials of the per-trial dB value: contralateral is the lower one in each column.
fig = figure('Position', [100 100 900 430], 'Color', 'w'); tiledlayout(1, 2, 'TileSpacing', 'compact');
for cl = 1:2
    sel = (cl == 1) * left + (cl == 2) * right; sel = sel > 0;
    nexttile; hold on; xregion(T_BASE(1), T_BASE(2), 'FaceColor', cBase, 'FaceAlpha', 0.6); xregion(ACTIVE(1), ACTIVE(2), 'FaceColor', cAct, 'FaceAlpha', 0.5);
    for ci = 1:2, plot(times, (movmean(mean(P(:, ci, sel), 3), sm) / base(ci) - 1) * 100, 'Color', chanCol(ci, :), 'LineWidth', 2); end
    yline(0, '--k'); xline(0, 'k'); ylim([-100 150]); xlabel('time from cue (s)'); ylabel('ERD % relative to baseline');
    legend('baseline', 'active', 'C3', 'C4', 'Location', 'northeast'); title(sprintf('%s-fist imagery', cueName(cl)));
end
figTitle('Stage 5: mu ERD over time'); saveFig(fig, fullfile(figDir, 'stage5_erd_curve.png'));
% The same as one number per trial and channel, then averaged over trials: mu ERD in dB in the active window.
dB = squeeze(10 * log10(mean(P(inAct, :, :), 1) ./ base));              % [chan x trial]
T5 = table(["C3"; "C4"], mean(dB(:, left), 2), mean(dB(:, right), 2), 'VariableNames', {'channel', 'left_imagery_dB', 'right_imagery_dB'});
disp(T5); disp('Left imagery should be most negative at C4, right imagery at C3.');

%% Stage 6: the same thing as PSD curves
% Instead of following one band over time, take the PSD in the baseline window and in the active window, per trial, and average over
% trials (Welch: 0.9 s Hann segments, 1.1 Hz resolution). Grey = baseline (all trials), colour = active window for one cue class.
% The baseline has a clear mu peak near 11 Hz. ERD is the gap between the two curves: in the contralateral panels (left fist at C4,
% right fist at C3) the peak is gone and beta (15-25 Hz) is lower too; in the ipsilateral panels the peak is partly kept (C3, left)
% or fully kept (C4, right). Below 8 Hz and above 30 Hz the curves roughly overlap: the effect is specific to these bands.
[S.f, S.Pb, S.Pa] = psdByTrial(D.Ef, times, T_BASE, ACTIVE, cfg.FS);   % [freq x chan x trial]
Pb = mean(S.Pb, 3);                                                   % one baseline spectrum per channel, pooled over all trials
fig = figure('Position', [100 100 900 600], 'Color', 'w'); tiledlayout(2, 2, 'TileSpacing', 'compact');
for cl = 1:2
    sel = (cl == 1) * left + (cl == 2) * right; sel = sel > 0;
    for ci = 1:2
        nexttile; hold on; xregion(cfg.MU(1), cfg.MU(2), 'FaceColor', 'k', 'FaceAlpha', 0.07); xregion(cfg.BETA(1), cfg.BETA(2), 'FaceColor', 'k', 'FaceAlpha', 0.03);
        plot(S.f, 10 * log10(Pb(:, ci)), 'Color', [0.45 0.45 0.45], 'LineWidth', 2);
        plot(S.f, 10 * log10(mean(S.Pa(:, ci, sel), 3)), 'Color', chanCol(ci, :), 'LineWidth', 2);
        xlim([4 35]); xlabel('frequency (Hz)'); ylabel('PSD (dB re 1 \muV^2/Hz)');
        contra = (cl == 1 && ci == 2) || (cl == 2 && ci == 1);
        title(sprintf('%s-fist imagery | %s%s', cueName(cl), chans(ci), contraTag(1 + contra)));
        legend('mu', 'beta', 'baseline', 'active', 'Location', 'northeast');
    end
end
figTitle('Stage 6: PSD before (grey) and after (colour) the cue'); saveFig(fig, fullfile(figDir, 'stage6_psd.png'));
% Dividing the two curves gives the ERD spectrum in dB: how much each frequency lost. The contralateral curve dips by about 9 dB
% at 11-12 Hz and by about 4 dB around 20-22 Hz.
fig = figure('Position', [100 100 900 430], 'Color', 'w'); tiledlayout(1, 2, 'TileSpacing', 'compact');
for cl = 1:2
    sel = (cl == 1) * left + (cl == 2) * right; sel = sel > 0;
    nexttile; hold on; xregion(cfg.MU(1), cfg.MU(2), 'FaceColor', 'k', 'FaceAlpha', 0.07); xregion(cfg.BETA(1), cfg.BETA(2), 'FaceColor', 'k', 'FaceAlpha', 0.03);
    for ci = 1:2, plot(S.f, 10 * log10(mean(S.Pa(:, ci, sel), 3) ./ Pb(:, ci)), 'Color', chanCol(ci, :), 'LineWidth', 2); end
    yline(0, '--k'); xlim([4 35]); xlabel('frequency (Hz)'); ylabel('ERD (dB)'); legend('mu', 'beta', 'C3', 'C4', 'Location', 'southeast');
    title(sprintf('%s-fist imagery', cueName(cl)));
end
figTitle('Stage 6b: ERD spectrum = 10 log10(active PSD / baseline PSD)'); saveFig(fig, fullfile(figDir, 'stage6b_erd_spectrum.png'));

%% Stage 7: the wavelet maps are the same quantity
% a2_tfr.m uses a Morlet wavelet instead of a band-pass or Welch. Averaged over the active window, the wavelet power divided by its
% baseline must give the same ERD spectrum as stage 6, and averaged over 8-13 Hz it must give the same time course as stage 5.
% Different estimators of one thing; if they disagreed, one of them would be wrong. Left: same shape and place; Welch dips deeper
% at the mu peak because the wavelet has a 2 Hz bandwidth against ~1.1 Hz for Welch, so it smooths the peak. Right: the time courses
% coincide (correlation printed below).
E = load(fullfile(cacheDir, sprintf('tfr_S%03d.mat', subject)), 'P', 'y', 'B', 'freqs');
assert(isequal(E.y(:), D.y(:)), 'trial sets differ between the walkthrough and the cube');
cube = [1 6];                                                       % C3, C4 rows in the cube (a2_tfr.m channel order)
fig = figure('Position', [100 100 900 430], 'Color', 'w'); tiledlayout(1, 2, 'TileSpacing', 'compact');
nexttile; hold on; pairs = [1 2; 2 1];                              % [class, channel]: left-C4 and right-C3 (contralateral)
for p = 1:2
    cl = pairs(p, 1); ci = pairs(p, 2); sel = D.y == cl - 1;
    wav = 10 * log10(reshape(mean(E.P(cube(ci), :, inAct, sel), [3 4]), [], 1) ./ E.B(cube(ci), :)');   % both 25 x 1
    wel = 10 * log10(mean(S.Pa(:, ci, sel), 3) ./ Pb(:, ci));
    plot(S.f, wel, '-', 'Color', chanCol(ci, :), 'LineWidth', 2); plot(E.freqs, wav, 'o--', 'Color', chanCol(ci, :), 'MarkerSize', 4);
end
yline(0, '--k'); xlim([6 30]); xlabel('frequency (Hz)'); ylabel('ERD (dB)'); title('ERD spectrum: Welch (line) vs wavelet (dots)');
legend('left imagery, C4 Welch', 'wavelet', 'right imagery, C3 Welch', 'wavelet', 'Location', 'southeast');
nexttile; hold on; xregion(T_BASE(1), T_BASE(2), 'FaceColor', cBase, 'FaceAlpha', 0.6); xregion(ACTIVE(1), ACTIVE(2), 'FaceColor', cAct, 'FaceAlpha', 0.5);
fm = E.freqs >= cfg.MU(1) & E.freqs <= cfg.MU(2);
for p = 1:1
    cl = pairs(p, 1); ci = pairs(p, 2); sel = D.y == cl - 1;
    wmu = squeeze(mean(mean(E.P(cube(ci), fm, :, sel), 4), 2));
    wb = mean(E.B(cube(ci), fm));
    plot(times, (movmean(wmu, sm) / wb - 1) * 100, 'Color', chanCol(ci, :), 'LineWidth', 2);
    plot(times, (movmean(mean(P(:, ci, sel), 3), sm) / base(ci) - 1) * 100, 'k--', 'LineWidth', 1.2);
    fprintf('Left imagery at C4: correlation of mu ERD time course, wavelet vs band-pass+Hilbert: %.4f\n', corr(movmean(wmu, sm), movmean(mean(P(:, ci, sel), 3), sm)));
end
yline(0, ':k'); xline(0, 'k'); ylim([-100 150]); xlabel('time from cue (s)'); ylabel('mu ERD %'); legend('baseline', 'active', 'wavelet', 'band-pass + Hilbert', 'Location', 'northeast');
title('mu ERD over time, left imagery at C4');
figTitle('Stage 7: wavelet, Welch and band-pass tell the same story'); saveFig(fig, fullfile(figDir, 'stage7_agreement.png'));

%% Stage 8: one number per trial
% The statistical test in the next step needs one ERD value per trial, channel and band: 10 log10(mean active power / baseline), in dB.
% Every dot is one trial (mu band). The bars are the means from stage 5. The contralateral channel sits lower in each group, but single
% trials overlap a lot (some C3 trials of left-fist imagery are above 0 dB): that is why the next step needs a test over trials.
fig = figure('Position', [100 100 800 480], 'Color', 'w'); hold on;
for cl = 1:2
    sel = (cl == 1) * left + (cl == 2) * right; sel = sel > 0;
    for ci = 1:2
        x = cl + (ci - 1.5) * 0.35; v = dB(ci, sel);
        swarmchart(x * ones(size(v)), v, 18, chanCol(ci, :), 'filled', 'MarkerFaceAlpha', 0.6, 'XJitterWidth', 0.25);
        plot(x + [-0.15 0.15], mean(v) * [1 1], 'k', 'LineWidth', 2.5);
    end
end
yline(0, '--k'); xticks([1 2]); xticklabels({'left-fist imagery', 'right-fist imagery'}); ylabel('mu ERD per trial (dB)');
h1 = plot(nan, nan, 'o', 'Color', cC3, 'MarkerFaceColor', cC3); h2 = plot(nan, nan, 'o', 'Color', cC4, 'MarkerFaceColor', cC4); legend([h1 h2], 'C3', 'C4');
title(sprintf('Stage 8: per-trial mu ERD, S%03d', subject)); saveFig(fig, fullfile(figDir, 'stage8_per_trial.png'));

%% Stage 9: C3 minus C4, and how the sign works
% The test in the next step asks one question per subject: does the C3 - C4 difference in ERD differ between left-fist and right-fist
% imagery? ERD in dB is NEGATIVE when power falls. Left-fist imagery drops C4 a lot (mean -7.1 dB) and C3 less (-3.1 dB), so
% C3 - C4 = -3.1 - (-7.1) = +4.0 dB: POSITIVE. Right-fist imagery drops C3 a lot (-4.8) and C4 little (-1.3), so C3 - C4 = -3.5 dB:
% NEGATIVE. So left gives a positive and right a negative C3 - C4. The same sign holds for plain power (P3 - P4 is positive when
% C4 power fell). The test statistic is T = mean(C3 - C4 | left) - mean(C3 - C4 | right): large and positive when the lateralisation
% is as predicted, about zero when the two channels behave alike. Top row: S072; bottom row: S099, labelled absent. Left column: the
% C3 - C4 difference over time (class means, smoothed 0.25 s). Right column: one value per trial.
fig = figure('Position', [100 100 900 700], 'Color', 'w'); tl = tiledlayout(2, 2, 'TileSpacing', 'compact');
for s9 = [subject 99]
    if s9 == subject, Ds = D; else, Ds = loadSubject(s9, edfDir, chans, cfg); end
    A = asymmetry(Ds, cfg, T_BASE, ACTIVE, sm);
    nexttile; hold on; xregion(T_BASE(1), T_BASE(2), 'FaceColor', cBase, 'FaceAlpha', 0.6); xregion(ACTIVE(1), ACTIVE(2), 'FaceColor', cAct, 'FaceAlpha', 0.5);
    plot(times, A.dtL, 'Color', cL, 'LineWidth', 2); plot(times, A.dtR, 'Color', cR, 'LineWidth', 2);
    yline(0, '--k'); xline(0, 'k'); ylim([-10 10]); xlabel('time from cue (s)'); ylabel(sprintf('S%03d  C3 - C4 (dB)', s9));
    if s9 == subject, legend('baseline', 'active', 'left-fist imagery', 'right-fist imagery', 'Location', 'northwest'); end
    nexttile; hold on;
    swarmchart(ones(sum(A.y == 0), 1), A.dTrial(A.y == 0), 18, cL, 'filled', 'MarkerFaceAlpha', 0.6, 'XJitterWidth', 0.4);
    swarmchart(2 * ones(sum(A.y == 1), 1), A.dTrial(A.y == 1), 18, cR, 'filled', 'MarkerFaceAlpha', 0.6, 'XJitterWidth', 0.4);
    plot([0.8 1.2], A.meanL * [1 1], 'k', 'LineWidth', 2.5); plot([1.8 2.2], A.meanR * [1 1], 'k', 'LineWidth', 2.5);
    yline(0, '--k'); xlim([0.4 2.6]); xticks([1 2]); xticklabels({'left-fist imagery', 'right-fist imagery'}); ylabel('C3 - C4 per trial (dB)');
    title(sprintf('means %+.1f and %+.1f dB: T = %+.1f dB', A.meanL, A.meanR, A.T));
    fprintf('S%03d: mean C3-C4 left %+.2f dB, right %+.2f dB, T = %+.2f dB\n', s9, A.meanL, A.meanR, A.T);
end
title(tl, 'Stage 9: C3 - C4 for left and right cues'); saveFig(fig, fullfile(figDir, 'stage9_c3_minus_c4.png'));

%% Contrast: the same view for a subject labelled "absent"
% Contralateral and ipsilateral are pooled over the two cue classes (left fist: C4 contra; right fist: C3 contra). Rows: the subject
% above (labelled present) and S099 (labelled absent in a2_lateralisation.csv). Columns: mu ERD over time, ERD spectrum, per-trial mu ERD.
% S099 shows no gap between the two channels in any column: no mu dip in the spectrum and equal means (printed below).
fig = figure('Position', [100 100 960 620], 'Color', 'w'); tiledlayout(2, 3, 'TileSpacing', 'compact');
for s = [subject 99]
    if s == subject, Ds = D; else, Ds = loadSubject(s, edfDir, chans, cfg); end
    R = contraIpsi(Ds, cfg, T_BASE, ACTIVE, sm);
    nexttile; hold on; xregion(T_BASE(1), T_BASE(2), 'FaceColor', cBase, 'FaceAlpha', 0.6); xregion(ACTIVE(1), ACTIVE(2), 'FaceColor', cAct, 'FaceAlpha', 0.5);
    plot(times, R.erdCon, 'Color', [0.1 0.1 0.1], 'LineWidth', 2); plot(times, R.erdIps, 'Color', [0.55 0.55 0.55], 'LineWidth', 2);
    yline(0, '--k'); xline(0, 'k'); ylim([-100 150]); ylabel(sprintf('S%03d  mu ERD %%', s)); xlabel('time from cue (s)');
    if s == subject, legend('baseline', 'active', 'contralateral', 'ipsilateral', 'Location', 'northeast'); end
    nexttile; hold on; xregion(cfg.MU(1), cfg.MU(2), 'FaceColor', 'k', 'FaceAlpha', 0.07); xregion(cfg.BETA(1), cfg.BETA(2), 'FaceColor', 'k', 'FaceAlpha', 0.03);
    plot(R.f, R.spCon, 'Color', [0.1 0.1 0.1], 'LineWidth', 2); plot(R.f, R.spIps, 'Color', [0.55 0.55 0.55], 'LineWidth', 2);
    yline(0, '--k'); xlim([4 35]); ylim([-10 5]); xlabel('frequency (Hz)'); ylabel('ERD (dB)');
    nexttile; hold on;
    swarmchart(ones(size(R.dbCon)), R.dbCon, 14, [0.1 0.1 0.1], 'filled', 'MarkerFaceAlpha', 0.5, 'XJitterWidth', 0.4);
    swarmchart(2 * ones(size(R.dbIps)), R.dbIps, 14, [0.55 0.55 0.55], 'filled', 'MarkerFaceAlpha', 0.5, 'XJitterWidth', 0.4);
    plot([0.8 1.2], mean(R.dbCon) * [1 1], 'r', 'LineWidth', 2.5); plot([1.8 2.2], mean(R.dbIps) * [1 1], 'r', 'LineWidth', 2.5);
    yline(0, '--k'); xticks([1 2]); xticklabels({'contralateral', 'ipsilateral'}); ylabel('mu ERD per trial (dB)'); xlim([0.4 2.6]);
    fprintf('S%03d: mean mu ERD contra %.2f dB, ipsi %.2f dB\n', s, mean(R.dbCon), mean(R.dbIps));
end
figTitle('Present (top) vs absent (bottom): same pipeline, same panels'); saveFig(fig, fullfile(figDir, 'contrast_present_absent.png'));

%% Local functions
function D = loadSubject(s, edfDir, chans, cfg)
% Raw, 1-40 Hz filtered, mu-filtered and mu-envelope signals per run, and the cue epochs cut from the filtered ones.
Ef = {}; Emu = {}; Eenv = {}; y = []; runId = []; onsetS = []; runList = [4 8 12];   % imagery runs (DECISIONS D1)
for k = 1:3
    r = runList(k);
    f = fullfile(edfDir, sprintf('S%03d', s), sprintf('S%03dR%02d.edf', s, r));
    [X, fs, code, onset, dur] = readRun(f, chans);
    Xf = bandpass(X, cfg.BAND, fs, 'ImpulseResponse', 'fir');
    Xm = bandpass(X, cfg.MU, fs, 'ImpulseResponse', 'fir');
    if fs ~= cfg.FS, Xf = resample(Xf, cfg.FS, fs); Xm = resample(Xm, cfg.FS, fs); end
    Xe = abs(hilbert(Xm));
    D.runs(k) = struct('fs', fs, 'raw', X, 'filt', Xf, 'code', code, 'onset', onset, 'dur', dur);
    for e = find(code == "T1" | code == "T2")'
        a = round(onset(e) * cfg.FS) + round(cfg.TMIN * cfg.FS);          % 0-based first sample, as a2_tfr.m and data.py
        if a < 0 || a + cfg.N_T > size(Xf, 1), continue; end
        Ef{end+1} = Xf(a+1:a+cfg.N_T, :); Emu{end+1} = Xm(a+1:a+cfg.N_T, :); Eenv{end+1} = Xe(a+1:a+cfg.N_T, :); %#ok<AGROW>
        y(end+1) = code(e) == "T2"; runId(end+1) = r; onsetS(end+1) = onset(e); %#ok<AGROW>
    end
end
D.Ef = cat(3, Ef{:}); D.Emu = cat(3, Emu{:}); D.Eenv = cat(3, Eenv{:}); D.y = y(:); D.run = runId(:); D.onset = onsetS(:);
end

function [X, fs, code, onset, dur] = readRun(f, chans)
info = edfinfo(f); tt = edfread(f);
labels = erase(string(info.SignalLabels), "."); vn = tt.Properties.VariableNames;
X = zeros(size(tt, 1) * double(info.NumSamples(1)), numel(chans));
for i = 1:numel(chans)
    j = find(strcmpi(labels, chans(i)), 1); assert(~isempty(j), 'channel %s not in %s', chans(i), f);
    X(:, i) = vertcat(tt.(vn{j}){:});
end
fs = double(info.NumSamples(1)) / seconds(info.DataRecordDuration);
code = string(info.Annotations.Annotations); onset = seconds(info.Annotations.Onset); dur = seconds(info.Annotations.Duration);
end

function [f, Pb, Pa] = psdByTrial(Ef, times, tBase, tAct, fs)
% Welch PSD per trial and channel in the baseline and the active window. The same 0.9 s Hann segment length (144 samples,
% ~1.1 Hz resolution, zero-padded to 512 points for a smooth curve) is used in both windows, so the two are comparable.
w = 144; nfft = 512;
ib = times >= tBase(1) & times <= tBase(2); ia = times >= tAct(1) & times <= tAct(2);
nTr = size(Ef, 3); [~, f] = pwelch(Ef(ib, :, 1), hann(w), 0, nfft, fs);
Pb = zeros(numel(f), size(Ef, 2), nTr); Pa = Pb;
for k = 1:nTr
    Pb(:, :, k) = pwelch(Ef(ib, :, k), hann(w), 0, nfft, fs);        % 145 samples, one 144-sample segment
    Pa(:, :, k) = pwelch(Ef(ia, :, k), hann(w), w / 2, nfft, fs);    % 561 samples, six overlapping segments
end
end

function R = contraIpsi(D, cfg, tBase, tAct, sm)
% mu ERD of the contralateral and ipsilateral channel, pooled over the cue classes: left fist -> C4 contralateral, right fist -> C3.
times = cfg.TMIN + (0:cfg.N_T-1) / cfg.FS; ib = times >= tBase(1) & times <= tBase(2); ia = times >= tAct(1) & times <= tAct(2);
P = D.Eenv .^ 2; base = reshape(mean(P(ib, :, :), [1 3]), 1, []);
nTr = numel(D.y); Pcon = zeros(cfg.N_T, nTr); Pips = Pcon;
for k = 1:nTr
    con = 1 + (D.y(k) == 0); ips = 3 - con;                                 % y=0 (left fist): contra = channel 2 (C4)
    Pcon(:, k) = P(:, con, k) / base(con); Pips(:, k) = P(:, ips, k) / base(ips);
end
R.erdCon = (movmean(mean(Pcon, 2), sm) - 1) * 100; R.erdIps = (movmean(mean(Pips, 2), sm) - 1) * 100;
R.dbCon = 10 * log10(mean(Pcon(ia, :), 1)); R.dbIps = 10 * log10(mean(Pips(ia, :), 1));
[R.f, Pb, Pa] = psdByTrial(D.Ef, times, tBase, tAct, cfg.FS); Pb = mean(Pb, 3);
aCon = zeros(numel(R.f), nTr); aIps = aCon;
for k = 1:nTr, con = 1 + (D.y(k) == 0); ips = 3 - con; aCon(:, k) = Pa(:, con, k); aIps(:, k) = Pa(:, ips, k); end
bCon = zeros(numel(R.f), nTr); bIps = bCon;
for k = 1:nTr, con = 1 + (D.y(k) == 0); ips = 3 - con; bCon(:, k) = Pb(:, con); bIps(:, k) = Pb(:, ips); end
R.spCon = 10 * log10(mean(aCon, 2) ./ mean(bCon, 2)); R.spIps = 10 * log10(mean(aIps, 2) ./ mean(bIps, 2));
end

function saveFig(fig, file)
% Same PNG as exportgraphics, without the interactive axes toolbar drawn into it.
for ax = findall(fig, 'Type', 'axes')', ax.Toolbar.Visible = 'off'; end
exportgraphics(fig, file, 'Resolution', 110);
end

function figTitle(str)
% Title belongs to the tiled layout, so it is centred over the panels and never clipped by the figure edge.
tl = findobj(gcf, 'Type', 'tiledlayout'); title(tl(1), str);
end

function A = asymmetry(D, cfg, tBase, tAct, sm)
% C3 - C4 in dB (channel 1 minus channel 2), over time for each cue class and as one value per trial.
times = cfg.TMIN + (0:cfg.N_T-1) / cfg.FS; ib = times >= tBase(1) & times <= tBase(2); ia = times >= tAct(1) & times <= tAct(2);
P = D.Eenv .^ 2; base = reshape(mean(P(ib, :, :), [1 3]), 1, []);             % pooled over all trials, label-blind
dbTime = @(sel) 10 * log10(movmean(mean(P(:, :, sel), 3), sm, 1) ./ base);    % [time x chan], ERD in dB of the class-mean power
tL = dbTime(D.y == 0); tR = dbTime(D.y == 1);
A.dtL = tL(:, 1) - tL(:, 2); A.dtR = tR(:, 1) - tR(:, 2);
db = squeeze(10 * log10(mean(P(ia, :, :), 1) ./ base));                       % [chan x trial]
A.dTrial = db(1, :)' - db(2, :)'; A.y = D.y;
A.meanL = mean(A.dTrial(D.y == 0)); A.meanR = mean(A.dTrial(D.y == 1)); A.T = A.meanL - A.meanR;
end
