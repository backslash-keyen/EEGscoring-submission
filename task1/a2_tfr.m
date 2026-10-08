%% A2, step 1: time-frequency power at C3, C4 and their neighbours
% Per subject, Morlet power around the imagery cue (runs 4, 8, 12), expressed relative to a pre-cue baseline.
% This is only the first bullet of A2. The spatial reference, the lateralisation index and the test come next,
% so the signals here are the channels as recorded (the recording reference is undocumented, DECISIONS D4).
% Needs base MATLAB + Signal Processing Toolbox. Run from anywhere: matlab -batch "run('task1/a2_tfr.m')"

%% Setup
% Every choice is a named constant with its reason beside it.
repo = fileparts(fileparts(mfilename('fullpath')));
dataRoot = fullfile(repo, 'data');
if ~isfolder(dataRoot), dataRoot = getenv('EEG_DATA_ROOT'); end    % git worktrees do not contain the git-ignored data
assert(isfolder(dataRoot), 'Data not found: put it in <repo>/data or set EEG_DATA_ROOT to the folder holding MNE-eegbci-data');
edfDir = fullfile(dataRoot, 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0');
cacheDir = fullfile(dataRoot, 'cache_matlab');     % per-subject power cubes; large, so outside git
outDir = fullfile(repo, 'outputs');
if ~isfolder(cacheDir), mkdir(cacheDir); end

subjects = 70:109;
runs = [4 8 12];              % imagery left/right fist (DECISIONS D1)
FS = 160;                     % analysis rate; 128 Hz-header subjects are resampled to it (DECISIONS D2)
TMIN = -1.5; TMAX = 4.0;      % epoch around the cue; windows past the recording end are dropped (DECISIONS D3)
N_T = floor((TMAX - TMIN) * FS) + 1;     % fixed epoch length, same as task1/data.py
BAND = [1 40];                % zero-phase band-pass on the continuous recording: drift and line noise out before wavelets
freqs = 6:1:30;               % covers mu (8-13 Hz) and beta (13-30 Hz) with 1 Hz steps, plus 6-8 Hz as context
T_BASE = [-1.0 -0.1];         % baseline inside the preceding rest, pooled over trials, label-blind (DECISIONS D5)
% C3, C4 and the 4 nearest electrodes of each by 3-D distance in the standard 10-10 montage. MATLAB has no
% montage built in, so the list is fixed here; it was checked against erd.neighbours() in task1/erd.py (DECISIONS D17).
chans = ["C3" "Cp3" "Fc3" "C5" "C1"   "C4" "Cp4" "Fc4" "C6" "C2"];
exampleSubject = 72;          % subject shown in the figures below
recompute = false;            % true: ignore cached cubes
times = TMIN + (0:N_T-1) / FS;
className = ["left" "right"];     % class 0 = T1 = imagine left fist, class 1 = T2 = imagine right fist

%% Check the wavelet code on a signal with a known answer
% A 10 Hz sine must peak at 10 Hz and carry almost no power at 20 Hz. If this fails, nothing downstream can be trusted.
fsT = 160; tT = (0:1/fsT:20)';
Pt = morletPower(sin(2*pi*10*tT), fsT, freqs);
mid = squeeze(mean(Pt(800:2400, 1, :), 1));
[~, iPk] = max(mid);
fprintf('Sine test: peak at %g Hz, power there %.3g, at 20 Hz %.3g\n', freqs(iPk), mid(iPk), mid(freqs == 20));
assert(freqs(iPk) == 10 && mid(freqs == 20) < 1e-3 * mid(iPk), 'Wavelet check failed');

%% Compute power for every subject
% Power is computed on the continuous filtered recording and only then cut into epochs. Cutting first (as task1/erd.py
% does) lets the wavelet run off both ends of each epoch onto zero padding. Measured on S072 against erd.py: power in the
% last 0.25 s is only ~0.4% lower there, so this is a precaution, not a correction (DECISIONS D15).
rows = cell(numel(subjects), 1);
for si = 1:numel(subjects)
    s = subjects(si);
    cf = fullfile(cacheDir, sprintf('tfr_S%03d.mat', s));
    if ~recompute && isfile(cf)
        S = load(cf, 'summary');
    else
        S = tfrSubject(s, edfDir, runs, chans, freqs, FS, TMIN, N_T, BAND, times, T_BASE);
        save(cf, '-v7', '-struct', 'S');
    end
    rows{si} = S.summary;
    fprintf('S%03d  %g Hz  L/R used %d/%d  dropped %d\n', s, S.summary.hdr_rate_hz, S.summary.n_left_used, S.summary.n_right_used, S.summary.n_dropped);
end
summ = vertcat(rows{:});
writetable(summ, fullfile(outDir, 'a2_tfr_matlab_summary.csv'));

%% Cross-check against the Python audit
% The trials entering the power cubes must be exactly the trials audit.csv says were used, per subject and class.
py = readtable(fullfile(repo, 'audit.csv'));
bad = find(py.n_left_used ~= summ.n_left_used | py.n_right_used ~= summ.n_right_used | py.sfreq_hz ~= summ.hdr_rate_hz);
if isempty(bad), disp('Trial counts and header rates match audit.csv for all 40 subjects.');
else, fprintf('MISMATCH vs audit.csv for subjects: %s\n', mat2str(summ.subject(bad)')); end
fprintf('Baseline power min over all subjects/channels/freqs: %.3g (must be > 0); non-finite values: %d\n', ...
    min(summ.min_baseline_power), sum(summ.n_nonfinite));

%% Time-frequency maps for one subject
% ERD% = mean power over trials of one cue class / baseline power - 1, per channel and frequency (negative = power fell).
% Expect left-fist imagery to be blue at C4 and right-fist imagery at C3 (contralateral). Rows: the C3 group, the C4 group;
% the first column is C3 or C4, the other four are its neighbours.
E = load(fullfile(cacheDir, sprintf('tfr_S%03d.mat', exampleSubject)), 'P', 'y', 'B');
for cl = 0:1
    M = (mean(E.P(:, :, :, E.y == cl), 4) ./ E.B - 1) * 100;       % [chan x freq x time], B broadcasts over time
    fig = figure('Position', [100 100 1300 480], 'Color', 'w');
    tl = tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
    for ci = 1:numel(chans)
        nexttile; imagesc(times, freqs, squeeze(M(ci, :, :))); axis xy; clim([-100 100]); colormap(bwr(256));
        xline(0, 'k'); title(chans(ci) + ternary(ci == 1 || ci == 6, "  (centre)", ""));
    end
    cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = 'ERD %  (relative to baseline)';
    title(tl, sprintf('S%03d, %s-fist imagery (n = %d trials)', exampleSubject, className(cl + 1), sum(E.y == cl)));
    xlabel(tl, 'time from cue (s)'); ylabel(tl, 'frequency (Hz)');
    exportgraphics(fig, fullfile(outDir, sprintf('a2_tfr_matlab_S%03d_%s.png', exampleSubject, className(cl + 1))), 'Resolution', 110);
end

%% Local functions
function out = tfrSubject(s, edfDir, runs, chans, freqs, FS, TMIN, N_T, band, times, tBase)
% All cue epochs of one subject as a power cube P [chan x freq x time x trial] (single, uV^2), plus labels and baseline.
segs = {}; y = []; runId = []; trialId = []; onsetS = []; fsHdr = []; nAnnot = 0;
for r = runs
    f = fullfile(edfDir, sprintf('S%03d', s), sprintf('S%03dR%02d.edf', s, r));
    [X, fs, code, onset] = readRun(f, chans);
    fsHdr(end+1) = fs; %#ok<AGROW>
    % FIR keeps the filter linear-phase like MNE's default; filtering at the header rate, then resampling, same order as data.py
    Xf = bandpass(X, band, fs, 'ImpulseResponse', 'fir');
    if fs ~= FS, Xf = resample(Xf, FS, fs); end
    Pc = morletPower(Xf, FS, freqs);                   % [time x chan x freq]
    k = 0;
    for e = find(code == "T1" | code == "T2")'
        nAnnot = nAnnot + 1;
        % 0-based first sample of the epoch, as task1/data.py. MATLAB rounds .5 away from zero, numpy to even:
        % at most one sample (6 ms) apart for the 128 Hz subjects.
        a = round(onset(e) * FS) + round(TMIN * FS);
        if a < 0 || a + N_T > size(Pc, 1), continue; end
        segs{end+1} = permute(Pc(a+1 : a+N_T, :, :), [2 3 1]); %#ok<AGROW>
        y(end+1) = code(e) == "T2"; runId(end+1) = r; trialId(end+1) = k; onsetS(end+1) = onset(e); k = k + 1; %#ok<AGROW>
    end
end
P = cat(4, segs{:});
% One baseline per channel and frequency, pooled over all trials and baseline samples; class labels are never used.
B = mean(P(:, :, times >= tBase(1) & times <= tBase(2), :), [3 4]);
out.P = P; out.y = y(:); out.run = runId(:); out.trial = trialId(:); out.onset = onsetS(:);
out.B = B; out.times = times; out.freqs = freqs; out.chans = chans;
out.summary = table(s, fsHdr(1), sum(y == 0), sum(y == 1), nAnnot, nAnnot - numel(y), min(B(:)), nnz(~isfinite(P)), ...
    'VariableNames', {'subject', 'hdr_rate_hz', 'n_left_used', 'n_right_used', 'n_annotated', 'n_dropped', 'min_baseline_power', 'n_nonfinite'});
end

function [X, fs, code, onset] = readRun(f, chans)
% Samples of the requested channels (uV, one column each), header rate, and the T0/T1/T2 annotations with onsets in s.
info = edfinfo(f);
tt = edfread(f);
labels = erase(string(info.SignalLabels), ".");        % EEGBCI labels carry trailing dots: 'C3..' -> 'C3'
vn = tt.Properties.VariableNames;                      % same order as the signal labels (annotations are not a variable)
X = zeros(size(tt, 1) * double(info.NumSamples(1)), numel(chans));
for i = 1:numel(chans)
    j = find(strcmpi(labels, chans(i)), 1);            % file uses 'Cp3', montages use 'CP3'
    assert(~isempty(j), 'channel %s not in %s', chans(i), f);
    X(:, i) = vertcat(tt.(vn{j}){:});
end
fs = double(info.NumSamples(1)) / seconds(info.DataRecordDuration);
code = string(info.Annotations.Annotations);
onset = seconds(info.Annotations.Onset);
end

function P = morletPower(X, fs, freqs)
% Morlet wavelet power, written to match mne.time_frequency.tfr_array_morlet (n_cycles = f/2, zero_mean = true) so the
% numbers port to Python unchanged. n_cycles = f/2 gives the same window width at every frequency (sigma_t = 1/(4 pi) = 0.08 s,
% i.e. sigma_f = 2 Hz): a constant absolute bandwidth, adequate for 1 Hz steps in mu/beta (DECISIONS D18).
[n, nCh] = size(X);
P = zeros(n, nCh, numel(freqs), 'single');
for k = 1:numel(freqs)
    f = freqs(k); sigT = (f / 2) / (2 * pi * f);
    t = 0:1/fs:5*sigT; t = [-fliplr(t), t(2:end)];                     % sample at exactly t = 0, out to 5 sigma
    w = (exp(2i*pi*f*t) - exp(-2 * (pi * f * sigT)^2)) .* exp(-t.^2 / (2 * sigT^2));   % zero mean => admissible wavelet
    w = w / (sqrt(0.5) * norm(w));                                     % MNE's scaling (unit norm for the real part)
    P(:, :, k) = single(abs(conv2(X, w(:), 'same')).^2);               % 'same' = central part, like MNE's offset (L-1)/2
end
end

function c = bwr(n)
% Blue (decrease) - white - red (increase), the same sense as the RdBu_r maps in task1/erd.py.
h = floor(n / 2); u = linspace(0, 1, h)';
c = [0.13 + 0.87*u, 0.40 + 0.60*u, 0.67 + 0.33*u; ones(h, 1), 1 - 0.85*u, 1 - 0.85*u];
end

function r = ternary(cond, a, b)
if cond, r = a; else, r = b; end
end
