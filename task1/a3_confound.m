%% A3 confound audit in MATLAB: can anything other than motor cortex separate left from right imagery?
% Re-implements task1/confound.py (headline decoders only) from the raw EDF files, with a CAUSAL pre-filter.
% Decoders: F1 frontal (eye movement), O1 occipital (visual), M1 sensorimotor mu/beta (the real signal), their pre-cue twins
% (F1_pre, O1_pre, M1_pre: the same decoder moved before the cue, where nothing can be decodable) and R2 (previous trial labels only, no EEG).
% Needs base MATLAB + Signal Processing Toolbox + Statistics and Machine Learning Toolbox (fitclinear).
% Run from anywhere: matlab -batch "run('task1/a3_confound.mlx')"   (the .mlx is the same code; MATLAB refuses run() on the .m while an .mlx
% of the same name sits beside it). About 70 s the first time (epochs are cached), about 50 s afterwards. Python reference: task1/confound.py.

%% Setup
repo = fileparts(fileparts(mfilename('fullpath')));
dataRoot = fullfile(repo, 'data');
if ~isfolder(dataRoot), dataRoot = getenv('EEG_DATA_ROOT'); end   % git worktrees do not contain the git-ignored data
assert(isfolder(dataRoot), 'Data not found: put it in <repo>/data or set EEG_DATA_ROOT to the folder holding MNE-eegbci-data');
edfDir = fullfile(dataRoot, 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0');
cacheDir = fullfile(dataRoot, 'cache_matlab_causal');          % epoched, causally filtered data; delete to force a re-read
outDir = fullfile(repo, 'outputs');
if ~isfolder(cacheDir), mkdir(cacheDir); end
if ~isfolder(outDir), mkdir(outDir); end

cfg.subjects = 70:109;
cfg.runs = [4 8 12];                  % imagery runs: T1 = imagine LEFT fist (label 0), T2 = RIGHT (label 1)
cfg.fs = 160; cfg.tmin = -1.5; cfg.tmax = 4.0; cfg.nT = round((cfg.tmax - cfg.tmin) * cfg.fs) + 1;   % 881 samples per epoch
cfg.band = [1 40]; cfg.filtOrder = 4; % Butterworth, run forward once (causal); see the Filter note below
cfg.C = 0.1;                          % logistic regression strength, fixed in advance, never tuned on test accuracy
cfg.alpha = 0.05;                     % chance threshold: smallest accuracy with exact binomial P(X >= k | p = 0.5) <= 5%
cfg.mu = [8 13]; cfg.beta = [13 30]; cfg.welchN = 128; cfg.welchStep = 64;   % Welch: Hann 128 samples, 64 overlap
cfg.binSamples = round(0.1 * cfg.fs); % 100 ms bins = 16 samples, exactly equal (Python's float masks give 15-17; see report)
cfg.prevLabels = 3;                   % order decoder sees the previous 3 labels of the same run

FRONT = ["Fp1" "Fpz" "Fp2" "Af7" "Af3" "Afz" "Af4" "Af8" "F5" "F6" "F7" "F8"];          % where eye movements show first
OCC = ["Po7" "Po3" "Poz" "Po4" "Po8" "O1" "Oz" "O2" "Iz"];                              % visual cortex
SENSORIMOTOR = ["Fc5" "Fc3" "Fc1" "Fcz" "Fc2" "Fc4" "Fc6" "C5" "C3" "C1" "Cz" "C2" "C4" "C6" "Cp5" "Cp3" "Cp1" "Cpz" "Cp2" "Cp4" "Cp6"];

% Each decoder = channel set + time window + feature type. Windows are [start end) in seconds around the cue.
% Pre-cue twins keep the same channels and feature type but sit before the cue: any accuracy there is a leak, not brain signal.
dec = struct('name', {'F1', 'F1_pre', 'O1', 'O1_pre', 'M1', 'M1_pre'}, ...
    'type', {'td', 'td', 'td', 'td', 'bp', 'bp'}, ...
    'chans', {FRONT, FRONT, OCC, OCC, SENSORIMOTOR, SENSORIMOTOR}, ...
    'win', {[0 0.5], [-0.5 0], [0 0.5], [-0.5 0], [0.5 4.0], [-1.5 -0.1]}, ...
    'base', {[-0.5 0], [-1.0 -0.5], [-0.5 0], [-1.0 -0.5], [], []});
withinNames = ["F1" "F1_pre" "O1" "O1_pre" "M1" "M1_pre" "R2"];   % within-subject table and figure (O1_pre added to the six asked for: it is the visual twin)
pooledNames = ["F1" "F1_pre" "O1" "M1" "R2"];            % pooled cross-subject set

%% Filter note: why the pre-filter must be causal
% A zero-phase filter (filtfilt, or MNE's default) smears the post-cue eye-movement response BACKWARDS in time, about 1.25 s into the
% pre-cue window (a synthetic step shows 58% leakage). That made the pre-cue "null" decodable, so the twin no longer measured chance.
% A causal filter only lets the past influence the present. We run a 4th-order Butterworth band-pass 1-40 Hz forward once as second-order
% sections, starting from its steady state on the first sample so the start of the recording does not ring. Python used MNE's
% minimum-phase FIR instead, so the two will not match exactly; that difference is reported, not hidden.

%% Epoch every subject and compute the features once
% Features are computed per subject right after loading, so the 40 subjects' raw epochs never sit in memory together.
nSub = numel(cfg.subjects);
S = repmat(struct('id', [], 'y', [], 'run', [], 'trial', [], 'F', struct()), nSub, 1);
tStart = tic;
for i = 1:nSub
    s = cfg.subjects(i); t0 = tic;
    D = loadSubject(s, edfDir, cacheDir, cfg);
    S(i).id = s; S(i).y = D.y; S(i).run = D.run; S(i).trial = D.trial;
    for d = dec
        ch = chanIndex(D.chNames, d.chans);
        if d.type == "td"
            S(i).F.(d.name) = featTD(D.X, ch, d.win, d.base, cfg);
        else
            S(i).F.(d.name) = featBP(D.X, D.chNames, d.chans, d.win, cfg);
        end
    end
    S(i).F.R2 = featOrder(D.y, D.run, cfg.prevLabels);
    fprintf('S%03d: %3d trials, features done (%.1f s, total %.0f s)\n', s, numel(D.y), toc(t0), toc(tStart));
end

%% Within-subject: leave-one-run-out
% Train on two imagery runs, test on the third, rotate, and score all held-out trials together. Runs are the natural fold because
% trials inside a run share drift and arousal; testing on a different run is the stricter test.
rows = {};
for i = 1:nSub
    n = numel(S(i).y); thr = binomThr(n, cfg.alpha);
    for name = withinNames
        acc = looAcc(S(i).F.(char(name)), S(i).y, S(i).run, cfg.C);
        rows(end+1, :) = {S(i).id, char(name), n, acc, thr, string(acc >= thr - 1e-12)}; %#ok<SAGROW>
    end
end
within = cell2table(rows, 'VariableNames', {'subject', 'decoder', 'n', 'acc', 'thr_binom', 'above_binom'});
within.above_binom = strrep(strrep(within.above_binom, "true", "True"), "false", "False");   % same spelling as the Python CSVs
writetable(within, fullfile(outDir, 'a3_matlab_within_subject.csv'));
for name = withinNames
    m = within.decoder == name;
    fprintf('%-7s mean within-subject acc %.3f, %2d / %d subjects above their binomial threshold (5%% by chance = %.1f)\n', ...
        name, mean(within.acc(m)), sum(within.above_binom(m) == "True"), nSub, cfg.alpha * nSub);
end

%% Pooled cross-subject
% Train on 7 folds of subjects and test on the held-out 5 subjects. The fold assignment is read from a3_folds.csv (the same one Python
% used) so both languages split subjects identically; the scaler sees training subjects only.
folds = readtable(fullfile(outDir, 'a3_folds.csv'));
foldOf = arrayfun(@(s) folds.fold(folds.subject == s), [S.id]);
Y = vertcat(S.y); subjOf = repelem([S.id]', arrayfun(@(x) numel(x.y), S));
foldTrial = foldOf(arrayfun(@(s) find([S.id] == s), subjOf));
nTot = numel(Y); thrPool = binomThr(nTot, cfg.alpha);
rows = {};
for name = pooledNames
    F = cell2mat(arrayfun(@(x) x.F.(char(name)), S, 'UniformOutput', false));
    correct = 0;
    for f = unique(foldOf(:))'
        te = foldTrial == f;
        correct = correct + sum(fitPredict(F(~te, :), Y(~te), F(te, :), cfg.C) == Y(te));
    end
    rows(end+1, :) = {char(name), nTot, correct / nTot, thrPool}; %#ok<SAGROW>
    fprintf('pooled %-7s acc %.3f (threshold %.3f, n = %d)\n', name, correct / nTot, thrPool, nTot);
end
pooled = cell2table(rows, 'VariableNames', {'decoder', 'n', 'acc', 'thr_binom'});
writetable(pooled, fullfile(outDir, 'a3_matlab_pooled.csv'));

%% Compare with the Python (causal-filter) results
% The two implementations share labels, trials, folds, features and classifier objective, and differ in the pre-filter (Butterworth IIR
% vs MNE minimum-phase FIR), the resampler for the three 128 Hz subjects, and the solver. Expect close but not identical numbers.
% R2 uses no EEG at all, so it is the control for the classifier and the trial bookkeeping: it should agree almost exactly.
pyW = readtable(fullfile(outDir, 'a3_within_subject.csv'));
pyP = readtable(fullfile(outDir, 'a3_cross_subject.csv'));
pyAbove = @(t) ismember(lower(string(t.above_binom)), ["true" "1"]);   % readtable may parse True/False as text or logical
nMismatch = 0;
rows = {};
for name = withinNames
    a = pyW(string(pyW.decoder) == name, :); b = within(within.decoder == name, :);
    a = sortrows(a, 'subject'); b = sortrows(b, 'subject');
    assert(isequal(a.subject, b.subject), 'subject lists differ for %s', name);
    nMismatch = nMismatch + sum(a.n ~= b.n);
    r = corr(a.acc, b.acc);
    pp = pyP(string(pyP.decoder) == name, :); mp = pooled(string(pooled.decoder) == name, :);
    pyPool = NaN; mlPool = NaN;
    if ~isempty(pp), pyPool = pp.acc; end
    if ~isempty(mp), mlPool = mp.acc; end
    rows(end+1, :) = {char(name), mean(a.acc), mean(b.acc), mean(abs(a.acc - b.acc)), sum(pyAbove(a)), sum(b.above_binom == "True"), ...
        r, pyPool, mlPool}; %#ok<SAGROW>
end
cmp = cell2table(rows, 'VariableNames', {'decoder', 'py_mean_acc', 'ml_mean_acc', 'mean_abs_diff_per_subject', 'py_n_above', 'ml_n_above', ...
    'pearson_r_per_subject_acc', 'py_pooled_acc', 'ml_pooled_acc'});
writetable(cmp, fullfile(outDir, 'a3_matlab_vs_python.csv'));
disp(cmp);
fprintf('Trial counts differing from Python: %d subject-decoder rows\n', nMismatch);

%% Figure: within-subject accuracy per decoder
% One dot per subject; black bar = mean; the grey band spans the lowest to highest per-subject chance threshold (it varies with n).
% The pre-cue twins (_pre) should sit on the 0.5 line with about 5% of dots above the band; if they do not, something leaks.
fig = figure('Position', [100 100 1000 480], 'Color', 'w'); hold on;
nd = numel(withinNames); thrAll = within.thr_binom;
patch([0.4 nd+0.6 nd+0.6 0.4], [min(thrAll) min(thrAll) max(thrAll) max(thrAll)], [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
yline(0.5, '--', 'Color', [0.4 0.4 0.4]);
for k = 1:nd
    m = within.decoder == withinNames(k); a = within.acc(m); above = within.above_binom(m) == "True";
    jit = (mod((1:numel(a))' * 37, 41) / 41 - 0.5) * 0.5;            % fixed jitter so the figure is reproducible without a random seed
    col = [0.35 0.35 0.35]; if endsWith(withinNames(k), "_pre"), col = [0.80 0.15 0.15]; end
    scatter(k + jit(~above), a(~above), 22, col, 'filled', 'MarkerFaceAlpha', 0.35);
    scatter(k + jit(above), a(above), 26, col, 'filled', 'MarkerEdgeColor', 'k');
    plot([k - 0.32 k + 0.32], mean(a) * [1 1], 'k', 'LineWidth', 2.5);
    text(k, 1.02, sprintf('%d/%d', sum(above), numel(a)), 'HorizontalAlignment', 'center', 'FontSize', 10);
end
xlim([0.4 nd + 0.6]); ylim([0.2 1.06]); xticks(1:nd); xticklabels(strrep(withinNames, '_', '\_'));
ylabel('leave-one-run-out accuracy'); grid on; box off;
title('A3 (MATLAB, causal filter): one dot per subject, black = mean, grey band = chance threshold range; outlined dots above threshold');
ax = gca; ax.Toolbar.Visible = 'off';   % keep the axes toolbar out of the exported PNG
exportgraphics(fig, fullfile(outDir, 'a3_matlab_summary.png'), 'Resolution', 150);
fprintf('Done in %.0f s\n', toc(tStart));

%% Local functions
function D = loadSubject(s, edfDir, cacheDir, cfg)
% Cue epochs (channels x samples x trials, microvolts) from the causally filtered continuous runs, cached per subject.
f = fullfile(cacheDir, sprintf('S%03d.mat', s));
if isfile(f), D = load(f); D.X = double(D.X); return; end
Xs = {}; y = []; runId = []; trial = []; onsetS = [];
for r = cfg.runs
    [X, fs, code, onset, chNames] = readRun(fullfile(edfDir, sprintf('S%03d', s), sprintf('S%03dR%02d.edf', s, r)));
    % filter at the recorded rate, then resample (as the Python pipeline did), so the 1-40 Hz band is defined on the original signal
    Xf = causalBandpass(X, fs, cfg.band, cfg.filtOrder);
    if fs ~= cfg.fs, Xf = resample(Xf, cfg.fs, fs); end    % subjects 88, 92, 100 are stored at 128 Hz
    k = 0;
    for e = find(code == "T1" | code == "T2")'
        a = round(onset(e) * cfg.fs) + round(cfg.tmin * cfg.fs);   % 0-based first sample of the epoch
        if a < 0 || a + cfg.nT > size(Xf, 1), continue; end        % window must fit in the recording (drops S104 run 8's last trial)
        Xs{end+1} = Xf(a+1:a+cfg.nT, :)'; %#ok<AGROW>
        y(end+1) = code(e) == "T2"; runId(end+1) = r; trial(end+1) = k; onsetS(end+1) = onset(e); k = k + 1; %#ok<AGROW>
    end
end
D.X = cat(3, Xs{:}); D.y = y(:); D.run = runId(:); D.trial = trial(:); D.onset = onsetS(:); D.chNames = chNames;
X = single(D.X); y = D.y; run = D.run; trial = D.trial; onset = D.onset; chNames = D.chNames; %#ok<NASGU>
save(f, 'X', 'y', 'run', 'trial', 'onset', 'chNames');
end

function [X, fs, code, onset, chNames] = readRun(f)
% All 64 EEG channels (samples x channels, microvolts) plus the annotation list. Channel names carry trailing dots in the EDF ('C3..').
info = edfinfo(f); tt = edfread(f);
chNames = erase(string(info.SignalLabels), ".");
vn = tt.Properties.VariableNames;
X = zeros(size(tt, 1) * double(info.NumSamples(1)), numel(chNames));
for i = 1:numel(chNames), X(:, i) = vertcat(tt.(vn{i}){:}); end
fs = double(info.NumSamples(1)) / seconds(info.DataRecordDuration);
code = string(info.Annotations.Annotations); onset = seconds(info.Annotations.Onset);
end

function Y = causalBandpass(X, fs, band, ord)
% One forward pass through a Butterworth band-pass as second-order sections: no filtfilt, so nothing leaks backwards in time.
[z, p, k] = butter(ord, band / (fs / 2), 'bandpass');
[sos, g] = zp2sos(z, p, k);
Y = X * g;
for i = 1:size(sos, 1)
    b = sos(i, 1:3); a = sos(i, 4:6);
    Y = filter(b, a, Y, lfilterZi(b, a) * Y(1, :), 1);     % steady-state start on the first sample: no switch-on transient
end
end

function zi = lfilterZi(b, a)
% Initial state that makes a constant input give a constant output (scipy.signal.lfilter_zi); uses only the first sample, so still causal.
b = b / a(1); a = a / a(1); n = numel(a);
C = zeros(n - 1); C(:, 1) = -a(2:end); C(1:n-2, 2:n-1) = eye(n - 2);
zi = (eye(n - 1) - C) \ (b(2:end) - a(2:end) * b(1))';
end

function ch = chanIndex(names, wanted)
% Positions of the wanted channels in the recording, matched case-insensitively (the EDF says 'Fc5', 'Cp3', 'Af7', 'Po7', 'Iz'...).
ch = zeros(1, numel(wanted));
for j = 1:numel(wanted)
    k = find(strcmpi(names, wanted(j)), 1); assert(~isempty(k), 'channel %s not found', wanted(j)); ch(j) = k;
end
end

function w = winIdx(win, cfg)
% Time window [t0, t1) in seconds around the cue -> 1-based sample indices (cue at sample 241).
w = (round((win(1) - cfg.tmin) * cfg.fs) + 1) : round((win(2) - cfg.tmin) * cfg.fs);
end

function F = featTD(X, ch, win, base, cfg)
% Baseline-subtracted bin means of the raw (filtered) voltage: shape of the post-cue deflection, where eye movements live.
% The baseline is subtracted per trial and channel so slow offsets between trials cannot masquerade as a class difference.
b = mean(X(ch, winIdx(base, cfg), :), 2);
w = winIdx(win, cfg); nb = floor(numel(w) / cfg.binSamples);
F = [];
for j = 1:nb
    seg = w((j-1) * cfg.binSamples + (1:cfg.binSamples));
    F = [F, squeeze(mean(X(ch, seg, :), 2) - b)']; %#ok<AGROW>    % trials x channels, one block per bin
end
end

function F = featBP(X, names, chans, win, cfg)
% Welch log10 power in mu and beta, LEFT MINUS RIGHT over mirror pairs: the classic lateralisation of sensorimotor rhythms.
% Mirror pair = odd electrode number k with k+1 of the same stem (C3-C4, Fc1-Fc2, ...); midline channels have no partner and are dropped.
[L, R] = mirrorPairs(names, chans);
w = winIdx(win, cfg); N = cfg.welchN; hop = cfg.welchStep;
nSeg = floor((numel(w) - N) / hop) + 1;
hw = 0.5 - 0.5 * cos(2 * pi * (0:N-1)' / N);                  % periodic Hann, as scipy's default window
nf = N / 2 + 1; fv = (0:nf-1) * cfg.fs / N;
P = 0;
for q = 1:nSeg
    seg = X(:, w((q-1) * hop + (1:N)), :);                    % channels x N x trials
    seg = (seg - mean(seg, 2)) .* hw';                        % constant detrend then window, as scipy welch
    sp = abs(fft(seg, N, 2)) .^ 2;
    P = P + sp(:, 1:nf, :);
end
P = P / nSeg / (cfg.fs * sum(hw .^ 2)); P(:, 2:nf-1, :) = 2 * P(:, 2:nf-1, :);   % density scaling, one-sided
F = [];
for band = {cfg.mu, cfg.beta}
    bd = band{1};
    bp = log10(squeeze(mean(P(:, fv >= bd(1) & fv <= bd(2), :), 2)) + 1e-30);   % channels x trials
    F = [F, (bp(L, :) - bp(R, :))']; %#ok<AGROW>
end
end

function [L, R] = mirrorPairs(names, chans)
% Index pairs (into the recording's channels) of the left and right member of each mirror pair inside `chans`.
L = []; R = []; low = cellstr(lower(chans));
for q = 1:numel(low)
    c = low{q}; num = regexp(c, '\d+$', 'match', 'once');
    if isempty(num) || mod(str2double(num), 2) == 0, continue; end
    mate = [c(1:end-numel(num)) num2str(str2double(num) + 1)];
    if ismember(mate, low)
        L(end+1) = find(strcmpi(names, c), 1); R(end+1) = find(strcmpi(names, mate), 1); %#ok<AGROW>
    end
end
end

function F = featOrder(y, run, nPrev)
% Previous nPrev labels of the same run coded -1 (left) / +1 (right), 0 where absent: the no-EEG control for sequence structure.
sgn = 2 * y - 1; F = zeros(numel(y), nPrev);
for i = 1:numel(y)
    for k = 1:nPrev
        j = i - k;
        if j >= 1 && run(j) == run(i), F(i, k) = sgn(j); end
    end
end
end

function p = fitPredict(Xtr, ytr, Xte, C)
% L2 logistic regression equal to sklearn LogisticRegression(C): sklearn minimises 0.5||w||^2 + C*sum(loss), fitclinear minimises
% mean(loss) + Lambda/2*||w||^2, so Lambda = 1/(C*n). Neither penalises the intercept. Scaler uses training rows only.
mu = mean(Xtr, 1); sd = std(Xtr, 1, 1); sd(sd == 0) = 1;
m = fitclinear((Xtr - mu) ./ sd, ytr, 'Learner', 'logistic', 'Regularization', 'ridge', 'Lambda', 1 / (C * numel(ytr)), ...
    'Solver', 'lbfgs', 'ClassNames', [0 1], 'BetaTolerance', 1e-10, 'GradientTolerance', 1e-8, 'IterationLimit', 5000);
p = predict(m, (Xte - mu) ./ sd);
end

function acc = looAcc(F, y, run, C)
% Leave-one-run-out accuracy pooled over the held-out runs.
correct = 0;
for r = unique(run)'
    te = run == r;
    correct = correct + sum(fitPredict(F(~te, :), y(~te), F(te, :), C) == y(te));
end
acc = correct / numel(y);
end

function thr = binomThr(n, alpha)
% Smallest accuracy k/n with exact binomial P(X >= k | n, 0.5) <= alpha. Upper tail summed directly (no 1 - cdf cancellation).
j = 0:n; pmf = exp(gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1) - n * log(2));
tail = flip(cumsum(flip(pmf)));                      % tail(k+1) = P(X >= k)
thr = (find(tail <= alpha, 1) - 1) / n;
end
