%% A1 data audit (MATLAB twin of task1/audit.py)
% One row per subject for EEGBCI subjects 70-109, imagery runs 4, 8, 12 -> outputs/audit_matlab.csv.
% Needs only base MATLAB + Signal Processing Toolbox (edfinfo/edfread, pwelch). Run from the repo root:
%   matlab -batch "run('task1/audit.m')"

repo = fileparts(fileparts(mfilename('fullpath')));
dataRoot = fullfile(repo, 'data');
if ~isfolder(dataRoot), dataRoot = getenv('EEG_DATA_ROOT'); end    % git worktrees do not contain the git-ignored data
assert(isfolder(dataRoot), 'Data not found: put it in <repo>/data or set EEG_DATA_ROOT to the folder holding MNE-eegbci-data');
dataDir = fullfile(dataRoot, 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0');
subjects = 70:109;
runs = [4 8 12];            % imagery left/right fist (DECISIONS D1)
FS_NOMINAL = 160;           % the rate every analysis uses; 128 Hz-header files are resampled to it (DECISIONS D2)
TMIN = -1.5; TMAX = 4.0;    % epoch window around the cue (DECISIONS D3)
N_T = floor((TMAX - TMIN) * FS_NOMINAL) + 1;   % same fixed epoch length as task1/data.py
CUE_NOMINAL = 4.1;
codeMeaning = "T0=rest,T1=imagine left fist,T2=imagine right fist";

rows = cell(numel(subjects), 1);
for si = 1:numel(subjects)
    s = subjects(si);
    nL = 0; nR = 0; nLused = 0; nRused = 0;
    fsList = []; sizeOk = true; durs = []; cues = []; mains = []; anomalies = strings(0);
    for r = runs
        f = fullfile(dataDir, sprintf('S%03d', s), sprintf('S%03dR%02d.edf', s, r));
        [fsHdr, sizeOkRun] = edfHeaderRate(f);    % straight from the file bytes, not from edfinfo
        info = edfinfo(f);
        fsInfo = double(info.NumSamples(1)) / seconds(info.DataRecordDuration);
        if abs(fsInfo - fsHdr) > 1e-6, anomalies(end+1) = sprintf('R%d: edfinfo rate != raw header rate', r); end %#ok<SAGROW>
        if ~sizeOkRun, anomalies(end+1) = sprintf('R%d: file size inconsistent with EDF header', r); end %#ok<SAGROW>
        fsList(end+1) = fsHdr; sizeOk = sizeOk && sizeOkRun; %#ok<SAGROW>

        a = info.Annotations;
        code = string(a.Annotations); onset = seconds(a.Onset); dur = seconds(a.Duration);
        extra = setdiff(unique(code), ["T0" "T1" "T2"]);
        if ~isempty(extra), anomalies(end+1) = sprintf('R%d: unexpected codes %s', r, strjoin(extra, '/')); end %#ok<SAGROW>
        isCue = code == "T1" | code == "T2";
        n1 = sum(code == "T1"); n2 = sum(code == "T2");
        nL = nL + n1; nR = nR + n2;
        cues = [cues; dur(isCue)]; %#ok<AGROW>
        runDur = double(info.NumDataRecords) * seconds(info.DataRecordDuration);
        durs(end+1) = runDur; %#ok<SAGROW>
        if n1 + n2 < 14 || (n1 + n2 > 16 && fsHdr == FS_NOMINAL)
            anomalies(end+1) = sprintf('R%d: %dL/%dR trials', r, n1, n2); %#ok<SAGROW>
        end

        % Trials whose analysis window runs past the end of the (resampled) recording are dropped, never silently.
        nTimes = round(runDur * fsHdr * FS_NOMINAL / fsHdr);   % samples after resampling to 160 Hz
        for k = find(isCue)'
            a0 = round(onset(k) * FS_NOMINAL + TMIN * FS_NOMINAL);
            if a0 < 0 || a0 + N_T > nTimes, continue; end
            if code(k) == "T1", nLused = nLused + 1; else, nRused = nRused + 1; end
        end

        % Independent rate check: US mains hum is 60 Hz whatever the header says (a mislabelled 160 Hz file
        % claiming 128 Hz would show it at 48 Hz). Strongest narrow line between 40 Hz and Nyquist.
        tt = edfread(f);
        chans = tt.Properties.VariableNames;
        X = cell2mat(cellfun(@(v) vertcat(tt.(v){:}), chans, 'UniformOutput', false));
        win = round(4 * fsHdr);
        [p, fq] = pwelch(X, hann(win), win / 2, win, fsHdr);
        p = mean(p, 2); m = fq >= 40 & fq < fsHdr / 2 - 0.5;
        [~, im] = max(p(m)); fm = fq(m); mains(end+1) = fm(im); %#ok<SAGROW>
    end

    if numel(unique(fsList)) > 1 || fsList(1) ~= FS_NOMINAL
        anomalies(end+1) = sprintf('header rate [%s] Hz', strjoin(string(unique(fsList)), ',')); end
    if max(durs) - min(durs) > 5
        anomalies(end+1) = sprintf('run durations [%s] s differ', strjoin(string(round(durs, 1)), ', ')); end
    odd = unique(round(cues(abs(cues - CUE_NOMINAL) > 0.2), 2));
    if ~isempty(odd) && all(fsList == FS_NOMINAL), anomalies(end+1) = sprintf('cue lengths [%s] s', strjoin(string(odd), ',')); end
    if isempty(anomalies), anomalies = "none"; end

    rows{si} = table(s, strjoin(string(runs), ','), fsList(1), round(sum(durs), 1), nL, nR, codeMeaning, ...
        strjoin(string(unique(round(cues, 2))'), ','), sizeOk, median(mains), strjoin(anomalies, '; '), nLused, nRused, nL + nR - nLused - nRused, ...
        'VariableNames', {'subject','runs_loaded','hdr_rate_hz','duration_s','n_left','n_right','event_code_meaning','cue_s', ...
        'hdr_size_consistent','mains_peak_hz','anomalies','n_left_used','n_right_used','n_dropped'});
    fprintf('S%03d  %g Hz  L/R %d/%d  used %d/%d  mains %.2f Hz  %s\n', s, fsList(1), nL, nR, nLused, nRused, median(mains), rows{si}.anomalies);
end

res = vertcat(rows{:});
writetable(res, fullfile(repo, 'outputs', 'audit_matlab.csv'));
fprintf('\ntotals L/R annotated %d/%d, used %d, dropped %d\n', sum(res.n_left), sum(res.n_right), ...
    sum(res.n_left_used + res.n_right_used), sum(res.n_dropped));

%% Cross-check against the Python audit (audit.csv): counts and rates must agree
py = readtable(fullfile(repo, 'audit.csv'));
bad = find(py.n_left ~= res.n_left | py.n_right ~= res.n_right | py.n_left_used ~= res.n_left_used | ...
    py.n_right_used ~= res.n_right_used | py.sfreq_hz ~= res.hdr_rate_hz);
if isempty(bad), disp('MATLAB audit matches audit.csv on rate, annotated and used trial counts for all subjects.');
else, fprintf('MISMATCH vs audit.csv for subjects: %s\n', mat2str(res.subject(bad)')); end

%% Local function: sampling rate straight from the raw EDF header bytes
function [fs, sizeOk] = edfHeaderRate(f)
% EDF layout: n_records at byte 237 (1-based), record duration at 245, n_signals at 253, then per-signal fields.
% samples/record is the 8-byte field after the 80-byte prefilter field. Rate = samples per record / record duration.
fid = fopen(f, 'r'); b = fread(fid, inf, 'uint8=>char')'; fclose(fid);
n = str2double(b(237:244)); recDur = str2double(b(245:252)); ns = str2double(b(253:256));
off = 256 + ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80);
spr = zeros(1, ns); lab = strings(1, ns);
for i = 1:ns
    spr(i) = str2double(b(off + (i-1)*8 + 1 : off + i*8));
    lab(i) = strtrim(b(256 + (i-1)*16 + 1 : 256 + i*16));
end
sig = spr(~contains(lab, 'Annot'));
fs = sig(1) / recDur;
% a header that lied about its rate would not match the number of bytes actually stored
sizeOk = numel(b) == 256 * (ns + 1) + n * sum(spr) * 2;
end
