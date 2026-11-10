%% RUN_NBS_THREE_GROUP  Edge-wise group differences with the Network-Based Statistic (NBS 1.2).
%
% Same computation as 'Run NBS' in the NBS GUI, called from a script (NBSglm + NBSstats).
%   Edges     : 63,903 edges (upper triangle of 358 x 358), Fisher-z FC (atanh; |r| clipped at 1 - 1e-7; diagonal 0)
%   Design    : no intercept; [HC, TC, BB indicators, Age, Sex (1/2), MeanFD]
%   Component : size = number of edges ('Extent'); FWE-corrected P from 5,000 permutations (Freedman-Lane), alpha 0.05
%   1. Omnibus F contrast [1 1 1 0 0 0] (reduced model = intercept + covariates, 2 df), threshold F = 3.1^2 = 9.61;
%      also the F threshold with the same P as |t| = 3.1 (sensitivity).
%   2. If the omnibus component is significant: six one-sided t contrasts (each athlete group vs non-athletes and the two
%      athlete groups against each other, both directions), threshold t = 3.1. Holm correction over these six P values is
%      done in pipeline/03_network_connectivity/s04_nbs_summary.py.
% Each run uses its own fixed random seed. The edges of every significant component are written to results/edges/.
%
% Inputs : <output_root>/02_gradients/glasser358_HCreference/FC.mat and 01_cohort/subject_covariates.csv
% Outputs: <output_root>/03_network_connectivity/nbs/
% Usage (MATLAB): run('<code_release>/matlab/nbs/run_nbs_three_group.m')

here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));
cfg = config_paths();
addpath(cfg.nbs_dir);
OUT = cfg.layout.nbs;
if ~exist(fullfile(OUT, 'edges'), 'dir'), mkdir(fullfile(OUT, 'edges')); end
N = 358;
ind_upper = find(triu(ones(N, N), 1));
PERMS = 5000;

%% inputs
S = load(fullfile(cfg.layout.grad_A, 'FC.mat'));
FC = double(S.FC);
ns = size(FC, 3);
ids = S.subject_ids;
y = zeros(ns, numel(ind_upper));
for s = 1:ns
    r = FC(:, :, s);
    r(1:N + 1:end) = 0;
    r = max(min(r, 1 - 1e-7), -(1 - 1e-7));
    Z = atanh(r);
    Z(1:N + 1:end) = 0;
    y(s, :) = Z(ind_upper);
end
C = readtable(fullfile(cfg.output_root, '01_cohort', 'subject_covariates.csv'), 'TextType', 'string');
assert(isequal(string(C.participant_id), string(ids(:))), 'covariate table and FC are in a different order');
g = string(C.Group);
X = [double(g == "HC"), double(g == "TC"), double(g == "BB"), C.Age, C.Sex, C.MeanFD];
save(fullfile(OUT, 'design_three_group.mat'), 'X', 'ids');

dfe = ns - size(X, 2);
p_t31 = 2 * tcdf(-3.1, dfe);
F_pmatched = finv(1 - p_t31, 2, dfe);
runs = { ...
    'omnibus_F_thr9.61',  'ftest', [1 1 1 0 0 0],  3.1^2; ...
    'omnibus_F_pmatched', 'ftest', [1 1 1 0 0 0],  F_pmatched; ...
    'HC_gt_TC',           'ttest', [1 -1 0 0 0 0], 3.1; ...
    'TC_gt_HC',           'ttest', [-1 1 0 0 0 0], 3.1; ...
    'HC_gt_BB',           'ttest', [1 0 -1 0 0 0], 3.1; ...
    'BB_gt_HC',           'ttest', [-1 0 1 0 0 0], 3.1; ...
    'TC_gt_BB',           'ttest', [0 1 -1 0 0 0], 3.1; ...
    'BB_gt_TC',           'ttest', [0 -1 1 0 0 0], 3.1};

%% run every contrast
summary = table();
for k = 1:size(runs, 1)
    name = runs{k, 1};
    GLM = struct('y', y, 'X', X, 'contrast', runs{k, 3}, 'test', runs{k, 2}, 'perms', PERMS);
    rng(20261003 + k);
    STATS = struct('thresh', runs{k, 4}, 'alpha', 0.05, 'size', 'Extent', 'N', N);
    STATS.test_stat = NBSglm(GLM);                 % row 1 = observed statistic, rows 2.. = permutations
    [nc, con, pv] = NBSstats(STATS, -1, GLM);
    obs_stat = zeros(N); obs_stat(ind_upper) = STATS.test_stat(1, :); obs_stat = obs_stat + obs_stat';
    fprintf('%-20s threshold %.3f: %d significant component(s)\n', name, runs{k, 4}, nc);
    res = struct('name', name, 'test', runs{k, 2}, 'contrast', runs{k, 3}, 'thresh', runs{k, 4}, 'perms', PERMS, ...
        'seed', 20261003 + k, 'n', nc, 'pval', pv, 'obs_stat', obs_stat);
    res.con_mat = con;
    save(fullfile(OUT, ['NBS_' name '.mat']), '-struct', 'res', '-v7.3');

    % edges of every significant component: node indices (1-based, Glasser-358 order), component, P, edge statistic
    rows = [];
    for c = 1:nc
        A = full(con{c}); A = triu(A | A', 1);
        [i, j] = find(A);
        rows = [rows; i, j, repmat(c, numel(i), 1), repmat(pv(c), numel(i), 1), obs_stat(sub2ind([N N], i, j))]; %#ok<AGROW>
    end
    if isempty(rows), rows = zeros(0, 5); end
    writetable(array2table(rows, 'VariableNames', {'Node_i', 'Node_j', 'Component', 'Component_P', 'Edge_stat'}), ...
        fullfile(OUT, 'edges', ['NBS_' name '_edges.csv']));

    ne = 0; p1 = NaN;
    if nc > 0, ne = nnz(triu(con{1})); p1 = pv(1); end
    summary = [summary; table(string(name), string(runs{k, 2}), string(mat2str(runs{k, 3})), runs{k, 4}, PERMS, nc, ne, p1, ...
        'VariableNames', {'Run', 'Test', 'Contrast', 'Threshold', 'Perms', 'N_components', 'Edges_largest', 'P_largest'})]; %#ok<AGROW>
end
writetable(summary, fullfile(OUT, 'NBS_three_group_run_summary.csv'));
disp(summary)
