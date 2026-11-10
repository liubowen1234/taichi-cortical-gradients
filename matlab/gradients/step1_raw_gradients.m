function step1_raw_gradients(FC, analysis_root)
% STEP1_RAW_GRADIENTS  Individual connectivity gradients with BrainSpace.
%
% Parameters (identical for every FC matrix, including the reference in Step 2):
%   input        individual Pearson-r FC (no Fisher z before the embedding), diagonal set to 0
%   sparsity     90: each row keeps its strongest 10% of connections (done once, inside BrainSpace fit)
%   kernel       cosine similarity (negative affinities set to 0 by BrainSpace)
%   embedding    diffusion maps, alpha = 0.5, diffusion time = 0
%   components   10; random_state = 0
%
% Output: <analysis_root>/Step1_raw_gradients/BrainSpace_formal_raw_gradients.mat
%   gradients_all  parcels x 10 x subjects     lambda_all  10 x subjects (eigenvalues, used as lambda1-lambda3)

kernel_name = 'cs';
approach_name = 'dm';
n_gradients = 10;
random_state = 0;
sparsity_percentile = 90;
alpha = 0.5;
diffusion_time = 0;

if min(FC(:)) < -1.0001 || max(FC(:)) > 1.0001
    error('FC must be Pearson r (values in [-1, 1]).');
end
[n_parcels, ~, n_subjects] = size(FC);
output_dir = fullfile(analysis_root, 'Step1_raw_gradients');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
diag_idx = 1:(n_parcels + 1):(n_parcels * n_parcels);

gradients_all = nan(n_parcels, n_gradients, n_subjects);
lambda_all = [];
fit_success = false(n_subjects, 1);
fprintf('Step 1: individual gradients\n');
for s = 1:n_subjects
    A = FC(:, :, s);
    A(diag_idx) = 0;
    gm = GradientMaps('kernel', kernel_name, 'approach', approach_name, 'n_components', n_gradients, ...
        'random_state', random_state);
    gm = gm.fit(A, 'sparsity', sparsity_percentile, 'alpha', alpha, 'diffusion_time', diffusion_time);
    if isempty(lambda_all)
        lambda_all = nan(numel(gm.lambda{1}), n_subjects);
    end
    gradients_all(:, :, s) = gm.gradients{1};
    lambda_all(:, s) = gm.lambda{1}(:);
    fit_success(s) = true;
end
assert(all(fit_success), 'Some individual gradients failed.');

subject_id = compose("sub-%03d", (1:n_subjects)');   % row number in the FC array
analysis_info = struct('kernel', 'cosine similarity', 'approach', 'diffusion maps', 'sparsity', sparsity_percentile, ...
    'alpha', alpha, 'diffusion_time', diffusion_time, 'n_components', n_gradients, 'random_state', random_state, ...
    'input', 'Pearson r, diagonal 0', 'matlab_version', version);
save(fullfile(output_dir, 'BrainSpace_formal_raw_gradients.mat'), 'gradients_all', 'lambda_all', 'fit_success', ...
    'subject_id', 'analysis_info', '-v7.3');
fprintf('Step 1 done: %d subjects\n', n_subjects);
end
