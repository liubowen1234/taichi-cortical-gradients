function step2_reference_alignment(FC, group, analysis_root)
% STEP2_REFERENCE_ALIGNMENT  Non-athlete reference embedding and Procrustes alignment of every subject.
%
% Reference: the HC (non-athlete) FC matrices are Fisher-z transformed, averaged and transformed back to r; the reference
% embedding is computed from this mean matrix with exactly the Step 1 parameters (10 components).
% Only non-athletes form the reference, so that each athlete group is measured against a common coordinate system that
% the athletes' own connectivity does not shape.
% Alignment: the first 3 raw components of each subject are rotated onto the first 3 reference components with one
% orthogonal Procrustes rotation (BrainSpace procrustes_alignment, nIterations = 1: a fixed reference, no scaling or
% translation).
%
% Output: <analysis_root>/Step2_reference_alignment/BrainSpace_HC_reference_aligned_gradients.mat

kernel_name = 'cs';
approach_name = 'dm';
raw_n_components = 10;
alignment_dims = 3;
random_state = 0;
sparsity_percentile = 90;
alpha = 0.5;
diffusion_time = 0;
n_iterations = 1;      % one rotation to the fixed reference (more iterations would turn it into generalised Procrustes)

hc_idx = find(group == "HC")';
tc_idx = find(group == "TC")';
bb_idx = find(group == "BB")';
[n_parcels, ~, n_subjects] = size(FC);
output_dir = fullfile(analysis_root, 'Step2_reference_alignment');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

S1 = load(fullfile(analysis_root, 'Step1_raw_gradients', 'BrainSpace_formal_raw_gradients.mat'));
raw10_all = S1.gradients_all;
lambda10_all = S1.lambda_all;

% ---- HC mean FC: r -> Fisher z -> mean -> r (diagonal 0; |r| clipped just below 1 so that atanh stays finite)
clip_eps = 1e-7;
diag_idx = 1:(n_parcels + 1):(n_parcels * n_parcels);
hc_z = nan(n_parcels, n_parcels, numel(hc_idx));
for k = 1:numel(hc_idx)
    A = FC(:, :, hc_idx(k));
    A(diag_idx) = 0;
    Z = atanh(min(max(A, -1 + clip_eps), 1 - clip_eps));
    Z(diag_idx) = 0;
    hc_z(:, :, k) = Z;
end
hc_mean_z = mean(hc_z, 3, 'omitnan');
hc_reference_fc = tanh(hc_mean_z);
hc_reference_fc(diag_idx) = 0;
hc_reference_fc = (hc_reference_fc + hc_reference_fc') / 2;     % exact symmetry

% ---- reference embedding (same parameters as Step 1)
gref = GradientMaps('kernel', kernel_name, 'approach', approach_name, 'n_components', raw_n_components, ...
    'random_state', random_state);
gref = gref.fit(hc_reference_fc, 'sparsity', sparsity_percentile, 'alpha', alpha, 'diffusion_time', diffusion_time);
reference_gradients_raw10 = gref.gradients{1};
reference_lambda = gref.lambda{1}(:);
reference_gradients = reference_gradients_raw10(:, 1:alignment_dims);

% ---- align every subject to the reference
raw_gradients = raw10_all(:, 1:alignment_dims, :);
aligned_gradients = nan(n_parcels, alignment_dims, n_subjects);
alignment_xfms = nan(alignment_dims, alignment_dims, n_subjects);
individual_lambda = lambda10_all(1:alignment_dims, :);
fit_success = false(n_subjects, 1);
for s = 1:n_subjects
    [aligned_cell, xfm_cell] = procrustes_alignment({raw_gradients(:, :, s)}, 'reference', reference_gradients, ...
        'nIterations', n_iterations);
    aligned_gradients(:, :, s) = aligned_cell{1};
    alignment_xfms(:, :, s) = xfm_cell{1};
    fit_success(s) = true;
end

% QC: spatial correlation of each aligned component with the reference component
template_similarity = nan(n_subjects, alignment_dims);
for s = 1:n_subjects
    for g = 1:alignment_dims
        template_similarity(s, g) = corr(aligned_gradients(:, g, s), reference_gradients(:, g));
    end
end

analysis_info = struct('reference', 'HC Pearson r -> Fisher z -> mean -> inverse z -> diagonal 0; Step 1 parameters', ...
    'alignment', 'one orthogonal Procrustes rotation of components 1-3 to the HC reference; no centring or scaling', ...
    'alignment_dimensions', alignment_dims, 'n_hc', numel(hc_idx));
save(fullfile(output_dir, 'BrainSpace_HC_reference_aligned_gradients.mat'), 'hc_reference_fc', 'reference_gradients_raw10', ...
    'reference_gradients', 'reference_lambda', 'raw_gradients', 'aligned_gradients', 'alignment_xfms', 'individual_lambda', ...
    'template_similarity', 'fit_success', 'hc_idx', 'tc_idx', 'bb_idx', 'analysis_info', '-v7.3');
fprintf('Step 2 done: reference from %d HC, %d subjects aligned\n', numel(hc_idx), n_subjects);
end
