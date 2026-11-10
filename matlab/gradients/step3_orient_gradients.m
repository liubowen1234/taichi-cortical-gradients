function step3_orient_gradients(analysis_root, atlas, group)
% STEP3_ORIENT_GRADIENTS  Name and orient the aligned components.
%
% G1 (sensory-association axis) and G2 (visual-somatosensory axis) are identified in the reference embedding with Yeo-7
% network anchors (orient_gradients.m); the same permutation and signs are applied to every subject. No second rotation.
% The remaining aligned component is kept as G3 (its sign is fixed later, in Python, by the CAB-NP modulation networks).
%
% Output: <analysis_root>/Step3_canonical_G1G2/BrainSpace_step3_G1G2_ONLY_results.mat
%   reference_canonical  parcels x 2 (G1, G2)          aligned_canonical   parcels x 2 x subjects
%   reference_canonical3d parcels x 3 (G1, G2, G3)     aligned_canonical3d parcels x 3 x subjects

S = load(fullfile(analysis_root, 'Step2_reference_alignment', 'BrainSpace_HC_reference_aligned_gradients.mat'));
reference_gradients = S.reference_gradients;
aligned_gradients = S.aligned_gradients;
fit_success = logical(S.fit_success);
hc_idx = S.hc_idx; tc_idx = S.tc_idx; bb_idx = S.bb_idx;
[~, n_components, n_subjects] = size(aligned_gradients);
output_dir = fullfile(analysis_root, 'Step3_canonical_G1G2');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

[reference_canonical, aligned_canonical, best_components, sign_flip, R] = ...
    orient_gradients(reference_gradients, aligned_gradients, atlas);
reference_vs_canonical = [R(best_components(1), 1) * sign_flip(1); R(best_components(2), 2) * sign_flip(2)];
fprintf('G1 <- reference component %d (sign %+d); G2 <- component %d (sign %+d)\n', ...
    best_components(1), sign_flip(1), best_components(2), sign_flip(2));

% G3 = the remaining aligned component (used for G3 metrics and 3D distances)
g3_component = setdiff(1:n_components, best_components, 'stable');
assert(numel(g3_component) == 1, 'Exactly one component should remain for G3.');
reference_canonical3d = [reference_canonical, reference_gradients(:, g3_component)];
aligned_canonical3d = nan(atlas.n_parcels, 3, n_subjects);
aligned_canonical3d(:, 1:2, :) = aligned_canonical;
aligned_canonical3d(:, 3, :) = aligned_gradients(:, g3_component, :);

% QC: similarity of each subject's G1 / G2 with the reference
template_similarity_canonical = nan(n_subjects, 2);
for s = 1:n_subjects
    template_similarity_canonical(s, 1) = corr(aligned_canonical(:, 1, s), reference_canonical(:, 1));
    template_similarity_canonical(s, 2) = corr(aligned_canonical(:, 2, s), reference_canonical(:, 2));
end
qc_table = table(compose("sub-%03d", (1:n_subjects)'), group(:), fit_success, template_similarity_canonical(:, 1), ...
    template_similarity_canonical(:, 2), 'VariableNames', {'Subject', 'Group', 'FitSuccess', 'TemplateSimilarity_G1', ...
    'TemplateSimilarity_G2'});
writetable(qc_table, fullfile(output_dir, 'canonical_template_similarity.csv'));

canonical_info = struct('orientation', 'Yeo-7 anchors: G1 default mode vs sensory; G2 visual vs somatosensory', ...
    'component_mapping', best_components, 'sign_flip', sign_flip, 'g3_component', g3_component);
save(fullfile(output_dir, 'BrainSpace_step3_G1G2_ONLY_results.mat'), 'reference_canonical', 'aligned_canonical', ...
    'reference_canonical3d', 'aligned_canonical3d', 'g3_component', 'R', 'best_components', 'sign_flip', ...
    'reference_vs_canonical', 'template_similarity_canonical', 'qc_table', 'hc_idx', 'tc_idx', 'bb_idx', 'fit_success', ...
    'canonical_info', '-v7.3');
fprintf('Step 3 done\n');
end
