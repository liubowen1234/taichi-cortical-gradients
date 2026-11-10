function run_gradient_pipeline(fc_file, analysis_root, atlas_file)
% RUN_GRADIENT_PIPELINE  Steps 1-4 of the connectivity-gradient analysis for one FC data set.
%
%   fc_file       FC file written by s02_build_fc.py (variables FC [parcels x parcels x subjects, Pearson r], group, subject_ids)
%   analysis_root output folder; every step writes a sub-folder (Step1_raw_gradients, ..., Step4_gradient_metrics)
%   atlas_file    parcel table (resources/atlas/*.tsv) in the same order as the FC rows; gives each parcel's Yeo-7 network
%
% Step 0  copy the FC to <analysis_root>/FC.mat (MATLAB v7.3, read by the Python scripts with h5py)
% Step 1  individual gradients (BrainSpace diffusion maps; 10 components)
% Step 2  non-athlete (HC) reference embedding and one Procrustes rotation of every subject's first 3 components to it
% Step 3  identify G1 and G2 with Yeo-7 network anchors and fix their signs; the remaining component is G3
% Step 4  subject-level and parcel-level gradient metrics (CSV tables for the statistics in Python)

if ~exist(analysis_root, 'dir'), mkdir(analysis_root); end
S = load(fc_file);
FC = double(S.FC);
group = strtrim(string(S.group(:)));
subject_ids = strtrim(string(S.subject_ids(:)));
assert(all(isfinite(FC(:))), 'FC contains NaN or Inf.');
fc73 = fullfile(analysis_root, 'FC.mat');
save(fc73, 'FC', 'group', 'subject_ids', '-v7.3');
clear S

atlas = read_atlas(atlas_file);
assert(size(FC, 1) == atlas.n_parcels, 'FC size does not match the atlas table.');
fprintf('\n=== %s: %d parcels, %d subjects (HC %d, TC %d, BB %d)\n', analysis_root, size(FC, 1), size(FC, 3), ...
    sum(group == "HC"), sum(group == "TC"), sum(group == "BB"));

step1_raw_gradients(FC, analysis_root);
step2_reference_alignment(FC, group, analysis_root);
step3_orient_gradients(analysis_root, atlas, group);
step4_gradient_metrics(analysis_root, atlas, group);
end
