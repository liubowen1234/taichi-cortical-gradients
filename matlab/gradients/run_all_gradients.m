%% RUN_ALL_GRADIENTS  Connectivity gradients of every analysis that needs the MATLAB steps.
%
% Run after pipeline/01_cohort/s02_build_fc.py has written the FC files to <output_root>/01_cohort/fc/.
% Each call of run_gradient_pipeline performs Steps 1-4 (see that function):
%   1. main analysis       : Glasser-358, 36P denoising, non-athlete (HC) reference
%   2. no-GSR sensitivity  : Glasser-358, 32P denoising (no global signal regression), HC reference rebuilt from 32P FC
%   3. atlas sensitivity   : Schaefer-400 and Schaefer-600 cortical parcels, HC reference
%
% Usage (MATLAB, from any folder):  run('<code_release>/matlab/gradients/run_all_gradients.m')

here = fileparts(mfilename('fullpath'));
addpath(here, fileparts(here));
cfg = config_paths();
addpath(genpath(cfg.brainspace_matlab));
L = cfg.layout;
atlas_dir = fullfile(cfg.resources, 'atlas');

run_gradient_pipeline(fullfile(L.fc, 'FC_glasser_36P.mat'), L.grad_A, fullfile(atlas_dir, 'atlas-Glasser358_dseg.tsv'));
run_gradient_pipeline(fullfile(L.fc, 'FC_glasser_32P.mat'), L.grad_noGSR, fullfile(atlas_dir, 'atlas-Glasser358_dseg.tsv'));
run_gradient_pipeline(fullfile(L.fc, 'FC_schaefer400_36P.mat'), L.schaefer400, fullfile(atlas_dir, 'atlas-Schaefer400_cortex.tsv'));
run_gradient_pipeline(fullfile(L.fc, 'FC_schaefer600_36P.mat'), L.schaefer600, fullfile(atlas_dir, 'atlas-Schaefer600_cortex.tsv'));
disp('All gradient analyses finished.');
