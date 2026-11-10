function cfg = config_paths()
% CONFIG_PATHS  Read config/config.json and return absolute paths for the MATLAB steps.
%
% Fields of cfg:
%   release_root      folder that holds this code
%   output_root       results folder (same as the Python scripts use)
%   brainspace_matlab BrainSpace MATLAB toolbox (folder that contains GradientMaps.m somewhere below it)
%   nbs_dir           Network-Based Statistic toolbox (NBS 1.2)
%   resources         resources/ folder of this code (atlas tables)
%   layout            output folders shared with the Python scripts (see common/config.py, class LAYOUT)

here = fileparts(mfilename('fullpath'));
cfg.release_root = fileparts(here);
config_file = fullfile(cfg.release_root, 'config', 'config.json');
if ~exist(config_file, 'file')
    error('config/config.json not found. Copy config/config_template.json to config/config.json and edit the paths.');
end
raw = jsondecode(fileread(config_file));
base = fileparts(config_file);

cfg.output_root = resolve(raw.paths.output_root, base);
cfg.brainspace_matlab = resolve(raw.paths.brainspace_matlab, base);
cfg.nbs_dir = resolve(raw.paths.nbs_dir, base);
cfg.resources = fullfile(cfg.release_root, 'resources');

% Output folders (must match common/config.py)
L.fc = fullfile(cfg.output_root, '01_cohort', 'fc');
L.grad_A = fullfile(cfg.output_root, '02_gradients', 'glasser358_HCreference');
L.grad_noGSR = fullfile(cfg.output_root, '06_robustness', 'no_GSR_32P', 'glasser358_HCreference');
L.schaefer400 = fullfile(cfg.output_root, '06_robustness', 'schaefer400');
L.schaefer600 = fullfile(cfg.output_root, '06_robustness', 'schaefer600');
L.nbs = fullfile(cfg.output_root, '03_network_connectivity', 'nbs');
cfg.layout = L;
end


function p = resolve(p, base)
% Relative paths in config.json are relative to the config folder.
p = char(p);
if startsWith(p, '~')
    p = fullfile(char(java.lang.System.getProperty('user.home')), p(2:end));
end
if ~(startsWith(p, '/') || ~isempty(regexp(p, '^[A-Za-z]:', 'once')))
    p = fullfile(base, p);
end
p = char(java.io.File(p).getCanonicalPath());
end
