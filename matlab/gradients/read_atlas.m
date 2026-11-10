function atlas = read_atlas(atlas_file)
% READ_ATLAS  Read a parcel table and the Yeo-7 network of each parcel.
%
% Two table formats are used:
%   Glasser-358  (atlas-Glasser358_dseg.tsv): Yeo-7 names in column community_yeo
%                ('visual', 'somatosensory', 'default mode', ...)
%   Schaefer     (atlas-Schaefer400/600_cortex.tsv): Yeo-7 names in column network_label ('Vis', 'SomMot', 'Default', ...)
% The rows must be in the same order as the FC rows (index = 1:n).
%
% Output fields: n_parcels, table, yeo7_labels (numeric code per parcel), yeo7_names, and the three network names used as
% anchors to orient G1 and G2 (anchor_default, anchor_visual, anchor_somato; lower case).

T = readtable(atlas_file, 'FileType', 'text', 'Delimiter', '\t', 'TextType', 'string');
n = height(T);
if ~isequal(double(T.index(:)), (1:n)')
    error('The atlas table must list the parcels in FC order (index = 1:%d).', n);
end
if ismember('community_yeo', T.Properties.VariableNames)
    yeo = lower(strtrim(string(T.community_yeo)));
    anchors = {'default mode', 'visual', 'somatosensory'};
else
    yeo = strtrim(string(T.network_label));
    anchors = {'default', 'vis', 'sommot'};
end
[names, ~, codes] = unique(yeo, 'stable');
if numel(names) ~= 7
    error('Expected 7 Yeo networks, found %d.', numel(names));
end
atlas = struct();
atlas.n_parcels = n;
atlas.file = atlas_file;
atlas.table = T;
atlas.yeo7_labels = double(codes(:));
atlas.yeo7_names = cellstr(names(:));
atlas.anchor_default = anchors{1};
atlas.anchor_visual = anchors{2};
atlas.anchor_somato = anchors{3};
end
