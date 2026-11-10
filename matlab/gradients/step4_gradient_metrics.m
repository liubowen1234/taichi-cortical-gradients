function step4_gradient_metrics(analysis_root, atlas, group)
% STEP4_GRADIENT_METRICS  Subject-level and parcel-level metrics of the oriented gradients (no statistics here).
%
% Single-axis metrics (G1 and G2 separately; over all parcels of a subject):
%   mean, median, SD, variance, min, max, full range, 5th/25th/75th/95th percentiles, robust range (= span,
%   95th - 5th percentile), IQR, median absolute deviation, mean |x|, RMS, L2 norm, skewness, excess kurtosis,
%   fraction of positive parcels.
% G1-G2 plane:
%   centroid; global dispersion (mean Euclidean distance of the parcels from their centroid) and its RMS;
%   eccentricity summaries; 2 x 2 covariance of the parcel coordinates -> total variance, major- and minor-axis variance
%   (larger / smaller eigenvalue), anisotropy index (major - minor) / (major + minor), axis ratio, 1-SD ellipse area;
%   convex hull area; mean / median / 95th-percentile pairwise parcel distance.
% 3D (G1, G2, G3): global dispersion (Euclidean, so the arbitrary sign of G3 does not matter).
% Similarity to the reference: Pearson and Spearman r per axis; mean / median / 95th / max / RMS displacement of the
%   parcels from their reference position; Frobenius distance.
%
% Outputs (<analysis_root>/Step4_gradient_metrics/):
%   subject_gradient_metrics.csv     one row per subject (Subject = sub-<row number>)
%   parcel_gradient_metrics_long.csv one row per subject x parcel (G1, G2, G3_ReferenceAligned, Eccentricity2D,
%                                    ReferenceG1/G2, displacement from the reference)
%   BrainSpace_step4_gradient_metrics.mat

S = load(fullfile(analysis_root, 'Step3_canonical_G1G2', 'BrainSpace_step3_G1G2_ONLY_results.mat'));
aligned_canonical = S.aligned_canonical;
reference_canonical = S.reference_canonical;
aligned_canonical3d = S.aligned_canonical3d;
fit_success = S.fit_success;
hc_idx = S.hc_idx; tc_idx = S.tc_idx; bb_idx = S.bb_idx;
[n_parcels, ~, n_subjects] = size(aligned_canonical);
assert(n_parcels == atlas.n_parcels);
output_dir = fullfile(analysis_root, 'Step4_gradient_metrics');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

subject_id = compose("sub-%03d", (1:n_subjects)');
metrics_table = table(subject_id, group(:), fit_success, 'VariableNames', {'Subject', 'Group', 'FitSuccess'});

%% 1. single-axis metrics
metric_names = {'Mean', 'Median', 'SD', 'Variance', 'Min', 'Max', 'FullRange', 'Q05', 'Q25', 'Q75', 'Q95', 'RobustRange', ...
    'IQR', 'MADMedian', 'MeanAbsoluteValue', 'RMS', 'L2Norm', 'Skewness', 'ExcessKurtosis', 'PositiveFraction'};
for g = 1:2
    for m = 1:numel(metric_names)
        metrics_table.(sprintf('G%d_%s', g, metric_names{m})) = nan(n_subjects, 1);
    end
end
for s = 1:n_subjects
    for g = 1:2
        x = aligned_canonical(:, g, s);
        x = x(isfinite(x));
        q05 = local_percentile(x, 5); q25 = local_percentile(x, 25);
        q75 = local_percentile(x, 75); q95 = local_percentile(x, 95);
        med = median(x);
        v = @(name) sprintf('G%d_%s', g, name);
        metrics_table.(v('Mean'))(s) = mean(x);
        metrics_table.(v('Median'))(s) = med;
        metrics_table.(v('SD'))(s) = std(x, 0);
        metrics_table.(v('Variance'))(s) = var(x, 0);
        metrics_table.(v('Min'))(s) = min(x);
        metrics_table.(v('Max'))(s) = max(x);
        metrics_table.(v('FullRange'))(s) = max(x) - min(x);
        metrics_table.(v('Q05'))(s) = q05;
        metrics_table.(v('Q25'))(s) = q25;
        metrics_table.(v('Q75'))(s) = q75;
        metrics_table.(v('Q95'))(s) = q95;
        metrics_table.(v('RobustRange'))(s) = q95 - q05;          % the "span" of the gradient
        metrics_table.(v('IQR'))(s) = q75 - q25;
        metrics_table.(v('MADMedian'))(s) = median(abs(x - med));
        metrics_table.(v('MeanAbsoluteValue'))(s) = mean(abs(x));
        metrics_table.(v('RMS'))(s) = sqrt(mean(x .^ 2));
        metrics_table.(v('L2Norm'))(s) = sqrt(sum(x .^ 2));
        metrics_table.(v('Skewness'))(s) = local_skewness(x);
        metrics_table.(v('ExcessKurtosis'))(s) = local_excess_kurtosis(x);
        metrics_table.(v('PositiveFraction'))(s) = mean(x > 0);
    end
end

%% 2. G1-G2 plane: centroid and dispersion
new_cols = {'Centroid_G1', 'Centroid_G2', 'GlobalDispersionMean', 'GlobalDispersionRMS', 'EccentricityMedian', ...
    'EccentricitySD', 'EccentricityQ95', 'EccentricityMax', 'GlobalDispersion3DMean', 'GlobalDispersion3DRMS', ...
    'TotalVariance2D', 'G1G2Covariance', 'G1G2Correlation', 'MajorVariance', 'MinorVariance', 'AnisotropyIndex', ...
    'AxisRatio', 'EllipseArea1SD', 'ConvexHullArea', 'MeanPairwiseDistance', 'MedianPairwiseDistance', 'Q95PairwiseDistance'};
for k = 1:numel(new_cols)
    metrics_table.(new_cols{k}) = nan(n_subjects, 1);
end
for s = 1:n_subjects
    X = squeeze(aligned_canonical(:, :, s));
    centroid = mean(X, 1);
    ecc = sqrt(sum((X - centroid) .^ 2, 2));          % distance of each parcel from the centroid
    metrics_table.Centroid_G1(s) = centroid(1);
    metrics_table.Centroid_G2(s) = centroid(2);
    metrics_table.GlobalDispersionMean(s) = mean(ecc);
    metrics_table.GlobalDispersionRMS(s) = sqrt(mean(ecc .^ 2));
    metrics_table.EccentricityMedian(s) = median(ecc);
    metrics_table.EccentricitySD(s) = std(ecc, 0);
    metrics_table.EccentricityQ95(s) = local_percentile(ecc, 95);
    metrics_table.EccentricityMax(s) = max(ecc);

    % 3D dispersion (G1, G2, G3)
    X3 = squeeze(aligned_canonical3d(:, :, s));
    r3 = sqrt(sum((X3 - mean(X3, 1)) .^ 2, 2));
    metrics_table.GlobalDispersion3DMean(s) = mean(r3);
    metrics_table.GlobalDispersion3DRMS(s) = sqrt(mean(r3 .^ 2));

    % covariance of the parcel cloud in the G1-G2 plane
    C = cov(X);
    ev = sort(eig(C), 'descend');
    major_var = ev(1); minor_var = ev(2);
    metrics_table.TotalVariance2D(s) = trace(C);
    metrics_table.G1G2Covariance(s) = C(1, 2);
    Rtmp = corrcoef(X(:, 1), X(:, 2));
    metrics_table.G1G2Correlation(s) = Rtmp(1, 2);
    metrics_table.MajorVariance(s) = major_var;
    metrics_table.MinorVariance(s) = minor_var;          % "short-axis variance"; does not change under in-plane rotation
    if major_var + minor_var > 0
        metrics_table.AnisotropyIndex(s) = (major_var - minor_var) / (major_var + minor_var);
    end
    if minor_var > 0
        metrics_table.AxisRatio(s) = sqrt(major_var / minor_var);
    end
    if major_var >= 0 && minor_var >= 0
        metrics_table.EllipseArea1SD(s) = pi * sqrt(major_var * minor_var);
    end

    % area covered by the parcels and their pairwise distances
    hull_idx = convhull(X(:, 1), X(:, 2));
    metrics_table.ConvexHullArea(s) = polyarea(X(hull_idx, 1), X(hull_idx, 2));
    d = local_pairwise_distances(X);
    metrics_table.MeanPairwiseDistance(s) = mean(d);
    metrics_table.MedianPairwiseDistance(s) = median(d);
    metrics_table.Q95PairwiseDistance(s) = local_percentile(d, 95);
end

%% 3. similarity to and displacement from the reference (QC / exploratory)
ref_cols = {'TemplateSimilarityPearson_G1', 'TemplateSimilarityPearson_G2', 'TemplateSimilaritySpearman_G1', ...
    'TemplateSimilaritySpearman_G2', 'ReferenceMAE_G1', 'ReferenceMAE_G2', 'ReferenceRMSE_G1', 'ReferenceRMSE_G2', ...
    'MeanParcelDisplacementToReference', 'MedianParcelDisplacementToReference', 'Q95ParcelDisplacementToReference', ...
    'MaxParcelDisplacementToReference', 'RMSParcelDisplacementToReference', 'FrobeniusDistanceToReference'};
for k = 1:numel(ref_cols)
    metrics_table.(ref_cols{k}) = nan(n_subjects, 1);
end
for s = 1:n_subjects
    X = squeeze(aligned_canonical(:, :, s));
    D = X - reference_canonical;
    for g = 1:2
        Rtmp = corrcoef(X(:, g), reference_canonical(:, g));
        metrics_table.(sprintf('TemplateSimilarityPearson_G%d', g))(s) = Rtmp(1, 2);
        metrics_table.(sprintf('TemplateSimilaritySpearman_G%d', g))(s) = local_spearman(X(:, g), reference_canonical(:, g));
        metrics_table.(sprintf('ReferenceMAE_G%d', g))(s) = mean(abs(D(:, g)));
        metrics_table.(sprintf('ReferenceRMSE_G%d', g))(s) = sqrt(mean(D(:, g) .^ 2));
    end
    disp2 = sqrt(sum(D .^ 2, 2));
    metrics_table.MeanParcelDisplacementToReference(s) = mean(disp2);
    metrics_table.MedianParcelDisplacementToReference(s) = median(disp2);
    metrics_table.Q95ParcelDisplacementToReference(s) = local_percentile(disp2, 95);
    metrics_table.MaxParcelDisplacementToReference(s) = max(disp2);
    metrics_table.RMSParcelDisplacementToReference(s) = sqrt(mean(disp2 .^ 2));
    metrics_table.FrobeniusDistanceToReference(s) = sqrt(sum(D(:) .^ 2));
end

%% 4. parcel-level long table
n_rows = n_subjects * n_parcels;
Subject = strings(n_rows, 1); Group = strings(n_rows, 1); Parcel = nan(n_rows, 1);
G1 = nan(n_rows, 1); G2 = nan(n_rows, 1); G3_ReferenceAligned = nan(n_rows, 1); Eccentricity2D = nan(n_rows, 1);
ReferenceG1 = nan(n_rows, 1); ReferenceG2 = nan(n_rows, 1);
DeltaG1_toReference = nan(n_rows, 1); DeltaG2_toReference = nan(n_rows, 1); Displacement2D_toReference = nan(n_rows, 1);
for s = 1:n_subjects
    X = squeeze(aligned_canonical(:, :, s));
    ecc = sqrt(sum((X - mean(X, 1)) .^ 2, 2));
    D = X - reference_canonical;
    rows = (s - 1) * n_parcels + (1:n_parcels);
    Subject(rows) = subject_id(s); Group(rows) = group(s); Parcel(rows) = 1:n_parcels;
    G1(rows) = X(:, 1); G2(rows) = X(:, 2); G3_ReferenceAligned(rows) = aligned_canonical3d(:, 3, s);
    Eccentricity2D(rows) = ecc;
    ReferenceG1(rows) = reference_canonical(:, 1); ReferenceG2(rows) = reference_canonical(:, 2);
    DeltaG1_toReference(rows) = D(:, 1); DeltaG2_toReference(rows) = D(:, 2);
    Displacement2D_toReference(rows) = sqrt(sum(D .^ 2, 2));
end
parcel_table = table(Subject, Group, Parcel, G1, G2, G3_ReferenceAligned, Eccentricity2D, ReferenceG1, ReferenceG2, ...
    DeltaG1_toReference, DeltaG2_toReference, Displacement2D_toReference);

writetable(parcel_table, fullfile(output_dir, 'parcel_gradient_metrics_long.csv'));
writetable(metrics_table, fullfile(output_dir, 'subject_gradient_metrics.csv'));
save(fullfile(output_dir, 'BrainSpace_step4_gradient_metrics.mat'), 'metrics_table', 'parcel_table', 'aligned_canonical', ...
    'aligned_canonical3d', 'reference_canonical', 'hc_idx', 'tc_idx', 'bb_idx', 'fit_success', '-v7.3');
fprintf('Step 4 done: %s\n', output_dir);
end


%% ===== helper functions =====
function q = local_percentile(x, p)
% Percentile with linear interpolation between order statistics (no Statistics Toolbox needed).
x = sort(x(:)); x = x(isfinite(x)); n = numel(x);
if n == 0, q = NaN; return; end
if n == 1, q = x; return; end
pos = 1 + (n - 1) * (p / 100);
lo = floor(pos); hi = ceil(pos);
if lo == hi
    q = x(lo);
else
    q = x(lo) + (pos - lo) * (x(hi) - x(lo));
end
end

function s = local_skewness(x)
% Standardised third central moment.
x = x(isfinite(x(:))); mu = mean(x); sigma = sqrt(mean((x - mu) .^ 2));
if sigma == 0, s = 0; else, s = mean(((x - mu) / sigma) .^ 3); end
end

function k = local_excess_kurtosis(x)
% Standardised fourth central moment minus 3 (0 for a normal distribution).
x = x(isfinite(x(:))); mu = mean(x); sigma = sqrt(mean((x - mu) .^ 2));
if sigma == 0, k = 0; else, k = mean(((x - mu) / sigma) .^ 4) - 3; end
end

function d = local_pairwise_distances(X)
% Euclidean distance of every pair of parcels (rows of X).
n = size(X, 1); d = nan(n * (n - 1) / 2, 1); k = 0;
for i = 1:n - 1
    di = sqrt(sum((X(i + 1:end, :) - X(i, :)) .^ 2, 2));
    d(k + (1:numel(di))) = di; k = k + numel(di);
end
end

function r = local_spearman(x, y)
% Spearman correlation = Pearson correlation of the ranks (ties get their average rank).
R = corrcoef(local_tiedrank(x), local_tiedrank(y)); r = R(1, 2);
end

function ranks = local_tiedrank(x)
x = x(:); [sx, order] = sort(x); rs = nan(size(sx)); i = 1;
while i <= numel(sx)
    j = i;
    while j < numel(sx) && sx(j + 1) == sx(i), j = j + 1; end
    rs(i:j) = mean(i:j); i = j + 1;
end
ranks = nan(size(x)); ranks(order) = rs;
end
