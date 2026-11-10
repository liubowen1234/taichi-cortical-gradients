function [reference_oriented, aligned_oriented, mapping, sign_flip, anchor_contrasts] = ...
    orient_gradients(reference_gradients, aligned_gradients, atlas)
% ORIENT_GRADIENTS  Decide which aligned components are G1 and G2 and fix their signs, using Yeo-7 network anchors.
%
%   G1 = the component with the largest contrast  default mode  vs  mean(visual, somatosensory); default mode positive.
%   G2 = the component with the largest contrast  visual  vs  somatosensory; visual positive.
% The best (G1, G2) pair maximises |G1 contrast| + |G2 contrast| over all ordered pairs of different components.
% Only permutation and sign changes are applied (no further rotation).
%
%   reference_gradients : parcels x k  (reference embedding)
%   aligned_gradients   : parcels x k x subjects (subjects already aligned to the reference)

n_components = size(reference_gradients, 2);
if n_components < 2
    error('At least two aligned components are needed.');
end
names = lower(string(atlas.yeo7_names));
is_default = atlas.yeo7_labels == find(names == atlas.anchor_default, 1);
is_visual = atlas.yeo7_labels == find(names == atlas.anchor_visual, 1);
is_somato = atlas.yeo7_labels == find(names == atlas.anchor_somato, 1);
if ~any(is_default) || ~any(is_visual) || ~any(is_somato)
    error('The atlas must contain default mode, visual and somatosensory networks.');
end

% contrast of every reference component: column 1 = G1 anchor contrast, column 2 = G2 anchor contrast
anchor_contrasts = nan(n_components, 2);
for cc = 1:n_components
    x = reference_gradients(:, cc);
    anchor_contrasts(cc, 1) = mean(x(is_default), 'omitnan') - mean(x(is_visual | is_somato), 'omitnan');
    anchor_contrasts(cc, 2) = mean(x(is_visual), 'omitnan') - mean(x(is_somato), 'omitnan');
end

best_score = -Inf;
mapping = [NaN NaN];
for c1 = 1:n_components
    for c2 = 1:n_components
        if c1 == c2, continue; end
        score = abs(anchor_contrasts(c1, 1)) + abs(anchor_contrasts(c2, 2));
        if score > best_score
            best_score = score;
            mapping = [c1 c2];
        end
    end
end
sign_flip = [sign(anchor_contrasts(mapping(1), 1)), sign(anchor_contrasts(mapping(2), 2))];
if any(sign_flip == 0)
    error('A network-anchor contrast is zero; the gradient sign cannot be defined.');
end

reference_oriented = [reference_gradients(:, mapping(1)) * sign_flip(1), reference_gradients(:, mapping(2)) * sign_flip(2)];
n_subjects = size(aligned_gradients, 3);
aligned_oriented = nan(atlas.n_parcels, 2, n_subjects);
for s = 1:n_subjects
    aligned_oriented(:, 1, s) = aligned_gradients(:, mapping(1), s) * sign_flip(1);
    aligned_oriented(:, 2, s) = aligned_gradients(:, mapping(2), s) * sign_flip(2);
end
end
