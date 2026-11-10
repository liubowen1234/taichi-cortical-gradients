"""Spatial permutation ("spin") nulls for parcellated cortical maps.

Parcel centroids are computed on the fs_LR 32k sphere. Each permutation draws a uniform random rotation (QR of a Gaussian
matrix); the right hemisphere uses the mirror of the left rotation (Alexander-Bloch et al., 2018). Rotated centroids are
re-assigned to original parcels either
  * 'hungarian' (Vasa et al., 2018): optimal one-to-one assignment, a permutation without replacement (default), or
  * 'nearest'   (Vazquez-Rodriguez et al., 2019): nearest original centroid, with replacement.
Hemispheres are rotated and assigned separately, so values never move between hemispheres.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def sphere_centroids(sphere_v, vidx, n_parcels):
    """Mean sphere coordinate of each parcel, renormalised to the unit sphere. vidx: vertex -> parcel (0-based, -1 = none)."""
    c = np.full((n_parcels, 3), np.nan)
    for p in np.unique(vidx[vidx >= 0]):
        m = sphere_v[vidx == p].mean(0); c[p] = m / np.linalg.norm(m)
    return c


def random_rotation(rng):
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0: q[:, 0] *= -1
    return q


def spin_permutations(cL, cR, idxL, idxR, n_perm=10000, method='hungarian', seed=1234):
    """Return (n_perm, n_parcels) integer array: perm[k, i] = parcel whose value moves to parcel i.
    cL/cR: unit-sphere centroids of the left/right parcels; idxL/idxR: their positions in the full parcel vector."""
    rng = np.random.default_rng(seed)
    n = len(idxL) + len(idxR)
    out = np.empty((n_perm, n), dtype=np.int32)
    F = np.diag([-1.0, 1.0, 1.0])
    for k in range(n_perm):
        R = random_rotation(rng)
        for c, idx, Rh in [(cL, idxL, R), (cR, idxR, F @ R @ F)]:
            rot = c @ Rh.T
            d = 1 - rot @ c.T                     # cosine distance: rows = rotated parcels, cols = original parcels
            if method == 'hungarian':
                r_, col = linear_sum_assignment(d)
                # rotated parcel r_ lands on original location col: value of r_ moves to col
                out[k, idx[col]] = idx[r_]
            else:
                out[k, idx] = idx[np.argmin(d, 0)]  # each original parcel takes the value of the nearest rotated parcel
    return out


def spin_corr(x, y, perms, method='pearson'):
    """Correlation between maps x and y, and two-sided spin p (x is rotated)."""
    from scipy.stats import rankdata
    ok = np.isfinite(x) & np.isfinite(y)
    if method == 'spearman':
        x = rankdata(x); y = rankdata(y)
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    xs = x[perms]                                 # n_perm x n
    null = np.array([np.corrcoef(xs[k][ok], y[ok])[0, 1] for k in range(len(perms))])
    p = (1 + np.sum(np.abs(null) >= abs(r))) / (1 + len(null))
    return r, p, null


def spin_network_means(x, labels, perms, names):
    """Mean of map x in each network vs spin null; returns observed means, null (n_perm x n_net), two-sided p, z."""
    xs = x[perms]
    obs = np.array([np.nanmean(x[labels == n]) for n in names])
    null = np.column_stack([np.nanmean(xs[:, labels == n], 1) for n in names])
    mu, sd = null.mean(0), null.std(0, ddof=1)
    p = (1 + np.sum(np.abs(null - mu) >= np.abs(obs - mu), 0)) / (1 + len(null))
    return obs, null, p, (obs - mu) / sd
