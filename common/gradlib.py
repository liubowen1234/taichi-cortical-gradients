"""Python versions of the MATLAB gradient steps, used where Python is more convenient
(alignment to the external template, specification-curve analysis).

  procrustes_R / align_subject : BrainSpace procrustes_analysis (single rotation to a fixed reference)
  orient                       : choose which aligned components are G1 and G2 and fix their signs with Yeo-7 anchors
                                 (same rule as matlab/gradients/orient_gradients.m)
  subject_metrics              : the G1-G2 metrics of matlab/gradients/step4_gradient_metrics.m
  network_values               : network mean scores and network-pair distances in the G1-G2 plane
  hc3                          : OLS with HC3 standard errors for one coefficient
The Python and MATLAB versions give the same numbers to floating-point precision.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import ConvexHull

# ----------------------------------------------------------------- alignment
def procrustes_R(X, target):
    """BrainSpace procrustes_analysis, one rotation to a fixed reference.
    X: parcels x k (source), target: parcels x m. Returns k x m transform."""
    U, _, Vt = np.linalg.svd(target.T @ X, full_matrices=False)
    V = Vt.T
    r = min(U.shape[1], V.shape[1])
    return V[:, :r] @ U[:, :r].T


def zcols(A):
    return (A - A.mean(0)) / A.std(0, ddof=1)


def align_subject(raw, reference, zspace=True):
    """Rotation solved (optionally) on column-z-scored data, applied to raw-scale components."""
    if zspace:
        R = procrustes_R(zcols(raw), zcols(reference))
    else:
        R = procrustes_R(raw, reference)
    return raw @ R, R


# --------------------------------------------------------------- orientation
def anchor_contrasts(ref, yeo_names_per_parcel):
    y = np.asarray(yeo_names_per_parcel)
    is_def, is_vis, is_som = y == 'default mode', y == 'visual', y == 'somatosensory'
    C = np.zeros((ref.shape[1], 2))
    for c in range(ref.shape[1]):
        x = ref[:, c]
        C[c, 0] = x[is_def].mean() - x[is_vis | is_som].mean()
        C[c, 1] = x[is_vis].mean() - x[is_som].mean()
    return C


def orient(ref, aligned_stack, yeo_names_per_parcel):
    """Choose the component pair (G1, G2) that maximises |DMN - sensory| + |visual - somatosensory| and fix the signs.
    aligned_stack: subjects x parcels x k."""
    C = anchor_contrasts(ref, yeo_names_per_parcel)
    k = ref.shape[1]; best = -np.inf; mapping = None
    for c1 in range(k):
        for c2 in range(k):
            if c1 == c2: continue
            s = abs(C[c1, 0]) + abs(C[c2, 1])
            if s > best: best, mapping = s, (c1, c2)
    sign = np.array([np.sign(C[mapping[0], 0]), np.sign(C[mapping[1], 1])])
    assert np.all(sign != 0)
    ref_o = np.column_stack([ref[:, mapping[0]] * sign[0], ref[:, mapping[1]] * sign[1]])
    al_o = np.stack([aligned_stack[:, :, mapping[0]] * sign[0], aligned_stack[:, :, mapping[1]] * sign[1]], 2)
    return ref_o, al_o, mapping, sign, C


# ------------------------------------------------------------------ subject metrics
def _pct(x, p):
    return float(np.percentile(x, p))  # linear interpolation, as in the MATLAB step 4


def _skew(x):
    mu = x.mean(); s = np.sqrt(np.mean((x - mu) ** 2))
    return 0.0 if s == 0 else float(np.mean(((x - mu) / s) ** 3))


def _kurt(x):
    mu = x.mean(); s = np.sqrt(np.mean((x - mu) ** 2))
    return 0.0 if s == 0 else float(np.mean(((x - mu) / s) ** 4) - 3)


SINGLE = ['Mean', 'Median', 'SD', 'Variance', 'Min', 'Max', 'FullRange', 'Q05', 'Q25', 'Q75', 'Q95',
          'RobustRange', 'IQR', 'MADMedian', 'MeanAbsoluteValue', 'RMS', 'L2Norm', 'Skewness',
          'ExcessKurtosis', 'PositiveFraction']


def subject_metrics(G, reference):
    """G: subjects x parcels x 2 (oriented G1/G2). reference: parcels x 2 (oriented)."""
    rows = []
    iu = np.triu_indices(G.shape[1], 1)
    for s in range(G.shape[0]):
        X = G[s]; d = {}
        for g in range(2):
            x = X[:, g]; q05, q25, q75, q95 = (_pct(x, p) for p in (5, 25, 75, 95)); med = np.median(x)
            vals = dict(Mean=x.mean(), Median=med, SD=x.std(ddof=1), Variance=x.var(ddof=1), Min=x.min(), Max=x.max(),
                        FullRange=x.max() - x.min(), Q05=q05, Q25=q25, Q75=q75, Q95=q95, RobustRange=q95 - q05,
                        IQR=q75 - q25, MADMedian=np.median(np.abs(x - med)), MeanAbsoluteValue=np.abs(x).mean(),
                        RMS=np.sqrt(np.mean(x ** 2)), L2Norm=np.sqrt(np.sum(x ** 2)), Skewness=_skew(x),
                        ExcessKurtosis=_kurt(x), PositiveFraction=np.mean(x > 0))
            for k in SINGLE: d[f'G{g + 1}_{k}'] = float(vals[k])
        c = X.mean(0); ecc = np.sqrt(((X - c) ** 2).sum(1))
        d.update(Centroid_G1=c[0], Centroid_G2=c[1], GlobalDispersionMean=ecc.mean(),
                 GlobalDispersionRMS=np.sqrt(np.mean(ecc ** 2)), EccentricityMedian=np.median(ecc),
                 EccentricitySD=ecc.std(ddof=1), EccentricityQ95=_pct(ecc, 95), EccentricityMax=ecc.max())
        C = np.cov(X.T); ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        d.update(TotalVariance2D=np.trace(C), G1G2Covariance=C[0, 1], G1G2Correlation=np.corrcoef(X.T)[0, 1],
                 MajorVariance=ev[0], MinorVariance=ev[1], AnisotropyIndex=(ev[0] - ev[1]) / (ev[0] + ev[1]),
                 AxisRatio=np.sqrt(ev[0] / ev[1]), EllipseArea1SD=np.pi * np.sqrt(ev[0] * ev[1]))
        d['ConvexHullArea'] = ConvexHull(X).volume
        pdist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))[iu]
        d.update(MeanPairwiseDistance=pdist.mean(), MedianPairwiseDistance=np.median(pdist),
                 Q95PairwiseDistance=_pct(pdist, 95))
        for g in range(2):
            a, b = X[:, g], reference[:, g]
            d[f'TemplateSimilarityPearson_G{g + 1}'] = np.corrcoef(a, b)[0, 1]
            d[f'TemplateSimilaritySpearman_G{g + 1}'] = stats.spearmanr(a, b).statistic
            d[f'TemplateSimilarityCosine_G{g + 1}'] = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        rows.append(d)
    return pd.DataFrame(rows)


def network_values(G, labels, names, prefix):
    """Network mean G1 / G2, within-network dispersion and network-pair distances (G1-G2 plane)."""
    out = {}
    ids = list(names)
    C1 = np.stack([G[:, labels == n, 0].mean(1) for n in ids], 1)
    C2 = np.stack([G[:, labels == n, 1].mean(1) for n in ids], 1)
    net, pair = {}, {}
    for i, n in enumerate(ids):
        P = G[:, labels == n, :]; c = P.mean(1, keepdims=True)
        net[(n, '', 'MeanG1')] = C1[:, i]
        net[(n, '', 'MeanG2')] = C2[:, i]
        net[(n, '', 'WithinDispersion2D')] = np.sqrt(((P - c) ** 2).sum(2)).mean(1)
    for a in range(len(ids) - 1):
        for b in range(a + 1, len(ids)):
            d1 = C1[:, b] - C1[:, a]; d2 = C2[:, b] - C2[:, a]
            pair[(ids[a], ids[b], 'DeltaG1_BminusA')] = d1
            pair[(ids[a], ids[b], 'DeltaG2_BminusA')] = d2
            pair[(ids[a], ids[b], 'CentroidDistance2D')] = np.sqrt(d1 ** 2 + d2 ** 2)
    return net, pair


# ------------------------------------------------------------------ regression
def hc3(y, X, term=1):
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1); y, X = y[ok], X[ok]
    n, p = X.shape; b = np.linalg.pinv(X) @ y; e = y - X @ b; ix = np.linalg.pinv(X.T @ X)
    h = np.minimum(np.sum((X @ ix) * X, 1), .999999)
    V = ix @ (X.T @ (X * ((e / (1 - h)) ** 2)[:, None])) @ ix
    se = np.sqrt(max(V[term, term], 0)); t = b[term] / se; df = n - p
    pv = 2 * stats.t.sf(abs(t), df); crit = stats.t.ppf(.975, df)
    return dict(N=n, Beta=b[term], SE_HC3=se, T_HC3=t, P_HC3=pv, DF=df,
                CI95_Lower=b[term] - crit * se, CI95_Upper=b[term] + crit * se)


