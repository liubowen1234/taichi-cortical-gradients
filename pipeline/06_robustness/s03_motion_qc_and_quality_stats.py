"""Step 6.3 - residual head-motion effects on connectivity (QC-FC) and data-quality covariates (Supplementary Fig. 8).

1. QC-FC: for each of the 63,903 edges, the partial correlation of Fisher-z FC with mean FD across all participants
   (adjusted for group, Age, Sex) and within each group (Age, Sex). Reported: median |r|, % edges with P < 0.05 and with
   FDR q < 0.05, and the Spearman correlation of QC-FC with the Euclidean distance between parcel centroids (distance
   dependence; centroids on the conte69 midthickness surface).
2. Motion map: per-parcel HC3 t of mean FD in  G2 ~ MeanFD + TC + BB + Age + Sex, correlated with the three G2 group-effect
   maps (spin test).
3. Group comparison of the signal-quality measures of s02 (three-group engine, one family).
4. Sensitivity: the primary metrics with VIS2 residual amplitude as an extra covariate; parcel-wise G2 group effects with
   each parcel's own residual amplitude as covariate.

Outputs (06_robustness/data_quality/): QCFC_summary_3group.csv, motion_map_vs_group_maps_3group.csv,
signal_quality_group_stats_3group.csv, sensitivity_VIS2_amplitude_covariate_3group.csv,
parcel_G2_with_own_amplitude_covariate_3group.csv
Usage: python pipeline/06_robustness/s03_motion_qc_and_quality_stats.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy import stats
from common.config import CFG, LAYOUT
from common import stats_engine as ts, figlib as fl
from common.data import covariates, cabnp_mapping, load_fc, row_ids, parcel_matrix, spin_permutations, KEEP358

OUT = LAYOUT.QUALITY
cov = covariates()
grp = cov.Group.values
NS = len(cov)
fd = cov.MeanFD.values
age = cov.Age.values.astype(float)
sex = cov.Sex.values - 1.0
net = cabnp_mapping().CABNP_Network.values
v2 = net == 'Visual2'
iu = np.triu_indices(358, 1)

# ---- 1. QC-FC
R = load_fc(LAYOUT.GRAD_A / 'FC.mat')
E = np.arctanh(np.clip(np.stack([R[s][iu] for s in range(NS)]), -1 + 1e-7, 1 - 1e-7))
del R
surf = CFG.path('brainspace_matlab') / 'datasets/surfaces'
lab = CFG.path('cabnp_dir') / 'SeparateHemispheres'
vL, _ = fl.load_surface(surf / 'conte69_32k_left_hemisphere.gii')
vR, _ = fl.load_surface(surf / 'conte69_32k_right_hemisphere.gii')
iL, iR = fl.glasser358_vertex_index(lab / 'cortex_ColeAnticevic_NetPartition_GlasserParcels_L.label.gii',
                                    lab / 'cortex_ColeAnticevic_NetPartition_GlasserParcels_R.label.gii')
cen = np.array([vL[iL == p].mean(0) if (iL == p).any() else vR[iR == p].mean(0) for p in range(358)])
dist = np.linalg.norm(cen[iu[0]] - cen[iu[1]], axis=1)          # mm
v2e = v2[iu[0]] | v2[iu[1]]                                      # edges with at least one VIS2 parcel


def partial_edges(Y, x, C):
    """Partial correlation of every column of Y with x, controlling for C."""
    Q = C @ np.linalg.pinv(C)
    ry = Y - Q @ Y; rx = x - Q @ x
    r = (rx @ ry) / np.sqrt((rx @ rx) * (ry ** 2).sum(0))
    df = len(x) - C.shape[1] - 1
    t = r * np.sqrt(df / (1 - r ** 2))
    return r, 2 * stats.t.sf(np.abs(t), df)


qc = []
for label, m, C in [('All (group dummies, Age, Sex)', np.ones(NS, bool),
                     np.column_stack([np.ones(NS), grp == 'TC', grp == 'BB', age, sex]).astype(float)),
                    ('HC only', grp == 'HC', None), ('TC only', grp == 'TC', None), ('BB only', grp == 'BB', None)]:
    if C is None:
        C = np.column_stack([np.ones(m.sum()), age[m], sex[m]])
    r, p = partial_edges(E[m], fd[m], C)
    q = ts.bh(p)
    for en, em in [('all edges', np.ones(len(r), bool)), ('VIS2 edges', v2e), ('non-VIS2 edges', ~v2e)]:
        qc.append(dict(Sample=label, Edges=en, N_subjects=int(m.sum()), median_abs_QCFC=np.median(np.abs(r[em])),
                       mean_QCFC=r[em].mean(), pct_p05=100 * (p[em] < .05).mean(), pct_q05_FDR=100 * (q[em] < .05).mean(),
                       distance_dependence_rho=stats.spearmanr(r[em], dist[em])[0]))
pd.DataFrame(qc).to_csv(OUT / 'QCFC_summary_3group.csv', index=False)
del E

# ---- 2. motion map vs group-effect maps
IDS = row_ids(NS)
G2A = parcel_matrix(pd.read_csv(LAYOUT.GRAD_A / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv'), 'G2', IDS)
Xm = np.column_stack([np.ones(NS), fd, grp == 'TC', grp == 'BB', age, sex]).astype(float)
XtXi = np.linalg.inv(Xm.T @ Xm)
B = XtXi @ Xm.T @ G2A
Res = G2A - Xm @ B
h = np.einsum('ij,jk,ik->i', Xm, XtXi, Xm)
se = np.sqrt(np.array([(XtXi @ (Xm.T * (Res[:, j] ** 2 / (1 - h) ** 2)) @ Xm @ XtXi)[1, 1] for j in range(358)]))
tmap = B[1] / se
PERM = spin_permutations()


def spin(x, y):
    r = np.corrcoef(x, y)[0, 1]
    Xn = x[PERM]; Xn = (Xn - Xn.mean(1, keepdims=True)) / Xn.std(1, keepdims=True)
    null = (Xn * ((y - y.mean()) / y.std())).mean(1)
    return r, (1 + np.sum(np.abs(null) >= abs(r))) / (1 + len(null))


TA = pd.read_csv(LAYOUT.GRAD_A / 'stats_three_group/Step5_parcel_G2_358_3group.csv').sort_values('Parcel')
MM = pd.DataFrame([dict(Comparison=f'motion t-map vs {k} g map', **dict(zip(['r', 'p_spin'], spin(tmap, TA[f'{k}_g'].values))))
                   for k in ('TCvsHC', 'BBvsHC', 'TCvsBB')])
MM['VIS2_mean_motion_t'] = tmap[v2].mean()
MM.to_csv(OUT / 'motion_map_vs_group_maps_3group.csv', index=False)

# ---- 3. signal quality, three groups
SQ = pd.read_csv(OUT / 'data/subject_signal_quality.csv')
assert list(SQ.Subject) == list(cov.participant_id)
AMP = pd.read_csv(OUT / 'data/parcel_residual_amplitude_Glasser360.csv').drop(columns='Subject').values[:, KEEP358]
ENG = ts.Engine(ts.design(cov), grp)
sq = [dict(Metric=m, **ENG.analyse(SQ[m].values.astype(float)))
      for m in ['n_censored', 'remaining_min', 'mean_coverage', 'n_zeroSD_cortex', 'cortex_median_amplitude', 'DVARS_median']]
SQt, _ = ts.finalize_family(pd.DataFrame(sq).assign(Family='Signal_quality'), 'Family')
SQt.to_csv(OUT / 'signal_quality_group_stats_3group.csv', index=False)

# ---- 4. residual-amplitude covariates
cov2 = cov.copy()
cov2['VIS2_amplitude'] = np.nanmean(AMP[:, v2], 1)
ENG2 = ts.Engine(ts.design(cov2, 'VIS2_amplitude'), grp)
A4 = pd.read_csv(LAYOUT.GRAD_A / 'Step4_gradient_metrics/subject_gradient_metrics.csv')
A91 = pd.read_csv(LAYOUT.GRAD_A / 'step_9.1/step9_1_additional_subject_metrics.csv')
prim = {'G2 range': A4.G2_RobustRange.values, 'λ2': A91.Eigenvalue_G2.values, 'VIS2 mean G2': G2A[:, v2].mean(1),
        'G1 range': A4.G1_RobustRange.values}
sens = []
for name, y in prim.items():
    a, b = ENG.analyse(y), ENG2.analyse(y)
    sens.append(dict(Metric=name, **{f'{k}_{c}{suf}': r[f'{k}_{c}'] for suf, r in (('', a), ('_ampcov', b))
                                     for k in ('TCvsHC', 'BBvsHC', 'TCvsBB') for c in ('Beta', 'P')},
                     Omnibus_P=a['Omnibus_P'], Omnibus_P_ampcov=b['Omnibus_P']))
pd.DataFrame(sens).to_csv(OUT / 'sensitivity_VIS2_amplitude_covariate_3group.csv', index=False)
prow = []
for j in range(358):
    cj = cov.copy(); cj['amp'] = AMP[:, j]
    prow.append(dict(Parcel=j + 1, Network=net[j], **ts.Engine(ts.design(cj, 'amp'), grp).analyse(G2A[:, j])))
PA, _ = ts.finalize_family(pd.DataFrame(prow).assign(Family='Parcel_G2_ampcov'), 'Family')
PA.to_csv(OUT / 'parcel_G2_with_own_amplitude_covariate_3group.csv', index=False)
print(pd.DataFrame(qc).to_string(index=False, float_format=lambda v: f'{v:.3g}'))
