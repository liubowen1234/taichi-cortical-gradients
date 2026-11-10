"""Step 6.4 - is any network-level G2 effect explained by head motion? (Supplementary Fig. 8)

A. Motion maps of parcel G2 (main analysis):
     pooled    : G2 ~ MeanFD + TC + BB + Age + Sex -> per-parcel partial r of mean FD (and HC3 t)
     per group : G2 ~ MeanFD + Age + Sex in HC, TC and BB separately
   Network enrichment of each map over the 12 CAB-NP networks against the spin null (two-sided P, BH-FDR over networks)
   and the spatial correlation of each motion map with the three G2 group-effect maps.
B. Network mean G2 (12 networks):
     association with mean FD (pooled partial r adjusted for group, Age, Sex; within-group Spearman);
     group effects in the motion subsets of common/motion.py;
     sensitivity models with mean FD squared and with the number of censored frames as extra covariates.

Outputs (06_robustness/data_quality/motion_network_enrichment/)
Usage: python pipeline/06_robustness/s04_motion_network_enrichment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy import stats
from common.config import LAYOUT, mkdir
from common import stats_engine as ts, spin
from common.data import covariates, cabnp_mapping, row_ids, parcel_matrix, spin_permutations, NETS
from common.motion import motion_subsets

OUT = mkdir(LAYOUT.QUALITY / 'motion_network_enrichment')
cov = covariates()
grp = cov.Group.values
NS = len(cov)
fd = cov.MeanFD.values
age = cov.Age.values.astype(float)
sex = cov.Sex.values - 1.0
net = cabnp_mapping().CABNP_Network.values
G2A = parcel_matrix(pd.read_csv(LAYOUT.GRAD_A / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv'), 'G2', row_ids(NS))
PERM = spin_permutations()
TA = pd.read_csv(LAYOUT.GRAD_A / 'stats_three_group/Step5_parcel_G2_358_3group.csv').sort_values('Parcel')


def resid(y, C):
    return y - C @ np.linalg.lstsq(C, y, rcond=None)[0]


def hc3_t(y, X, j):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    h = np.minimum(np.einsum('ij,jk,ik->i', X, XtXi, X), .999999)
    V = XtXi @ (X.T * (e / (1 - h)) ** 2) @ X @ XtXi
    t = b[j] / np.sqrt(V[j, j]); df = len(y) - X.shape[1]
    return t, 2 * stats.t.sf(abs(t), df)


def motion_map(mask, covars):
    """Per-parcel partial r (and HC3 t, P) of mean FD with parcel G2 within `mask`, adjusting for `covars`."""
    C = np.column_stack([np.ones(mask.sum())] + [c[mask] for c in covars])
    X = np.column_stack([C[:, :1], fd[mask], C[:, 1:]])
    rx = resid(fd[mask], C)
    out = []
    for p in range(358):
        y = G2A[mask, p]
        t, pv = hc3_t(y, X, 1)
        out.append((np.corrcoef(resid(y, C), rx)[0, 1], t, pv))
    return np.array(out)


def spin_corr(x, y):
    r = np.corrcoef(x, y)[0, 1]
    Xn = x[PERM]; Xn = (Xn - Xn.mean(1, keepdims=True)) / Xn.std(1, keepdims=True)
    null = (Xn * ((y - y.mean()) / y.std())).mean(1)
    return r, (1 + np.sum(np.abs(null) >= abs(r))) / (1 + len(null))


# ---- A. motion maps and network enrichment
TCd, BBd = (grp == 'TC').astype(float), (grp == 'BB').astype(float)
MAPS = {'pooled': motion_map(np.ones(NS, bool), [TCd, BBd, age, sex])}
for g in ('HC', 'TC', 'BB'):
    MAPS[g] = motion_map(grp == g, [age, sex])
pd.DataFrame({'Parcel': np.arange(1, 359), 'Network': net, **{f'partial_r_{k}': v[:, 0] for k, v in MAPS.items()},
              **{f't_HC3_{k}': v[:, 1] for k, v in MAPS.items()}, **{f'P_{k}': v[:, 2] for k, v in MAPS.items()}}) \
    .to_csv(OUT / 'parcel_G2_motion_maps_3group.csv', index=False)
erows, crows = [], []
for k, v in MAPS.items():
    obs, null, pv, z = spin.spin_network_means(v[:, 0], net, PERM, NETS)
    q = ts.bh(pv)
    for i, n in enumerate(NETS):
        erows.append(dict(Map=k, Network=n, N_parcels=int((net == n).sum()), Observed_mean_partial_r=obs[i], Null_mean=null[:, i].mean(),
                          Null_2p5=np.percentile(null[:, i], 2.5), Null_97p5=np.percentile(null[:, i], 97.5), z_spin=z[i],
                          p_spin=pv[i], q_spin_FDR=q[i]))
    for c in ('TCvsHC', 'BBvsHC', 'TCvsBB'):
        r, ps = spin_corr(v[:, 0], TA[f'{c}_g'].values)
        crows.append(dict(Map=k, Compared_with=f'{c} parcel G2 g map', r=r, p_spin=ps))
pd.DataFrame(erows).to_csv(OUT / 'motion_map_network_enrichment_3group.csv', index=False)
pd.DataFrame(crows).to_csv(OUT / 'motion_map_vs_group_maps_3group.csv', index=False)

# ---- B. network mean G2
NM = {n: G2A[:, net == n].mean(1) for n in NETS}
C0 = np.column_stack([np.ones(NS), TCd, BBd, age, sex])
X0 = np.column_stack([C0[:, :1], fd, C0[:, 1:]])
b1 = []
for n in NETS:
    y = NM[n]; t, pv = hc3_t(y, X0, 1)
    row = dict(Network=n, pooled_partial_r=np.corrcoef(resid(y, C0), resid(fd, C0))[0, 1], pooled_t_HC3=t, pooled_P=pv)
    for g in ('HC', 'TC', 'BB'):
        m = grp == g
        rho, pr = stats.spearmanr(fd[m], y[m])
        row.update({f'{g}_spearman_rho': rho, f'{g}_spearman_P': pr})
    b1.append(row)
B1 = pd.DataFrame(b1)
B1['pooled_Q_FDR'] = ts.bh(B1.pooled_P.values)
B1.to_csv(OUT / 'network_meanG2_vs_MeanFD_3group.csv', index=False)

b2 = []
for sname, k in motion_subsets(cov).items():
    E = ts.Engine(ts.design(cov[k]), grp[k])
    for n in NETS:
        r = E.analyse(NM[n][k])
        b2.append(dict(Subset=sname, N=int(k.sum()), Network=n, Omnibus_P=r['Omnibus_P'],
                       **{f'{c}_{x}': r[f'{c}_{x}'] for c in ('TCvsHC', 'BBvsHC', 'TCvsBB') for x in ('g', 'P_Holm', 'P_TOST')}))
pd.DataFrame(b2).to_csv(OUT / 'network_meanG2_motion_subsets_3group.csv', index=False)

SQ = pd.read_csv(LAYOUT.QUALITY / 'data/subject_signal_quality.csv').set_index('Subject').loc[cov.participant_id]
cv = cov.copy()
cv['MeanFD_sq'] = (fd - fd.mean()) ** 2
cv['Censored'] = SQ['n_censored'].values
b3 = []
for lab, extra in [('primary', None), ('+ MeanFD^2', 'MeanFD_sq'), ('+ n_censored', 'Censored')]:
    E = ts.Engine(ts.design(cv, extra), grp)
    for n in NETS:
        r = E.analyse(NM[n])
        b3.append(dict(Model=lab, Network=n, Omnibus_P=r['Omnibus_P'],
                       **{f'{c}_{x}': r[f'{c}_{x}'] for c in ('TCvsHC', 'BBvsHC', 'TCvsBB') for x in ('g', 'P_Holm')}))
pd.DataFrame(b3).to_csv(OUT / 'network_meanG2_motion_covariates_3group.csv', index=False)
print('done')
