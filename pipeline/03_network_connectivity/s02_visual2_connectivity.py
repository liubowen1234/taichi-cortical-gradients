"""Step 3.2 - Visual2 (VIS2) connectivity and the relation between VIS2 segregation and its G2 position.

1. VIS2 connectivity profile: mean positive-only Fisher-z FC within VIS2 and between VIS2 and each of the other 11
   CAB-NP networks (block means, diagonal excluded). The 12 measures form one family (omnibus BH-FDR, Holm pairwise).
   Two further families used in the text: VIS2 x {SMN, CON, AUD, DAN} (4 tests), and the five measures VIS2 within +
   those four partners.
2. Segregation -> gradient relation (consistency check; both measures come from the same FC):
       VIS2 mean G2 ~ VIS2 segregation + TC + BB + Age + Sex + MeanFD   (HC3)
   with G2 span as a secondary outcome, and the group difference in slope (Segregation x TC, Segregation x BB; 2-df Wald F).
   Within-group Pearson / Spearman correlations. Gradients of the main analysis (A) and of the template alignment (B).

Outputs (03_network_connectivity/visual2_connectivity/)
  subject_Visual2_networkFC_and_bridge_inputs.csv  subject values (VIS2 x partner FC, segregation, VIS2 mean G2, G2 span)
  subject_Visual2_all_partners_FC.csv
  Visual2_all_partners_FC_3group.csv, Visual2_networkpair_FC_3group.csv, Visual2_networkpair_FC_family5_q.csv
  segregation_gradient_regression_3group.csv, within_group_correlations_3group.csv

Usage: python pipeline/03_network_connectivity/s02_visual2_connectivity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy import stats
from common.config import LAYOUT, mkdir
from common import stats_engine as ts
from common.data import covariates, cabnp_mapping, load_fc, row_ids, parcel_matrix

OUT = mkdir(LAYOUT.VIS2)
cov = covariates()
grp = cov.Group.values
NS = len(cov)
net = cabnp_mapping().CABNP_Network.values
R = load_fc(LAYOUT.GRAD_A / 'FC.mat')
Z = np.arctanh(np.clip(R, -1 + 1e-7, 1 - 1e-7))
for s in range(NS):
    np.fill_diagonal(Z[s], 0.0)


def network_fc(z, a, b):
    """Mean positive-only Fisher-z FC between parcel sets a and b (within a set: upper triangle only)."""
    x = np.maximum(z, 0.0); np.fill_diagonal(x, np.nan)
    if np.array_equal(a, b):
        block = x[np.ix_(a, a)]
        return float(np.nanmean(block[np.triu_indices(len(a), k=1)]))
    return float(np.nanmean(x[np.ix_(a, b)]))


ENG = ts.Engine(ts.design(cov), grp)
v2 = np.where(net == 'Visual2')[0]

# ---- 1a. VIS2 with all 12 networks
ALL = {f'Visual2_{n}': np.array([network_fc(Z[s], v2, np.where(net == n)[0]) for s in range(NS)])
       for n in dict.fromkeys(net)}
T = pd.DataFrame([{'Metric': k, 'Family': 'VIS2 within + 11 partners (12)', **ENG.analyse(y)} for k, y in ALL.items()])
T, _ = ts.finalize_family(T, 'Family')
T.to_csv(OUT / 'Visual2_all_partners_FC_3group.csv', index=False)
pd.DataFrame({'Subject': cov.participant_id, 'Group': grp, **ALL}).to_csv(OUT / 'subject_Visual2_all_partners_FC.csv', index=False)

# ---- 1b. VIS2 within (pre-specified single test) and VIS2 x four partners (one family)
PAIRS = [('Visual2_Visual2', 'Visual2'), ('Visual2_Somatomotor', 'Somatomotor'), ('Visual2_CinguloOpercular', 'Cingulo-Opercular'),
         ('Visual2_Auditory', 'Auditory'), ('Visual2_DorsalAttention', 'Dorsal-Attention')]
seg = pd.read_csv(LAYOUT.SEG / 'subject_network_segregation.csv')
assert list(seg.Subject) == list(cov.participant_id)
F = pd.DataFrame({'Subject': cov.participant_id, 'Group': grp, **{k: ALL[f'Visual2_{b}'] for k, b in PAIRS}})
assert np.abs(F.Visual2_Visual2 - seg.Visual2_Within).max() < 1e-12
rows = [{'Metric': k, 'NetworkA': 'Visual2', 'NetworkB': b,
         'Family': 'Visual2 cross-network FC (4)' if b != 'Visual2' else 'Visual2 within (pre-specified, no FDR)',
         **ENG.analyse(F[k].values)} for k, b in PAIRS]
T = pd.DataFrame(rows)
cross, _ = ts.finalize_family(T[T.Family.str.startswith('Visual2 cross')].copy(), 'Family')
within = T[~T.Family.str.startswith('Visual2 cross')].copy()
within['Omnibus_Q_FDR'] = within.Omnibus_P
within['Posthoc_inferential'] = (within.Omnibus_P < .05).astype(int)
for a, c, _ in ts.PAIRS:
    within[f'{a}vs{c}_Sig'] = ((within.Omnibus_P < .05) & (within[f'{a}vs{c}_P_Holm'] < .05)).astype(int)
T = pd.concat([within, cross])
T.to_csv(OUT / 'Visual2_networkpair_FC_3group.csv', index=False)
T[['Metric', 'Omnibus_F', 'Omnibus_P']].assign(q_family5=ts.bh(T.Omnibus_P.values)) \
    .to_csv(OUT / 'Visual2_networkpair_FC_family5_q.csv', index=False)

# ---- gradient inputs (A = main analysis, B = external template)
gA = pd.read_csv(LAYOUT.GRAD_A / 'Step4_gradient_metrics/subject_gradient_metrics.csv')
G2A = parcel_matrix(pd.read_csv(LAYOUT.GRAD_A / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv'), 'G2', row_ids(NS))
gB = pd.read_csv(LAYOUT.GRAD_B / 'Step4_subject_gradient_metrics.csv')
G2B = parcel_matrix(pd.read_csv(LAYOUT.GRAD_B / 'Step4_parcel_gradients_long.csv'), 'G2', row_ids(NS))
isv2 = net == 'Visual2'
F['Segregation_Visual2'] = seg.Visual2_Segregation.values
F['A_Visual2_MeanG2'] = G2A[:, isv2].mean(1); F['A_G2_RobustRange'] = gA.G2_RobustRange.values
F['B_Visual2_MeanG2'] = G2B[:, isv2].mean(1); F['B_G2_RobustRange'] = gB.G2_RobustRange.values
F.to_csv(OUT / 'subject_Visual2_networkFC_and_bridge_inputs.csv', index=False)


# ---- 2. segregation -> gradient regressions
def hc3_general(y, X):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    h = np.minimum(np.einsum('ij,jk,ik->i', X, XtXi, X), .999999)
    V = XtXi @ (X.T * ((e / (1 - h)) ** 2)) @ X @ XtXi
    return b, V, len(y) - X.shape[1]


tc, bb = (grp == 'TC').astype(float), (grp == 'BB').astype(float)
base = [cov.Age.values, cov.Sex.values - 1.0, cov.MeanFD.values]
sv = F.Segregation_Visual2.values
reg, corr = [], []
for scheme in ('A', 'B'):
    for role, outcome in [('Primary', 'Visual2_MeanG2'), ('Secondary', 'G2_RobustRange')]:
        y = F[f'{scheme}_{outcome}'].values
        X = np.column_stack([np.ones(NS), sv, tc, bb] + base)
        b, V, df = hc3_general(y, X)
        se = np.sqrt(V[1, 1]); t = b[1] / se
        reg.append(dict(Scheme=scheme, Model=role, Outcome=outcome, Term='Segregation_Visual2',
                        Formula=f'{outcome} ~ Seg + TC + BB + Age + Sex + MeanFD', Beta=b[1], SE_HC3=se, T=t,
                        P=2 * stats.t.sf(abs(t), df), DF=df))
        # slope differences between groups (HC slope = reference)
        X = np.column_stack([np.ones(NS), sv, tc, bb, sv * tc, sv * bb] + base)
        b, V, df = hc3_general(y, X)
        Rm = np.zeros((2, X.shape[1])); Rm[0, 4] = Rm[1, 5] = 1
        rb = Rm @ b
        Fv = float(rb @ np.linalg.solve(Rm @ V @ Rm.T, rb)) / 2
        out = dict(Scheme=scheme, Model='Interaction (exploratory)', Outcome=outcome, Term='Seg x Group (2 df)',
                   Formula=f'{outcome} ~ Seg * Group + Age + Sex + MeanFD', F=Fv, P=stats.f.sf(Fv, 2, df), DF=df,
                   Slope_HC=b[1], Slope_TC=b[1] + b[4], Slope_BB=b[1] + b[5])
        for nm, L in [('TCminusHC', [0, 0, 0, 0, 1, 0]), ('BBminusHC', [0, 0, 0, 0, 0, 1]), ('TCminusBB', [0, 0, 0, 0, 1, -1])]:
            L = np.r_[L, np.zeros(len(base))]
            est = L @ b; s_ = np.sqrt(L @ V @ L)
            out[f'SlopeDiff_{nm}'] = est; out[f'SlopeDiff_{nm}_P'] = 2 * stats.t.sf(abs(est / s_), df)
        reg.append(out)
        for g in ('HC', 'TC', 'BB'):
            m = grp == g
            pr = stats.pearsonr(sv[m], y[m]); sr = stats.spearmanr(sv[m], y[m])
            corr.append(dict(Scheme=scheme, Group=g, Outcome=outcome, N=int(m.sum()), Pearson_r=pr.statistic, Pearson_p=pr.pvalue,
                             Spearman_rho=sr.statistic, Spearman_p=sr.pvalue))
pd.DataFrame(reg).to_csv(OUT / 'segregation_gradient_regression_3group.csv', index=False)
pd.DataFrame(corr).to_csv(OUT / 'within_group_correlations_3group.csv', index=False)
print(pd.DataFrame(reg).to_string(index=False, float_format=lambda v: f'{v:.3g}'))
