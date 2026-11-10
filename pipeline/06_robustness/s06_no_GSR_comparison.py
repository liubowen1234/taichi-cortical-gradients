"""Step 6.6 - denoising without global signal regression (32P = the 36P model minus the 4 global-signal regressors).

Inputs: the 32P gradients (matlab run_all_gradients.m, then pipeline/02_gradients/s01_extra_subject_metrics.py noGSR)
and the 32P FC. Everything is computed exactly as in the main analysis:
  gradients    : HC reference rebuilt from 32P FC; G2 span, lambda1-3, minor-axis variance, VIS2 mean G2, G1 / G3 span
  connectivity : Fisher z, negative edges set to 0, VIS2 segregation and block means (as in 03_network_connectivity)
  statistics   : three-group engine; also with mean FC as an extra covariate (mean FC differs more without GSR)
  dose         : Metric ~ TrainYears + Age + Sex + MeanFD within each athlete group
  spatial      : parcel-wise G2 group effects; similarity of the 32P and 36P maps and reference G2
Pre-specified criterion: for G2 span and lambda2, TC > HC and TC > BB (Holm P < 0.05) and BB vs HC not significant.
Also writes the rows of main Fig. 6 that come from this step and from s05 (no GSR; + global signal amplitude).

Outputs (06_robustness/no_GSR_32P/): subject_metrics_32P.csv, group_stats_36P_vs_32P_3group.csv, criterion_check.csv,
training_dose_36P_vs_32P.csv, parcel_G2_358_3group_32P.csv, spatial_similarity_36P_vs_32P.csv, fig6_extra_rows.csv
Usage: python pipeline/06_robustness/s06_no_GSR_comparison.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from common.config import LAYOUT
from common import stats_engine as ts
from common.data import covariates, training_years, cabnp_mapping, load_fc, row_ids, parcel_matrix

OUT = LAYOUT.NOGSR
NEW, OLD = LAYOUT.GRAD_NOGSR, LAYOUT.GRAD_A          # NEW = 32P (no GSR), OLD = 36P main analysis
cov = covariates()
grp = cov.Group.values
NS = len(cov)
yrs = training_years(cov)
net = cabnp_mapping().CABNP_Network.values
v2 = net == 'Visual2'
IDS = row_ids(NS)

# ---- connectivity metrics from the 32P FC
FC = load_fc(NEW / 'FC.mat')                             # subjects x 358 x 358
iu = np.triu_indices(358, 1)


def block(x, a, b):
    x = x.copy(); np.fill_diagonal(x, np.nan)
    if a is b:
        blk = x[np.ix_(a, a)]
        return float(np.nanmean(blk[np.triu_indices(a.sum(), 1)]))
    return float(np.nanmean(x[np.ix_(a, b)]))


conn = []
for s in range(NS):
    R = FC[s]
    Z = np.arctanh(np.clip(R, -1 + 1e-7, 1 - 1e-7)); np.fill_diagonal(Z, 0.0)
    A = np.maximum(Z, 0)
    W = A[np.ix_(v2, v2)][np.triu_indices(v2.sum(), 1)].mean()
    B = A[np.ix_(v2, ~v2)].mean()
    conn.append(dict(MeanFC_32P=R[iu].mean(), Segregation_Visual2=(W - B) / W, Visual2_Somatomotor=block(A, v2, net == 'Somatomotor'),
                     Visual2_CinguloOpercular=block(A, v2, net == 'Cingulo-Opercular')))
C = pd.DataFrame(conn)
del FC


# ---- gradient metrics of both pipelines
def grad(root):
    s4 = pd.read_csv(root / 'Step4_gradient_metrics/subject_gradient_metrics.csv'); assert (s4.Group.values == grp).all()
    s91 = pd.read_csv(root / 'step_9.1/step9_1_additional_subject_metrics.csv')
    s92 = pd.read_csv(root / 'step_9.2/step9_2_G3_subject_metrics.csv')
    L = pd.read_csv(root / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv')
    G2 = parcel_matrix(L, 'G2', IDS)
    refG2 = L[L.Subject == IDS[0]].sort_values('Parcel').ReferenceG2.values
    m = {'G2 range': s4.G2_RobustRange.values, 'lambda2': s91.Eigenvalue_G2.values, 'Short-axis variance': s4.MinorVariance.values,
         'VIS2 mean G2': G2[:, v2].mean(1), 'G1 range': s4.G1_RobustRange.values, 'lambda1': s91.Eigenvalue_G1.values,
         'G3 range': s92.G3_RobustRange.values, 'lambda3': s92.Eigenvalue_G3.values}
    return m, G2, refG2


mo, G2o, refo = grad(OLD)
mn, G2n, refn = grad(NEW)
br = pd.read_csv(LAYOUT.VIS2 / 'subject_Visual2_networkFC_and_bridge_inputs.csv')
mo.update({'VIS2 segregation': br.Segregation_Visual2.values, 'VIS2-SMN FC': br.Visual2_Somatomotor.values,
           'VIS2-CON FC': br.Visual2_CinguloOpercular.values})
mn.update({'VIS2 segregation': C.Segregation_Visual2.values, 'VIS2-SMN FC': C.Visual2_Somatomotor.values,
           'VIS2-CON FC': C.Visual2_CinguloOpercular.values})
pd.DataFrame({'participant_id': cov.participant_id, 'Group': grp, 'MeanFC_32P': C.MeanFC_32P.values, **mn}) \
    .to_csv(OUT / 'subject_metrics_32P.csv', index=False)

# ---- group statistics
ENG = ts.Engine(ts.design(cov), grp)
cov2 = cov.copy(); cov2['MeanFC32P'] = C.MeanFC_32P.values
ENG2 = ts.Engine(ts.design(cov2, 'MeanFC32P'), grp)
rows = [dict(Metric=name, Pipeline=pipe, **eng.analyse(m[name])) for name in mo
        for pipe, m, eng in (('36P (main)', mo, ENG), ('32P (no GSR)', mn, ENG), ('32P + mean FC covariate', mn, ENG2))]
T = pd.DataFrame(rows)
T.to_csv(OUT / 'group_stats_36P_vs_32P_3group.csv', index=False)


def criterion_met(d):
    sig = lambda k: d[f'{k}_P_Holm'] < .05
    return bool(d['Omnibus_P'] < .05 and sig('TCvsHC') and d.TCvsHC_Beta > 0 and sig('TCvsBB') and d.TCvsBB_Beta > 0 and not sig('BBvsHC'))


T[T.Metric.isin(['G2 range', 'lambda2'])].assign(Criterion_met=lambda x: x.apply(criterion_met, axis=1)) \
    [['Metric', 'Pipeline', 'Criterion_met']].to_csv(OUT / 'criterion_check.csv', index=False)


# ---- training dose (raw Sex coding 1/2 changes only the intercept)
def dose(y, g):
    m = grp == g
    X = np.column_stack([np.ones(m.sum()), yrs[m], cov.Age.values[m], cov.Sex.values[m], cov.MeanFD.values[m]])
    r = ts.hc3(y[m], X)
    return dict(N=int(m.sum()), Beta_std=r['Beta'] * yrs[m].std(ddof=1) / y[m].std(ddof=1), P=r['P'])


pd.DataFrame([dict(Group=g, Metric=n, Pipeline=p, **dose(m[n], g)) for g in ('TC', 'BB') for n in ('G2 range', 'lambda2', 'VIS2 mean G2')
              for p, m in (('36P (main)', mo), ('32P (no GSR)', mn))]).to_csv(OUT / 'training_dose_36P_vs_32P.csv', index=False)

# ---- spatial: parcel G2 effects and reference G2
pe = [dict(Parcel=j + 1, Network=net[j], **{k: v for k, v in ENG.analyse(G2n[:, j]).items()
                                             if k.startswith(('Omnibus_P', 'TCvsHC', 'BBvsHC', 'TCvsBB'))}) for j in range(358)]
PE, _ = ts.finalize_family(pd.DataFrame(pe).assign(Family='Parcel_G2'), 'Family')
PE.to_csv(OUT / 'parcel_G2_358_3group_32P.csv', index=False)
TA = pd.read_csv(OLD / 'stats_three_group/Step5_parcel_G2_358_3group.csv').sort_values('Parcel')
SP = [dict(Map='HC reference G2', r=np.corrcoef(refo, refn)[0, 1])]
SP += [dict(Map=f'{k} parcel G2 g', r=np.corrcoef(TA[f'{k}_g'].values, PE[f'{k}_g'].values)[0, 1]) for k in ('TCvsHC', 'BBvsHC', 'TCvsBB')]
for k in ('TCvsHC', 'BBvsHC', 'TCvsBB'):
    SP.append(dict(Map=f'{k} significant parcels 32P', r=np.nan, sig=int(PE[f'{k}_Sig'].sum()),
                   sig_pos_VIS2=int(((PE[f'{k}_Sig'] == 1) & (PE[f'{k}_Beta'] > 0) & (net == 'Visual2')).sum())))
pd.DataFrame(SP).to_csv(OUT / 'spatial_similarity_36P_vs_32P.csv', index=False)


# ---- rows for main Fig. 6 (same format as 06_robustness/robustness_summary/fig6_*.csv)
def fig_row(setting, metric, r):
    d = dict(Setting=setting, Metric=metric, Omnibus_P=r['Omnibus_P'])
    for c in ('TCvsHC', 'BBvsHC', 'TCvsBB'):
        d.update({f'{c}_g': r[f'{c}_g'], f'{c}_lo': r[f'{c}_g_CI95_Lower'], f'{c}_hi': r[f'{c}_g_CI95_Upper'],
                  f'{c}_t': r[f'{c}_T_HC3'], f'{c}_P_Holm': r[f'{c}_P_Holm'],
                  f'{c}_sig': bool(r['Omnibus_P'] < .05 and r[f'{c}_P_Holm'] < .05)})
    return d


rows = [fig_row('No GSR (32P)', lab, T[(T.Metric == m) & (T.Pipeline == '32P (no GSR)')].iloc[0])
        for m, lab in [('G2 range', 'G2 range'), ('Short-axis variance', 'Minor-axis variance'), ('VIS2 mean G2', 'VIS2 mean G2')]]
gsa = pd.read_csv(LAYOUT.GSA / 'subject_global_signal_amplitude.csv')
assert list(gsa.participant_id) == list(cov.participant_id)
cov3 = cov.copy(); cov3['GSA'] = gsa.GSA_bandpass.values
rows.append(fig_row('+ global signal amplitude', 'G2 range', ts.Engine(ts.design(cov3, 'GSA'), grp).analyse(br.A_G2_RobustRange.values)))
pd.DataFrame(rows).to_csv(OUT / 'fig6_extra_rows.csv', index=False)
print(T[['Metric', 'Pipeline', 'Omnibus_P', 'TCvsHC_g', 'BBvsHC_g', 'TCvsBB_g']].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
