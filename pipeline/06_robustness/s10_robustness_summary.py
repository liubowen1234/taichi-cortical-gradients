"""Step 6.10 - the key metrics re-analysed in the same way in every robustness setting (main Fig. 6; Supplementary Table 6).

Settings: main analysis; + intracranial volume; external template; four head-motion subsets; Schaefer-400; Schaefer-600.
Metrics: G2 span, lambda2, minor-axis variance of the G1-G2 plane, VIS2 mean G2 (Glasser; for Schaefer the Yeo-7 visual
network is used in the figure panels); training dose of G2 span within Tai Chi and within basketball.
In each setting the omnibus P of the four group metrics are BH-FDR corrected together and the pre-specified pattern is
assigned (common/stats_engine.classify). Because this common family differs from the step-specific families, a few q
values differ slightly from the step tables.
Additional covariate settings (residual BOLD amplitude of VIS2 and of the cortex, DVARS, % censored frames) are analysed
for all four metrics in the same way.

Outputs (06_robustness/robustness_summary/)
  robustness_heatmap_data.csv               patterns and statistics per setting and metric
  covariate_settings_four_metrics_3group.csv
  fig6_refs_atlas.csv, fig6_covariates.csv, fig6_motion.csv   data of main Fig. 6 panels (g [95% CI], t, Holm P)
Usage: python pipeline/06_robustness/s10_robustness_summary.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import h5py
from common.config import LAYOUT, RESOURCES, mkdir
from common import stats_engine as ts
from common.data import covariates, training_years, cabnp_mapping, row_ids, parcel_matrix
from common.motion import motion_subsets

OUT = mkdir(LAYOUT.SUMMARY)
cov = covariates()
grp = cov.Group.values
N = len(cov)
years = training_years(cov)
cab = cabnp_mapping()
v2 = cab.CABNP_Network.values == 'Visual2'
SID = row_ids(N)
lam2_glasser = pd.read_csv(LAYOUT.GRAD_A / 'step_9.1/step9_1_additional_subject_metrics.csv').Eigenvalue_G2.values
C3 = ['TCvsHC', 'BBvsHC', 'TCvsBB']


def span(X):
    return np.percentile(X, 95, axis=1) - np.percentile(X, 5, axis=1)


def minor_var(g1, g2):
    return np.array([np.linalg.eigvalsh(np.cov(np.vstack([g1[i], g2[i]])))[0] for i in range(len(g1))])


def glasser(path):
    L = pd.read_csv(path)
    g1, g2 = parcel_matrix(L, 'G1', SID), parcel_matrix(L, 'G2', SID)
    return {'G2 range': span(g2), 'λ2': lam2_glasser, 'Minor-axis variance': minor_var(g1, g2), 'VIS2 mean G2': g2[:, v2].mean(1)}


def schaefer(k):
    d = LAYOUT.SCHAEFER[k]
    S = pd.read_csv(d / 'Step4_gradient_metrics/subject_gradient_metrics.csv')
    assert (S.Subject.values == SID).all() and (S.Group.values == grp).all()
    with h5py.File(d / 'Step1_raw_gradients/BrainSpace_formal_raw_gradients.mat', 'r') as h:
        lam = np.array(h['lambda_all'])
    lam = lam if lam.shape[0] == N else lam.T
    return {'G2 range': S.G2_RobustRange.values, 'λ2': lam[:, 1], 'Minor-axis variance': S.MinorVariance.values, 'VIS2 mean G2': None}


def schaefer_visual(k):
    """Mean G2 of the Yeo-7 visual network (Schaefer atlas; used in place of VIS2 in the figure panels)."""
    at = pd.read_csv(RESOURCES / f'atlas/atlas-Schaefer{k}_cortex.tsv', sep='\t').sort_values('index')
    L = pd.read_csv(LAYOUT.SCHAEFER[k] / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv')
    g2 = L.pivot(index='Subject', columns='Parcel', values='G2').loc[SID]
    assert (g2.columns.values == at['index'].values).all()
    return g2.values[:, at.network_label.values == 'Vis'].mean(1)


base = {'A': glasser(LAYOUT.GRAD_A / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv'),
        'B': glasser(LAYOUT.GRAD_B / 'Step4_parcel_gradients_long.csv'), 'S400': schaefer(400), 'S600': schaefer(600)}
SUB = motion_subsets(cov)


def dose(y, keep, g, extra):
    m = keep & (grp == g)
    cols = [np.ones(m.sum()), years[m], cov.Age.values[m], cov.Sex.values[m] - 1, cov.MeanFD.values[m]]
    if extra:
        cols.append(cov[extra].values[m])
    r = ts.hc3(y[m], np.column_stack(cols))
    return r['Beta'] * years[m].std(ddof=1) / y[m].std(ddof=1), r['P'], int(m.sum())


# ---- 1. uniform summary across settings
SETTINGS = [('Primary (HC reference)', 'A', None, None), ('+ intracranial volume', 'A', 'eTIV', None),
            ('External template', 'B', None, None), ('Motion-matched subset', 'A', None, 'SM_motion_matched'),
            ('Drop top 10% motion', 'A', None, 'P10_dropTop10pct_each'), ('Drop top 20% motion', 'A', None, 'P20_dropTop20pct_each'),
            ('Athletes FD ≤ HC maximum', 'A', None, 'SF_athletes_le_HCmax'), ('Schaefer-400', 'S400', None, None),
            ('Schaefer-600', 'S600', None, None)]
rows = []
for lab, key, extra, sub in SETTINGS:
    keep = np.ones(N, bool) if sub is None else SUB[sub]
    E = ts.Engine(ts.design(cov[keep], extra), grp[keep])
    res = {m: (None if y is None else E.analyse(np.asarray(y)[keep])) for m, y in base[key].items()}
    names = [m for m in res if res[m] is not None]
    for m, qq in zip(names, ts.bh([res[m]['Omnibus_P'] for m in names])):
        r = res[m]; r['Omnibus_Q_FDR'] = qq
        rows.append(dict(Setting=lab, Metric=m, N=int(keep.sum()), Pattern=ts.classify(r), Omnibus_Q=qq,
                         **{f'{c}_{k}': r[f'{c}_{k}'] for c in C3 for k in ['g', 'P_Holm', 'P_TOST']}))
    for m in res:
        if res[m] is None:
            rows.append(dict(Setting=lab, Metric=m, N=int(keep.sum()), Pattern='not applicable'))
    for g in ['TC', 'BB']:
        b, p, n = dose(np.asarray(base[key]['G2 range']), keep, g, extra)
        rows.append(dict(Setting=lab, Metric=f'Dose {g}', N=n, Pattern=('significant' if p < .05 else 'n.s.'), beta_std=b, P=p))
pd.DataFrame(rows).to_csv(OUT / 'robustness_heatmap_data.csv', index=False)

# ---- covariates used below
SQ = pd.read_csv(LAYOUT.QUALITY / 'data/subject_signal_quality.csv').set_index('Subject').loc[cov.participant_id]
AMP = pd.read_csv(LAYOUT.QUALITY / 'data/parcel_residual_amplitude_Glasser360.csv').set_index('Subject').loc[cov.participant_id]
cov['VIS2_amp'] = AMP[cab.loc[v2, 'GlasserLabel'].values].mean(1).values
cov['cortex_amp'] = SQ.cortex_median_amplitude.values
cov['DVARS'] = SQ.DVARS_median.values
cov['censored'] = SQ.pct_censored.values
COVSET = [('Primary (HC reference)', None), ('+ intracranial volume', 'eTIV'), ('+ VIS2 amplitude', 'VIS2_amp'),
          ('+ cortex amplitude', 'cortex_amp'), ('+ DVARS', 'DVARS'), ('+ censored volumes', 'censored')]

# ---- 2. covariate settings, four metrics
rows = []
for lab, extra in COVSET:
    E = ts.Engine(ts.design(cov, extra), grp)
    res = {m: E.analyse(np.asarray(y)) for m, y in base['A'].items()}
    for m, qq in zip(res, ts.bh([res[m]['Omnibus_P'] for m in res])):
        r = res[m]; r['Omnibus_Q_FDR'] = qq
        rows.append(dict(Setting=lab, Covariate=extra or '', Metric=m, N=N, Pattern=ts.classify(r), Omnibus_P=r['Omnibus_P'],
                         Omnibus_Q=qq, **{f'{c}_{k}': r[f'{c}_{k}'] for c in C3
                                           for k in ['g', 'g_CI95_Lower', 'g_CI95_Upper', 'T_HC3', 'P', 'P_Holm', 'P_TOST', 'BF01']}))
pd.DataFrame(rows).to_csv(OUT / 'covariate_settings_four_metrics_3group.csv', index=False)


# ---- 3. data of the Fig. 6 panels ('sig' = omnibus P < .05 and Holm P < .05, single metric; display only)
def fig_row(o):
    d = {'Omnibus_P': o['Omnibus_P']}
    for c in C3:
        d.update({f'{c}_g': o[f'{c}_g'], f'{c}_lo': o[f'{c}_g_CI95_Lower'], f'{c}_hi': o[f'{c}_g_CI95_Upper'],
                  f'{c}_t': o[f'{c}_T_HC3'], f'{c}_P_Holm': o[f'{c}_P_Holm'], f'{c}_sig': bool(o['Omnibus_P'] < .05 and o[f'{c}_P_Holm'] < .05)})
    return d


for k in ['S400', 'S600']:
    base[k]['VIS2 mean G2'] = schaefer_visual(int(k[1:]))
E = ts.Engine(ts.design(cov), grp)
pd.DataFrame([dict(Setting=lab, Metric=m, **fig_row(E.analyse(np.asarray(base[key][m]))))
              for lab, key in [('HC reference (primary)', 'A'), ('External template', 'B'), ('Schaefer-400', 'S400'), ('Schaefer-600', 'S600')]
              for m in ['G2 range', 'Minor-axis variance', 'VIS2 mean G2']]).to_csv(OUT / 'fig6_refs_atlas.csv', index=False)
pd.DataFrame([dict(Setting=lab, Metric=m, **fig_row(ts.Engine(ts.design(cov, extra), grp).analyse(np.asarray(base['A'][m]))))
              for lab, extra in [('Primary', None)] + COVSET[1:] for m in ['G2 range', 'VIS2 mean G2']]) \
    .to_csv(OUT / 'fig6_covariates.csv', index=False)
rows = []
for lab, sub in [('Full', None), ('Motion-\nmatched', 'SM_motion_matched'), ('Drop top\n10%', 'P10_dropTop10pct_each'),
                 ('Drop top\n20%', 'P20_dropTop20pct_each'), ('Athletes\nFD ≤ HC max', 'SF_athletes_le_HCmax')]:
    keep = np.ones(N, bool) if sub is None else SUB[sub]
    Es = ts.Engine(ts.design(cov[keep]), grp[keep])
    for m in ['G2 range', 'Minor-axis variance']:
        rows.append(dict(Subset=lab, N=int(keep.sum()), Metric=m, **fig_row(Es.analyse(np.asarray(base['A'][m])[keep]))))
pd.DataFrame(rows).to_csv(OUT / 'fig6_motion.csv', index=False)
print(pd.read_csv(OUT / 'robustness_heatmap_data.csv').pivot(index='Setting', columns='Metric', values='Pattern').to_string())
