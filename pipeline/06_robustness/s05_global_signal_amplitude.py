"""Step 6.5 - global signal amplitude (GSA) as a covariate.

GSA per participant, from the fMRIPrep global_signal regressor in the XCP-D 36P design matrix (raw scanner units,
non-steady-state volumes already removed): SD of the signal / its mean x 100 (% signal), over the frames retained after
FD censoring. Two versions: linear detrend only; detrend + band-pass 0.008-0.09 Hz (FFT, TR = 2 s), the main version.
Then: three-group comparison of GSA (with and without mean FD as covariate); what GSA correlates with; the key results
with GSA as an extra covariate; the training-dose models with GSA as an extra covariate.

Outputs (06_robustness/global_signal_amplitude/): subject_global_signal_amplitude.csv, GSA_group_stats_3group.csv,
GSA_group_stats_noMeanFD_3group.csv, GSA_correlations.csv, key_results_with_GSA_covariate_3group.csv,
training_dose_with_GSA_covariate.csv
Usage: python pipeline/06_robustness/s05_global_signal_amplitude.py
"""
import sys, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy import stats
from common.config import CFG, LAYOUT, mkdir
from common import stats_engine as ts
from common.data import covariates, training_years

OUT = mkdir(LAYOUT.GSA)
TR, LO, HI = 2.0, 0.008, 0.09
cov = covariates()
grp = cov.Group.values


def read_tsv(p):
    r = list(csv.reader(open(p), delimiter='\t'))
    return r[0], np.array(r[1:], float)


def bandpass(x):
    n = len(x); f = np.fft.rfftfreq(n, TR); X = np.fft.rfft(x)
    X[(f < LO) | (f > HI)] = 0
    return np.fft.irfft(X, n)


# ---- 1. GSA per participant
rows = []
for sid, g in zip(cov.participant_id, grp):
    d = next(x['dir'] for x in CFG.datasets('36P') if (x['dir'] / sid).is_dir())
    h, a = read_tsv(d / sid / 'func' / f'{sid}_task-rest_design.tsv')
    gs = a[:, h.index('global_signal')]
    _, o = read_tsv(d / sid / 'func' / f'{sid}_task-rest_outliers.tsv')
    keep = o[:, 0] == 0
    assert len(keep) == len(gs), sid
    t = np.arange(len(gs))
    det = gs - np.polyval(np.polyfit(t, gs, 1), t)
    m = gs.mean()
    rows.append(dict(participant_id=sid, Group=g, NFrames=len(gs), NRetained=int(keep.sum()), GS_mean_raw=m,
                     GSA_detrended=det[keep].std(ddof=1) / m * 100, GSA_bandpass=bandpass(det)[keep].std(ddof=1) / m * 100))
gs = pd.DataFrame(rows)
gs.to_csv(OUT / 'subject_global_signal_amplitude.csv', index=False)

# ---- inputs for the covariate analyses
br = pd.read_csv(LAYOUT.VIS2 / 'subject_Visual2_networkFC_and_bridge_inputs.csv')
s91 = pd.read_csv(LAYOUT.GRAD_A / 'step_9.1/step9_1_additional_subject_metrics.csv')
s4 = pd.read_csv(LAYOUT.GRAD_A / 'Step4_gradient_metrics/subject_gradient_metrics.csv')
sq = pd.read_csv(LAYOUT.QUALITY / 'data/subject_signal_quality.csv')
assert list(br.Subject) == list(s91.Subject) == list(sq.Subject) == list(cov.participant_id) and (s4.Group.values == grp).all()
yrs = training_years(cov)

# ---- 2. group comparison of GSA (Age, Sex, MeanFD), and without MeanFD (motion may lie on the causal path)
ENG = ts.Engine(ts.design(cov), grp)
G = pd.DataFrame([dict(Metric=m, **ENG.analyse(gs[m].values)) for m in ('GSA_bandpass', 'GSA_detrended')]).assign(Family='GSA')
G, _ = ts.finalize_family(G, 'Family')
G.to_csv(OUT / 'GSA_group_stats_3group.csv', index=False)
X0 = ts.design(cov)[:, :5]                               # intercept, TC, BB, Age, Sex
G0 = pd.DataFrame([dict(Metric=m, **ts.Engine(X0, grp).analyse(gs[m].values)) for m in ('GSA_bandpass', 'GSA_detrended')])
G0.to_csv(OUT / 'GSA_group_stats_noMeanFD_3group.csv', index=False)

# ---- 3. what GSA tracks
cor = []
for name, v in [('MeanFD', cov.MeanFD.values), ('n_censored', sq.n_censored.values), ('cortex_median_amplitude', sq.cortex_median_amplitude.values),
                ('G2_RobustRange', br.A_G2_RobustRange.values), ('Visual2_Segregation', br.Segregation_Visual2.values)]:
    for g in ('All', 'HC', 'TC', 'BB'):
        m = np.ones(len(grp), bool) if g == 'All' else grp == g
        r, p = stats.spearmanr(gs.GSA_bandpass.values[m], v[m])
        cor.append(dict(Variable=name, Sample=g, N=int(m.sum()), rho=r, P=p))
pd.DataFrame(cor).to_csv(OUT / 'GSA_correlations.csv', index=False)

# ---- 4. key results with GSA as covariate
key = {'G2 range': br.A_G2_RobustRange.values, 'lambda2': s91.Eigenvalue_G2.values, 'Short-axis variance': s4.MinorVariance.values,
       'VIS2 mean G2': br.A_Visual2_MeanG2.values, 'VIS2 segregation': br.Segregation_Visual2.values,
       'VIS2-SMN FC': br.Visual2_Somatomotor.values, 'VIS2-CON FC': br.Visual2_CinguloOpercular.values, 'G1 range': s4.G1_RobustRange.values}
cov2 = cov.copy(); cov2['GSA'] = gs.GSA_bandpass.values
ENG2 = ts.Engine(ts.design(cov2, 'GSA'), grp)
rows = []
for name, y in key.items():
    a, b = ENG.analyse(y), ENG2.analyse(y)
    rows.append(dict(Metric=name, Omnibus_P=a['Omnibus_P'], Omnibus_P_GSAcov=b['Omnibus_P'],
                     **{f'{k}_{c}{suf}': r[f'{k}_{c}'] for k in ('TCvsHC', 'BBvsHC', 'TCvsBB') for suf, r in (('', a), ('_GSAcov', b))
                        for c in ('Beta', 'P_Holm', 'P_TOST')}))
pd.DataFrame(rows).to_csv(OUT / 'key_results_with_GSA_covariate_3group.csv', index=False)


# ---- 5. training dose with GSA as covariate (raw Sex coding 1/2 changes only the intercept)
def dose(y, m, extra=None):
    d = np.column_stack([np.ones(m.sum()), yrs[m], cov.Age.values[m], cov.Sex.values[m], cov.MeanFD.values[m]] +
                        ([extra[m]] if extra is not None else []))
    r = ts.hc3(y[m], d)
    return int(m.sum()), r['Beta'] * d[:, 1].std(ddof=1) / y[m].std(ddof=1), r['P']


D = []
for g in ('TC', 'BB'):
    m = grp == g
    for name in ('G2 range', 'lambda2', 'VIS2 mean G2'):
        n, bs, p = dose(key[name], m)
        _, bs2, p2 = dose(key[name], m, gs.GSA_bandpass.values)
        D.append(dict(Group=g, Metric=name, N=n, Beta_std=bs, P=p, Beta_std_GSAcov=bs2, P_GSAcov=p2))
    r, p = stats.spearmanr(yrs[m], gs.GSA_bandpass.values[m])
    D.append(dict(Group=g, Metric='GSA vs training years (Spearman)', N=int(m.sum()), Beta_std=r, P=p))
pd.DataFrame(D).to_csv(OUT / 'training_dose_with_GSA_covariate.csv', index=False)
print(G[['Metric', 'Omnibus_P', 'TCvsHC_g', 'BBvsHC_g', 'TCvsBB_g']].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
