"""Step 5 - training dose within each athlete group.

Within-group model (Tai Chi and basketball separately):
    Metric ~ TrainYears + Age + Sex + MeanFD, OLS with HC3 standard errors (t test, df = n - 5)
Family of seven metrics per group (BH-FDR): G1 / G2 / G3 span, lambda2, VIS2 mean G2, VIS2 segregation, VIS2-SMN FC.
Standardised beta = beta x SD(years) / SD(metric). Partial Spearman = Pearson r of the ranks of metric and years after
regressing the covariates out of each (robustness check). Gradients: main analysis (A); key metrics also with the
template alignment (B).
Further analyses
  - G2 span: leave-one-out range of P (each participant left out once).
  - Basketball: log10 cumulative training hours (years x weekly hours x 52) as an alternative predictor.
  - Pooled athletes: Metric ~ TrainYears * Sport + Age + Sex + MeanFD; the interaction tests whether the slopes differ
    (BH-FDR over the seven metrics).
  - Dose maps: parcel-wise partial r of G2 with training years (Age, Sex, MeanFD partialled out) in each athlete group;
    spatial correlation (spin test, raw and with the reference G2 axis removed) with the group-effect maps.
  - Training years vs mean FD within each athlete group (Spearman, Pearson).
  - Exploratory: the two VIS2 connections that differed between basketball players and non-athletes (VIS2-CON, VIS2-VIS1),
    same within-group model, not part of the seven-metric family.

Outputs (05_training_dose/): dose_within_group_3group.csv, dose_G2range_leave_one_out.csv, dose_BB_cumulative_hours.csv,
dose_pooled_athletes_interaction.csv, parcel_G2_dose_maps_TC_BB.csv, dose_map_spatial_correlations.csv,
trainyears_vs_meanFD_3group.csv, dose_exploratory_VIS2_CON_VIS1.csv
Usage: python pipeline/05_training_dose/s01_training_dose.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy import stats
from common.config import LAYOUT, mkdir
from common import stats_engine as ts
from common.data import covariates, training, training_years, cabnp_mapping, row_ids, parcel_matrix, hc_reference_axis, \
    spin_permutations

OUT = mkdir(LAYOUT.DOSE)
cov = covariates()
grp = cov.Group.values
NS = len(cov)
yrs = training_years(cov)
tr = training().set_index('participant_id')
wk = np.array([tr.WeeklyHours.get(s, np.nan) for s in cov.participant_id], float)
cab = cabnp_mapping()
v2 = cab.CABNP_Network.values == 'Visual2'
IDS = row_ids(NS)

# ---- subject metrics (rows in the analysis order)
A4 = pd.read_csv(LAYOUT.GRAD_A / 'Step4_gradient_metrics/subject_gradient_metrics.csv')
A91 = pd.read_csv(LAYOUT.GRAD_A / 'step_9.1/step9_1_additional_subject_metrics.csv')
A92 = pd.read_csv(LAYOUT.GRAD_A / 'step_9.2/step9_2_G3_subject_metrics.csv')
G2A = parcel_matrix(pd.read_csv(LAYOUT.GRAD_A / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv'), 'G2', IDS)
B4 = pd.read_csv(LAYOUT.GRAD_B / 'Step4_subject_gradient_metrics.csv')
G2B = parcel_matrix(pd.read_csv(LAYOUT.GRAD_B / 'Step4_parcel_gradients_long.csv'), 'G2', IDS)
SEG = pd.read_csv(LAYOUT.SEG / 'subject_network_segregation.csv')
NF = pd.read_csv(LAYOUT.VIS2 / 'subject_Visual2_networkFC_and_bridge_inputs.csv')
AP = pd.read_csv(LAYOUT.VIS2 / 'subject_Visual2_all_partners_FC.csv')
assert list(SEG.Subject) == list(cov.participant_id) == list(NF.Subject) == list(A91.Subject) == list(AP.Subject)
METRICS = {'A': [('G1 range', A4.G1_RobustRange.values), ('G2 range', A4.G2_RobustRange.values), ('G3 range', A92.G3_RobustRange.values),
                 ('λ2', A91.Eigenvalue_G2.values), ('VIS2 mean G2', G2A[:, v2].mean(1)),
                 ('VIS2 segregation', SEG.Visual2_Segregation.values), ('VIS2–SMN FC', NF.Visual2_Somatomotor.values)],
           'B': [('G1 range', B4.G1_RobustRange.values), ('G2 range', B4.G2_RobustRange.values), ('λ2', A91.Eigenvalue_G2.values),
                 ('VIS2 mean G2', G2B[:, v2].mean(1))]}


def covariates_matrix(m):
    """intercept, Age, Sex (0/1), MeanFD for the rows in mask m."""
    return np.column_stack([np.ones(m.sum()), cov.Age.values[m], cov.Sex.values[m] - 1.0, cov.MeanFD.values[m]]).astype(float)


def with_years(C, x):
    """Insert the predictor x as the second column (after the intercept)."""
    return np.column_stack([C[:, :1], x, C[:, 1:]])


def partial(y, x, C, rank=False):
    """Partial correlation of y and x given C (Pearson, or Spearman = on ranks)."""
    if rank:
        y, x = stats.rankdata(y), stats.rankdata(x)
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    r = np.corrcoef(ry, rx)[0, 1]
    df = len(y) - C.shape[1] - 1
    return r, 2 * stats.t.sf(abs(r * np.sqrt(df / (1 - r ** 2))), df)


# ---- within-group dose, seven-metric family
rows = []
for scheme, mets in METRICS.items():
    for g in ('TC', 'BB'):
        m = grp == g
        C = covariates_matrix(m)
        x = yrs[m]
        X = with_years(C, x)
        part = []
        for name, y in mets:
            y = np.asarray(y, float)[m]
            r = ts.hc3(y, X)
            rs, ps = partial(y, x, C, rank=True)
            part.append(dict(Scheme=scheme, Group=g, Metric=name, N=int(m.sum()), **r,
                             StandardizedBeta=r['Beta'] * x.std(ddof=1) / y.std(ddof=1), partial_Spearman=rs, partial_Spearman_P=ps))
        for d, q in zip(part, ts.bh([d['P'] for d in part])):
            d['Q_FDR_family'] = q
        rows += part
pd.DataFrame(rows).to_csv(OUT / 'dose_within_group_3group.csv', index=False)

# ---- leave-one-out (G2 span)
loo = []
for scheme in ('A', 'B'):
    y_all = np.asarray(METRICS[scheme][1][1], float)
    for g in ('TC', 'BB'):
        idx = np.where(grp == g)[0]
        ps = []
        for j in idx:
            mm = np.zeros(NS, bool); mm[idx] = True; mm[j] = False
            ps.append(ts.hc3(y_all[mm], with_years(covariates_matrix(mm), yrs[mm]))['P'])
        loo.append(dict(Scheme=scheme, Group=g, Metric='G2 range', LOO_P_min=min(ps), LOO_P_max=max(ps)))
pd.DataFrame(loo).to_csv(OUT / 'dose_G2range_leave_one_out.csv', index=False)

# ---- basketball: cumulative hours
mb = grp == 'BB'
cum = np.log10(yrs[mb] * wk[mb] * 52)
C = covariates_matrix(mb)
cumrows = []
for name, y in METRICS['A']:
    y = np.asarray(y, float)[mb]
    r = ts.hc3(y, with_years(C, cum))
    rs, ps = partial(y, cum, C, rank=True)
    cumrows.append(dict(Group='BB', Predictor='log10 cumulative hours', Metric=name, **r, partial_Spearman=rs, partial_Spearman_P=ps))
CUM = pd.DataFrame(cumrows)
CUM['Q_FDR_family'] = ts.bh(CUM.P.values)
CUM.to_csv(OUT / 'dose_BB_cumulative_hours.csv', index=False)

# ---- pooled athletes: does the slope differ between sports?
ath = grp != 'HC'
sport = (grp[ath] == 'BB').astype(float)
C = covariates_matrix(ath)
x = yrs[ath]
prow = []
for scheme, mets in METRICS.items():
    for name, y in mets:
        y = np.asarray(y, float)[ath]
        X = np.column_stack([C[:, :1], x, sport, x * sport, C[:, 1:]])
        ri = ts.hc3(y, X, 3)                                              # interaction = slope(BB) - slope(TC)
        rc = ts.hc3(y, np.column_stack([C[:, :1], x, sport, C[:, 1:]]), 1)   # common slope
        prow.append(dict(Scheme=scheme, Metric=name, N=int(ath.sum()), Slope_TC=ts.hc3(y, X, 1)['Beta'],
                         Slope_BB=ts.hc3(y, X, 1)['Beta'] + ri['Beta'], Interaction_Beta=ri['Beta'], Interaction_P=ri['P'],
                         Common_slope_Beta=rc['Beta'], Common_slope_P=rc['P']))
POOL = pd.DataFrame(prow)
for s in POOL.Scheme.unique():
    m = POOL.Scheme == s
    POOL.loc[m, 'Interaction_Q_FDR'] = ts.bh(POOL.loc[m, 'Interaction_P'].values)
POOL.to_csv(OUT / 'dose_pooled_athletes_interaction.csv', index=False)

# ---- dose maps and their spatial correspondence
PERM = spin_permutations()
AXIS = hc_reference_axis()
Hm = np.column_stack([np.ones(358), AXIS])
Hm = np.eye(358) - Hm @ np.linalg.pinv(Hm)


def spin(x, y, axis_removed=False):
    if axis_removed:
        xr, yr = Hm @ x, Hm @ y
        r = np.corrcoef(xr, yr)[0, 1]
        Xn = x[PERM] @ Hm.T; yz = (yr - yr.mean()) / yr.std()
    else:
        r = np.corrcoef(x, y)[0, 1]
        Xn = x[PERM]; yz = (y - y.mean()) / y.std()
    Xn = (Xn - Xn.mean(1, keepdims=True)) / Xn.std(1, keepdims=True)
    null = (Xn * yz).mean(1)
    return r, (1 + np.sum(np.abs(null) >= abs(r))) / (1 + len(null))


dmap = {}
for g in ('TC', 'BB'):
    m = grp == g
    C = covariates_matrix(m)
    dmap[g] = np.array([partial(G2A[m, p], yrs[m], C)[0] for p in range(358)])
TA = pd.read_csv(LAYOUT.GRAD_A / 'stats_three_group/Step5_parcel_G2_358_3group.csv').sort_values('Parcel')
pd.DataFrame({'Parcel358': np.arange(1, 359), 'Network': cab.CABNP_Network.values, 'TC_dose_partial_r': dmap['TC'],
              'BB_dose_partial_r': dmap['BB']}).to_csv(OUT / 'parcel_G2_dose_maps_TC_BB.csv', index=False)
srows = []
for a, b, lab in [(dmap['TC'], TA.TCvsHC_g.values, 'TC dose map vs TC–HC g map'), (dmap['BB'], TA.BBvsHC_g.values, 'BB dose map vs BB–HC g map'),
                  (dmap['BB'], dmap['TC'], 'BB dose map vs TC dose map'), (dmap['BB'], TA.TCvsHC_g.values, 'BB dose map vs TC–HC g map')]:
    r, p = spin(a, b)
    r2, p2 = spin(a, b, True)
    srows.append(dict(Comparison=lab, r=r, p_spin=p, r_G2axis_removed=r2, p_spin_G2axis_removed=p2))
pd.DataFrame(srows).to_csv(OUT / 'dose_map_spatial_correlations.csv', index=False)

# ---- training years vs mean FD
rows = []
for g in ('TC', 'BB'):
    m = grp == g
    rs, ps = stats.spearmanr(yrs[m], cov.MeanFD.values[m])
    rp, pp = stats.pearsonr(yrs[m], cov.MeanFD.values[m])
    rows.append(dict(Group=g, N=int(m.sum()), Spearman_rho=rs, Spearman_P=ps, Pearson_r=rp, Pearson_P=pp))
pd.DataFrame(rows).to_csv(OUT / 'trainyears_vs_meanFD_3group.csv', index=False)

# ---- exploratory: VIS2-CON and VIS2-VIS1 (raw Sex coding 1/2 changes only the intercept)
rows = []
for g in ('TC', 'BB'):
    m = grp == g
    X = np.column_stack([np.ones(m.sum()), yrs[m], cov.Age.values[m], cov.Sex.values[m], cov.MeanFD.values[m]])
    for name, y in [('VIS2–SMN FC (check)', NF.Visual2_Somatomotor.values), ('VIS2–CON FC', NF.Visual2_CinguloOpercular.values),
                    ('VIS2–VIS1 FC', AP['Visual2_Visual1'].values)]:
        r = ts.hc3(y[m], X)
        rows.append(dict(Group=g, Metric=name, N=int(m.sum()), StandardizedBeta=r['Beta'] * yrs[m].std(ddof=1) / y[m].std(ddof=1),
                         T=r['T'], P=r['P']))
pd.DataFrame(rows).to_csv(OUT / 'dose_exploratory_VIS2_CON_VIS1.csv', index=False)

pd.set_option('display.width', 250)
print(pd.DataFrame([r for r in pd.read_csv(OUT / 'dose_within_group_3group.csv').to_dict('records') if r['Scheme'] == 'A'])
      [['Group', 'Metric', 'StandardizedBeta', 'P', 'Q_FDR_family']].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
