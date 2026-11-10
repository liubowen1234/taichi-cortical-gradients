"""Step 6.1 - robustness of the group effects to head motion.

The models are refitted in four motion subsets (common/motion.py): drop the 10% / 20% highest-FD participants of each
group; drop athletes above the largest non-athlete mean FD; and a motion-matched subset. Gradients are not recomputed,
because every subject is aligned to a fixed reference.
In each subset the original families are rebuilt in full for the key tables (G1 / G2 span and dispersion; step 9.1
metrics; 12 networks x mean G1 / G2 / G3; network segregation; network PC; VIS2 cross-network FC) and the pre-specified
pattern is assigned to the primary metrics. Within-group training dose (G2 span) per subset. Within-group Spearman
correlations of mean FD with the key metrics (full sample).

Outputs (06_robustness/motion_subsets/): subset_motion_summary_3group.csv, key_metrics_by_subset_3group.csv,
training_dose_G2range_by_subset_3group.csv, within_group_MeanFD_metric_correlations_3group.csv
Usage: python pipeline/06_robustness/s01_motion_subsets.py
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy import stats
from common.config import LAYOUT, mkdir
from common import stats_engine as ts
from common.data import covariates, training_years, cabnp_mapping, row_ids, parcel_matrix, NETS
from common.motion import motion_subsets

OUT = mkdir(LAYOUT.MOTION)
cov = covariates()
grp = cov.Group.values
NS = len(cov)
fd = cov.MeanFD.values
yrs = training_years(cov)
net = cabnp_mapping().CABNP_Network.values
IDS = row_ids(NS)

# ---- subject-level data
A4 = pd.read_csv(LAYOUT.GRAD_A / 'Step4_gradient_metrics/subject_gradient_metrics.csv')
A91 = pd.read_csv(LAYOUT.GRAD_A / 'step_9.1/step9_1_additional_subject_metrics.csv')
LA = pd.read_csv(LAYOUT.GRAD_A / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv')
g3sign = json.load(open(LAYOUT.GRAD_A / 'step_9.2/step9_2_G3_orientation.json'))['sign_flip']
G1A, G2A = parcel_matrix(LA, 'G1', IDS), parcel_matrix(LA, 'G2', IDS)
G3A = g3sign * parcel_matrix(LA, 'G3_ReferenceAligned', IDS)
B4 = pd.read_csv(LAYOUT.GRAD_B / 'Step4_subject_gradient_metrics.csv')
SEG = pd.read_csv(LAYOUT.SEG / 'subject_network_segregation.csv')
PCp = pd.read_csv(LAYOUT.SEG / 'subject_parcel_PC.csv').drop(columns='Subject').values
NF = pd.read_csv(LAYOUT.VIS2 / 'subject_Visual2_networkFC_and_bridge_inputs.csv')
for T in (A91, SEG, NF):
    assert list(T.iloc[:, 0]) == list(cov.participant_id)

FAMILIES = {
    'Core_Global': {m: A4[m].values for m in ['G1_RobustRange', 'G2_RobustRange', 'GlobalDispersionMean']},
    'Step9_1': {'TemplateSimilarityCosine_G1': A91.TemplateSimilarityCosine_G1.values,
                'TemplateSimilarityCosine_G2': A91.TemplateSimilarityCosine_G2.values, 'MeanFCdegree': A91.MeanFCdegree.values,
                'Eigenvalue_G1': A91.Eigenvalue_G1.values, 'Eigenvalue_G2': A91.Eigenvalue_G2.values},
    'Network_MeanG123': {f'{n}::{m}': G[:, net == n].mean(1) for n in NETS for m, G in [('MeanG1', G1A), ('MeanG2', G2A), ('MeanG3', G3A)]},
    'Network_Segregation': {n: SEG[f'{n}_Segregation'].values for n in NETS},
    'Network_PC': {n: PCp[:, net == n].mean(1) for n in NETS},
    'Visual2_crossFC': {k: NF[k].values for k in ['Visual2_Somatomotor', 'Visual2_CinguloOpercular', 'Visual2_Auditory', 'Visual2_DorsalAttention']},
    'SchemeB_Core': {'B_G2_RobustRange': B4.G2_RobustRange.values, 'B_G1_RobustRange': B4.G1_RobustRange.values,
                     'B_GlobalDispersionMean': B4.GlobalDispersionMean.values},
}
KEY = [('Core_Global', 'G2_RobustRange', 'G2 range'), ('Step9_1', 'Eigenvalue_G2', 'λ2'), ('Network_MeanG123', 'Visual2::MeanG2', 'VIS2 mean G2'),
       ('Core_Global', 'G1_RobustRange', 'G1 range'), ('Network_Segregation', 'Visual2', 'VIS2 segregation'), ('Network_PC', 'Visual2', 'VIS2 PC'),
       ('Visual2_crossFC', 'Visual2_Somatomotor', 'VIS2–SMN FC'), ('Visual2_crossFC', 'Visual2_CinguloOpercular', 'VIS2–CON FC'),
       ('SchemeB_Core', 'B_G2_RobustRange', 'G2 range (scheme B)')]
PATTERN_METRICS = ('G2 range', 'λ2', 'VIS2 mean G2', 'G1 range', 'G2 range (scheme B)')

SUB = motion_subsets(cov)
srows = [dict(Subset=name, N=int(k.sum()), **{f'N_{g}': int((k & (grp == g)).sum()) for g in ('HC', 'TC', 'BB')},
              **{f'MeanFD_{g}': fd[k & (grp == g)].mean() for g in ('HC', 'TC', 'BB')},
              Welch_ANOVA_P=ts.welch_anova_p([fd[k & (grp == g)] for g in ('HC', 'TC', 'BB')]),
              Excluded=';'.join(cov.participant_id[~k])) for name, k in SUB.items()]
pd.DataFrame(srows).to_csv(OUT / 'subset_motion_summary_3group.csv', index=False)

rows, drow = [], []
for sname, k in SUB.items():
    ENG = ts.Engine(ts.design(cov[k]), grp[k])
    res = {}
    for fam, vals in FAMILIES.items():
        O = pd.DataFrame([{'Subset': sname, 'Family': fam, 'Metric': m, **ENG.analyse(np.asarray(y, float)[k])} for m, y in vals.items()])
        O, _ = ts.finalize_family(O, 'Family')
        res[fam] = O.set_index('Metric')
    for fam, met, lab in KEY:
        r = res[fam].loc[met]
        rows.append(dict(Subset=sname, Metric=lab,
                         **{c: r[c] for c in ['Omnibus_P', 'Omnibus_Q_FDR'] + [f'{p}_{s}' for p in ('TCvsHC', 'BBvsHC', 'TCvsBB')
                                                                             for s in ('g', 'P_Holm', 'P_TOST')]},
                         Pattern=ts.classify(r) if lab in PATTERN_METRICS else ''))
    for g in ('TC', 'BB'):                         # training dose of G2 span within the subset
        m = k & (grp == g)
        X = np.column_stack([np.ones(m.sum()), yrs[m], cov.Age.values[m], cov.Sex.values[m] - 1.0, fd[m]])
        y = A4.G2_RobustRange.values[m]
        r = ts.hc3(y, X)
        drow.append(dict(Subset=sname, Group=g, N=int(m.sum()), Beta=r['Beta'],
                         StandardizedBeta=r['Beta'] * yrs[m].std(ddof=1) / y.std(ddof=1), P=r['P']))
pd.DataFrame(rows).to_csv(OUT / 'key_metrics_by_subset_3group.csv', index=False)
pd.DataFrame(drow).to_csv(OUT / 'training_dose_G2range_by_subset_3group.csv', index=False)

crow = []
for g in ('HC', 'TC', 'BB'):
    m = grp == g
    for fam, met, lab in KEY:
        r = stats.spearmanr(fd[m], np.asarray(FAMILIES[fam][met], float)[m])
        crow.append(dict(Group=g, Metric=lab, Spearman_rho=r.statistic, P=r.pvalue))
pd.DataFrame(crow).to_csv(OUT / 'within_group_MeanFD_metric_correlations_3group.csv', index=False)
pd.set_option('display.width', 250)
print(pd.DataFrame(srows).drop(columns='Excluded').to_string(index=False, float_format=lambda v: f'{v:.3g}'))
