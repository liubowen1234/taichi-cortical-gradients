"""Step 2.2 - three-group statistics of all gradient metrics (main analysis, non-athlete reference).

Model for every metric: Metric ~ TC + BB + Age + Sex + MeanFD (HC = reference), OLS with HC3 standard errors
(common/stats_engine.py). Multiple comparisons are controlled hierarchically:
  1. omnibus group effect (2-df Wald F), Benjamini-Hochberg FDR within each family of tests;
  2. pairwise contrasts (TC-HC, BB-HC, TC-BB) only where omnibus q < 0.05, Holm-corrected within the metric.

The families of tests are listed in resources/test_families/glasser358/ (one file = one table; a 'Family' column, where
present, splits a table into several families). Each row names the metric, e.g.
  Step5_core_global_metrics       G1 span, G2 span, global dispersion (G1-G2 plane)
  step9_1_additional_global_stats similarity to the reference, mean FC degree, lambda1, lambda2
  step9_2_eigenvalue_3axes_stats  lambda1-lambda3
  step9_3_tail_stats              5th and 95th percentile of G2 (and G1)
  Step5_exploratory_global_geometry  manifold descriptors (anisotropy, convex hull area, pairwise distance, total / major /
                                  minor-axis variance)
  step9_2_network_MeanG123_stats  12 CAB-NP networks x mean G1 / G2 / G3
  step9_2_range_3axes_stats       G1-G3 span (the family used to test G3)
  Step5_parcel_G2_358             parcel-wise G2 (358 tests)
  Step5_1_CABNP12_* / Step5_Yeo7_* network means, within-network dispersion and network-pair distances

Outputs (<scheme folder>/stats_three_group[_plus<covariate>]/): one <table>_3group.csv per table,
family_summary_3group.csv, primary_metrics_classification.csv (pre-specified pattern of G2 span, lambda2, VIS2 mean G2, G1 span).

Usage: python pipeline/02_gradients/s02_group_statistics.py [main|noGSR] [--covar eTIV]
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from common.config import LAYOUT, RESOURCES, mkdir
from common import stats_engine as ts
from common.data import covariates, glasser_atlas, cabnp_mapping, row_ids, parcel_matrix

args = sys.argv[1:]
which = args[0] if args and not args[0].startswith('--') else 'main'
ROOT = {'main': LAYOUT.GRAD_A, 'noGSR': LAYOUT.GRAD_NOGSR}[which]
EXTRA = args[args.index('--covar') + 1] if '--covar' in args else None
OUT = mkdir(ROOT / ('stats_three_group' + (f'_plus{EXTRA}' if EXTRA else '')))

cov = covariates()
S4 = pd.read_csv(ROOT / 'Step4_gradient_metrics/subject_gradient_metrics.csv')
S91 = pd.read_csv(ROOT / 'step_9.1/step9_1_additional_subject_metrics.csv')
S92 = pd.read_csv(ROOT / 'step_9.2/step9_2_G3_subject_metrics.csv')
LONG = pd.read_csv(ROOT / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv')
yeo = glasser_atlas().community_yeo.str.strip().values
cabl = cabnp_mapping().CABNP_Network.values
g3sign = json.load(open(ROOT / 'step_9.2/step9_2_G3_orientation.json'))['sign_flip']

N = len(cov)
IDX = row_ids(N)
assert list(S4.Subject) == IDX and list(S91.Subject) == list(cov.participant_id) and list(S92.Subject) == list(cov.participant_id)
assert (S4.Group.values == cov.Group.values).all()
grp = cov.Group.values
ENG = ts.Engine(ts.design(cov, EXTRA), grp)
MODEL = 'Group(HC ref) + Age + Sex + MeanFD' + (f' + {EXTRA}' if EXTRA else '')

G1, G2, G3a, ECC = (parcel_matrix(LONG, c, IDX) for c in ('G1', 'G2', 'G3_ReferenceAligned', 'Eccentricity2D'))
XYZ = np.stack([G1, G2, G3a], 2)


# ---------------------------------------------------------------- subject values for one row of a family table
def subject_metric(name):
    for T in (S4, S91, S92):
        if name in T.columns:
            return T[name].values.astype(float)
    raise KeyError(name)


def net_mask(labels, name):
    m = labels == name
    assert m.any(), name
    return m


def network_value(labels, a, metric):
    m = net_mask(labels, a)
    if metric == 'MeanG1': return G1[:, m].mean(1)
    if metric == 'MeanG2': return G2[:, m].mean(1)
    if metric == 'MeanG3': return g3sign * G3a[:, m].mean(1)
    if metric == 'WithinDispersion2D':                 # mean distance of the network's parcels from their centroid
        Q = np.stack([G1[:, m], G2[:, m]], 2); c = Q.mean(1, keepdims=True)
        return np.sqrt(((Q - c) ** 2).sum(2)).mean(1)
    if metric == 'WithinDispersionSS3D':               # sum of squared distances in (G1, G2, G3)
        Q = XYZ[:, m, :]; c = Q.mean(1, keepdims=True)
        return ((Q - c) ** 2).sum(2).sum(1)
    raise KeyError(metric)


def pair_value(labels, a, b, metric):
    ma, mb = net_mask(labels, a), net_mask(labels, b)
    if metric == 'CentroidDistance3D':
        ca, cb = XYZ[:, ma, :].mean(1), XYZ[:, mb, :].mean(1)
        return np.sqrt(((cb - ca) ** 2).sum(1))
    d1 = G1[:, mb].mean(1) - G1[:, ma].mean(1)
    d2 = G2[:, mb].mean(1) - G2[:, ma].mean(1)
    if metric == 'DeltaG1_BminusA': return d1
    if metric == 'DeltaG2_BminusA': return d2
    if metric == 'CentroidDistance2D': return np.sqrt(d1 ** 2 + d2 ** 2)
    raise KeyError(metric)


def row_values(table_name, r):
    if table_name.startswith('Step5_parcel_'):        # e.g. Step5_parcel_G2_358: one test per parcel
        return {'G1': G1, 'G2': G2, 'Eccentricity': ECC}[table_name.split('_')[2]][:, int(r.Parcel) - 1]
    if 'Metric' in r.index and 'NetworkA' not in r.index and 'Network' not in r.index:
        return subject_metric(r.Metric)
    if 'Network' in r.index:
        return network_value(cabl, r.Network, r.Metric)
    labels = yeo if 'Yeo7' in table_name else cabl
    if isinstance(r.NetworkB, str) and r.NetworkB not in ('', 'nan'):
        return pair_value(labels, r.NetworkA, r.NetworkB, r.MetricType)
    return network_value(labels, r.NetworkA, r.MetricType)


# ---------------------------------------------------------------- run every family
fam_rows, all_tables = [], {}
for f in sorted((RESOURCES / 'test_families/glasser358').glob('*.csv')):
    D = pd.read_csv(f, keep_default_na=False, na_values=[''])
    rows = [{**r.to_dict(), 'Model': MODEL, 'N': N, **ENG.analyse(row_values(f.name, r))} for _, r in D.iterrows()]
    O, summ = ts.finalize_family(pd.DataFrame(rows), 'Family', f.stem)
    fam_rows += [dict(Table=f.name, **d) for d in summ]
    name = f'{f.stem}_3group.csv'
    O.to_csv(OUT / name, index=False)
    all_tables[name] = O
pd.DataFrame(fam_rows).to_csv(OUT / 'family_summary_3group.csv', index=False)


# ---------------------------------------------------------------- pre-specified classification of the primary metrics
def pick(table, **kw):
    T = all_tables[table]
    m = np.ones(len(T), bool)
    for k, v in kw.items():
        m &= (T[k].astype(str) == str(v)).values
    assert m.sum() == 1, (table, kw)
    return T[m].iloc[0]


PRIMARY = [('G2 range (VS)', pick('Step5_core_global_metrics_3group.csv', Metric='G2_RobustRange')),
           ('lambda2', pick('step9_1_additional_global_stats_3group.csv', Metric='Eigenvalue_G2')),
           ('VIS2 mean G2', pick('step9_2_network_MeanG123_stats_3group.csv', Network='Visual2', Metric='MeanG2')),
           ('G1 range (SA, control axis)', pick('Step5_core_global_metrics_3group.csv', Metric='G1_RobustRange'))]
rows = [dict(Metric=lab, Omnibus_F=r.Omnibus_F, Omnibus_P=r.Omnibus_P, Omnibus_Q=r.Omnibus_Q_FDR,
             **{f'{k}_{s}': r[f'{k}_{s}'] for k in ('TCvsHC', 'BBvsHC', 'TCvsBB') for s in ('g', 'P_Holm', 'P_TOST', 'BF01')},
             Pattern=ts.classify(r)) for lab, r in PRIMARY]
C = pd.DataFrame(rows)
C.to_csv(OUT / 'primary_metrics_classification.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(C.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
print(pd.DataFrame(fam_rows).to_string(index=False))
