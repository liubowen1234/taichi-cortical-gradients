"""Step 6.7 - atlas robustness: the gradient analysis repeated with Schaefer-400 and Schaefer-600 cortical parcels.

Inputs: matlab run_all_gradients.m output in 06_robustness/schaefer<N>/ (same MATLAB steps, HC reference).
Families: resources/test_families/schaefer<N>/ (span and dispersion, manifold geometry, parcel G1 / G2 / eccentricity,
Yeo-7 and Yeo-17 network means, dispersion and pair distances). Networks use the Yeo labels of the Schaefer atlas
(the visual network replaces CAB-NP Visual2). G3 is used with the sign of the aligned component.
Statistics: common/stats_engine.py.

Outputs (06_robustness/schaefer<N>/stats_three_group/): <table>_3group.csv, family_summary_3group.csv,
key_metrics_classification.csv
Usage: python pipeline/06_robustness/s07_schaefer_statistics.py 400   (and 600)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from common.config import LAYOUT, RESOURCES, mkdir
from common import stats_engine as ts
from common.data import covariates, row_ids, parcel_matrix

NPc = int(sys.argv[1])
ROOT = LAYOUT.SCHAEFER[NPc]
OUT = mkdir(ROOT / 'stats_three_group')
cov = covariates()
grp = cov.Group.values
N = len(cov)
S4 = pd.read_csv(ROOT / 'Step4_gradient_metrics/subject_gradient_metrics.csv')
LONG = pd.read_csv(ROOT / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv')
atlas = pd.read_csv(RESOURCES / f'atlas/atlas-Schaefer{NPc}_cortex.tsv', sep='\t').sort_values('index').reset_index(drop=True)
IDX = row_ids(N)
assert list(S4.Subject) == IDX and (S4.Group.values == grp).all()
G1, G2, G3a, ECC = (parcel_matrix(LONG, c, IDX) for c in ('G1', 'G2', 'G3_ReferenceAligned', 'Eccentricity2D'))
XYZ = np.stack([G1, G2, G3a], 2)
g3sign = 1
yeo7 = atlas.network_label.astype(str).values
yeo17 = atlas.network_label_17network.astype(str).values


def net_mask(labels, name):
    m = labels == name
    assert m.any(), name
    return m


def network_value(labels, a, metric):
    m = net_mask(labels, a)
    if metric == 'MeanG1': return G1[:, m].mean(1)
    if metric == 'MeanG2': return G2[:, m].mean(1)
    if metric == 'MeanG3': return g3sign * G3a[:, m].mean(1)
    if metric == 'WithinDispersion2D':
        Q = np.stack([G1[:, m], G2[:, m]], 2); c = Q.mean(1, keepdims=True)
        return np.sqrt(((Q - c) ** 2).sum(2)).mean(1)
    if metric == 'WithinDispersionSS3D':
        Q = XYZ[:, m, :]; c = Q.mean(1, keepdims=True)
        return ((Q - c) ** 2).sum(2).sum(1)
    raise KeyError(metric)


def pair_value(labels, a, b, metric):
    ma, mb = net_mask(labels, a), net_mask(labels, b)
    if metric == 'CentroidDistance3D':
        return np.sqrt(((XYZ[:, mb, :].mean(1) - XYZ[:, ma, :].mean(1)) ** 2).sum(1))
    d1 = G1[:, mb].mean(1) - G1[:, ma].mean(1)
    d2 = G2[:, mb].mean(1) - G2[:, ma].mean(1)
    if metric == 'DeltaG1_BminusA': return d1
    if metric == 'DeltaG2_BminusA': return d2
    if metric == 'CentroidDistance2D': return np.sqrt(d1 ** 2 + d2 ** 2)
    raise KeyError(metric)


def row_values(name, r):
    if name.startswith('Step5_parcel_'):
        return {'G1': G1, 'G2': G2, 'Eccentricity': ECC}[name.split('_')[2]][:, int(r.Parcel) - 1]
    if 'Metric' in r.index and 'NetworkA' not in r.index and 'Network' not in r.index:
        return S4[r.Metric].values.astype(float)
    labels = yeo7 if 'Yeo7' in name else yeo17
    if isinstance(r.NetworkB, str) and r.NetworkB not in ('', 'nan'):
        return pair_value(labels, r.NetworkA, r.NetworkB, r.MetricType)
    return network_value(labels, r.NetworkA, r.MetricType)


ENG = ts.Engine(ts.design(cov), grp)
fam_rows, tabs = [], {}
for f in sorted((RESOURCES / f'test_families/schaefer{NPc}').glob('*.csv')):
    D = pd.read_csv(f, keep_default_na=False, na_values=[''])
    rows = [{**r.to_dict(), 'N': N, **ENG.analyse(row_values(f.name, r))} for _, r in D.iterrows()]
    O, summ = ts.finalize_family(pd.DataFrame(rows), 'Family', f.stem)
    fam_rows += [dict(Table=f.name, **d) for d in summ]
    nm = f'{f.stem}_3group.csv'
    O.to_csv(OUT / nm, index=False)
    tabs[nm] = O
pd.DataFrame(fam_rows).to_csv(OUT / 'family_summary_3group.csv', index=False)
core = tabs['Step5_core_global_metrics_3group.csv'].set_index('Metric')
exp = tabs['Step5_exploratory_global_geometry_3group.csv'].set_index('Metric')
rows = [dict(Metric=nm, Omnibus_Q=r.Omnibus_Q_FDR, **{f'{k}_{s}': r[f'{k}_{s}'] for k in ('TCvsHC', 'BBvsHC', 'TCvsBB')
                                                      for s in ('g', 'P_Holm', 'P_TOST')}, Pattern=ts.classify(r))
        for nm, r in [('G2 range', core.loc['G2_RobustRange']), ('G1 range', core.loc['G1_RobustRange']),
                      ('MinorVariance (2D, rotation-invariant)', exp.loc['MinorVariance'])]]
pd.DataFrame(rows).to_csv(OUT / 'key_metrics_classification.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.3g}'))
