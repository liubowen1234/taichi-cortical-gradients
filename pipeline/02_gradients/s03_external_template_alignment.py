"""Step 2.3 - robustness of the alignment reference: alignment to the external BrainSpace template.

The main analysis aligns 3 components to a non-athlete reference. Here each subject's first 2 raw components (MATLAB
Step 1) are aligned to the 2 group-level FC gradients that ship with BrainSpace (conte69 32k template, averaged within
the Glasser-358 parcels). The rotation is solved on column-z-scored components and applied to the raw-scale components;
G1 / G2 are then identified and signed with the same Yeo-7 anchor rule as the main analysis.
Each subject is aligned to the fixed template on its own, so no group contributes to the coordinate system.

Statistics: the families of the main analysis that are defined in two dimensions (span, manifold geometry, parcels,
networks, step 9.1) with the three-group engine.

Outputs (02_gradients/glasser358_external_template/)
  template_reference_358.csv            template G1 / G2 per Glasser parcel
  Step4_subject_gradient_metrics.csv    subject metrics (same definitions as MATLAB Step 4)
  Step4_parcel_gradients_long.csv       parcel G1, G2 and eccentricity
  stats_three_group[_plus<cov>]/        Step5_<family>_3group.csv, family_summary_3group.csv, primary_metrics_classification.csv

Usage: python pipeline/02_gradients/s03_external_template_alignment.py [--covar eTIV]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import h5py
from common.config import CFG, LAYOUT, mkdir
from common import stats_engine as ts, gradlib as gl, figlib as fl
from common.data import covariates, glasser_atlas, cabnp_mapping, row_ids, NETS

EXTRA = sys.argv[sys.argv.index('--covar') + 1] if '--covar' in sys.argv else None
OUT = mkdir(LAYOUT.GRAD_B)
ST = mkdir(OUT / ('stats_three_group' + (f'_plus{EXTRA}' if EXTRA else '')))

atlas = glasser_atlas()
yeo = atlas.community_yeo.str.strip().str.lower().values
yeo_names = list(pd.unique(yeo))
cab_labels = cabnp_mapping().CABNP_Network.values
cov = covariates()
N = len(cov); grp = cov.Group.values
SUBJ = row_ids(N)

# ---- template: BrainSpace conte69 FC gradients 1 and 2, mean over the vertices of each Glasser parcel
lab_dir = CFG.path('cabnp_dir') / 'SeparateHemispheres'
iL, iR = fl.glasser358_vertex_index(lab_dir / 'cortex_ColeAnticevic_NetPartition_GlasserParcels_L.label.gii',
                                    lab_dir / 'cortex_ColeAnticevic_NetPartition_GlasserParcels_R.label.gii')
vidx = np.concatenate([iL, iR])                   # 64,984 vertices (left then right, as in the template files) -> parcel
tdir = CFG.path('brainspace_matlab') / 'datasets/template_gradients'
T2 = np.full((358, 2), np.nan)
for j, nm in enumerate(['conte69_32k_fc_gradient1', 'conte69_32k_fc_gradient2']):
    v = np.loadtxt(tdir / f'{nm}.csv').ravel()
    assert len(v) == 64984
    for p in range(358):
        T2[p, j] = np.nanmean(v[vidx == p])
assert np.isfinite(T2).all()
pd.DataFrame({'parcel_index': np.arange(1, 359), 'FC_G1_template': T2[:, 0], 'FC_G2_template': T2[:, 1]}) \
    .to_csv(OUT / 'template_reference_358.csv', index=False)

# ---- align, orient, subject metrics
with h5py.File(LAYOUT.GRAD_A / 'Step1_raw_gradients/BrainSpace_formal_raw_gradients.mat', 'r') as h:
    RAW = h['gradients_all'][:].transpose(0, 2, 1)   # subjects x 358 x 10
    LAM = h['lambda_all'][:]
s91 = pd.read_csv(LAYOUT.GRAD_A / 'step_9.1/step9_1_additional_subject_metrics.csv')
aligned = np.stack([gl.align_subject(RAW[s, :, :2], T2, zspace=True)[0] for s in range(N)])
ref_o, Gc, mapping, sign, C = gl.orient(T2, aligned, yeo)
M = gl.subject_metrics(Gc, ref_o)
M.insert(0, 'Group', grp); M.insert(0, 'Subject', SUBJ); M.insert(1, 'participant_id', cov.participant_id.values)
ecc = np.sqrt(((Gc - Gc.mean(1, keepdims=True)) ** 2).sum(2))
M.to_csv(OUT / 'Step4_subject_gradient_metrics.csv', index=False)
pd.DataFrame({'Subject': np.repeat(SUBJ, 358), 'Group': np.repeat(grp, 358), 'Parcel': np.tile(np.arange(1, 359), N),
              'G1': Gc[:, :, 0].ravel(), 'G2': Gc[:, :, 1].ravel(), 'Eccentricity2D': ecc.ravel()}) \
    .to_csv(OUT / 'Step4_parcel_gradients_long.csv', index=False)
print('component mapping', mapping, 'sign', sign)

# ---- statistics
ENG = ts.Engine(ts.design(cov, EXTRA), grp)
MODEL = 'Group(HC ref) + Age + Sex + MeanFD' + (f' + {EXTRA}' if EXTRA else '')


def family_table(values, family, keys=('Metric',)):
    rows = []
    for k, y in values.items():
        kk = k if isinstance(k, tuple) else (k,)
        rows.append({'Family': family, **dict(zip(keys, kk)), 'Model': MODEL, 'N': N, **ENG.analyse(y)})
    return ts.finalize_family(pd.DataFrame(rows), 'Family')


fam = {}
fam['Core_Global'] = {m: M[m].values for m in ['G1_RobustRange', 'G2_RobustRange', 'GlobalDispersionMean']}
fam['Exploratory_Global_Geometry'] = {m: M[m].values for m in ['AnisotropyIndex', 'ConvexHullArea', 'MeanPairwiseDistance',
                                                               'TotalVariance2D', 'MajorVariance', 'MinorVariance']}
fam['Parcel_G1'] = {p + 1: Gc[:, p, 0] for p in range(358)}
fam['Parcel_G2'] = {p + 1: Gc[:, p, 1] for p in range(358)}
fam['Parcel_Eccentricity'] = {p + 1: ecc[:, p] for p in range(358)}
ynet, ypair = gl.network_values(Gc, yeo, yeo_names, 'Yeo7')
cnet, cpair = gl.network_values(Gc, cab_labels, NETS, 'CAB')
fam['Yeo7_Network'], fam['Yeo7_Pairs'], fam['CABNP12_Network'], fam['CABNP12_NetworkPairs'] = ynet, ypair, cnet, cpair
fam['Step9_1_Additional_Global'] = {'TemplateSimilarityCosine_G1': M.TemplateSimilarityCosine_G1.values,
                                    'TemplateSimilarityCosine_G2': M.TemplateSimilarityCosine_G2.values,
                                    'MeanFCdegree': s91.MeanFCdegree.values, 'Eigenvalue_G1': LAM[:, 0], 'Eigenvalue_G2': LAM[:, 1]}
keys = {'Parcel_G1': ('Parcel',), 'Parcel_G2': ('Parcel',), 'Parcel_Eccentricity': ('Parcel',)}
for f in ['Yeo7_Network', 'Yeo7_Pairs', 'CABNP12_Network', 'CABNP12_NetworkPairs']:
    keys[f] = ('NetworkA', 'NetworkB', 'MetricType')
summ_all, res = [], {}
for f, vals in fam.items():
    O, summ = family_table(vals, f, keys.get(f, ('Metric',)))
    if f.startswith('Parcel'):
        O.insert(2, 'Label', atlas.label.values); O.insert(3, 'Yeo7', yeo); O.insert(4, 'CABNP12', cab_labels)
    O.to_csv(ST / f'Step5_{f}_3group.csv', index=False)
    res[f] = O
    summ_all += summ
pd.DataFrame(summ_all).to_csv(ST / 'family_summary_3group.csv', index=False)


def pick(f, **kw):
    T = res[f]; m = np.ones(len(T), bool)
    for k, v in kw.items():
        m &= (T[k].astype(str) == str(v)).values
    assert m.sum() == 1
    return T[m].iloc[0]


PRIMARY = [('G2 range (VS)', pick('Core_Global', Metric='G2_RobustRange')),
           ('lambda2', pick('Step9_1_Additional_Global', Metric='Eigenvalue_G2')),
           ('VIS2 mean G2', pick('CABNP12_Network', NetworkA='Visual2', MetricType='MeanG2')),
           ('G1 range (SA, control axis)', pick('Core_Global', Metric='G1_RobustRange'))]
rows = [dict(Metric=l, Omnibus_F=r.Omnibus_F, Omnibus_P=r.Omnibus_P, Omnibus_Q=r.Omnibus_Q_FDR,
             **{f'{k}_{s}': r[f'{k}_{s}'] for k in ('TCvsHC', 'BBvsHC', 'TCvsBB') for s in ('g', 'P_Holm', 'P_TOST', 'BF01')},
             Pattern=ts.classify(r)) for l, r in PRIMARY]
Cl = pd.DataFrame(rows)
Cl.to_csv(ST / 'primary_metrics_classification.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(Cl.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
