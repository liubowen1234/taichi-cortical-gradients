"""Step 6.9 - how much do the results depend on the alignment reference? (Supplementary Fig. 1)

A = non-athlete reference (main analysis), B = external BrainSpace template (2D alignment).
1. Key metrics under A and B with the three-group engine (no family FDR here; families are in each scheme's tables):
   G2 span, G1 span, lambda2, minor-axis variance, VIS2 mean G2, G2 poles (95th / 5th percentile); and the Tai Chi
   training dose of G2 span.
2. Frame angle of B relative to A: for each subject, the 2D orthogonal Procrustes rotation from the A (G1, G2) to the
   B (G1, G2) coordinates; median angle across subjects.
3. Rotation curve: every subject's A (G1, G2) rotated in the G1-G2 plane by -35 to +35 degrees (equivalent to rotating the
   reference frame) and the G1 / G2 span group effects recomputed. The minor-axis variance does not change with rotation.
4. Similarity of the parcel-wise G2 effect maps (Hedges' g) under A and B (Pearson r over 358 parcels).

Outputs (06_robustness/alignment_comparison/): key_metrics_by_scheme.csv, frame_angle_vs_schemeA.json,
rotation_curve_schemeA.csv, schemeA_vs_schemeB_G2_effectmap_similarity.csv
Usage: python pipeline/06_robustness/s09_alignment_comparison.py
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy import stats
from common.config import LAYOUT, mkdir
from common import stats_engine as ts
from common.data import covariates, training_years, cabnp_mapping, row_ids, parcel_matrix

OUT = mkdir(LAYOUT.ALIGN)
cov = covariates()
grp = cov.Group.values
N = len(cov)
yrs = training_years(cov)
v2 = cabnp_mapping().CABNP_Network.values == 'Visual2'
lam2 = pd.read_csv(LAYOUT.GRAD_A / 'step_9.1/step9_1_additional_subject_metrics.csv').Eigenvalue_G2.values
SCHEMES = {'A': LAYOUT.GRAD_A / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv',
           'B': LAYOUT.GRAD_B / 'Step4_parcel_gradients_long.csv'}
SID = row_ids(N)


def load(path):
    L = pd.read_csv(path)
    assert (L.drop_duplicates('Subject').set_index('Subject').loc[SID].Group.values == grp).all()
    return parcel_matrix(L, 'G1', SID), parcel_matrix(L, 'G2', SID)


def span(X):
    return np.percentile(X, 95, axis=1) - np.percentile(X, 5, axis=1)


def minor_var(g1, g2):
    return np.array([np.linalg.eigvalsh(np.cov(np.vstack([g1[i], g2[i]])))[0] for i in range(len(g1))])


def metrics(g1, g2):
    return {'G2 range': span(g2), 'G1 range': span(g1), 'λ2': lam2, 'Minor-axis variance': minor_var(g1, g2),
            'VIS2 mean G2': g2[:, v2].mean(1), 'G2 visual pole (Q95)': np.percentile(g2, 95, axis=1),
            'G2 somatosensory pole (Q05)': np.percentile(g2, 5, axis=1)}


E = ts.Engine(ts.design(cov), grp)
DATA = {k: load(v) for k, v in SCHEMES.items()}
rows = []
for sch, (g1, g2) in DATA.items():
    for m, y in metrics(g1, g2).items():
        o = E.analyse(y)
        r = dict(Scheme=sch, Metric=m, Omnibus_P=o['Omnibus_P'])
        for c in ['TCvsHC', 'BBvsHC', 'TCvsBB']:
            r.update({f'{c}_g': o[f'{c}_g'], f'{c}_g_lo': o[f'{c}_g_CI95_Lower'], f'{c}_g_hi': o[f'{c}_g_CI95_Upper'],
                      f'{c}_P_Holm': o[f'{c}_P_Holm'], f'{c}_P_TOST': o[f'{c}_P_TOST']})
        rows.append(r)
    m = grp == 'TC'                                     # training dose in Tai Chi (standardised beta stored in TCvsHC_g)
    y = span(g2)[m]
    X = np.column_stack([np.ones(m.sum()), yrs[m], cov.Age.values[m], cov.Sex.values[m] - 1, cov.MeanFD.values[m]])
    r = ts.hc3(y, X)
    rows.append(dict(Scheme=sch, Metric='Dose: G2 range ~ years (TC)', Omnibus_P=np.nan,
                     TCvsHC_g=r['Beta'] * X[:, 1].std(ddof=1) / y.std(ddof=1), TCvsHC_P_Holm=r['P']))
pd.DataFrame(rows).to_csv(OUT / 'key_metrics_by_scheme.csv', index=False)

# ---- frame angle of each scheme relative to A
gA1, gA2 = DATA['A']
ang = {}
for sch, (g1, g2) in DATA.items():
    th, refl = [], []
    for i in range(N):
        Xa = np.column_stack([gA1[i], gA2[i]]); Xs = np.column_stack([g1[i], g2[i]])
        Xa = Xa - Xa.mean(0); Xs = Xs - Xs.mean(0)
        U, _, Vt = np.linalg.svd(Xa.T @ Xs)
        Rm = U @ Vt                                     # Xa @ Rm ~ Xs
        refl.append(np.linalg.det(Rm) < 0)
        th.append(np.degrees(np.arctan2(Rm[0, 1], Rm[0, 0])))
    ang[sch] = dict(median_angle_deg=float(np.median(th)), IQR=[float(np.percentile(th, 25)), float(np.percentile(th, 75))],
                    reflections=int(np.sum(refl)))
json.dump(ang, open(OUT / 'frame_angle_vs_schemeA.json', 'w'), indent=1)

# ---- rotation curve on the A gradients
cur = []
for th in np.arange(-35, 35.5, 1.0):
    t = np.radians(th); c, s = np.cos(t), np.sin(t)
    r1 = c * gA1 + s * gA2; r2 = -s * gA1 + c * gA2    # frame rotated by +theta (G1 towards G2)
    for m, y in [('G2 range', span(r2)), ('G1 range', span(r1))]:
        o = E.analyse(y)
        cur.append(dict(theta=th, Metric=m, **{f'{k}_g': o[f'{k}_g'] for k in ['TCvsHC', 'BBvsHC', 'TCvsBB']},
                        **{f'{k}_P': o[f'{k}_P'] for k in ['TCvsHC', 'BBvsHC', 'TCvsBB']}))
pd.DataFrame(cur).to_csv(OUT / 'rotation_curve_schemeA.csv', index=False)

# ---- similarity of the parcel-wise G2 effect maps under A and B
a = pd.read_csv(LAYOUT.GRAD_A / 'stats_three_group/Step5_parcel_G2_358_3group.csv')
b = pd.read_csv(LAYOUT.GRAD_B / 'stats_three_group/Step5_Parcel_G2_3group.csv')
assert len(a) == len(b) == 358 and (a.Parcel.values == b.Parcel.values).all()
pd.DataFrame([dict(Contrast=c, N_parcels=358, Pearson_r=stats.pearsonr(a[f'{c}_g'], b[f'{c}_g'])[0]) for c in ['TCvsHC', 'TCvsBB', 'BBvsHC']]) \
    .to_csv(OUT / 'schemeA_vs_schemeB_G2_effectmap_similarity.csv', index=False)
print(json.dumps(ang, indent=1))
