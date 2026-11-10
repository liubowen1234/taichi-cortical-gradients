"""Step 4.5 - position of subcortical and cerebellar regions on the cortical gradients (Supplementary Fig. 2).

For each subcortical / cerebellar CAB-NP parcel and participant (censored FC, Fisher z):
  G1 / G2 / G3 position = weighted mean of the reference cortical gradient scores over the parcel's strongest 10% positive
                          cortical connections (weights = Fisher-z FC);
  VIS2 / SMN coupling   = mean Fisher-z FC with the Visual2 / Somatomotor cortex.
Parcels are averaged (weighted by grayordinate count) into thalamus, basal ganglia and cerebellum.
Primary family: 3 regions x {G2 position, VIS2 coupling, SMN coupling} = 9 tests (three-group engine). G1 / G3 positions
are exploratory. Sensitivity versions: all frames (no censoring), each subject's own aligned gradients, top 5% / 20%.
Exploratory: region-by-network subdivisions and single parcels with >= 10 grayordinates (FDR per metric).

Outputs (04_spatial/subcortical_projection/): subject_roi_metrics_primary_3group.csv, roi_group_effects_3group.csv,
roi_network_subdivisions_exploratory_3group.csv, parcel_group_effects_exploratory_3group.csv, family_summary_3group.csv
Usage: python pipeline/04_spatial/s05_subcortical_projection.py
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import h5py
from common.config import LAYOUT
from common import stats_engine as ts
from common.data import covariates, cabnp_mapping, glasser_atlas

OUT = LAYOUT.SUBCORT
TOPFRAC = 0.10
cov = covariates()
grp = cov.Group.values
NS = len(cov)
D = np.load(OUT / 'data/cabnp718_timeseries.npz')
subs = D['subjects']
assert (subs == cov.participant_id.values).all()
TS = D['ts'].astype(np.float64)
CEN = D['censor']
info = pd.read_csv(OUT / 'data/cabnp718_parcel_info.csv')
net = cabnp_mapping().CABNP_Network.values
cifti358 = list(glasser_atlas().cifti_label)
glab = list(info.GLASSERLABELNAME[:360])
ctx_idx = np.array([glab.index(c) for c in cifti358])          # CAB-NP index of every Glasser-358 parcel
assert (info.NETWORK.values[ctx_idx] == net).all()
with h5py.File(LAYOUT.GRAD_A / 'Step3_canonical_G1G2/BrainSpace_step3_G1G2_ONLY_results.mat', 'r') as h:
    A3 = h['aligned_canonical3d'][:].transpose(0, 2, 1).copy()   # subjects x 358 x 3
    R3 = h['reference_canonical3d'][:].T.copy()                  # 358 x 3
g3sign = json.load(open(LAYOUT.GRAD_A / 'step_9.2/step9_2_G3_orientation.json'))['sign_flip']
A3[:, :, 2] *= g3sign; R3[:, 2] *= g3sign                        # G3 with the modulation pole positive
sub_idx = np.arange(360, 718)
sinfo = info.iloc[sub_idx].reset_index(drop=True)
sinfo['Structure'] = sinfo.LABEL.str.split('_').str[-1].str.split('-').str[-1]
ROI = {'Thalamus': ['Thalamus'], 'Basal ganglia': ['Caudate', 'Putamen', 'Pallidum', 'Accumbens'], 'Cerebellum': ['Cerebellum']}
w_vox = sinfo.n_grayordinates.values.astype(float)
v2c, smc = net == 'Visual2', net == 'Somatomotor'


def metrics(fc_version='censored', grad='ref', topfrac=TOPFRAC):
    k = int(round(topfrac * 358))
    out = {m: np.zeros((NS, 358)) for m in ['G1', 'G2', 'G3', 'VIS2', 'SMN']}
    for s in range(NS):
        x = TS[s][~CEN[s]] if fc_version == 'censored' else TS[s]
        with np.errstate(invalid='ignore', divide='ignore'):
            z = np.corrcoef(x.T)
        Z = np.arctanh(np.clip(z[np.ix_(sub_idx, ctx_idx)], -0.999999, 0.999999))   # subcortical x cortical
        gr = R3 if grad == 'ref' else A3[s]
        thr = np.sort(Z, 1)[:, -k][:, None]
        W = np.where((Z >= thr) & (Z > 0), Z, 0.0)                                 # strongest positive connections
        pos = (W @ gr) / W.sum(1, keepdims=True)
        for j, m in enumerate(['G1', 'G2', 'G3']):
            out[m][s] = pos[:, j]
        out['VIS2'][s] = Z[:, v2c].mean(1)
        out['SMN'][s] = Z[:, smc].mean(1)
    return out


def roi_values(M):
    res = {}
    for r, st in ROI.items():
        m = sinfo.Structure.isin(st).values
        res[r] = (M[:, m] * w_vox[m]).sum(1) / w_vox[m].sum()
    return res


VERS = {'primary': metrics('censored', 'ref'), 'allframes': metrics('allframes', 'ref'),
        'own_gradients': metrics('censored', 'own'), 'top05': metrics('censored', 'ref', 0.05),
        'top20': metrics('censored', 'ref', 0.20)}
ROIV = pd.DataFrame({f'{r}__{m}': v for m in ['G1', 'G2', 'G3', 'VIS2', 'SMN'] for r, v in roi_values(VERS['primary'][m]).items()})
ROIV.insert(0, 'Subject', subs); ROIV.insert(1, 'Group', grp)
ROIV.to_csv(OUT / 'subject_roi_metrics_primary_3group.csv', index=False)

ENG = ts.Engine(ts.design(cov), grp)
rows = []
for vname, M in VERS.items():
    for m in ['G2', 'VIS2', 'SMN', 'G1', 'G3']:
        for r, y in roi_values(M[m]).items():
            fam = f"{vname}::{'Primary' if m in ('G2', 'VIS2', 'SMN') else 'Exploratory'}"
            rows.append(dict(Version=vname, ROI=r, Metric=m, Family=fam, **ENG.analyse(y)))
R, summ = ts.finalize_family(pd.DataFrame(rows), 'Family')
R.to_csv(OUT / 'roi_group_effects_3group.csv', index=False)

srows = []
for r, st in ROI.items():
    for n in sorted(sinfo.NETWORK[sinfo.Structure.isin(st)].unique()):
        m_ = sinfo.Structure.isin(st).values & (sinfo.NETWORK.values == n)
        if w_vox[m_].sum() < 20:
            continue
        for met in ['G2', 'VIS2', 'SMN']:
            y = (VERS['primary'][met][:, m_] * w_vox[m_]).sum(1) / w_vox[m_].sum()
            srows.append(dict(ROI=r, Network=n, n_parcels=int(m_.sum()), n_grayordinates=int(w_vox[m_].sum()), Metric=met,
                              Family=f'Subdivision::{met}', **ENG.analyse(y)))
SUBD, s2 = ts.finalize_family(pd.DataFrame(srows), 'Family')
SUBD.to_csv(OUT / 'roi_network_subdivisions_exploratory_3group.csv', index=False)
prow = []
for met in ['G2', 'VIS2', 'SMN']:
    for j in np.where(w_vox >= 10)[0]:
        prow.append(dict(Metric=met, Family=f'Parcel::{met}', Label=sinfo.LABEL[j], Structure=sinfo.Structure[j],
                         Network=sinfo.NETWORK[j], n_grayordinates=int(w_vox[j]), **ENG.analyse(VERS['primary'][met][:, j])))
PP, s3 = ts.finalize_family(pd.DataFrame(prow), 'Family')
PP.to_csv(OUT / 'parcel_group_effects_exploratory_3group.csv', index=False)
pd.DataFrame(summ + s2 + s3).to_csv(OUT / 'family_summary_3group.csv', index=False)
cols = ['ROI', 'Metric', 'Omnibus_P', 'Omnibus_Q_FDR', 'TCvsHC_g', 'BBvsHC_g', 'TCvsBB_g']
print(R[R.Version == 'primary'][cols].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
