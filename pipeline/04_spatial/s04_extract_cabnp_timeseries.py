"""Step 4.4 - whole-brain CAB-NP time series (718 parcels: 360 cortical + 358 subcortical / cerebellar) for the
projection of subcortical and cerebellar regions onto the cortical gradients.

For each participant: parcel-mean time series over all grayordinates of the parcel (as wb_command -cifti-parcellate
-method MEAN) from the XCP-D denoised, unsmoothed CIFTI data; the XCP-D censoring mask (FD > 0.5 mm); and the fraction
of zero-variance grayordinates (signal dropout) of each parcel.
Needs only numpy (CIFTI files are read with common/cifti_min.py).

Outputs (04_spatial/subcortical_projection/data/)
  cabnp718_timeseries.npz  ts (subjects x frames x 718, float32), censor (subjects x frames), subjects, labels
  cabnp718_parcel_info.csv label key + number of grayordinates + mean / max dropout fraction
Usage: python pipeline/04_spatial/s04_extract_cabnp_timeseries.py
"""
import sys, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from common.config import CFG, LAYOUT, mkdir
from common.cifti_min import read_cifti, brain_models
from common.data import covariates

CAB = CFG.path('cabnp_dir')
OUT = mkdir(LAYOUT.SUBCORT / 'data')
subs = list(covariates().participant_id)
key = list(csv.DictReader(open(CAB / 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'), delimiter='\t'))
assert len(key) == 718
keyval = np.array([int(k['KEYVALUE']) for k in key])
lab, lxml, _ = read_cifti(CAB / 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii')
lab = np.rint(lab[0]).astype(int)


def func_dir(sub):
    for d in CFG.datasets('36P'):
        if (d['dir'] / sub / 'func').is_dir():
            return d['dir'] / sub / 'func'
    raise FileNotFoundError(sub)


def gray_table(xml):
    """(structure, vertex) or voxel ijk -> grayordinate index of a CIFTI file."""
    t = {}
    for bm in brain_models(xml):
        if bm['model'] == 'CIFTI_MODEL_TYPE_SURFACE':
            for i, v in enumerate(bm['verts']):
                t[(bm['structure'], 'v', int(v))] = bm['offset'] + i
        else:
            for i, v in enumerate(bm['ijk']):
                t[('VOL', tuple(int(q) for q in v))] = bm['offset'] + i
    return t


def dtseries(sub):
    return func_dir(sub) / f'{sub}_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii'


# map the CAB-NP labels onto the grayordinates of the denoised data (all participants share one layout; checked)
_, dxml, _ = read_cifti(dtseries(subs[0]), want_data=False)
td = gray_table(dxml)
for s in (subs[-1],):
    assert gray_table(read_cifti(dtseries(s), want_data=False)[1]) == td, 'grayordinate layout differs between participants'
tl = gray_table(lxml)
labd = np.zeros(len(td), int)
n_unmatched = 0
for k_, i in tl.items():
    if k_ in td:
        labd[td[k_]] = lab[i]
    elif lab[i] > 0:
        n_unmatched += 1
members = [np.where(labd == kv)[0] for kv in keyval]
nvox = np.array([len(m) for m in members])
print('unmatched labelled grayordinates:', n_unmatched, '| empty parcels:', int((nvox == 0).sum()))

TS = DROP = CEN = None
for si, s in enumerate(subs):
    d, _, _ = read_cifti(dtseries(s))
    out = np.array([int(float(r['framewise_displacement'])) for r in
                    csv.DictReader(open(func_dir(s) / f'{s}_task-rest_outliers.tsv'), delimiter='\t')], bool)
    if TS is None:
        TS = np.zeros((len(subs), len(d), 718), np.float32)
        CEN = np.zeros((len(subs), len(d)), bool)
        DROP = np.zeros((len(subs), 718), np.float32)
    d = d.astype(np.float64)
    TS[si] = np.column_stack([d[:, m].mean(1) if len(m) else np.full(len(d), np.nan) for m in members])
    sd = d.std(0)
    DROP[si] = [(sd[m] < 1e-6).mean() if len(m) else np.nan for m in members]
    CEN[si] = out
    print(si + 1, s, int(out.sum()), 'censored frames', flush=True)

np.savez_compressed(OUT / 'cabnp718_timeseries.npz', ts=TS, censor=CEN, subjects=np.array(subs),
                    labels=np.array([k['LABEL'] for k in key]))
with open(OUT / 'cabnp718_parcel_info.csv', 'w', newline='') as fh:
    cols = list(key[0]) + ['n_grayordinates', 'mean_dropout_fraction', 'max_dropout_fraction']
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for i, k in enumerate(key):
        w.writerow({**k, 'n_grayordinates': int(nvox[i]), 'mean_dropout_fraction': f'{np.nanmean(DROP[:, i]):.4f}',
                    'max_dropout_fraction': f'{np.nanmax(DROP[:, i]):.4f}'})
print('saved', OUT)
