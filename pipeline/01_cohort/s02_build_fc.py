"""Step 1.2 - parcel-wise functional connectivity (FC) of all included participants.

For each participant the FC matrix is the Pearson correlation between parcel-mean time series after removing the frames
that XCP-D flagged for censoring (FD > 0.5 mm). Two equivalent sources are supported (set per XCP-D folder in config.json):
  fc_source = "pconn"    : the parcellated connectivity file written by XCP-D (same computation, read as float32);
  fc_source = "dtseries" : computed here from the unsmoothed denoised CIFTI: average the surface vertices of each parcel,
                           drop the censored frames, Pearson correlation, stored at float32 precision.
On participants that have both, the two sources agree to < 1e-6.

Atlases
  glasser     : 360 HCP-MMP parcels; the bilateral hippocampal parcels (R_H, L_H) have almost no signal and are removed -> 358.
  schaefer400 / schaefer600 : cortical parcels of the XCP-D 4S456 / 4S656 atlases (first 400 / 600 parcels).

Output (01_cohort/fc/FC_<atlas>_<pipeline>.mat, MATLAB v5): FC (parcels x parcels x subjects, Pearson r), subject_ids, group,
parcel_labels. Subject order = 01_cohort/subject_covariates.csv.

Usage: python pipeline/01_cohort/s02_build_fc.py --atlas glasser --pipeline 36P
       (also run: --atlas glasser --pipeline 32P; --atlas schaefer400 --pipeline 36P; --atlas schaefer600 --pipeline 36P)
"""
import sys, argparse, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io as sio
from common.config import CFG, LAYOUT, mkdir
from common.data import covariates, KEEP358

warnings.filterwarnings('ignore', 'invalid value encountered in divide')   # zero-variance H parcels give NaN (removed later)
ap = argparse.ArgumentParser()
ap.add_argument('--atlas', choices=['glasser', 'schaefer400', 'schaefer600'], default='glasser')
ap.add_argument('--pipeline', choices=['36P', '32P'], default='36P')
args = ap.parse_args()

ATLAS = {'glasser': ('glasser_dlabel', 'Glasser', 360), 'schaefer400': ('schaefer400_dlabel', '4S456Parcels', 400),
         'schaefer600': ('schaefer600_dlabel', '4S656Parcels', 600)}
cfg_key, xcpd_name, n_parcels = ATLAS[args.atlas]
dlabel_file = CFG.path(cfg_key)                                   # tpl-fsLR_atlas-<name>_den-91k_dseg.dlabel.nii
tsv = pd.read_csv(str(dlabel_file).replace('.dlabel.nii', '.tsv'), sep='\t')
parcel_ids = tsv['index'].values[:n_parcels]
name_col = 'cifti_label' if 'cifti_label' in tsv else 'name'

# vertex -> parcel label of the dlabel file, keyed by (surface structure, vertex number)
lab = nib.load(dlabel_file)
L = np.asarray(lab.get_fdata()).ravel().astype(int)
lab_keys, lab_vals = [], []
for name, slc, bm in lab.header.get_axis(1).iter_structures():
    if bm.volume_mask.any():
        continue
    for i, v in enumerate(bm.vertex):
        lab_keys.append((name, int(v))); lab_vals.append(L[slc.start + i])
lab_vals = np.array(lab_vals)


def parcel_timeseries(dtseries_file):
    """Frames x parcels: mean over the surface vertices of each parcel (same vertex matching as XCP-D)."""
    img = nib.load(dtseries_file)
    pos = {}
    for name, slc, bm in img.header.get_axis(1).iter_structures():
        if bm.volume_mask.any():
            continue
        for i, v in enumerate(bm.vertex):
            pos[(name, int(v))] = slc.start + i
    idx = np.array([pos[k] for k in lab_keys])
    D = np.asarray(img.get_fdata(dtype=np.float64))[:, idx]
    return np.column_stack([D[:, lab_vals == k].mean(1) for k in parcel_ids])


def subject_fc(sub):
    for d in CFG.datasets(args.pipeline):
        f = d['dir'] / sub / 'func'
        if f.is_dir():
            break
    else:
        raise FileNotFoundError(sub)
    prefix = f / f'{sub}_task-rest_'
    if d['fc_source'] == 'pconn':
        pconn = Path(f'{prefix}space-fsLR_atlas-{xcpd_name}_den-91k_stat-pearsoncorrelation_boldmap.pconn.nii')
        return np.asarray(nib.load(pconn).get_fdata(dtype=np.float32))[:n_parcels, :n_parcels].astype(np.float64)
    ts = parcel_timeseries(f'{prefix}space-fsLR_den-91k_desc-denoised_bold.dtseries.nii')
    censored = pd.read_csv(f'{prefix}outliers.tsv', sep='\t')['framewise_displacement'].values.astype(bool)
    return np.corrcoef(ts[~censored].T).astype(np.float32).astype(np.float64)


cov = covariates()
keep = KEEP358 if args.atlas == 'glasser' else np.arange(n_parcels)
if args.atlas == 'glasser':
    assert list(tsv[name_col].values[[119, 299]]) == ['R_H_ROI', 'L_H_ROI']
FC = np.zeros((len(keep), len(keep), len(cov)))
for i, sub in enumerate(cov.participant_id):
    M = subject_fc(sub)[np.ix_(keep, keep)]
    assert np.isfinite(M).all(), f'non-finite FC for {sub}'
    FC[:, :, i] = M
    print(i + 1, sub, f'mean FC = {M[np.triu_indices(len(keep), 1)].mean():.4f}', flush=True)

out = mkdir(LAYOUT.FC) / f'FC_{args.atlas}_{args.pipeline}.mat'
sio.savemat(out, dict(FC=FC, subject_ids=np.array(list(cov.participant_id), dtype=object),
                      group=np.array(list(cov.Group), dtype=object),
                      parcel_labels=np.array(tsv[name_col].values[keep], dtype=object)),
            do_compression=True, format='5')
print('saved', out, FC.shape)
