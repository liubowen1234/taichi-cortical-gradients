"""Step 6.2 - signal-quality measures of every participant (from the XCP-D outputs of the main 36P pipeline).

Per participant (uncensored frames only):
  number / percentage of censored frames, remaining scan time, mean FD;
  zero-variance grayordinates (signal dropout) in the cortex;
  residual BOLD amplitude = temporal SD of each grayordinate; median over the cortex, and per Glasser parcel the median
  over its non-dropout vertices;
  DVARS = RMS of the frame-to-frame signal change over cortical non-dropout grayordinates (pairs of consecutive retained
  frames), median over frames;
  mean parcel coverage = 1 - fraction of zero-variance vertices (equals the XCP-D coverage value to 1e-6).
Needs only numpy (CIFTI files are read with common/cifti_min.py).

Outputs (06_robustness/data_quality/data/): subject_signal_quality.csv, parcel_residual_amplitude_Glasser360.csv,
parcel_zeroSD_fraction_Glasser360.csv (360 parcels, XCP-D order)
Usage: python pipeline/06_robustness/s02_signal_quality.py
"""
import sys, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from common.config import CFG, LAYOUT, mkdir
from common.cifti_min import read_cifti, brain_models
from common.data import covariates

OUT = mkdir(LAYOUT.QUALITY / 'data')
TR_SEC = 2.0
subs = list(covariates().participant_id)
ATL = Path(str(CFG.path('glasser_dlabel')).replace('.dlabel.nii', ''))


def func_dir(sub):
    for d in CFG.datasets('36P'):
        if (d['dir'] / sub / 'func').is_dir():
            return d['dir'] / sub / 'func'
    raise FileNotFoundError(sub)


def vertex_table(xml):
    t = {}
    for bm in brain_models(xml):
        if bm['model'] == 'CIFTI_MODEL_TYPE_SURFACE':
            for i, v in enumerate(bm['verts']):
                t[(bm['structure'], int(v))] = bm['offset'] + i
    return t


# Glasser label of every grayordinate of the denoised data (0 = not cortex)
lab, lxml, _ = read_cifti(Path(str(ATL) + '.dlabel.nii'))
lab = np.rint(lab[0]).astype(int)
_, dxml, _ = read_cifti(func_dir(subs[0]) / f'{subs[0]}_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii', want_data=False)
tl, td = vertex_table(lxml), vertex_table(dxml)
lab91 = np.zeros(91282, int)
for key, i in tl.items():
    if key in td:
        lab91[td[key]] = lab[i]
assert (lab91 > 0).sum() == (lab > 0).sum()
names = [r['name'] for r in csv.DictReader(open(str(ATL) + '.tsv'), delimiter='\t')]
ctx = lab91 > 0
members = [np.where(lab91 == k)[0] for k in range(1, 361)]

S, AMP, ZERO = [], [], []
for s in subs:
    f = func_dir(s)
    out = np.array([int(float(r['framewise_displacement'])) for r in csv.DictReader(open(f / f'{s}_task-rest_outliers.tsv'), delimiter='\t')])
    fd = np.array([float(r['framewise_displacement']) for r in csv.DictReader(open(f / f'{s}_task-rest_motion.tsv'), delimiter='\t')])
    d, _, _ = read_cifti(f / f'{s}_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii')
    keep = out == 0
    sd = d[keep].astype(np.float64).std(0, ddof=1)
    zero = sd < 1e-6
    amp = np.array([np.median(sd[m][~zero[m]]) if (~zero[m]).any() else np.nan for m in members])
    zf = np.array([zero[m].mean() for m in members])
    pairs = np.where(keep[1:] & keep[:-1])[0]
    dv = np.sqrt(np.mean((d[pairs + 1][:, ctx & ~zero] - d[pairs][:, ctx & ~zero]).astype(np.float64) ** 2, 1))
    S.append(dict(Subject=s, n_volumes=len(out), n_censored=int(out.sum()), pct_censored=100 * out.mean(),
                  remaining_min=keep.sum() * TR_SEC / 60, meanFD_xcpd=float(np.nanmean(fd[1:])),
                  n_zeroSD_cortex=int(zero[ctx].sum()), n_zeroSD_all=int(zero.sum()),
                  cortex_median_amplitude=float(np.median(sd[ctx & ~zero])), DVARS_median=float(np.median(dv)),
                  mean_coverage=float(np.mean(1 - zf))))
    AMP.append(amp); ZERO.append(zf)
    print(s, S[-1]['n_censored'], round(S[-1]['cortex_median_amplitude'], 3), flush=True)

pd.DataFrame(S).to_csv(OUT / 'subject_signal_quality.csv', index=False)
for nm, M in [('parcel_residual_amplitude_Glasser360', AMP), ('parcel_zeroSD_fraction_Glasser360', ZERO)]:
    B = pd.DataFrame(np.array(M), columns=names)
    B.insert(0, 'Subject', subs)
    B.to_csv(OUT / f'{nm}.csv', index=False)
