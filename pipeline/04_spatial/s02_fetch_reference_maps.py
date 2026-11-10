"""Step 4.2 - download the reference maps and project them to the fs_LR 32k surface.

Needs neuromaps (0.0.7), NiMARE (0.21) and Connectome Workbench (wb_command on PATH, used by the neuromaps transforms).
Downloads (~1-2 GB) are cached in the neuromaps / NiMARE data folders given in config.json.

  1. neuromaps annotations: HCP S1200 T1w/T2w ratio (myelin); gene expression PC1 (abagen); evolutionary and
     developmental cortical expansion (Hill 2010; right hemisphere only); sensorimotor-association axis (Sydnor 2021);
     the first two FC gradients of Margulies 2016. Maps not in fs_LR 32k are transformed with neuromaps.
  2. Neurosynth v7 (abstract terms) with NiMARE: for each pre-specified term with >= 30 studies, an MKDA chi-square
     association map (studies with the term vs all other studies; z_desc-association), MNI152 -> fs_LR 32k.

Outputs (04_spatial/map_contextualization/data/): neuromaps_fsLR32k.npz, neurosynth_terms_fsLR32k.npz,
neurosynth_term_availability.csv, fetch_log.txt. Each array has 64,984 values (left hemisphere, then right).
Parcel averaging and statistics are done in s03_map_contextualization.py.
Usage: python pipeline/04_spatial/s02_fetch_reference_maps.py
"""
import sys, os, time, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from common.config import CFG, LAYOUT, mkdir

MIN_STUDIES = 30
OUT = mkdir(LAYOUT.CONTEXT / 'data')
if CFG.path('neuromaps_data'):
    os.environ['NEUROMAPS_DATA'] = str(CFG.path('neuromaps_data'))
LOG = open(OUT / 'fetch_log.txt', 'a')


def log(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


if shutil.which('wb_command') is None:
    sys.exit('wb_command not found on PATH (needed by the neuromaps transforms).')
import nibabel as nib
import neuromaps, nimare
from neuromaps.datasets import fetch_annotation
from neuromaps import transforms
log('neuromaps', neuromaps.__version__, 'nimare', nimare.__version__, 'wb_command', shutil.which('wb_command'))


def to_array(gii_pair):
    arr = [np.asarray(nib.load(g).agg_data() if isinstance(g, (str, os.PathLike)) else g.agg_data(), float).ravel() for g in gii_pair]
    assert all(a.size == 32492 for a in arr), [a.size for a in arr]
    return np.concatenate(arr)


# ------------------------------------------------------------------ 1. neuromaps
ANN = [('myelin_T1wT2w', 'hcps1200', 'myelinmap', 'fsLR', '32k'),
       ('genePC1', 'abagen', 'genepc1', 'fsaverage', '10k'),
       ('evo_expansion', 'hill2010', 'evoexp', 'fsLR', '164k'),
       ('dev_expansion', 'hill2010', 'devexp', 'fsLR', '164k'),
       ('SA_axis', 'sydnor2021', 'SAaxis', 'fsLR', '32k'),
       ('Margulies_G1', 'margulies2016', 'fcgradient01', 'fsLR', '32k'),
       ('Margulies_G2', 'margulies2016', 'fcgradient02', 'fsLR', '32k')]
NM = {}
for name, src, desc, space, den in ANN:
    try:
        f = fetch_annotation(source=src, desc=desc, space=space, den=den)
        f = f if isinstance(f, (list, tuple)) else [f]
        if len(f) == 1 and space != 'MNI152':
            # e.g. Hill 2010 expansion maps: right hemisphere only -> left filled with NaN
            hemi = 'R' if ('hemi-R' in str(f[0]) or 'hemi' not in str(f[0])) else 'L'
            g = f if den == '32k' else transforms.fslr_to_fslr(f[0], '32k', hemi=hemi)
            one = np.asarray(nib.load(g[0]).agg_data() if isinstance(g[0], (str, os.PathLike)) else g[0].agg_data(), float).ravel()
            assert one.size == 32492, one.size
            nanh = np.full(32492, np.nan)
            arr = np.concatenate([nanh, one] if hemi == 'R' else [one, nanh])
            log('single hemisphere map', name, 'hemi', hemi, Path(str(f[0])).name)
        elif space == 'fsLR' and den == '32k':
            arr = to_array(f)
        elif space == 'fsLR':
            arr = to_array(transforms.fslr_to_fslr(f, '32k'))
        elif space == 'fsaverage':
            arr = to_array(transforms.fsaverage_to_fslr(f, '32k'))
        else:
            arr = to_array(transforms.mni152_to_fslr(f[0], '32k'))
        NM[name] = arr.astype(np.float32); log('neuromaps ok', name, src, desc, space, den, 'n_nonzero', int((arr != 0).sum()))
    except Exception as e:
        log('neuromaps FAILED', name, src, desc, repr(e))
np.savez_compressed(OUT / 'neuromaps_fsLR32k.npz', **NM)

# ------------------------------------------------------------------ 2. Neurosynth terms
from nimare.extract import fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma.mkda import MKDAChi2

TERMS = ['visual', 'motion', 'visual motion', 'eye movements', 'visuospatial', 'spatial attention', 'attention',
         'motor', 'sensorimotor', 'somatosensory', 'action', 'navigation', 'balance', 'posture', 'postural', 'vestibular',
         'self motion', 'optic flow', 'mindfulness', 'meditation', 'default']
NDIR = CFG.path('nimare_data'); NDIR.mkdir(parents=True, exist_ok=True)


def load_dataset(vocab):
    pkl = NDIR / f'neurosynth_v7_{vocab}_dataset.pkl.gz'
    if pkl.exists():
        return nimare.dataset.Dataset.load(str(pkl))
    fetch_neurosynth(data_dir=str(NDIR), version='7', overwrite=False, source='abstract', vocab=vocab)   # downloads only
    root = NDIR / 'neurosynth'
    def one(pattern):
        hits = sorted(root.rglob(pattern))
        if not hits: sys.exit(f'Neurosynth file not found: {pattern} under {root}')
        return str(hits[0])
    coords = one('data-neurosynth_version-7_coordinates.tsv.gz')
    meta = one('data-neurosynth_version-7_metadata.tsv.gz')
    feats = one(f'data-neurosynth_version-7_vocab-{vocab}_source-abstract_type-*_features.npz')
    vocabf = one(f'data-neurosynth_version-7_vocab-{vocab}_vocabulary.txt')
    log('building NiMARE dataset from', Path(coords).name, Path(feats).name)
    ds = convert_neurosynth_to_dataset(coordinates_file=coords, metadata_file=meta,
                                       annotations_files={'features': feats, 'vocabulary': vocabf})
    ds.save(str(pkl)); return ds


def run_meta(ds, labels, out_name, pretty, thr=0.001):
    maps, rows = {}, []
    for lab, nm in zip(labels, pretty):
        ids = ds.get_studies_by_label(lab, label_threshold=thr)
        rows.append(dict(term=nm, label=lab, n_studies=len(ids)))
        if len(ids) < MIN_STUDIES:
            log('skip (few studies)', nm, len(ids)); continue
        t0 = time.time()
        rest = sorted(set(ds.ids) - set(ids))
        res = MKDAChi2().fit(ds.slice(ids), ds.slice(rest))
        img = res.get_map('z_desc-association')
        maps[nm] = to_array(transforms.mni152_to_fslr(img, '32k')).astype(np.float32)
        log('meta ok', nm, len(ids), f'{time.time() - t0:.0f}s')
        np.savez_compressed(OUT / out_name, **maps)          # save progressively
    return pd.DataFrame(rows)


ds = load_dataset('terms')
labels_all = ds.get_labels()
prefix = [l for l in labels_all if '__' in l]
log('Neurosynth terms dataset:', len(ds.ids), 'studies,', len(prefix), 'terms')
avail = {l.split('__', 1)[1]: l for l in prefix}
use = [(avail[t], t) for t in TERMS if t in avail]
missing = [t for t in TERMS if t not in avail]
log('terms not in vocabulary:', missing)
av = run_meta(ds, [u[0] for u in use], 'neurosynth_terms_fsLR32k.npz', [u[1] for u in use])
av = pd.concat([av, pd.DataFrame(dict(term=missing, label='', n_studies=0))])
av.to_csv(OUT / 'neurosynth_term_availability.csv', index=False)
log('done')
