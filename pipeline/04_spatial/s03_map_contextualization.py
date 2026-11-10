"""Step 4.3 - spatial correspondence of the G2 group-effect maps with published cortical maps.

Reference maps: the vertex-wise maps of s02_fetch_reference_maps.py averaged within the Glasser-358 parcels
(neuromaps: canonical G1 and G2, T1w/T2w, gene PC1, sensorimotor-association axis, evolutionary and developmental
expansion; Neurosynth term association maps).
Targets: parcel-wise Hedges' g maps of TC - HC, BB - HC and TC - BB (A = main analysis, B = template alignment).
Test: Pearson r with a two-sided spin P (10,000 rotations, s01). Because a general widening of G2 would make every effect
map resemble the G2 axis itself, each correlation is also computed after regressing the non-athlete reference G2 map
out of both maps (each rotated copy is residualised against the fixed axis); not done for the canonical gradients.
Right-hemisphere-only maps (Hill 2010) are tested on the 179 right-hemisphere parcels with their rotations.
BH-FDR within each family (neuromaps; Neurosynth terms) and target map.

Outputs (04_spatial/map_contextualization/): parcellated_reference_maps_Glasser358.csv, spatial_correspondence_3group.csv
Usage: python pipeline/04_spatial/s03_map_contextualization.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from common.config import CFG, LAYOUT
from common import stats_engine as ts, figlib as fl
from common.data import cabnp_mapping, hc_reference_axis, spin_permutations

OUT = LAYOUT.CONTEXT
DATA = OUT / 'data'
PERM = spin_permutations()
AXIS = hc_reference_axis()
lab = CFG.path('cabnp_dir') / 'SeparateHemispheres'
iL, iR = fl.glasser358_vertex_index(lab / 'cortex_ColeAnticevic_NetPartition_GlasserParcels_L.label.gii',
                                    lab / 'cortex_ColeAnticevic_NetPartition_GlasserParcels_R.label.gii')
vidx = np.concatenate([iL, iR])                     # 64,984 vertices -> parcel (0-based) or -1 (medial wall)


def parcellate(v):
    """Mean of the finite vertex values inside each parcel."""
    out = np.full(358, np.nan)
    ok = (vidx >= 0) & np.isfinite(v)
    s = np.bincount(vidx[ok], weights=v[ok], minlength=358)
    n = np.bincount(vidx[ok], minlength=358)
    out[n > 0] = s[n > 0] / n[n > 0]
    return out


def resid(x, m):
    A = np.column_stack([np.ones(m.sum()), AXIS[m]])
    return x[m] - A @ np.linalg.lstsq(A, x[m], rcond=None)[0]


def spin_r(x, y, m):
    """Correlation of maps x and y over the parcels in mask m; x is rotated. Two-sided spin P."""
    idx = np.where(m)[0]
    pos = -np.ones(358, int); pos[idx] = np.arange(len(idx))
    Pm = pos[PERM[:, idx]]
    assert (Pm >= 0).all()
    xs, ys = x[idx], y[idx]
    r = np.corrcoef(xs, ys)[0, 1]
    Xn = xs[Pm]; Xn = (Xn - Xn.mean(1, keepdims=True)) / Xn.std(1, keepdims=True)
    yz = (ys - ys.mean()) / ys.std()
    null = (Xn * yz).mean(1)                        # correlation of every rotated x with y
    return r, (1 + np.sum(np.abs(null) >= abs(r))) / (1 + len(null))


def spin_r_axis_removed(x, y, m):
    """As spin_r, after regressing the reference G2 axis out of both maps (the axis itself is not rotated)."""
    idx = np.where(m)[0]
    pos = -np.ones(358, int); pos[idx] = np.arange(len(idx))
    Pm = pos[PERM[:, idx]]
    xr, yr = resid(x, m), resid(y, m)
    r = np.corrcoef(xr, yr)[0, 1]
    A = np.column_stack([np.ones(len(idx)), AXIS[idx]]); H = A @ np.linalg.pinv(A)
    Xn = x[idx][Pm]; Xn = Xn - Xn @ H.T
    Xn = (Xn - Xn.mean(1, keepdims=True)) / Xn.std(1, keepdims=True)
    yz = (yr - yr.mean()) / yr.std()
    null = (Xn * yz).mean(1)
    return r, (1 + np.sum(np.abs(null) >= abs(r))) / (1 + len(null))


# ---- parcellated reference maps
PARC = {}
for fam, fn in [('neuromaps', 'neuromaps_fsLR32k.npz'), ('neurosynth_terms', 'neurosynth_terms_fsLR32k.npz')]:
    Z = np.load(DATA / fn)
    for name in Z.files:
        PARC[f'{fam}::{name}'] = parcellate(Z[name].astype(float))
REF = pd.DataFrame(PARC).assign(Parcel358=np.arange(1, 359), Network=cabnp_mapping().CABNP_Network.values)
REF.to_csv(OUT / 'parcellated_reference_maps_Glasser358.csv', index=False)

# ---- targets: parcel-wise g maps of the three contrasts
targets = {}
for scheme, f in [('A', LAYOUT.GRAD_A / 'stats_three_group/Step5_parcel_G2_358_3group.csv'),
                  ('B', LAYOUT.GRAD_B / 'stats_three_group/Step5_Parcel_G2_3group.csv')]:
    T = pd.read_csv(f).sort_values('Parcel')
    for k in ('TCvsHC', 'BBvsHC', 'TCvsBB'):
        targets[f'{scheme}_{k}_g'] = T[f'{k}_g'].values

rows = []
for col, x in PARC.items():
    fam, name = col.split('::')
    m = np.isfinite(x)
    hemi = 'both' if m.all() else ('right' if m[:179].all() and not m[179:].any() else 'partial')
    for tname, y in targets.items():
        r, p = spin_r(x, y, m)
        ra, pa = (np.nan, np.nan) if name.startswith('Margulies') else spin_r_axis_removed(x, y, m)
        rows.append(dict(Family=fam, Map=name, Target=tname, hemispheres=hemi, n_parcels=int(m.sum()), r=r, p_spin=p,
                         r_axis_removed=ra, p_spin_axis_removed=pa))
R = pd.DataFrame(rows)
for (fam, t), d in R.groupby(['Family', 'Target']):
    R.loc[d.index, 'q_spin'] = ts.bh(d.p_spin.values)
    ok = d.p_spin_axis_removed.notna()
    R.loc[d.index[ok], 'q_spin_axis_removed'] = ts.bh(d.p_spin_axis_removed[ok].values)
R.to_csv(OUT / 'spatial_correspondence_3group.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(R[R.Target.str.startswith('A_')].pivot_table(index=['Family', 'Map'], columns='Target', values='r').round(2))
