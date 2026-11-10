"""Step 4.1 - spatial permutations ("spin test") for Glasser-358 maps.

Parcel centroids are computed on the fs_LR 32k sphere (BrainSpace conte69 sphere surfaces). Each permutation applies a
uniform random rotation to the left hemisphere and its mirror image to the right hemisphere (Alexander-Bloch et al.,
2018); rotated parcels are re-assigned to the original parcels one-to-one with the Hungarian algorithm (Vasa et al., 2018),
so each permutation is a reordering of the 358 parcels without replacement. Hemispheres are rotated separately.
10,000 permutations, fixed seed.

Output: 04_spatial/spin_permutations/spin_perms_hungarian_10000.npy (10,000 x 358 integers: perm[k, i] = the parcel whose
value moves to parcel i). Takes several minutes.
Usage: python pipeline/04_spatial/s01_make_spin_permutations.py
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from common.config import CFG, LAYOUT, mkdir
from common import figlib as fl, spin

NPERM = CFG.option('spin_permutations')        # 10000
SEED = CFG.option('spin_seed')
OUT = mkdir(LAYOUT.SPIN)
surf = CFG.path('brainspace_matlab') / 'datasets/surfaces'
lab = CFG.path('cabnp_dir') / 'SeparateHemispheres'

iL, iR = fl.glasser358_vertex_index(lab / 'cortex_ColeAnticevic_NetPartition_GlasserParcels_L.label.gii',
                                    lab / 'cortex_ColeAnticevic_NetPartition_GlasserParcels_R.label.gii')
sL, _ = fl.load_surface(surf / 'conte69_32k_left_sphere.gii')
sR, _ = fl.load_surface(surf / 'conte69_32k_right_sphere.gii')
cL_all = spin.sphere_centroids(sL, iL, 358)
cR_all = spin.sphere_centroids(sR, iR, 358)
idxL = np.unique(iL[iL >= 0]); idxR = np.unique(iR[iR >= 0])
assert len(idxL) == 179 and len(idxR) == 179 and idxR.max() == 178 and idxL.min() == 179   # right hemisphere first
t0 = time.time()
perm = spin.spin_permutations(cL_all[idxL], cR_all[idxR], idxL, idxR, NPERM, 'hungarian', seed=SEED)
assert all(np.array_equal(np.sort(perm[k]), np.arange(358)) for k in range(0, NPERM, 997))
np.save(OUT / f'spin_perms_hungarian_{NPERM}.npy', perm)
print(f'{NPERM} permutations in {time.time() - t0:.0f} s ->', OUT)
