"""Step 2.4 - similarity of the parcel-wise G2 effect maps of the two athlete groups (Supplementary Fig. 9).

Maps: parcel-wise Hedges' g (unadjusted) of TC - HC, BB - HC and TC - BB.
Why a label-permutation null: the TC - HC and BB - HC maps share the same HC group, so sampling noise of the HC means
enters both maps with the same sign and makes them correlate even when there is no true group effect. A spin null
(rotating one map) treats the maps as independent and misses this. Here the group labels are shuffled across all
participants (group sizes fixed), both maps are recomputed and correlated, so the null distribution contains the
shared-reference correlation. The same is done for TC - HC vs TC - BB (shared TC group).
Reported: observed r, null mean and 95th percentile, one-sided P, raw and after regressing the reference G2 axis out of
both maps.

Output: 02_gradients/effect_map_similarity_labelperm.csv
Usage: python pipeline/02_gradients/s04_effect_map_similarity.py [n_perm (default 5000)]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from common.config import LAYOUT
from common.data import covariates, row_ids, parcel_matrix, hc_reference_axis

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
cov = covariates()
grp0 = cov.Group.values
N = len(grp0)
AXIS = hc_reference_axis()
A = np.column_stack([np.ones(358), AXIS])
Hm = np.eye(358) - A @ np.linalg.pinv(A)        # residual-forming matrix: removes the reference G2 axis from a map


def gmap(Y, grp, a, b):
    """Parcel-wise Hedges' g of group a minus group b (Y: subjects x parcels)."""
    ya, yb = Y[grp == a], Y[grp == b]
    n1, n2 = len(ya), len(yb)
    sp = np.sqrt(((n1 - 1) * ya.var(0, ddof=1) + (n2 - 1) * yb.var(0, ddof=1)) / (n1 + n2 - 2))
    return (ya.mean(0) - yb.mean(0)) / sp * (1 - 3 / (4 * (n1 + n2) - 9))


def corr(x, y):
    return np.corrcoef(x, y)[0, 1]


rows = []
for scheme, f in [('A_HCref', LAYOUT.GRAD_A / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv'),
                  ('B_template2D', LAYOUT.GRAD_B / 'Step4_parcel_gradients_long.csv')]:
    Y = parcel_matrix(pd.read_csv(f), 'G2', row_ids(N))
    rng = np.random.default_rng(20261003)
    for (a1, b1), (a2, b2) in [(('TC', 'HC'), ('BB', 'HC')), (('TC', 'HC'), ('TC', 'BB'))]:
        def stat(grp):
            x, y = gmap(Y, grp, a2, b2), gmap(Y, grp, a1, b1)
            return corr(x, y), corr(Hm @ x, Hm @ y)
        r_obs, r2_obs = stat(grp0)
        null = np.array([stat(rng.permutation(grp0)) for _ in range(NP)])
        rows.append(dict(Scheme=scheme, MapX=f'{a2}vs{b2}', MapY=f'{a1}vs{b1}', n_perm=NP,
                         r=r_obs, null_mean=null[:, 0].mean(), null_95pct=np.percentile(null[:, 0], 95),
                         p_perm=(1 + (null[:, 0] >= r_obs).sum()) / (1 + NP),
                         r_G2axis_removed=r2_obs, null_mean_axis_removed=null[:, 1].mean(),
                         null_95pct_axis_removed=np.percentile(null[:, 1], 95),
                         p_perm_axis_removed=(1 + (null[:, 1] >= r2_obs).sum()) / (1 + NP)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(LAYOUT.GRAD / 'effect_map_similarity_labelperm.csv', index=False)
