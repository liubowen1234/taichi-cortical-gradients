"""Step 3.3 - do the two athlete groups differ in WHICH partner network VIS2 is decoupled from?

1. Connectivity: with two partner networks per subject, the group x partner interaction equals the group effect on the
   within-subject difference  d = FC(VIS2-SMN) - FC(VIS2-CON). d is analysed with the three-group engine; the key
   contrast is TC vs BB. Versions: raw block means; each connection standardised to the non-athlete mean and SD before
   differencing; raw version in the motion-matched subset.
2. Gradient-space analogue: distances between the network centroids of VIS2, SMN and CON in the G1-G2 plane
   (|VIS2-SMN| - |VIS2-CON|; rotation-invariant) and along G2 only, for the main analysis and the template alignment.

Outputs (03_network_connectivity/visual2_connectivity/)
  Visual2_partner_interaction_3group.csv, Visual2_partner_distance_gradient_space_3group.csv,
  subject_partner_distance_gradient_space.csv

Usage: python pipeline/03_network_connectivity/s03_visual2_partner_contrast.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from common.config import LAYOUT
from common import stats_engine as ts
from common.data import covariates, cabnp_mapping, row_ids, parcel_matrix
from common.motion import motion_subsets

OUT = LAYOUT.VIS2
cov = covariates()
grp = cov.Group.values
eng = ts.Engine(ts.design(cov), grp)

# ---- 1. connectivity: SMN vs CON partner
F = pd.read_csv(OUT / 'subject_Visual2_networkFC_and_bridge_inputs.csv')
assert list(F.Subject) == list(cov.participant_id)
hc = grp == 'HC'


def z_hc(x):
    return (x - x[hc].mean()) / x[hc].std(ddof=1)


D = {'raw_SMN_minus_CON': F.Visual2_Somatomotor.values - F.Visual2_CinguloOpercular.values,
     'z_SMN_minus_CON': z_hc(F.Visual2_Somatomotor.values) - z_hc(F.Visual2_CinguloOpercular.values)}
rows = [{'Version': k, 'Subset': 'full', 'N': len(y), **eng.analyse(y)} for k, y in D.items()]
keep = motion_subsets(cov)['SM_motion_matched']
c2 = cov[keep].reset_index(drop=True)
e2 = ts.Engine(ts.design(c2), c2.Group.values)
rows.append({'Version': 'raw_SMN_minus_CON', 'Subset': 'SM_motion_matched', 'N': int(keep.sum()),
             **e2.analyse(D['raw_SMN_minus_CON'][keep])})
pd.DataFrame(rows).to_csv(OUT / 'Visual2_partner_interaction_3group.csv', index=False)

# ---- 2. gradient space
cab = cabnp_mapping().set_index('FC_Index_358')
SRC = {'A_HCref': LAYOUT.GRAD_A / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv',
       'B_template2D': LAYOUT.GRAD_B / 'Step4_parcel_gradients_long.csv'}
rows, subj = [], []
for scheme, f in SRC.items():
    L = pd.read_csv(f)
    ids = row_ids(len(cov))
    assert (L.drop_duplicates('Subject').set_index('Subject').loc[ids, 'Group'].values == grp).all()
    G = {k: parcel_matrix(L, k, ids) for k in ('G1', 'G2')}
    net = cab.loc[np.arange(1, 359), 'CABNP_Network'].values
    c = {n: np.c_[G['G1'][:, net == n].mean(1), G['G2'][:, net == n].mean(1)] for n in ('Visual2', 'Somatomotor', 'Cingulo-Opercular')}
    dS = np.linalg.norm(c['Visual2'] - c['Somatomotor'], axis=1)
    dC = np.linalg.norm(c['Visual2'] - c['Cingulo-Opercular'], axis=1)
    gS = np.abs(c['Visual2'][:, 1] - c['Somatomotor'][:, 1])
    gC = np.abs(c['Visual2'][:, 1] - c['Cingulo-Opercular'][:, 1])
    M = {'dist2D_VIS2_SMN': dS, 'dist2D_VIS2_CON': dC, 'dist2D_diff_SMN_minus_CON': dS - dC,
         'G2sep_VIS2_SMN': gS, 'G2sep_VIS2_CON': gC, 'G2sep_diff_SMN_minus_CON': gS - gC}
    rows += [{'Scheme': scheme, 'Metric': k, **eng.analyse(y)} for k, y in M.items()]
    subj.append(pd.DataFrame({'participant_id': cov.participant_id, 'Group': cov.Group, 'Scheme': scheme, **M}))
pd.DataFrame(rows).to_csv(OUT / 'Visual2_partner_distance_gradient_space_3group.csv', index=False)
pd.concat(subj).to_csv(OUT / 'subject_partner_distance_gradient_space.csv', index=False)
print(pd.DataFrame(rows)[['Scheme', 'Metric', 'Omnibus_P', 'TCvsBB_g', 'TCvsBB_P_Holm']].round(4).to_string(index=False))
