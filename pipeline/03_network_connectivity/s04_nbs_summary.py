"""Step 3.4 - summary of the NBS components (run matlab/nbs/run_nbs_three_group.m first).

- Holm correction over the component P values of the six directional t contrasts (interpreted only if the omnibus F
  component is significant). An NBS P of 0 means that none of the 5,000 permutations exceeded the observed size; it is
  reported as P < 0.0002 and entered into the Holm correction as 1/5000.
- Network composition of each component (CAB-NP): % of edges that involve VIS2, number of VIS2-SMN edges, top network pairs.
- Edge overlap between components (shared edges and Jaccard index).

Outputs (03_network_connectivity/nbs/): NBS_three_group_components_summary.csv, NBS_component_overlaps.csv,
edges/NBS_<run>_edges_labelled.csv
Usage: python pipeline/03_network_connectivity/s04_nbs_summary.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from common.config import LAYOUT
from common import stats_engine as ts
from common.data import cabnp_mapping

R = LAYOUT.NBS
cab = cabnp_mapping()
AB = {'Visual1': 'VIS1', 'Visual2': 'VIS2', 'Somatomotor': 'SMN', 'Cingulo-Opercular': 'CON', 'Dorsal-Attention': 'DAN',
      'Language': 'LAN', 'Frontoparietal': 'FPN', 'Auditory': 'AUD', 'Default': 'DMN', 'Posterior-Multimodal': 'PMM',
      'Ventral-Multimodal': 'VMM', 'Orbito-Affective': 'ORA'}
net = np.array([AB[x] for x in cab.CABNP_Network.values])
lab = cab.GlasserLabel.values
RUNS = ['omnibus_F_thr9.61', 'omnibus_F_pmatched', 'HC_gt_TC', 'TC_gt_HC', 'HC_gt_BB', 'BB_gt_HC', 'TC_gt_BB', 'BB_gt_TC']
summ = pd.read_csv(R / 'NBS_three_group_run_summary.csv').set_index('Run')

E, rows = {}, []
for r in RUNS:
    D = pd.read_csv(R / f'edges/NBS_{r}_edges.csv')
    D['NetA'] = net[D.Node_i - 1]; D['NetB'] = net[D.Node_j - 1]
    D['LabelA'] = lab[D.Node_i - 1]; D['LabelB'] = lab[D.Node_j - 1]
    D['Pair'] = ['–'.join(sorted([a, b])) for a, b in zip(D.NetA, D.NetB)]
    D.to_csv(R / f'edges/NBS_{r}_edges_labelled.csv', index=False)
    E[r] = set(zip(D.Node_i, D.Node_j))
    for c in sorted(D.Component.unique()):
        d = D[D.Component == c]
        v2 = (d.NetA == 'VIS2') | (d.NetB == 'VIS2')
        top = d.Pair.value_counts().head(5)
        rows.append(dict(Run=r, Component=int(c), Component_P=float(d.Component_P.iloc[0]), Edges=len(d),
                         Nodes=len(set(d.Node_i) | set(d.Node_j)), Pct_edges_with_VIS2=100 * v2.mean(),
                         VIS2_SMN_edges=int((d.Pair == 'SMN–VIS2').sum()), Top_pairs='; '.join(f'{k} {v}' for k, v in top.items())))
S = pd.DataFrame(rows)
pw = S[~S.Run.str.startswith('omnibus')].copy()
pw['P_Holm_6dir'] = ts.holm(np.maximum(pw.Component_P.values, 1 / 5000))
S = S.merge(pw[['Run', 'Component', 'P_Holm_6dir']], on=['Run', 'Component'], how='left')
S['Gate_omnibus_significant'] = bool(summ.loc['omnibus_F_thr9.61', 'P_largest'] < .05)
S['Component_P_report'] = ['< 0.0002' if p == 0 else f'{p:.4f}' for p in S.Component_P]
S.to_csv(R / 'NBS_three_group_components_summary.csv', index=False)

ov = []
for a, b in [('HC_gt_TC', 'HC_gt_BB'), ('TC_gt_HC', 'BB_gt_HC'), ('HC_gt_TC', 'BB_gt_TC'), ('HC_gt_BB', 'TC_gt_BB')]:
    s = E[a] & E[b]
    ov.append(dict(A=a, B=b, Edges_A=len(E[a]), Edges_B=len(E[b]), Shared=len(s),
                   Jaccard=len(s) / len(E[a] | E[b]) if E[a] | E[b] else np.nan))
pd.DataFrame(ov).to_csv(R / 'NBS_component_overlaps.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
print(S[['Run', 'Component', 'Component_P_report', 'P_Holm_6dir', 'Edges', 'Pct_edges_with_VIS2', 'VIS2_SMN_edges']].to_string(index=False))
