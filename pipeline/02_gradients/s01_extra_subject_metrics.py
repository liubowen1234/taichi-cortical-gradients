"""Step 2.1 - extra subject-level gradient metrics that the MATLAB Step 4 does not write.

Run after matlab/gradients/run_all_gradients.m.

  eigenvalues lambda1-lambda3 of each subject's own (unaligned) embedding, so they do not depend on the reference;
  cosine similarity of each subject's G1 / G2 / G3 with the reference;
  mean FC degree: per parcel, the sum of its strongest 10% connections (by absolute value), averaged over parcels;
  G3 orientation and span. G3 is the aligned component left after G1 and G2. Its sign is fixed once, in the reference
  embedding: the modulation networks (CAB-NP frontoparietal, dorsal attention, cingulo-opercular) are made positive
  relative to the representation networks (default, visual 1 and 2, somatomotor, auditory). The same sign is applied
  to every subject, so G3 runs along the modulation-representation axis.

Outputs (<scheme folder>/step_9.1 and step_9.2)
  step9_1_additional_subject_metrics.csv  cosine similarity G1/G2, mean FC degree, lambda1, lambda2
  step9_2_G3_subject_metrics.csv          G3 span (95th - 5th percentile), cosine similarity G3, lambda3
  step9_2_G3_orientation.json             reference contrast and the sign applied to G3

Usage: python pipeline/02_gradients/s01_extra_subject_metrics.py [main|noGSR]
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import h5py
from common.config import LAYOUT, mkdir
from common.data import covariates, cabnp_mapping

which = sys.argv[1] if len(sys.argv) > 1 else 'main'
ROOT = {'main': LAYOUT.GRAD_A, 'noGSR': LAYOUT.GRAD_NOGSR}[which]
cov = covariates()
N = len(cov)
net = cabnp_mapping().CABNP_Network.to_numpy()
mkdir(ROOT / 'step_9.1'); mkdir(ROOT / 'step_9.2')


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# h5py reads MATLAB arrays with the axes reversed: aligned_canonical3d becomes subjects x 3 x parcels
with h5py.File(ROOT / 'Step3_canonical_G1G2/BrainSpace_step3_G1G2_ONLY_results.mat', 'r') as h:
    ac3 = h['aligned_canonical3d'][:]; rc3 = h['reference_canonical3d'][:]
    ac = h['aligned_canonical'][:]; rc = h['reference_canonical'][:]
with h5py.File(ROOT / 'Step1_raw_gradients/BrainSpace_formal_raw_gradients.mat', 'r') as h:
    lam = h['lambda_all'][:]                         # subjects x 10
with h5py.File(ROOT / 'FC.mat', 'r') as h:
    FC = h['FC'][:]                                  # subjects x parcels x parcels
P = FC.shape[1]
assert ac3.shape == (N, 3, P) and lam.shape[0] == N

# ---- step 9.1: similarity to the reference, mean FC degree, eigenvalues
cosG1 = np.array([cosine(ac[i, 0, :], rc[0, :]) for i in range(N)])
cosG2 = np.array([cosine(ac[i, 1, :], rc[1, :]) for i in range(N)])
k = int(np.ceil(0.1 * (P - 1)))                       # strongest 10% of each parcel's connections
deg = np.empty(N)
for i in range(N):
    M = FC[i].copy(); np.fill_diagonal(M, 0.0)
    rowsum = np.empty(P)
    for r in range(P):
        row = M[r]
        top = np.argpartition(np.abs(row), -k)[-k:]
        rowsum[r] = row[top].sum()
    deg[i] = rowsum.mean()
s91 = pd.DataFrame({'Subject': cov.participant_id.values, 'Group': cov.Group.values,
                    'TemplateSimilarityCosine_G1': cosG1, 'TemplateSimilarityCosine_G2': cosG2, 'MeanFCdegree': deg,
                    'Eigenvalue_G1': lam[:, 0], 'Eigenvalue_G2': lam[:, 1]})
s91.to_csv(ROOT / 'step_9.1/step9_1_additional_subject_metrics.csv', index=False)

# ---- step 9.2: G3 orientation (modulation pole positive) and G3 metrics
G3_raw = ac3[:, 2, :]
refG3 = rc3[2, :]
mod_nets = ['Frontoparietal', 'Dorsal-Attention', 'Cingulo-Opercular']
rep_nets = ['Default', 'Visual1', 'Visual2', 'Somatomotor', 'Auditory']
contrast = refG3[np.isin(net, mod_nets)].mean() - refG3[np.isin(net, rep_nets)].mean()
sign = 1.0 if contrast >= 0 else -1.0
G3 = sign * G3_raw
refG3o = sign * refG3
span = np.percentile(G3, 95, axis=1) - np.percentile(G3, 5, axis=1)
json.dump(dict(axis='G3 (modulation-representation)', reference_contrast_before=float(contrast), sign_flip=float(sign),
               note='reference G3; modulation pole (FPN + DAN + CON) positive'),
          open(ROOT / 'step_9.2/step9_2_G3_orientation.json', 'w'), indent=2)
s92 = pd.DataFrame({'Subject': cov.participant_id.values, 'Group': cov.Group.values, 'G3_RobustRange': span,
                    'TemplateSimilarityCosine_G3': [cosine(G3[i], refG3o) for i in range(N)],
                    'Eigenvalue_G3': lam[:, 2], 'MeanFCdegree': deg})
s92.to_csv(ROOT / 'step_9.2/step9_2_G3_subject_metrics.csv', index=False)
print('G3 sign', sign, '; written for N =', N)
