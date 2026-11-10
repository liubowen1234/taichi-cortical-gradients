"""Small loaders shared by the analysis scripts.

Subject order: every subject-level array in this pipeline has the same row order, written by
01_cohort/s01_motion_and_participants.py: non-athletes (HC), then Tai Chi practitioners (TC), then basketball players (BB),
ascending participant ID within each group.

Parcel order (Glasser-358): the order of resources/atlas/atlas-Glasser358_dseg.tsv, i.e. the 360 HCP-MMP parcels in the
XCP-D order (right hemisphere first) with the two hippocampal parcels (R_H, L_H) removed.

The MATLAB gradient steps label subjects by their row number (sub-001, sub-002, ...), not by participant ID.
"""
import numpy as np
import pandas as pd
import h5py

from common.config import LAYOUT, RESOURCES

GROUPS = ['HC', 'TC', 'BB']
# The 12 Cole-Anticevic (CAB-NP) cortical networks
NETS = ['Visual1', 'Visual2', 'Somatomotor', 'Cingulo-Opercular', 'Dorsal-Attention', 'Language', 'Frontoparietal',
        'Auditory', 'Default', 'Posterior-Multimodal', 'Ventral-Multimodal', 'Orbito-Affective']
KEEP358 = np.setdiff1d(np.arange(360), [119, 299])     # 0-based positions of the 358 kept parcels in the 360-parcel order


def covariates():
    """One row per included participant: participant_id, Group, Age, Sex (1 = female, 2 = male), MeanFD, eTIV."""
    return pd.read_csv(LAYOUT.COHORT / 'subject_covariates.csv')


def training():
    """Athletes only: participant_id, Group, TrainYears, OnsetAge, WeeklyHours, Level."""
    return pd.read_csv(LAYOUT.COHORT / 'subject_training.csv')


def training_years(cov):
    """Training years aligned to the rows of `cov` (NaN for non-athletes)."""
    tr = training().set_index('participant_id')
    return np.array([tr.TrainYears.get(s, np.nan) for s in cov.participant_id], float)


def row_ids(n):
    """Row labels used by the MATLAB steps: sub-001 ... sub-<n>."""
    return [f'sub-{i:03d}' for i in range(1, n + 1)]


def glasser_atlas():
    """Glasser-358 table: index, label, cifti_label, community_yeo (Yeo-7 name of each parcel)."""
    return pd.read_csv(RESOURCES / 'atlas/atlas-Glasser358_dseg.tsv', sep='\t')


def cabnp_mapping():
    """Glasser-358 parcels with their CAB-NP network (columns FC_Index_358, GlasserLabel, CiftiLabel, Hemisphere, CABNP_Network)."""
    return pd.read_csv(RESOURCES / 'atlas/CABNP12_Glasser358_mapping.csv').sort_values('FC_Index_358').reset_index(drop=True)


def network_of_parcels():
    return cabnp_mapping().CABNP_Network.values


def load_fc(mat_file):
    """Read the FC array saved as MATLAB v7.3 by the gradient runner. Returns subjects x parcels x parcels (Pearson r)."""
    with h5py.File(mat_file, 'r') as h:
        return h['FC'][:]


def parcel_matrix(long_table, column, subjects):
    """Turn a long parcel table (Subject, Parcel, <column>) into a subjects x parcels matrix in the given subject order."""
    return long_table.pivot(index='Subject', columns='Parcel', values=column).loc[subjects].values.astype(float)


def hc_reference_axis():
    """G2 of the non-athlete reference embedding (main analysis), used to remove the G2 axis from spatial maps."""
    with h5py.File(LAYOUT.GRAD_A / 'Step3_canonical_G1G2/BrainSpace_step3_G1G2_ONLY_results.mat', 'r') as h:
        return h['reference_canonical3d'][:][1]


def spin_permutations():
    """Spin permutations (n_perm x 358) written by 04_spatial/s01_make_spin_permutations.py."""
    return np.load(LAYOUT.SPIN / 'spin_perms_hungarian_10000.npy')
