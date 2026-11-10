"""Step 1.1 - head motion, exclusion and the participant tables used by all later steps.

Input
  config "participants_table": one row per scanned participant with the columns
      participant_id, group (HC / TC / BB), age, sex (1 = female, 2 = male), education_years, eTIV_mL,
      train_years (athletes), weekly_hours (optional), level (optional)
  XCP-D outputs of the main (36P) pipeline: <sub>/func/<sub>_task-rest_motion.tsv and _outliers.tsv

What it does
  1. Mean framewise displacement (FD) of every participant = mean of the XCP-D FD column over frames 2..end
     (XCP-D sets the first frame to 0, so it is left out).
  2. Participants with mean FD > 0.25 mm are excluded (before any connectivity or gradient analysis).
  3. The included participants are ordered HC -> TC -> BB, ascending participant ID within each group.
     This order is used for every subject-level array of the pipeline.

Outputs (01_cohort/)
  motion_summary.csv      all scanned participants: mean FD, max FD, number of censored / retained frames, excluded yes/no
  exclusion_log.csv       the excluded participants
  subject_covariates.csv  model covariates of the included participants (Age, Sex, MeanFD; eTIV for a sensitivity model)
  subject_demographics.csv  descriptive variables (adds education)
  subject_training.csv    athletes only: training years, age at training onset (= age - training years), weekly hours, level

Usage: python pipeline/01_cohort/s01_motion_and_participants.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from common.config import CFG, LAYOUT, mkdir
from common.data import GROUPS

THR = CFG.option('motion_exclusion_meanFD_mm')          # 0.25 mm
OUT = mkdir(LAYOUT.COHORT)

part = pd.read_csv(CFG.path('participants_table'))
part['group'] = part['group'].str.strip()
assert set(part.group) <= set(GROUPS), 'group must be HC, TC or BB'


def find_func_dir(sub):
    """The func/ folder of a participant in one of the 36P XCP-D output folders."""
    for d in CFG.datasets('36P'):
        f = d['dir'] / sub / 'func'
        if f.is_dir():
            return f
    raise FileNotFoundError(f'no XCP-D output for {sub}')


rows = []
for sub, grp in zip(part.participant_id, part.group):
    f = find_func_dir(sub)
    fd = pd.read_csv(f / f'{sub}_task-rest_motion.tsv', sep='\t')['framewise_displacement'].values.astype(float)
    censored = pd.read_csv(f / f'{sub}_task-rest_outliers.tsv', sep='\t')['framewise_displacement'].values.astype(bool)
    rows.append(dict(participant_id=sub, Group=grp, MeanFD=fd[1:].mean(), MaxFD=np.nanmax(fd),
                     n_frames_FD_gt_0p5=int((fd > 0.5).sum()), n_censored=int(censored.sum()), n_retained=int((~censored).sum())))
mot = pd.DataFrame(rows)
mot['exclude'] = mot.MeanFD > THR
mot['reason'] = np.where(mot.exclude, f'MeanFD > {THR} mm', '')
mot.to_csv(OUT / 'motion_summary.csv', index=False)
mot.loc[mot.exclude, ['participant_id', 'Group', 'MeanFD', 'n_censored', 'reason']].to_csv(OUT / 'exclusion_log.csv', index=False)

# ---- included participants in the analysis order (HC -> TC -> BB, ascending ID)
inc = part[part.participant_id.isin(mot.participant_id[~mot.exclude])].copy()
inc['order'] = inc.group.map({g: i for i, g in enumerate(GROUPS)})
inc = inc.sort_values(['order', 'participant_id']).reset_index(drop=True)
m = mot.set_index('participant_id').loc[inc.participant_id]

cov = pd.DataFrame(dict(participant_id=inc.participant_id, Group=inc.group, Age=inc.age, Sex=inc.sex,
                        MeanFD=m.MeanFD.values, eTIV=inc.eTIV_mL))
assert set(cov.Sex) <= {1, 2}, 'sex must be coded 1 = female, 2 = male'
cov.to_csv(OUT / 'subject_covariates.csv', index=False)

demo = cov.copy()
demo['Education'] = inc.education_years.values
demo.to_csv(OUT / 'subject_demographics.csv', index=False)

ath = inc[inc.group != 'HC']
tr = pd.DataFrame(dict(participant_id=ath.participant_id, Group=ath.group, TrainYears=ath.train_years,
                       OnsetAge=ath.age - ath.train_years,
                       WeeklyHours=ath.get('weekly_hours', pd.Series(np.nan, index=ath.index)),
                       Level=ath.get('level', pd.Series('', index=ath.index))))
tr.to_csv(OUT / 'subject_training.csv', index=False)

print(mot.groupby('Group').agg(n=('MeanFD', 'size'), n_excluded=('exclude', 'sum'), MeanFD_max=('MeanFD', 'max')))
print('included:', cov.Group.value_counts().reindex(GROUPS).to_dict())
