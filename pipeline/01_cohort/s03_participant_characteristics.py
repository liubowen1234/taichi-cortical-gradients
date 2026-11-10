"""Step 1.3 - participant characteristics (Supplementary Table 1).

Continuous measures: Welch t test for each pair of groups, Welch ANOVA across the three groups.
Censoring / retained data: Mann-Whitney U (pairs) and Kruskal-Wallis (three groups).
Sex: chi-square test (2 x 2 with continuity correction for pairs; 3 x 2 across groups).
Athletes only: training years and age at training onset (Tai Chi vs basketball).
P values are descriptive (uncorrected, no covariates).

Output: 01_cohort/demographics.csv
Usage: python pipeline/01_cohort/s03_participant_characteristics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy import stats
from common.config import LAYOUT
from common.data import GROUPS, training

demo = pd.read_csv(LAYOUT.COHORT / 'subject_demographics.csv')
mot = pd.read_csv(LAYOUT.COHORT / 'motion_summary.csv').set_index('participant_id')
tr = training()
TR_SEC = 2.0
demo['n_censored'] = mot.loc[demo.participant_id, 'n_censored'].values
demo['retained_min'] = mot.loc[demo.participant_id, 'n_retained'].values * TR_SEC / 60
PAIRS = [('TC', 'HC'), ('BB', 'HC'), ('TC', 'BB')]


def welch_anova(groups):
    """Welch's one-way ANOVA: F, df1, df2, P."""
    k = len(groups)
    n = np.array([len(g) for g in groups]); m = np.array([g.mean() for g in groups]); v = np.array([g.var(ddof=1) for g in groups])
    w = n / v
    mw = (w * m).sum() / w.sum()
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    F = a / b
    df2 = (k ** 2 - 1) / (3 * lam)
    return F, k - 1, df2, stats.f.sf(F, k - 1, df2)


def summary(x):
    return dict(mean=x.mean(), sd=x.std(ddof=1), min=x.min(), max=x.max(), median=np.median(x))


rows = []


def continuous(label, col, rank=False, data=demo, groups=GROUPS, pairs=PAIRS):
    xs = {g: data.loc[data.Group == g, col].astype(float).values for g in groups}
    r = dict(Measure=label)
    for g in groups:
        for k, v in summary(xs[g]).items():
            r[f'{g}_{k}'] = v
    if len(groups) == 3:
        if rank:
            h = stats.kruskal(*xs.values())
            r.update(omnibus='Kruskal-Wallis H', omnibus_stat=h.statistic, omnibus_df='2', omnibus_P=h.pvalue)
        else:
            F, d1, d2, p = welch_anova(list(xs.values()))
            r.update(omnibus='Welch F', omnibus_stat=F, omnibus_df=f'{d1}, {d2:.1f}', omnibus_P=p)
    for a, b in pairs:
        t = stats.mannwhitneyu(xs[a], xs[b]) if rank else stats.ttest_ind(xs[a], xs[b], equal_var=False)
        r[f'{a}_vs_{b}_test'] = 'Mann-Whitney U' if rank else 'Welch t'
        r[f'{a}_vs_{b}_stat'] = t.statistic
        r[f'{a}_vs_{b}_P'] = t.pvalue
    rows.append(r)


continuous('Age, years', 'Age')
# sex (1 = female, 2 = male)
counts = {g: [(demo[demo.Group == g].Sex == 1).sum(), (demo[demo.Group == g].Sex == 2).sum()] for g in GROUPS}
c3 = stats.chi2_contingency(np.array([counts[g] for g in GROUPS]))
r = dict(Measure='Sex, female/male', **{f'{g}_count': f'{counts[g][0]}/{counts[g][1]}' for g in GROUPS},
         omnibus='chi-square', omnibus_stat=c3[0], omnibus_df='2', omnibus_P=c3[1])
for a, b in PAIRS:
    c = stats.chi2_contingency(np.array([counts[a], counts[b]]))       # Yates continuity correction (scipy default for 2 x 2)
    r.update({f'{a}_vs_{b}_test': 'chi-square (Yates)', f'{a}_vs_{b}_stat': c[0], f'{a}_vs_{b}_P': c[1]})
rows.append(r)
continuous('Education, years', 'Education')
continuous('Mean FD, mm', 'MeanFD')
continuous('eTIV, mL', 'eTIV')
continuous('Censored volumes (FD > 0.5 mm)', 'n_censored', rank=True)
continuous('Data retained after censoring, min', 'retained_min', rank=True)
continuous('Training, years', 'TrainYears', data=tr, groups=['TC', 'BB'], pairs=[('TC', 'BB')])
continuous('Age at training onset, years', 'OnsetAge', data=tr, groups=['TC', 'BB'], pairs=[('TC', 'BB')])
out = pd.DataFrame(rows)
out.to_csv(LAYOUT.COHORT / 'demographics.csv', index=False)
pd.set_option('display.width', 250)
print(out[['Measure'] + [c for c in out.columns if c.endswith(('_mean', '_count', 'omnibus_P'))]].to_string(index=False))
