"""Head-motion subsets used by the robustness analyses.

Gradients are not recomputed in a subset: every subject is aligned to a fixed reference (the non-athlete reference or
the external template), so leaving people out does not change anybody's gradient values. Only the models are refitted.

  S0_full                 all participants
  P10 / P20               drop the 10% / 20% of each group with the highest mean FD
  SF_athletes_le_HCmax    drop athletes whose mean FD exceeds the largest mean FD among non-athletes
  SM_motion_matched       repeatedly drop the highest-FD participant of the group with the highest mean FD until the
                          three-group Welch ANOVA on mean FD has P >= 0.5
"""
import numpy as np

from common.stats_engine import welch_anova_p

GROUPS = ('HC', 'TC', 'BB')


def motion_subsets(cov):
    """Dict subset name -> boolean mask over the rows of cov."""
    grp, fd, n = cov.Group.values, cov.MeanFD.values, len(cov)

    def drop_top(frac):
        keep = np.ones(n, bool)
        for g in GROUPS:
            idx = np.where(grp == g)[0]
            k = int(round(frac * len(idx)))
            keep[idx[np.argsort(-fd[idx])[:k]]] = False
        return keep

    def welch_p(keep):
        return welch_anova_p([fd[keep & (grp == g)] for g in GROUPS])

    subsets = {'S0_full': np.ones(n, bool), 'P10_dropTop10pct_each': drop_top(.10), 'P20_dropTop20pct_each': drop_top(.20),
               'SF_athletes_le_HCmax': ~((grp != 'HC') & (fd > fd[grp == 'HC'].max()))}
    keep = np.ones(n, bool)
    while welch_p(keep) < .5:
        means = {g: fd[keep & (grp == g)].mean() for g in GROUPS}
        g = max(means, key=means.get)
        idx = np.where(keep & (grp == g))[0]
        keep[idx[np.argmax(fd[idx])]] = False
    subsets['SM_motion_matched'] = keep
    return subsets
