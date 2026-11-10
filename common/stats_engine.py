"""Three-group statistics used by every group comparison in the paper.

Model (one outcome at a time):
    y = b0 + b1*TC + b2*BB + b3*Age + b4*Sex + b5*MeanFD [+ extra covariate] + e
TC and BB are indicator variables (non-athletes, HC, are the reference). Ordinary least squares with heteroscedasticity-
consistent (HC3) standard errors; t and F reference distributions with df = n - number of columns.

For each outcome Engine.analyse() returns
  - the omnibus group effect: 2-df Wald F test of b1 = b2 = 0;
  - three pairwise contrasts (TC - HC = b1, BB - HC = b2, TC - BB = b1 - b2): Wald t test, Holm-corrected within the outcome;
  - Hedges' g of each pair (unadjusted group means, pooled SD of the pair) with a large-sample 95% CI;
  - an equivalence test (TOST) of the covariate-adjusted contrast with bounds +/- 0.5 x pooled SD of the pair;
  - the JZS Bayes factor BF01 (Rouder et al., 2009; Cauchy prior scale r = 0.707) from the unadjusted two-sample t.

finalize_family() then applies Benjamini-Hochberg FDR to the omnibus P values within a family of tests; a pairwise
contrast counts as significant only if the omnibus q < 0.05 and its Holm P < 0.05.
The engine was checked against statsmodels (OLS with HC3, f_test / t_test) and agrees to 1e-10.
"""
import numpy as np
from scipy import stats, integrate

SESOI = 0.5      # smallest effect size of interest for the equivalence test (in pooled SD units)
# Contrast weights on [intercept, TC, BB]
PAIRS = [('TC', 'HC', np.r_[0, 1, 0]), ('BB', 'HC', np.r_[0, 0, 1]), ('TC', 'BB', np.r_[0, 1, -1])]


def design(cov, extra=None):
    """Design matrix with columns: intercept, TC, BB, Age, Sex (0 = female, 1 = male), MeanFD [, extra covariate]."""
    g = cov.Group.values
    cols = [np.ones(len(cov)), (g == 'TC').astype(float), (g == 'BB').astype(float), cov.Age.values, cov.Sex.values - 1.0,
            cov.MeanFD.values]
    if extra:
        cols.append(cov[extra].values)
    return np.column_stack(cols).astype(float)


def bh(p):
    """Benjamini-Hochberg adjusted P values (q values)."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p, kind='mergesort')
    r = p[o] * m / np.arange(1, m + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    q = np.empty(m)
    q[o] = np.minimum(r, 1)
    return q


def holm(p):
    """Holm-Bonferroni adjusted P values."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p, kind='mergesort')
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o])
    out = np.empty(m)
    out[o] = np.minimum(adj, 1)
    return out


def jzs_bf10(t, n1, n2, r=0.707):
    """JZS Bayes factor in favour of a group difference, from a two-sample t (Rouder et al., 2009)."""
    nu = n1 + n2 - 2
    neff = n1 * n2 / (n1 + n2)

    def integrand(g):
        return ((1 + neff * g) ** -0.5 * (1 + t ** 2 / ((1 + neff * g) * nu)) ** (-(nu + 1) / 2)
                * r * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-r ** 2 / (2 * g)))

    return integrate.quad(integrand, 0, np.inf, limit=200)[0] / (1 + t ** 2 / nu) ** (-(nu + 1) / 2)


class Engine:
    """Fits the same design matrix X to many outcomes. grp = group label of each row ('HC', 'TC' or 'BB')."""

    def __init__(self, X, grp):
        self.X, self.grp = X, np.asarray(grp)
        self.XtXi = np.linalg.inv(X.T @ X)
        # leverage of each observation (capped to avoid division by zero in HC3)
        self.H = np.minimum(np.einsum('ij,jk,ik->i', X, self.XtXi, X), .999999)
        self.DF = X.shape[0] - X.shape[1]

    def fit(self, y):
        """OLS coefficients b and their HC3 covariance matrix V."""
        X, XtXi = self.X, self.XtXi
        b = XtXi @ X.T @ y
        e = y - X @ b
        V = XtXi @ (X.T * ((e / (1 - self.H)) ** 2)) @ X @ XtXi
        return b, V

    def analyse(self, y):
        y = np.asarray(y, float)
        grp, DF, p = self.grp, self.DF, self.X.shape[1]
        b, V = self.fit(y)
        out = {}
        for g in ('HC', 'TC', 'BB'):
            v = y[grp == g]
            out[f'Mean_{g}'] = v.mean(); out[f'SD_{g}'] = v.std(ddof=1); out[f'N_{g}'] = len(v)
        # omnibus: b_TC = b_BB = 0 (2 df)
        R = np.zeros((2, p)); R[0, 1] = R[1, 2] = 1
        rb = R @ b
        F = float(rb @ np.linalg.solve(R @ V @ R.T, rb)) / 2
        out.update(Omnibus_F=F, Omnibus_df1=2, Omnibus_df2=DF, Omnibus_P=stats.f.sf(F, 2, DF))
        tcrit = stats.t.ppf(.975, DF)
        praw = []
        for a, c, w in PAIRS:
            L = np.zeros(p); L[:3] = w
            est = L @ b
            se = np.sqrt(L @ V @ L)
            t = est / se
            pv = 2 * stats.t.sf(abs(t), DF)
            praw.append(pv)
            ya, yc = y[grp == a], y[grp == c]
            n1, n2 = len(ya), len(yc)
            sp = np.sqrt(((n1 - 1) * ya.var(ddof=1) + (n2 - 1) * yc.var(ddof=1)) / (n1 + n2 - 2))   # pooled SD of the pair
            gH = (ya.mean() - yc.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9))                      # Hedges' g
            gse = np.sqrt((n1 + n2) / (n1 * n2) + gH ** 2 / (2 * (n1 + n2)))
            d = SESOI * sp                                                                          # equivalence bound
            p_tost = max(stats.t.sf((est + d) / se, DF), stats.t.cdf((est - d) / se, DF))
            t_u = (ya.mean() - yc.mean()) / (sp * np.sqrt(1 / n1 + 1 / n2))                         # unadjusted t (for BF01)
            k = f'{a}vs{c}'
            out.update({f'{k}_Beta': est, f'{k}_SE_HC3': se, f'{k}_CI95_Lower': est - tcrit * se, f'{k}_CI95_Upper': est + tcrit * se,
                        f'{k}_T_HC3': t, f'{k}_P': pv, f'{k}_g': gH, f'{k}_g_CI95_Lower': gH - 1.96 * gse,
                        f'{k}_g_CI95_Upper': gH + 1.96 * gse, f'{k}_P_TOST': p_tost, f'{k}_BF01': 1 / jzs_bf10(t_u, n1, n2)})
        for (a, c, _), ph in zip(PAIRS, holm(praw)):
            out[f'{a}vs{c}_P_Holm'] = ph
        return out


def finalize_family(O, fam_col=None, fam_name='family'):
    """BH-FDR of the omnibus P within each family; pairwise contrasts are flagged only where omnibus q < 0.05.
    O: one row per test (output of Engine.analyse plus identifier columns). Returns (O, summary rows)."""
    fams = O[fam_col] if fam_col and fam_col in O else None
    O['Omnibus_Q_FDR'] = np.nan
    groups = fams.unique() if fams is not None else [fam_name]
    summ = []
    for fam in groups:
        m = (fams == fam).values if fams is not None else np.ones(len(O), bool)
        O.loc[m, 'Omnibus_Q_FDR'] = bh(O.loc[m, 'Omnibus_P'].values)
        summ.append(dict(Family=fam, NTests=int(m.sum()), N_Omnibus_FDR05=int((O.loc[m, 'Omnibus_Q_FDR'] < .05).sum()),
                         **{f'N_{a}vs{c}_Holm05_given_omnibus':
                            int(((O.loc[m, 'Omnibus_Q_FDR'] < .05) & (O.loc[m, f'{a}vs{c}_P_Holm'] < .05)).sum())
                            for a, c, _ in PAIRS}))
    O['Posthoc_inferential'] = (O.Omnibus_Q_FDR < .05).astype(int)
    for a, c, _ in PAIRS:
        O[f'{a}vs{c}_Sig'] = ((O.Omnibus_Q_FDR < .05) & (O[f'{a}vs{c}_P_Holm'] < .05)).astype(int)
    return O, summ


def classify(r):
    """Pre-specified pattern label for a primary metric (one result row with omnibus q, Holm P, TOST P and betas)."""
    if r['Omnibus_Q_FDR'] >= .05:
        return 'no group effect'

    def sig(k): return r[f'{k}_P_Holm'] < .05
    def up(k): return r[f'{k}_Beta'] > 0
    def eq(k): return r[f'{k}_P_TOST'] < .05

    tc_hc, bb_hc, tc_bb = sig('TCvsHC') and up('TCvsHC'), sig('BBvsHC') and up('BBvsHC'), sig('TCvsBB')
    if tc_hc and bb_hc:
        if tc_bb:
            return 'graded (common + difference in degree)'
        return 'common' if eq('TCvsBB') else 'common (degree undetermined)'
    if tc_hc and tc_bb and up('TCvsBB') and not sig('BBvsHC'):
        return 'TC-specific' if eq('BBvsHC') else 'leaning TC-specific (undetermined)'
    if tc_hc and not sig('BBvsHC') and not tc_bb:
        return 'undetermined'
    if sig('BBvsHC') and not sig('TCvsHC'):
        return 'BB-only difference'
    return 'other pattern (see table)'


def hc3(y, X, term=1):
    """OLS with HC3 standard errors for one coefficient (used by the within-group training-dose models)."""
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, XtXi, X)
    V = XtXi @ (X.T * (e ** 2 / (1 - h) ** 2)) @ X @ XtXi
    se = np.sqrt(np.diag(V)); t = b / se; df = len(y) - X.shape[1]; tc = stats.t.ppf(.975, df)
    return dict(Beta=b[term], SE_HC3=se[term], T=t[term], P=2 * stats.t.sf(abs(t[term]), df),
                CI95_Lower=b[term] - tc * se[term], CI95_Upper=b[term] + tc * se[term], DF=df)


def welch_anova_p(groups):
    """P value of Welch's one-way ANOVA (unequal variances)."""
    k = len(groups)
    n = np.array([len(g) for g in groups]); m = np.array([g.mean() for g in groups]); v = np.array([g.var(ddof=1) for g in groups])
    w = n / v
    mw = (w * m).sum() / w.sum()
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    F = a / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
    return stats.f.sf(F, k - 1, (k ** 2 - 1) / (3 * lam))
