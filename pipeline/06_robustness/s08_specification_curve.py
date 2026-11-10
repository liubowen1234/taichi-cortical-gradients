"""Step 6.8 - specification-curve (multiverse) analysis of the G2 findings.

Specifications (all aligned to the non-athlete reference; 2 x 5 x 5 x 5 x 4 = 1,000):
  FC input     Pearson r* | Fisher z
  kernel       cosine* | normalised angle | Pearson | Spearman | Gaussian
  embedding    diffusion map alpha = 0.5* | alpha = 0 | alpha = 1 | Laplacian eigenmaps | PCA
  sparsity     90* | 0 | 50 | 75 | 95  (percentile of each connectivity profile kept)
  n aligned    3* | 2 | 5 | 10 components rotated onto the reference
  (* = main analysis). Embeddings use common/embedlib.py (Python version of BrainSpace).
Outcomes per specification: G2 span and VIS2 mean G2 -> Metric ~ TC + BB + Age + Sex + MeanFD (HC3): t, P and Hedges' g
of TC - HC, BB - HC, TC - BB, omnibus P; within-group training dose of G2 span (TC, BB); lambda2 of the raw embedding.
Inference across specifications (Simonsohn et al., 2020): for each contrast the labels of the two groups involved are
permuted (the third group stays fixed); for dose, training years are permuted within the group; the references stay fixed.
500 permutations; statistics = median OLS t and number of specifications with P < 0.05 in the observed (positive) direction.
The full run takes many hours; MV_QUICK=1 runs the main embedding only (for testing).

Outputs (06_robustness/multiverse/): multiverse_specifications_HCref_3group.csv, multiverse_lambda2_3group.csv,
multiverse_subject_values_HCref_3group.npz, multiverse_inference_HCref_3group.json
Usage: python pipeline/06_robustness/s08_specification_curve.py [n_perm (default 500)]
"""
import sys, os, json, time, itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import h5py
from scipy import stats
from common.config import LAYOUT, mkdir
from common import gradlib as gl, embedlib as el
from common.data import covariates, training_years, glasser_atlas, cabnp_mapping, load_fc

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = mkdir(LAYOUT.MULTIVERSE / ('quick' if os.environ.get('MV_QUICK') else ''))
FC = load_fc(LAYOUT.GRAD_A / 'FC.mat')                    # subjects x 358 x 358
NS = FC.shape[0]
yeo = glasser_atlas().community_yeo.str.strip().str.lower().values
v2 = cabnp_mapping().CABNP_Network.values == 'Visual2'
cov = covariates()
grp = cov.Group.values
age, sex, fd = cov.Age.values.astype(float), cov.Sex.values - 1.0, cov.MeanFD.values
yrs = training_years(cov)
tcm, bbm, hcm = grp == 'TC', grp == 'BB', grp == 'HC'
X = np.column_stack([np.ones(NS), tcm, bbm, age, sex, fd]).astype(float)
CONTR = {'TCvsHC': np.r_[0, 1, 0, 0, 0, 0], 'BBvsHC': np.r_[0, 0, 1, 0, 0, 0], 'TCvsBB': np.r_[0, 1, -1, 0, 0, 0]}
PAIRG = {'TCvsHC': ('TC', 'HC'), 'BBvsHC': ('BB', 'HC'), 'TCvsBB': ('TC', 'BB')}
INPUTS = ['r', 'z']; KERNELS = ['cosine', 'angle', 'pearson', 'spearman', 'gaussian']
APPROACH = ['dm', 'dm_a0', 'dm_a1', 'le', 'pca']; SPARS = [90, 0, 50, 75, 95]
NCOMP = [3, 2, 5, 10]
if os.environ.get('MV_QUICK'):
    INPUTS, KERNELS, APPROACH, SPARS = ['r'], ['cosine'], ['dm'], [90]
PRIMARY = dict(input='r', kernel='cosine', approach='dm', sparsity=90, alignment='HC reference', n_aligned=3)
with h5py.File(LAYOUT.GRAD_A / 'Step3_canonical_G1G2/BrainSpace_step3_G1G2_ONLY_results.mat', 'r') as h:
    R3P = h['reference_canonical'][:].T                  # main-analysis reference G1 / G2 (358 x 2)
Z = np.arctanh(np.clip(FC, -0.999999, 0.999999)); np.einsum('sii->si', Z)[:] = 0
R0 = FC.copy(); np.einsum('sii->si', R0)[:] = 0
del FC


def ref_matrix(idx, inp):
    """Mean FC of the reference subjects (Fisher-z average; back to r for r input)."""
    m = Z[idx].mean(0)
    M = np.tanh(m) if inp == 'r' else m
    np.fill_diagonal(M, 0)
    return (M + M.T) / 2


def hc3_contrasts(y, Xd):
    XtXi = np.linalg.inv(Xd.T @ Xd); b = XtXi @ Xd.T @ y; e = y - Xd @ b
    h = np.minimum(np.einsum('ij,jk,ik->i', Xd, XtXi, Xd), .999999)
    V = XtXi @ (Xd.T * ((e / (1 - h)) ** 2)) @ Xd @ XtXi; df = len(y) - Xd.shape[1]
    out = {}
    for k, L in CONTR.items():
        t = (L @ b) / np.sqrt(L @ V @ L)
        out[k] = (t, 2 * stats.t.sf(abs(t), df))
    Rm = np.zeros((2, Xd.shape[1])); Rm[0, 1] = Rm[1, 2] = 1; rb = Rm @ b
    F = float(rb @ np.linalg.solve(Rm @ V @ Rm.T, rb)) / 2
    return out, stats.f.sf(F, 2, df)


def hedges(y, a, c):
    ya, yc = y[grp == a], y[grp == c]; n1, n2 = len(ya), len(yc)
    sp = np.sqrt(((n1 - 1) * ya.var(ddof=1) + (n2 - 1) * yc.var(ddof=1)) / (n1 + n2 - 2))
    return (ya.mean() - yc.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9))


def dose(y, m):
    Xd = np.column_stack([np.ones(m.sum()), yrs[m], age[m], sex[m], fd[m]])
    r = gl.hc3(y[m], Xd)
    return r['Beta'] * yrs[m].std(ddof=1) / y[m].std(ddof=1), r['T_HC3'], r['P_HC3']


rows, YR, YV, LAMROWS = [], [], [], []
t0 = time.time()
for inp, ker, app, sp in itertools.product(INPUTS, KERNELS, APPROACH, SPARS):
    src = R0 if inp == 'r' else Z
    try:
        emb = [el.embed(src[s], ker, sp, app, 10) for s in range(NS)]
        raws = [e[0] for e in emb]
        lam2 = np.array([e[1][1] for e in emb])
        ref = el.embed(ref_matrix(np.where(hcm)[0], inp), ker, sp, app, 10)[0]
    except Exception as ex:
        rows.append(dict(input=inp, kernel=ker, approach=app, sparsity=sp, status=f'failed: {ex}'))
        continue
    lc, lp = hc3_contrasts(lam2, X)
    LAMROWS.append(dict(input=inp, kernel=ker, approach=app, sparsity=sp, omnibus_p=lp,
                        **{f'lambda2_{k}_{s}': v for k in CONTR for s, v in zip(('t', 'p'), lc[k])},
                        **{f'lambda2_{k}_g': hedges(lam2, *PAIRG[k]) for k in CONTR}))
    for k in NCOMP:
        refk = ref[:, :k]
        al = np.stack([r[:, :k] @ gl.procrustes_R(r[:, :k], refk) for r in raws])
        try:
            ref_o, G2d, mapping, sign, _ = gl.orient(refk, al, yeo)
        except AssertionError:
            rows.append(dict(input=inp, kernel=ker, approach=app, sparsity=sp, alignment='HC reference', n_aligned=k,
                             status='orientation failed'))
            continue
        g2 = G2d[:, :, 1]
        span = np.percentile(g2, 95, axis=1) - np.percentile(g2, 5, axis=1)
        vis = g2[:, v2].mean(1)
        row = dict(input=inp, kernel=ker, approach=app, sparsity=sp, alignment='HC reference', n_aligned=k, status='ok',
                   G2_component=int(mapping[1]) + 1, G2_ref_r_with_primary=float(np.corrcoef(ref_o[:, 1], R3P[:, 1])[0, 1]))
        for nm, y in (('range', span), ('vis2', vis)):
            c, op = hc3_contrasts(y, X)
            row[f'{nm}_omnibus_p'] = op
            for kk in CONTR:
                row[f'{nm}_{kk}_t'], row[f'{nm}_{kk}_p'] = c[kk]
                row[f'{nm}_{kk}_g'] = hedges(y, *PAIRG[kk])
        for gname, m in (('TC', tcm), ('BB', bbm)):
            row[f'dose_{gname}_beta_std'], row[f'dose_{gname}_t'], row[f'dose_{gname}_p'] = dose(span, m)
        rows.append(row); YR.append(span); YV.append(vis)
    print(f'{inp} {ker:9s} {app:6s} {sp:3d}  {time.time() - t0:6.0f}s', flush=True)

S = pd.DataFrame(rows)
ok = S.status == 'ok'
S['is_primary'] = ok & np.all([S[k] == v for k, v in PRIMARY.items()], axis=0)
S.to_csv(OUT / 'multiverse_specifications_HCref_3group.csv', index=False)
pd.DataFrame(LAMROWS).to_csv(OUT / 'multiverse_lambda2_3group.csv', index=False)
YR, YV = np.array(YR), np.array(YV)
np.savez_compressed(OUT / 'multiverse_subject_values_HCref_3group.npz', G2_range=YR.astype(np.float32), VIS2_meanG2=YV.astype(np.float32))
# stored as float32 and read back, so that the inference uses exactly the stored subject values
YR, YV = YR.astype(np.float32).astype(float), YV.astype(np.float32).astype(float)


# ---- specification-curve inference
def ols_t(Y, Xd, L):
    """OLS t of contrast L for every row of Y (specifications x subjects)."""
    XtXi = np.linalg.inv(Xd.T @ Xd); B = Y @ (XtXi @ Xd.T).T; E = Y - B @ Xd.T; df = Xd.shape[0] - Xd.shape[1]
    s2 = (E ** 2).sum(1) / df
    return (B @ L) / np.sqrt(s2 * (L @ XtXi @ L))


def summ(T, crit):
    return float(np.median(T)), int((T > crit).sum())


crit = stats.t.ppf(.975, NS - 6)
rngp = np.random.default_rng(20261003)
INF = {}
for nm, Y in (('range', YR), ('vis2', YV)):
    for k, L in CONTR.items():
        a, c = PAIRG[k]
        obs = summ(ols_t(Y, X, L), crit)
        idx = np.where((grp == a) | (grp == c))[0]
        null = []
        for _ in range(NPERM):
            gp = grp.copy(); gp[idx] = rngp.permutation(grp[idx])
            Xp = np.column_stack([np.ones(NS), gp == 'TC', gp == 'BB', age, sex, fd]).astype(float)
            null.append(summ(ols_t(Y, Xp, L), crit))
        nmn = np.array(null)
        INF[f'{nm}_{k}'] = dict(observed_median_t=obs[0], observed_n_sig_positive=obs[1], n_specs=int(len(Y)),
                                p_median_t=float((1 + (nmn[:, 0] >= obs[0]).sum()) / (1 + NPERM)),
                                p_n_sig=float((1 + (nmn[:, 1] >= obs[1]).sum()) / (1 + NPERM)),
                                null_median_t_95=float(np.percentile(nmn[:, 0], 95)), null_n_sig_95=float(np.percentile(nmn[:, 1], 95)))
for gname, m in (('TC', tcm), ('BB', bbm)):
    Cd = np.column_stack([np.ones(m.sum()), yrs[m], age[m], sex[m], fd[m]]); Ld = np.r_[0, 1, 0, 0, 0]
    critd = stats.t.ppf(.975, m.sum() - 5)
    obs = summ(ols_t(YR[:, m], Cd, Ld), critd)
    null = []
    for _ in range(NPERM):
        Cp = Cd.copy(); Cp[:, 1] = rngp.permutation(yrs[m]); null.append(summ(ols_t(YR[:, m], Cp, Ld), critd))
    nmn = np.array(null)
    INF[f'dose_{gname}'] = dict(observed_median_t=obs[0], observed_n_sig_positive=obs[1],
                                p_median_t=float((1 + (nmn[:, 0] >= obs[0]).sum()) / (1 + NPERM)),
                                p_n_sig=float((1 + (nmn[:, 1] >= obs[1]).sum()) / (1 + NPERM)))
so = S[ok]
sig = lambda c, sgn=1: float(100 * ((so[f'{c}_p'] < .05) & (np.sign(so[f'{c}_t']) == sgn)).mean())
INF['summary'] = dict(n_attempted=int((S.alignment == 'HC reference').sum()), n_ok=int(len(so)), n_perm=NPERM,
                      **{f'{nm}_{k}_pct_sig_pos': sig(f'{nm}_{k}') for nm in ('range', 'vis2') for k in CONTR},
                      **{f'{nm}_{k}_pct_sig_neg': sig(f'{nm}_{k}', -1) for nm in ('range', 'vis2') for k in CONTR},
                      **{f'{nm}_{k}_median_g': float(so[f'{nm}_{k}_g'].median()) for nm in ('range', 'vis2') for k in CONTR},
                      **{f'{nm}_{k}_pct_abs_g_lt_0.5': float(100 * (so[f'{nm}_{k}_g'].abs() < .5).mean()) for nm in ('range', 'vis2') for k in CONTR},
                      dose_TC_pct_sig_pos=float(100 * ((so.dose_TC_p < .05) & (so.dose_TC_t > 0)).mean()),
                      dose_BB_pct_sig_pos=float(100 * ((so.dose_BB_p < .05) & (so.dose_BB_t > 0)).mean()),
                      dose_TC_median_beta_std=float(so.dose_TC_beta_std.median()), dose_BB_median_beta_std=float(so.dose_BB_beta_std.median()))
s2 = so[so.approach != 'le']
INF['summary_excl_LE'] = {f'{nm}_{k}_pct_sig_pos': float(100 * ((s2[f'{nm}_{k}_p'] < .05) & (s2[f'{nm}_{k}_t'] > 0)).mean())
                          for nm in ('range', 'vis2') for k in CONTR}
json.dump(INF, open(OUT / 'multiverse_inference_HCref_3group.json', 'w'), indent=2, default=float)
print(json.dumps(INF['summary'], indent=1, default=float))
