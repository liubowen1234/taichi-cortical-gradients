"""Step 3.1 - network segregation and participation coefficient (12 CAB-NP networks).

Edge handling (main version): Fisher-z FC, diagonal excluded, negative weights set to 0.
  Segregation of network k = (W - B) / W, W = mean FC between parcels inside k, B = mean FC of k's parcels with all others.
  Participation coefficient (PC) of a parcel = 1 - sum over networks of (strength to that network / total strength)^2,
  averaged over the parcels of each network.
  Whole-cortex segregation (same formula with all within- and between-network edges) and modularity Q of the fixed
  CAB-NP partition; both again with Visual2 removed, to test whether the whole-cortex effect comes from Visual2.
Also computed: signed-edge versions, within-module degree z (WMD), and proportional thresholds (5-40% density) as
sensitivity analyses.
Families: each metric across the 12 networks; the two whole-cortex metrics; the two whole-cortex metrics without Visual2;
358 parcels (PC, WMD). Statistics: common/stats_engine.py.

Outputs (03_network_connectivity/segregation/)
  subject_network_segregation.csv, subject_density_sensitivity.csv, subject_parcel_PC.csv (subject-level values)
  stats_three_group[_plus<cov>]/stats_<family>_3group.csv, family_summary_3group.csv

Usage: python pipeline/03_network_connectivity/s01_network_segregation.py [--covar eTIV]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from common.config import LAYOUT, mkdir
from common import stats_engine as ts
from common.data import covariates, cabnp_mapping, load_fc, NETS

EXTRA = sys.argv[sys.argv.index('--covar') + 1] if '--covar' in sys.argv else None
MOD = mkdir(LAYOUT.SEG)
OUT = mkdir(MOD / ('stats_three_group' + (f'_plus{EXTRA}' if EXTRA else '')))
DENS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

cab = cabnp_mapping()
lab = cab.CABNP_Network.values
ci = np.array([NETS.index(x) for x in lab])             # network index of every parcel
cov = covariates()
grp = cov.Group.values
NS = len(cov)
R = load_fc(LAYOUT.GRAD_A / 'FC.mat')                    # subjects x 358 x 358, Pearson r
Z = np.arctanh(np.clip(R, -1 + 1e-7, 1 - 1e-7))
for s in range(NS):
    np.fill_diagonal(Z[s], 0.0)
N = 358
iu = np.triu_indices(N, 1)
same = ci[:, None] == ci[None, :]                        # True for within-network pairs


# ---------------------------------------------------------------- metric functions
def segregation(A, k):
    m = ci == k
    Wm = A[np.ix_(m, m)][np.triu_indices(m.sum(), 1)].mean()
    Bm = A[np.ix_(m, ~m)].mean()
    return Wm, Bm, (Wm - Bm) / Wm


def global_segregation(A):
    w = A[iu][same[iu]].mean()
    b = A[iu][~same[iu]].mean()
    return w, b, (w - b) / w


def global_segregation_excluding(A, k_excl):
    keep = ci != k_excl
    Ak = A[np.ix_(keep, keep)]; cik = ci[keep]; iu_ = np.triu_indices(keep.sum(), 1)
    sm = (cik[:, None] == cik[None, :])[iu_]; v = Ak[iu_]
    w, b = v[sm].mean(), v[~sm].mean()
    return (w - b) / w


def modularity_fixed(A):
    """Modularity Q of the fixed CAB-NP partition (weighted, undirected)."""
    s = A.sum(1); v = s.sum()
    return float(((A - np.outer(s, s) / v) * same).sum() / v)


def modularity_fixed_excluding(A, k_excl):
    keep = ci != k_excl
    Ak = A[np.ix_(keep, keep)]; cik = ci[keep]
    s_ = Ak.sum(1); v = s_.sum()
    return float(((Ak - np.outer(s_, s_) / v) * (cik[:, None] == cik[None, :])).sum() / v)


def participation(A):
    k = A.sum(1)
    kis = np.stack([A[:, ci == m].sum(1) for m in range(12)], 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        pc = 1 - ((kis / k[:, None]) ** 2).sum(1)
    pc[k == 0] = 0
    return pc


def wmd_z(A):
    """Within-module degree z score of every parcel."""
    z = np.zeros(N)
    for m in range(12):
        idx = np.where(ci == m)[0]
        kin = A[np.ix_(idx, idx)].sum(1)
        sd = kin.std()
        z[idx] = (kin - kin.mean()) / sd if sd > 0 else 0
    return z


def threshold_proportional(W, p):
    """Keep the strongest proportion p of all edges (signed values), zero the rest."""
    W = W.copy(); np.fill_diagonal(W, 0)
    vals = W[iu]; n_keep = int(round(p * len(vals)))
    order = np.argsort(vals)[::-1]
    keep = np.zeros(len(vals), bool); keep[order[:n_keep]] = True
    out = np.zeros_like(W); out[iu[0][keep], iu[1][keep]] = vals[keep]
    return out + out.T


# ---------------------------------------------------------------- subject level
rows, dens_rows, pc_all, wmd_all, pc15_all = [], [], [], [], []
for s in range(NS):
    Ap = np.maximum(Z[s], 0); np.fill_diagonal(Ap, 0)        # positive-only (main version)
    As = Z[s].copy()                                          # signed (sensitivity)
    r = {'Subject': cov.participant_id[s], 'Group': grp[s]}
    for k, n in enumerate(NETS):
        w, b, sg = segregation(Ap, k); r[f'{n}_Within'] = w; r[f'{n}_Between'] = b; r[f'{n}_Segregation'] = sg
        w, b, sg = segregation(As, k); r[f'{n}_Within_signed'] = w; r[f'{n}_Between_signed'] = b; r[f'{n}_Segregation_signed'] = sg
    w, b, sg = global_segregation(Ap); r.update(Global_Within=w, Global_Between=b, Global_Segregation=sg)
    r['Modularity_Q_CABNP'] = modularity_fixed(Ap)
    r['Global_Segregation_exclVisual2'] = global_segregation_excluding(Ap, 1)
    r['Modularity_Q_exclVisual2'] = modularity_fixed_excluding(Ap, 1)
    rows.append(r)
    pc_all.append(participation(Ap)); wmd_all.append(wmd_z(Ap))
    for p in DENS:
        At = threshold_proportional(Z[s], p)
        w, b, sg = segregation(At, 1)
        dens_rows.append(dict(Subject=r['Subject'], Group=grp[s], Density=p, Visual2_Within=w, Visual2_Between=b,
                              Visual2_Segregation=sg, Global_Segregation=global_segregation(At)[2]))
        if p == 0.15:
            pc15_all.append(participation(At))
S = pd.DataFrame(rows); D = pd.DataFrame(dens_rows)
PC = np.array(pc_all); WMD = np.array(wmd_all); PC15 = np.array(pc15_all)
if not EXTRA:
    S.to_csv(MOD / 'subject_network_segregation.csv', index=False)
    D.to_csv(MOD / 'subject_density_sensitivity.csv', index=False)
    pd.DataFrame(PC, columns=cab.GlasserLabel.values).assign(Subject=S.Subject).to_csv(MOD / 'subject_parcel_PC.csv', index=False)

# ---------------------------------------------------------------- statistics
ENG = ts.Engine(ts.design(cov, EXTRA), grp)
MODEL = 'Group(HC ref) + Age + Sex + MeanFD' + (f' + {EXTRA}' if EXTRA else '')


def family_table(values, family, keys=('Metric',)):
    rows_ = []
    for k, y in values.items():
        kk = k if isinstance(k, tuple) else (k,)
        rows_.append({'Family': family, **dict(zip(keys, kk)), 'Model': MODEL, **ENG.analyse(np.asarray(y, float))})
    return ts.finalize_family(pd.DataFrame(rows_), 'Family')


fam, summ_all = {}, []
spec = [('Network_Segregation', {n: S[f'{n}_Segregation'] for n in NETS}, ('Network',)),
        ('Network_Within', {n: S[f'{n}_Within'] for n in NETS}, ('Network',)),
        ('Network_Between', {n: S[f'{n}_Between'] for n in NETS}, ('Network',)),
        ('Network_Segregation_signed', {n: S[f'{n}_Segregation_signed'] for n in NETS}, ('Network',)),
        ('Network_Within_signed', {n: S[f'{n}_Within_signed'] for n in NETS}, ('Network',)),
        ('Network_Between_signed', {n: S[f'{n}_Between_signed'] for n in NETS}, ('Network',)),
        ('Global', {'Global_Segregation': S.Global_Segregation, 'Modularity_Q_CABNP': S.Modularity_Q_CABNP}, ('Metric',)),
        ('Global_exclVisual2', {'Global_Segregation_exclVisual2': S.Global_Segregation_exclVisual2,
                                'Modularity_Q_exclVisual2': S.Modularity_Q_exclVisual2}, ('Metric',)),
        ('Network_PC', {n: PC[:, ci == k].mean(1) for k, n in enumerate(NETS)}, ('Network',)),
        ('Network_PC_dens15', {n: PC15[:, ci == k].mean(1) for k, n in enumerate(NETS)}, ('Network',))]
for name, vals, keys in spec:
    fam[name], s_ = family_table(vals, name, keys); summ_all += s_
parc = pd.DataFrame({'Parcel': np.arange(1, 359), 'Label': cab.GlasserLabel.values, 'Network': lab})
for nm, M_ in [('PC', PC), ('WMD', WMD), ('PC_dens15', PC15)]:
    O, s_ = family_table({p + 1: M_[:, p] for p in range(N)}, f'Parcel_{nm}', ('Parcel',)); summ_all += s_
    O = parc.merge(O, on='Parcel')
    v2 = O.Network == 'Visual2'
    O['Omnibus_Q_FDR_Visual2only'] = np.nan
    O.loc[v2, 'Omnibus_Q_FDR_Visual2only'] = ts.bh(O.loc[v2, 'Omnibus_P'].values)
    fam[f'Parcel_{nm}'] = O
dres = []                                                   # density sensitivity: reported without FDR
for p in DENS:
    d = D[D.Density == p]
    for m in ['Visual2_Within', 'Visual2_Between', 'Visual2_Segregation', 'Global_Segregation']:
        dres.append({'Density': p, 'Metric': m, **ENG.analyse(d[m].values)})
fam['Density_sensitivity'] = pd.DataFrame(dres)
for k_, v_ in fam.items():
    v_.to_csv(OUT / f'stats_{k_}_3group.csv', index=False)
pd.DataFrame(summ_all).to_csv(OUT / 'family_summary_3group.csv', index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
show = ['Omnibus_Q_FDR', 'TCvsHC_g', 'TCvsHC_P_Holm', 'BBvsHC_g', 'BBvsHC_P_Holm', 'BBvsHC_P_TOST', 'TCvsBB_g', 'TCvsBB_P_Holm']
for k_ in ['Network_Segregation', 'Network_PC']:
    print('\n==', k_); print(fam[k_][['Network'] + show].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
