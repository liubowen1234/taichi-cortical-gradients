"""Fig. 5 - the visual-somatosensory axis scales with years of practice only in Tai Chi practitioners.

a  G2 span vs training years, TC | BB (group-coloured points; black OLS line + grey 95% CI band if the within-group
   FDR q < .05, dashed line without band otherwise)
b  VIS2-SMN FC vs training years, TC | BB (same encoding)
c  dose maps: partial r of parcel G2 with training years (age, sex, mean FD controlled), TC and BB rows
d  standardised beta [95% CI] of the seven pre-specified dose metrics, TC and BB (filled = within-group FDR q < .05)
e  TC dose map vs TC - HC group-effect map; points coloured by canonical G2; VIS2 outlined
Numbers are read from the result tables (no recomputation).
Usage: python pipeline/07_figures/fig5_training_dose.py [--raster-surfaces] -> 07_figures/Fig5_training_dose.pdf / .png
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, cm
from matplotlib.lines import Line2D
from scipy import stats
import fig_common as tg
import sydnor_palette as SP

tg.setup()
plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'text.color': SP.INK, 'axes.labelcolor': SP.INK, 'xtick.color': SP.INK, 'ytick.color': SP.INK,
                     'axes.edgecolor': SP.INK})
OUT = tg.OUT
NAME = 'Fig5_training_dose'
L = tg.LAYOUT; R6 = L.DOSE

# ------------------------------------------------------------------ data
cov = pd.read_csv(L.COHORT / 'subject_covariates.csv'); grp = cov.Group.values
TR = pd.read_csv(L.COHORT / 'subject_training.csv').set_index('participant_id')
years = TR.reindex(cov.participant_id).TrainYears.values
SM = tg.subject_metrics()
BR = pd.read_csv(L.VIS2 / 'subject_Visual2_networkFC_and_bridge_inputs.csv')
assert (SM.Subject.values == cov.participant_id.values).all() and (BR.Subject.values == cov.participant_id.values).all()
DOSE = pd.read_csv(R6 / 'dose_within_group_3group.csv'); DOSE = DOSE[DOSE.Scheme == 'A'].set_index(['Group', 'Metric'])
LOO = pd.read_csv(R6 / 'dose_G2range_leave_one_out.csv')
MAPS = pd.read_csv(R6 / 'parcel_G2_dose_maps_TC_BB.csv').sort_values('Parcel358')
SPC = pd.read_csv(R6 / 'dose_map_spatial_correlations.csv').set_index('Comparison')
P2 = pd.read_csv(tg.SA / 'Step5_parcel_G2_358_3group.csv').sort_values('Parcel')
REF = pd.read_csv(L.CONTEXT / 'parcellated_reference_maps_Glasser358.csv').sort_values('Parcel358')
net = tg.network_of_parcels(); v2 = net == 'Visual2'
assert (MAPS.Network.values == net).all() and (REF.Parcel358.values == P2.Parcel.values).all()
src = {'note': 'Fig. 5 source values (main analysis). beta_std, P, q from dose_within_group_3group.csv; '
               'spatial r from dose_map_spatial_correlations.csv.'}

W, H = 180, 104
fig, ax_, _, _ = tg.newfig(W, H)
LAB = lambda x, y, s: fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight='bold', va='top')
TXT = lambda x, y, s, **k: fig.text(x / W, 1 - y / H, s, **{**dict(fontsize=6.5, color=SP.INK, va='top'), **k})


def fmt_p(p):
    if p >= 0.001:
        return f'{p:.2g}'
    e = int(np.floor(np.log10(p))); return f'{p / 10 ** e:.1f} × 10$^{{−{-e}}}$'


def fit_band(ax, x, y, sig, n=60):
    X = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b; s2 = e @ e / (len(x) - 2); iXX = np.linalg.inv(X.T @ X)
    xs = np.linspace(x.min(), x.max(), n); Xs = np.column_stack([np.ones(n), xs])
    se = np.sqrt(np.sum(Xs @ iXX * Xs, 1) * s2); t = stats.t.ppf(.975, len(x) - 2)
    if sig:
        ax.fill_between(xs, Xs @ b - t * se, Xs @ b + t * se, color=SP.CIBAND, alpha=0.3, lw=0, zorder=1)
    ax.plot(xs, Xs @ b, color=SP.INK, lw=0.8, ls='-' if sig else (0, (3, 2)), zorder=4)


def dose_scatter(ax, g, y, metric, ylab=None):
    m = (grp == g) & np.isfinite(years)
    x, yy = years[m], y[m]
    jit = np.random.default_rng(5).uniform(-0.15, 0.15, m.sum())          # integer years: small horizontal jitter
    ax.scatter(x + jit, yy, s=6, color=SP.GROUPCOL[g], ec=SP.GROUPEDGE[g], lw=0.3, zorder=3)
    r = DOSE.loc[(g, metric)]; sig = r.Q_FDR_family < .05
    fit_band(ax, x, yy, sig)
    ax.text(0.04, 0.97, f'β$_{{\\mathrm{{std}}}}$ = {r.StandardizedBeta:.2f}'.replace('-', '−') + f'\nP = {fmt_p(r.P)}',
            transform=ax.transAxes, fontsize=6, va='top', linespacing=1.3)
    ax.set_title(f'{g} (n = {m.sum()})', fontsize=6.5, fontweight='bold', pad=2)
    ax.set_xlim(3.8, 15.2); ax.set_xticks([5, 8, 11, 14])
    if ylab:
        ax.set_ylabel(ylab)
    else:
        ax.set_yticklabels([])
    src[f'{g}|{metric}'] = dict(beta_std=float(r.StandardizedBeta), P=float(r.P), q=float(r.Q_FDR_family), n=int(m.sum()),
                                years_range=[float(x.min()), float(x.max())])


def pair(x0, y, metric, vals, ylab, label):
    axs = [ax_(x0 + k * 35, Y1, 32, H1) for k in range(2)]
    for k, g in enumerate(['TC', 'BB']):
        dose_scatter(axs[k], g, vals, metric, ylab if k == 0 else None)
    yl = (min(a.get_ylim()[0] for a in axs), max(a.get_ylim()[1] for a in axs))
    for a in axs:
        a.set_ylim(yl[0], yl[1] + (yl[1] - yl[0]) * 0.25)
    TXT(x0 + 33.5, Y1 + H1 + 5.5, 'Training (years)', ha='center')
    LAB(x0 - 13, 1, label)


# ------------------------------------------------------------------ a, b dose scatters
Y1, H1 = 6, 30
pair(15, Y1, 'G2 range', SM.G2_RobustRange.values, 'G2 range (Q95 − Q05)', 'a')
pair(109, Y1, 'VIS2–SMN FC', BR.Visual2_Somatomotor.values, 'VIS2–SMN FC (Fisher z)', 'b')
src['G2range_LOO'] = LOO[LOO.Scheme == 'A'].set_index('Group')[['LOO_P_min', 'LOO_P_max']].to_dict('index')

# ------------------------------------------------------------------ c dose maps
Y2 = 55
B = tg.Brain(); vmax = 0.5
BW = 14.5; BH = BW * 22.5 / 31; BX = 9
NORM = mcolors.Normalize(-vmax, vmax)
for k, (_, _, title) in enumerate(tg.VIEWS):
    TXT(BX + k * (BW + 0.4) + BW / 2, Y2 - 2.6, title.replace('Left ', 'L ').replace('Right ', 'R '), ha='center', fontsize=6)
for r_, g in enumerate(['TC', 'BB']):
    yb = Y2 + r_ * (BH + 1.5)
    B.row(ax_, BX, yb, BW, BH, SP.DIV(NORM(np.clip(MAPS[f'{g}_dose_partial_r'].values, -vmax, vmax)))[:, :3], gap=0.4)
    TXT(BX - 1, yb + BH / 2 + 1.2, g, ha='right', fontweight='bold')
cb = fig.colorbar(cm.ScalarMappable(norm=NORM, cmap=SP.DIV), cax=ax_(BX + 8, Y2 + 2 * BH + 5, 30, 1.3), orientation='horizontal')
cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, width=0.5, length=2, pad=1)
cb.solids.set_rasterized(False); cb.solids.set_edgecolor('face'); cb.set_ticks([-0.5, 0, 0.5]); cb.set_ticklabels(['−0.5', '0', '0.5'])
TXT(BX + 39.5, Y2 + 2 * BH + 4.8, 'Partial r (parcel G2 ~ years)', fontsize=6)
LAB(1, Y2 - 6, 'c')
src['c'] = {f'{g}_VIS2_mean_partial_r': float(MAPS.loc[v2, f'{g}_dose_partial_r'].mean()) for g in ['TC', 'BB']}

# ------------------------------------------------------------------ d standardised beta forest
mets = ['G2 range', 'λ2', 'VIS2 mean G2', 'VIS2–SMN FC', 'VIS2 segregation', 'G1 range', 'G3 range']
DX, DY, DW, DH = 95, Y2 - 1, 30, 38
ax = ax_(DX, DY, DW, DH)
ax.axvline(0, color='#888888', lw=0.5, ls=':'); ax.grid(axis='x', color=SP.GRID, lw=0.5); ax.set_axisbelow(True)
for i, m in enumerate(mets):
    for g, off in [('TC', -0.17), ('BB', 0.17)]:
        r = DOSE.loc[(g, m)]; k = r.StandardizedBeta / r.Beta
        lo, hi = sorted([r.CI95_Lower * k, r.CI95_Upper * k]); sig = r.Q_FDR_family < .05
        ax.hlines(i + off, lo, hi, color=SP.GROUPEDGE[g], lw=0.7, zorder=2)
        ax.scatter(r.StandardizedBeta, i + off, s=14, color=SP.GROUPCOL[g] if sig else 'white', ec=SP.GROUPEDGE[g], lw=0.7, zorder=3)
        src[f'd|{g}|{m}'] = dict(beta_std=round(float(r.StandardizedBeta), 3), ci=[round(lo, 3), round(hi, 3)], q=float(r.Q_FDR_family))
ax.set_yticks(range(len(mets))); ax.set_yticklabels(mets); ax.set_ylim(len(mets) - 0.5, -0.6)
ax.tick_params(axis='y', length=0, pad=1); ax.spines['left'].set_visible(False)
ax.set_xlim(-0.85, 0.85); ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8]); ax.set_xticklabels(['−0.8', '', '0', '', '0.8'])
ax.set_xlabel('Standardised β [95% CI]')
h_ = [Line2D([], [], marker='o', ls='', ms=3.8, mfc=SP.GROUPCOL[g], mec=SP.GROUPEDGE[g]) for g in ['TC', 'BB']] + \
     [Line2D([], [], marker='o', ls='', ms=3.8, mfc='white', mec='#666666')]
ax.legend(h_, ['TC', 'BB', 'q ≥ 0.05'], frameon=False, fontsize=6, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3,
          handletextpad=0.1, columnspacing=0.8, borderaxespad=0.2)
LAB(DX - 25, Y2 - 6, 'd')

# ------------------------------------------------------------------ e dose map vs group-effect map (TC)
EX, EW = 143, 34
ax = ax_(EX, DY, EW, DH)
x = P2.TCvsHC_g.values; y = MAPS.TC_dose_partial_r.values; z = REF['neuromaps::Margulies_G2'].values
cn = mcolors.Normalize(np.percentile(z, 2), np.percentile(z, 98))
ax.axhline(0, color=SP.GRID, lw=0.6, zorder=0); ax.axvline(0, color=SP.GRID, lw=0.6, zorder=0)
ax.scatter(x[~v2], y[~v2], s=4, c=SP.SEQ(cn(z[~v2])), lw=0, zorder=2)
ax.scatter(x[v2], y[v2], s=5, c=SP.SEQ(cn(z[v2])), ec=SP.INK, lw=0.4, zorder=3)
s = SPC.loc['TC dose map vs TC–HC g map']
ax.text(0.04, 0.97, f'r = {s.r:.2f}, $P_{{\\mathrm{{spin}}}}$ = 1 × 10$^{{−4}}$\nG2 removed: r = {s.r_G2axis_removed:.2f}', transform=ax.transAxes,
        fontsize=6, va='top', linespacing=1.4)
ax.text(0.97, 0.03, 'Outlined: VIS2', transform=ax.transAxes, fontsize=6, ha='right', va='bottom')
yl = ax.get_ylim(); ax.set_ylim(yl[0], yl[1] + 0.28 * (yl[1] - yl[0]))
ax.set_xlabel('TC − HC (Hedges g)'); ax.set_ylabel('TC dose map (partial r)')
LAB(EX - 11, Y2 - 6, 'e')
src['e'] = SPC.round(4).to_dict('index')

tg.save(fig, OUT, NAME)
(OUT / f'{NAME}_source_values.json').write_text(json.dumps(src, indent=1, ensure_ascii=False, default=float))
