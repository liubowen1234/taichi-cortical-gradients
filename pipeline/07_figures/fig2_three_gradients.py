"""Fig. 2 - global measures of the three gradients.

a  reference G1 (sensory-association), G2 (visual-somatosensory), G3 (modulation-representation) on the left hemisphere
b  eigenvalues lambda1-3      c  G1-G3 span (95th - 5th percentile)
d  G1-G2 manifold: mean parcel positions of the non-athletes (coloured by G2) and, per group, an ellipse with semi-axes
   sqrt(mean major- / minor-axis variance), oriented along the principal axis of the non-athlete mean manifold (sketch)
e  major- and minor-axis variance (log scale)      f  anisotropy index
Half-eye plots: density (right), median (white dot), IQR (thick bar), 5th-95th percentile (thin bar), participants (left).
Brackets: Holm P < .001 ***, < .01 **, < .05 * (only when the omnibus FDR q < .05); 'equiv.' when TOST (+/-0.5 g) P < .05;
otherwise 'ns'. Exact g and P are written to Fig2_three_gradients_source_values.json. Nothing is re-tested.
Usage: python pipeline/07_figures/fig2_three_gradients.py -> 07_figures/Fig2_three_gradients.pdf / .png
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
import fig_common as tg
from common import figlib as fl
import sydnor_palette as SP

tg.setup()
plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'text.color': SP.INK, 'axes.labelcolor': SP.INK, 'xtick.color': SP.INK, 'ytick.color': SP.INK,
                     'axes.edgecolor': SP.INK})
OUT = tg.OUT
NAME = 'Fig2_three_gradients'
G = tg.GROUPS
CONTR = [('TCvsHC', 0, 1), ('TCvsBB', 1, 2), ('BBvsHC', 0, 2)]
TIERS = (0.77, 0.85, 0.93)

# ------------------------------------------------------------------ data (main analysis, non-athlete reference)
SM = tg.subject_metrics(); grp = SM.Group.values
s92 = pd.read_csv(tg.GRAD_A / 'step_9.2/step9_2_G3_subject_metrics.csv')
assert (s92.Group.values == grp).all()
SM = SM.assign(G3_RobustRange=s92.G3_RobustRange.values, Eigenvalue_G3=s92.Eigenvalue_G3.values)
G1m = tg.parcel_long_matrix('G1'); G2m = tg.parcel_long_matrix('G2')
with h5py.File(tg.GRAD_A / 'Step3_canonical_G1G2/BrainSpace_step3_G1G2_ONLY_results.mat', 'r') as h:
    R3 = np.array(h['reference_canonical3d']).T
sgn = json.load(open(tg.GRAD_A / 'step_9.2/step9_2_G3_orientation.json'))['sign_flip']
REF = np.column_stack([R3[:, 0], R3[:, 1], sgn * R3[:, 2]])
R = {f'Eigenvalue_G{k}': tg.stat_row('step9_2_eigenvalue_3axes_stats_3group.csv', f'Eigenvalue_G{k}') for k in (1, 2, 3)}
R |= {f'G{k}_RobustRange': tg.stat_row('step9_2_range_3axes_stats_3group.csv', f'G{k}_RobustRange') for k in (1, 2, 3)}
R |= {m: tg.stat_row('Step5_exploratory_global_geometry_3group.csv', m) for m in ('AnisotropyIndex', 'MajorVariance', 'MinorVariance')}


class LeftBrain:
    """Left lateral + medial views of the cortex (Glasser-358 parcels)."""

    def __init__(self):
        self.b = tg.Brain()

    def row(self, ax_, x0, y, w, h, rgb, gap=0.4):
        return self.b.row(ax_, x0, y, w, h, rgb, gap=gap, views=tg.VIEWS[:2])


# ------------------------------------------------------------------ helpers
def stars(r, c):
    p = r[f'{c}_P_Holm']
    if r['Omnibus_Q_FDR'] < .05 and p < .05:
        return '***' if p < .001 else '**' if p < .01 else '*'
    return 'equiv.' if r[f'{c}_P_TOST'] < .05 else 'ns'


src = {'note': 'Fig. 2 source values. Group summaries are of the plotted participant values; statistics are read from '
               'the main-analysis result tables (Metric ~ group + age + sex + mean FD, HC3; omnibus FDR, Holm pairwise, '
               'TOST +/-0.5 g). Bracket labels: ***/**/* Holm P < .001/.01/.05 after omnibus q < .05; equiv. = TOST P < .05.',
       'panels': {}}


def halfeye(ax, col, title, ylab='', log=False, seed=0, panel=''):
    rng = np.random.default_rng(seed); r = R[col]
    allv = SM[col].values.astype(float)
    summ = {}
    for i, g in enumerate(G):
        v = SM.loc[grp == g, col].values.astype(float); v = v[np.isfinite(v)]
        tv = np.log10(v) if log else v
        ys = np.linspace(tv.min(), tv.max(), 200); d = gaussian_kde(tv, 0.35)(ys); d = d / d.max() * 0.40
        yy = 10 ** ys if log else ys
        ax.fill_betweenx(yy, i, i + d, color=SP.GROUPCOL[g], alpha=0.55, lw=0, zorder=1)
        ax.plot(i + d, yy, color=SP.GROUPEDGE[g], lw=0.5, zorder=2)
        ax.scatter(i - 0.05 - rng.uniform(0, 0.2, len(v)), v, s=2.2, color=SP.GROUPCOL[g], ec=SP.GROUPEDGE[g], lw=0.2, zorder=3)
        q1, md, q3 = np.percentile(v, [25, 50, 75]); p5, p95 = np.percentile(v, [5, 95])
        ax.plot([i, i], [p5, p95], color=SP.INK, lw=0.6, zorder=4, solid_capstyle='butt')
        ax.plot([i, i], [q1, q3], color=SP.INK, lw=1.8, zorder=4, solid_capstyle='butt')
        ax.scatter(i, md, s=9, color='white', ec=SP.INK, lw=0.6, zorder=5)
        summ[g] = dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)), median=float(md),
                       q25=float(q1), q75=float(q3), p05=float(p5), p95=float(p95))
    if log:
        ax.set_yscale('log'); lo, hi = np.log10(np.nanmin(allv)), np.log10(np.nanmax(allv))
        ax.set_ylim(10 ** (lo - 0.05 * (hi - lo)), 10 ** (lo + (hi - lo) / 0.72))
    else:
        lo, hi = np.nanmin(allv), np.nanmax(allv); y0 = lo - 0.05 * (hi - lo)
        ax.set_ylim(y0, y0 + (hi - y0) / 0.72)
    tr = ax.get_xaxis_transform(); lab = {}
    for (c, a, b), yf in zip(CONTR, TIERS):
        s = stars(r, c); sig = s.startswith('*'); lab[c] = s
        ax.plot([a, a, b, b], [yf - 0.025, yf, yf, yf - 0.025], color=SP.INK, lw=0.5, transform=tr, clip_on=False)
        ax.text((a + b) / 2, yf - (0.02 if sig else -0.005), s, ha='center', va='bottom', fontsize=6.5 if sig else 5.5,
                transform=tr, clip_on=False)
    ax.set_xticks(range(3)); ax.set_xticklabels(G); ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.35, 2.5); ax.set_title(title, fontsize=6.5, pad=2)
    if ylab: ax.set_ylabel(ylab)
    src['panels'][f'{panel} {title}'] = {
        'metric': col, 'groups': summ, 'omnibus_q_FDR': float(r['Omnibus_Q_FDR']), 'bracket': lab,
        **{c: {'g': float(r[f'{c}_g']), 'P_Holm': float(r[f'{c}_P_Holm']), 'P_TOST': float(r[f'{c}_P_TOST'])}
           for c, _, _ in CONTR}}


# ------------------------------------------------------------------ figure (180 x 146 mm)
W, H = 180, 146
fig, ax_, _, _ = tg.newfig(W, H)
LAB = lambda x, y, s: fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight='bold', va='top')
TXT = lambda x, y, s, **k: fig.text(x / W, 1 - y / H, s, **{**dict(fontsize=6.5, color=SP.INK, va='top'), **k})

# a reference gradients
B = LeftBrain(); BW, BH = 25, 25 * 22.5 / 31
ENDS = {0: ('Sensory', 'Association'), 1: ('Somatosensory', 'Visual'), 2: ('Representation', 'Modulation')}
src['panels']['a reference gradients'] = {}
for j in range(3):
    v = REF[:, j]; vmax = np.percentile(np.abs(v), 98); vmin = -9.0 if j == 0 else -vmax
    x0 = 6 + j * 59
    B.row(ax_, x0, 6, BW, BH, SP.SEQ(mcolors.Normalize(vmin, vmax)(np.clip(v, vmin, vmax)))[:, :3])
    TXT(x0 + BW + 0.2, 2.0, ['G1 (SA axis)', 'G2 (VS axis)', 'G3 (MR axis)'][j], ha='center', fontweight='bold')
    yc = 6 + BH + 1.0; cax = ax_(x0 + 9, yc, 32, 1.3)
    cax.imshow(SP.SEQ(np.linspace(0, 1, 256))[None, :, :3], aspect='auto'); cax.set_xticks([]); cax.set_yticks([])
    [s_.set_linewidth(0.3) for s_ in cax.spines.values()]
    TXT(x0 + 9, yc + 1.9, ENDS[j][0], fontsize=5.5); TXT(x0 + 41, yc + 1.9, ENDS[j][1], fontsize=5.5, ha='right')
    src['panels']['a reference gradients'][f'G{j + 1}'] = {'colour_limits': [float(vmin), float(vmax)],
                                                           'parcel_values_RH_first': [float(x) for x in v]}
LAB(2, 2, 'a')

# b eigenvalues, c ranges
Y2, PW, PH = 40, 22, 36
xs = [12, 39, 66, 104, 131, 158]
cols = [('Eigenvalue_G1', 'λ1 (G1)', 'Eigenvalue', 'b'), ('Eigenvalue_G2', 'λ2 (G2)', '', 'b'), ('Eigenvalue_G3', 'λ3 (G3)', '', 'b'),
        ('G1_RobustRange', 'G1 range', 'Range (Q95 − Q05)', 'c'), ('G2_RobustRange', 'G2 range', '', 'c'),
        ('G3_RobustRange', 'G3 range', '', 'c')]
for k, (x, (c, t, yl, p)) in enumerate(zip(xs, cols)):
    halfeye(ax_(x, Y2, PW, PH), c, t, yl, seed=k, panel=p)
LAB(2, Y2 - 5, 'b'); LAB(94, Y2 - 5, 'c')

# d manifold
Y3 = 94
ax = ax_(12, Y3, 42, 38)
x0, y0 = G1m[grp == 'HC'].mean(0), G2m[grp == 'HC'].mean(0)
ev0, evec0 = np.linalg.eigh(np.cov(np.vstack([x0, y0]))); ang = np.degrees(np.arctan2(evec0[1, 1], evec0[0, 1]))
ax.scatter(x0 - x0.mean(), y0 - y0.mean(), s=3, c=SP.SEQ(mcolors.Normalize(-6, 6)(np.clip(y0 - y0.mean(), -6, 6))), lw=0, alpha=0.8)
ell = {}
for g in G:
    a_ = np.sqrt(SM.loc[grp == g, 'MajorVariance'].mean()); b_ = np.sqrt(SM.loc[grp == g, 'MinorVariance'].mean())
    ax.add_patch(Ellipse((0, 0), 2 * a_, 2 * b_, angle=ang, fill=False, ec=SP.GROUPEDGE[g] if g == 'TC' else SP.GROUPCOL[g], lw=1.1))
    ell[g] = {'semi_major': float(a_), 'semi_minor': float(b_)}
ax.set_xlim(-10, 10); ax.set_ylim(-8, 8); ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('G1 (SA axis)'); ax.set_ylabel('G2 (VS axis)'); ax.set_title('G1–G2 manifold', fontsize=6.5, pad=2)
for i, g in enumerate(G):
    yy = 7.0 - 1.7 * i; ax.plot([5.0, 6.4], [yy] * 2, color=SP.GROUPEDGE[g] if g == 'TC' else SP.GROUPCOL[g], lw=1.1)
    ax.text(6.8, yy, g, fontsize=6, va='center')
src['panels']['d manifold'] = {'ellipse_angle_deg': float(ang), 'ellipses': ell,
                               'HC_mean_parcel_G1_centred': [float(x) for x in x0 - x0.mean()],
                               'HC_mean_parcel_G2_centred': [float(x) for x in y0 - y0.mean()]}
LAB(2, Y3 - 5, 'd')

# e long / short axis variance, f anisotropy
halfeye(ax_(70, Y3, PW, 38), 'MajorVariance', 'Long-axis variance', 'Variance (log scale)', log=True, seed=7, panel='e')
halfeye(ax_(97, Y3, PW, 38), 'MinorVariance', 'Short-axis variance', '', log=True, seed=8, panel='e')
LAB(60, Y3 - 5, 'e')
halfeye(ax_(140, Y3, 28, 38), 'AnisotropyIndex', 'Anisotropy', 'Anisotropy index', seed=9, panel='f')
LAB(128, Y3 - 5, 'f')

tg.save(fig, OUT, NAME)
(OUT / f'{NAME}_source_values.json').write_text(json.dumps(src, indent=1, ensure_ascii=False))
