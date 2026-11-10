"""Fig. 3 - where on the cortex the visual-somatosensory (G2) axis differs between groups.

a  parcels where Tai Chi practitioners (TC) differ from both non-athletes (HC) and basketball players (BB) in the same
   direction while BB do not differ from HC (toward the visual pole = purple, toward the somatosensory pole = yellow)
b  distribution of parcel G2 scores (subject-level KDE, group mean +/- s.e.m.; dashed = group-mean 5th / 95th
   percentile); arrows mark the shift of both TC tails (TC - HC g of the two percentiles, step9_3 tail table)
c  TC - HC map of parcel G2 vs the canonical G2 (Margulies et al., 2016); points coloured by canonical G2; VIS2 outlined
d  parcel G2 Hedges' g for TC - HC, BB - HC, TC - BB (black outline = significant in that contrast)
e  network mean G1 | G2 | G3 (Hedges' g; outline and value = significant)
All statistics are read from the result tables; nothing is recomputed.
Usage: python pipeline/07_figures/fig3_G2_location.py -> 07_figures/Fig3_G2_location.pdf / .png
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, cm
from scipy.stats import gaussian_kde
import fig_common as tg
from common import figlib as fl
import sydnor_palette as SP

tg.setup()
plt.rcParams.update({'font.size': 6, 'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42,
                     'text.color': SP.INK, 'axes.labelcolor': SP.INK, 'xtick.color': SP.INK, 'ytick.color': SP.INK,
                     'axes.edgecolor': SP.INK, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})
OUT = tg.OUT
NAME = 'Fig3_G2_location'
CONTR = [('TCvsHC', 'TC − HC'), ('BBvsHC', 'BB − HC'), ('TCvsBB', 'TC − BB')]
LIM = 0.8; NORM = mcolors.Normalize(-LIM, LIM)
LINECOL = {'HC': SP.GROUPEDGE['HC'], 'TC': SP.GROUPEDGE['TC'], 'BB': SP.GROUPCOL['BB']}
TCUP, TCDN = '#7E308E', '#FFC125'            # toward the visual pole (purple) / somatosensory pole (yellow)
TCDN_LINE = SP.GROUPEDGE['TC']               # dark yellow for lines / text on white

# ------------------------------------------------------------------ data
SM = tg.subject_metrics(); grp = SM.Group.values
G2 = tg.parcel_long_matrix('G2')
P2 = pd.read_csv(tg.SA / 'Step5_parcel_G2_358_3group.csv').sort_values('Parcel')
NM = pd.read_csv(tg.SA / 'step9_2_network_MeanG123_stats_3group.csv')
NETG = {k: NM[NM.Metric == k].set_index('Network').loc[tg.NETS] for k in ('MeanG1', 'MeanG2', 'MeanG3')}
om = P2.Omnibus_Q_FDR.values < .05
SIGC = {c: om & (P2[f'{c}_Sig'].values == 1) for c, _ in CONTR}
gTH, gTB = P2.TCvsHC_g.values, P2.TCvsBB_g.values
TCBOTH = SIGC['TCvsHC'] & SIGC['TCvsBB'] & (np.sign(gTH) == np.sign(gTB)) & ~SIGC['BBvsHC']
NUP, NDN = int((TCBOTH & (gTH > 0)).sum()), int((TCBOTH & (gTH < 0)).sum())
RES = tg.LAYOUT.CONTEXT
SC = pd.read_csv(RES / 'spatial_correspondence_3group.csv')
REF = pd.read_csv(tg.LAYOUT.CONTEXT / 'parcellated_reference_maps_Glasser358.csv').sort_values('Parcel358')
net = tg.network_of_parcels(); v2 = net == 'Visual2'
assert (REF.Parcel358.values == P2.Parcel.values).all() and (REF.Network.values == net).all()
src = {'note': 'Fig. 3 source values. Main analysis (non-athlete reference). Parcel order: Glasser 358, right first.',
       'a_TC_differs_from_both': dict(n=int(TCBOTH.sum()), visual_ward=NUP, somatosensory_ward=NDN,
                                      parcels=[int(p) for p in P2.Parcel.values[TCBOTH]]),
       'd_parcel_maps': {}}


class Brain4:
    def __init__(self):
        self.b = tg.Brain()

    def row(self, ax_, x0, y, w, h, rgb, thin=None, gap=0.4, titles=False, tfs=6):
        b = self.b
        for c_, (hemi, view, title) in enumerate(tg.VIEWS):
            v, f, ii, bd = (b.sL, b.fL, b.iL, b.bL) if hemi == 'L' else (b.sR, b.fR, b.iR, b.bR)
            ax = ax_(x0 + c_ * (w + gap), y, w, h)
            o1 = None if thin is None else fl.set_boundary_faces(f, np.where(ii >= 0, thin[np.maximum(ii, 0)], False))
            fl.render_surface(ax, v, f, fl.values_to_vertex_colors(ii, rgb, tg.MEDIAL), view, boundary=bd,
                              boundary_color=tg.BOUNDARY, outline=o1, outline_color=(0, 0, 0))
            if tg.VECTOR_SURF:
                for c in ax.collections: c.set_rasterized(False)
            if titles: ax.set_title(title, fontsize=tfs, pad=0)


def vbar(fig, ax_, x, y, h, w=1.4, lim=None):
    lim = lim or LIM
    cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(-lim, lim), cmap=SP.DIV), cax=ax_(x, y, w, h), orientation='vertical')
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, width=0.5, length=2, pad=1)
    cb.solids.set_rasterized(False); cb.solids.set_edgecolor('face')
    cb.set_ticks([-lim, 0, lim]); cb.set_ticklabels([f'−{lim:g}', '0', f'{lim:g}'])
    cb.set_label('Hedges g', fontsize=6, labelpad=1)
    return cb


# ------------------------------------------------------------------ layout (mm)
W = 180
AX0 = 4; AW = (176 - AX0 - 3 * 0.8) / 4; AH = AW * 22.5 / 31      # a: four large views
YA = 7
YB = YA + AH + 11                                                 # b / c row
HBC = 52
YD = YB + HBC + 17                                                # d / e row
DX = 14; DW = 23.4; DH = DW * 22.5 / 31
YDR = [YD + 4 + r * (DH + 1.0) for r in range(3)]
CS = 3.9
H = int(round(YDR[2] + DH + 10))
fig, ax_, _, _ = tg.newfig(W, H)
LAB = lambda x, y, s: fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight='bold', va='top')
TXT = lambda x, y, s, **k: fig.text(x / W, 1 - y / H, s, **{**dict(fontsize=7, color=SP.INK, va='top'), **k})
B = Brain4()

# ------------------------------------------------------------------ a TC differs from both groups (large)
rgb = np.tile(np.array(mcolors.to_rgb('#EFEFEF')), (358, 1))
rgb[TCBOTH & (gTH > 0)] = mcolors.to_rgb(TCUP); rgb[TCBOTH & (gTH < 0)] = mcolors.to_rgb(TCDN)
B.row(ax_, AX0, YA, AW, AH, rgb, gap=0.8, titles=True)
TXT(10, 1.6, 'TC differs from both HC and BB', fontweight='bold')
ky = YA + AH + 3.2
for k_, (col, t_) in enumerate([(TCUP, f'TC further toward the visual pole ({NUP} parcels)'),
                                (TCDN, f'TC further toward the somatosensory pole ({NDN} parcels)')]):
    xk = 30 + k_ * 66
    ax = ax_(xk, ky - 1.4, 4.5, 2.8)
    ax.add_patch(plt.Rectangle((0.08, 0.1), 0.84, 0.8, fc=col, ec='none'))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    TXT(xk + 5.5, ky, t_, fontsize=6, va='center')
LAB(1.5, 1.5, 'a')

# ------------------------------------------------------------------ b G2 score distribution, both tails shift outward
ax = ax_(14, YB + 4, 70, HBC - 4)
ctr = np.linspace(-16, 19, 351)
for g in ['HC', 'BB', 'TC']:
    Hd = np.array([gaussian_kde(G2[s], bw_method=0.25)(ctr) for s in np.where(grp == g)[0]])
    m = Hd.mean(0); se = Hd.std(0, ddof=1) / np.sqrt(len(Hd))
    ax.fill_between(ctr, np.maximum(m - se, 1e-6), m + se, color=SP.GROUPCOL[g], alpha=0.35, lw=0)
    ax.plot(ctr, m, color=LINECOL[g], lw=1.0, label=g)
    for q in ('G2_Q05', 'G2_Q95'):
        ax.axvline(SM.loc[grp == g, q].mean(), color=LINECOL[g], lw=0.6, ls=(0, (3, 2)))
ax.set_ylim(0, 0.165); ax.set_xlim(-14, 17)
ax.set_xlabel('Parcel G2 score'); ax.set_ylabel('Density')
h, l = ax.get_legend_handles_labels(); order = [l.index(g) for g in tg.GROUPS]
ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=6, loc='center right', handlelength=1.4,
          borderaxespad=0.1)
ax.text(0.0, -0.12, 'Somatosensory', transform=ax.transAxes, fontsize=6, ha='left', va='top')
ax.text(1.0, -0.12, 'Visual', transform=ax.transAxes, fontsize=6, ha='right', va='top')
ax.set_title('Both poles extend in TC', pad=2)
for q, col in (('G2_Q05', TCDN_LINE), ('G2_Q95', TCUP)):
    r = tg.stat_row('step9_3_tail_stats_3group.csv', q)
    src[f'b_{q}'] = {c: dict(g=float(r[f'{c}_g']), P_Holm=float(r[f'{c}_P_Holm'])) for c, _ in CONTR} | {'omnibus_q': float(r.Omnibus_Q_FDR)}
    xh, xt = SM.loc[grp == 'HC', q].mean(), SM.loc[grp == 'TC', q].mean()
    ya = 0.148
    ax.annotate('', xy=(xt, ya), xytext=(xh, ya), arrowprops=dict(arrowstyle='-|>', lw=1.0, color=col, mutation_scale=6,
                                                                  shrinkA=0, shrinkB=0))
    ax.text(xt + (-0.5 if q == 'G2_Q05' else 0.5), ya, tg.minus(f'g = {r.TCvsHC_g:.2f}'), fontsize=6, va='center',
            ha='right' if q == 'G2_Q05' else 'left', color=SP.INK)
LAB(1.5, YB - 1, 'b')

# ------------------------------------------------------------------ c TC - HC vs canonical G2
ax = ax_(110, YB + 4, 50, HBC - 4)
x = REF['neuromaps::Margulies_G2'].values; y = P2.TCvsHC_g.values
cn = mcolors.Normalize(np.percentile(x, 2), np.percentile(x, 98))
ax.scatter(x[~v2], y[~v2], s=4, c=SP.SEQ(cn(x[~v2])), lw=0, zorder=2)
ax.scatter(x[v2], y[v2], s=5, c=SP.SEQ(cn(x[v2])), ec=SP.INK, lw=0.4, zorder=3)
bb = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 100)
Xd = np.c_[np.ones_like(x), x]; res = y - Xd @ bb[::-1]; s2 = res @ res / (len(x) - 2)
XtXi = np.linalg.inv(Xd.T @ Xd); Xs = np.c_[np.ones_like(xs), xs]; se = np.sqrt(np.einsum('ij,jk,ik->i', Xs, XtXi, Xs) * s2)
ax.fill_between(xs, Xs @ bb[::-1] - 1.96 * se, Xs @ bb[::-1] + 1.96 * se, color=SP.CIBAND, alpha=0.3, lw=0, zorder=1)
ax.plot(xs, Xs @ bb[::-1], color=SP.INK, lw=1.0, zorder=4)
ax.axhline(0, color=SP.GRID, lw=0.6, zorder=0)
row = SC[(SC.Family == 'neuromaps') & (SC.Target == 'A_TCvsHC_g') & (SC.Map == 'Margulies_G2')]
assert len(row) == 1
rA, pA = float(row.r.iloc[0]), float(row.p_spin.iloc[0])
e_ = int(np.floor(np.log10(pA))); m_ = pA / 10 ** e_
ax.text(0.03, 0.98, f'r = {rA:.2f}\n$P_{{\\mathrm{{spin}}}}$ = {m_:.0f} × 10$^{{{e_}}}$'.replace('{-', '{−'), transform=ax.transAxes,
        fontsize=6, va='top', linespacing=1.4)
ax.text(0.97, 0.03, 'Outlined: VIS2', transform=ax.transAxes, fontsize=6, ha='right', va='bottom')
ax.set_xlabel('Canonical G2 (Margulies)'); ax.set_ylabel('TC − HC (Hedges g)')
ax.set_title('TC − HC follows the canonical G2', pad=2)
LAB(97, YB - 1, 'c')
src['c'] = dict(r=rA, p_spin=pA, slope=float(bb[0]), intercept=float(bb[1]))

# ------------------------------------------------------------------ d pairwise parcel maps
for r_, (c, lbl) in enumerate(CONTR):
    gv = P2[f'{c}_g'].values.astype(float)
    B.row(ax_, DX, YDR[r_], DW, DH, SP.DIV(NORM(np.clip(gv, -LIM, LIM)))[:, :3], SIGC[c], titles=(r_ == 0), tfs=5.5)
    TXT(1.5, YDR[r_] + DH / 2, lbl, fontsize=6, fontweight='bold', va='center')
    s = SIGC[c]
    src['d_parcel_maps'][lbl] = dict(n_sig=int(s.sum()), n_sig_pos=int((s & (gv > 0)).sum()), n_sig_neg=int((s & (gv < 0)).sum()),
                                     g=[round(float(v), 4) for v in gv], significant=[bool(v) for v in s])
vbar(fig, ax_, DX + 4 * DW + 3 * 0.4 + 1.5, YDR[0] + DH * 0.5, 2 * DH)
LAB(1.5, YD - 1, 'd')
TXT(DX, YD - 1, 'Parcel G2, pairwise contrasts (outline: significant)', fontsize=7)

# ------------------------------------------------------------------ e network mean G1 | G2 | G3
HNORM = mcolors.Normalize(-1, 1); HX, HY = 128, YD + 5
for b_, (key, sub) in enumerate([('MeanG1', 'G1 (SA)'), ('MeanG2', 'G2 (VS)'), ('MeanG3', 'G3 (MR)')]):
    hx = HX + b_ * (3 * CS + 1.6)
    ax = ax_(hx, HY, 3 * CS, 12 * CS)
    T = NETG[key]
    for j, (c, _) in enumerate(CONTR):
        for i, nw in enumerate(tg.NETS):
            g = T.loc[nw, f'{c}_g']; sig = T.loc[nw, f'{c}_Sig'] == 1
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, facecolor=SP.DIV(HNORM(np.clip(g, -1, 1))),
                                       edgecolor=SP.INK if sig else 'none', lw=0.8 if sig else 0, zorder=3))
            if sig:
                ax.text(j, i, f'{g:.2f}'.replace('-', '−').replace('0.', '.'), ha='center', va='center', fontsize=5,
                        color='white' if g > 0.55 else SP.INK, zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels([ct for _, ct in CONTR], rotation=90, fontsize=6)
    ax.set_yticks(range(12))
    ax.set_yticklabels([tg.ABBR[nw] for nw in tg.NETS] if b_ == 0 else [], fontsize=6)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(11.5, -0.5); ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(sub, fontsize=6, pad=2, fontweight='bold')
    src[f'e_network_{key}'] = T[[f'{c}_g' for c, _ in CONTR] + [f'{c}_Sig' for c, _ in CONTR]].round(4).to_dict('index')
TXT(HX + 4.5 * CS + 1.6, YD - 1, 'Network mean scores', ha='center')
vbar(fig, ax_, HX + 9 * CS + 3.2 + 2.0, HY + 12 * CS * 0.2, 12 * CS * 0.6, lim=1)
LAB(HX - 11, YD - 1, 'e')

tg.save(fig, OUT, NAME)
(OUT / f'{NAME}_source_values.json').write_text(json.dumps(src, indent=1, ensure_ascii=False))
print('H', H, 'TC vs both', NUP, NDN, {k: (v['n_sig'], v['n_sig_pos'], v['n_sig_neg']) for k, v in src['d_parcel_maps'].items()})
