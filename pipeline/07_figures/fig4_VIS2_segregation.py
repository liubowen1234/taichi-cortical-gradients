"""Fig. 4 - segregation of higher-order visual cortex (VIS2) in both athlete groups, from different partner networks.

a  VIS2 segregation (half-eye plot, brackets as in Fig. 2)
b  VIS2 network-pair FC: Hedges' g [95% CI]; contrast = colour + marker (TC - HC circle, BB - HC square, TC - BB
   triangle); filled = significant
c  schematic of the two segregation routes (edge colour = g vs HC; dashed = weaker)
d  VIS2 partner maps: per parcel, number of edges to VIS2 in the HC > TC / HC > BB NBS components
e  NBS chord diagrams of the HC > TC and HC > BB components
Inputs: 03_network_connectivity (segregation, visual2_connectivity, nbs).
Usage: python pipeline/07_figures/fig4_VIS2_segregation.py [--raster-surfaces] -> 07_figures/Fig4_VIS2_segregation.pdf / .png
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Circle, Wedge, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import fig_common as tg
import sydnor_palette as SP

tg.setup()
plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'text.color': SP.INK, 'axes.labelcolor': SP.INK, 'xtick.color': SP.INK, 'ytick.color': SP.INK,
                     'axes.edgecolor': SP.INK})
OUT = tg.OUT
NAME = 'Fig4_VIS2_segregation'
L = tg.LAYOUT
G = tg.GROUPS
CONTR = [('TCvsHC', 'TC − HC'), ('BBvsHC', 'BB − HC'), ('TCvsBB', 'TC − BB')]
DNORM = mcolors.Normalize(-0.8, 0.8)
# NBS ring / edge colours (Sydnor palette): partner network of VIS2 edges
RING = {'Visual2': SP.CONTRAST['BBvsHC'][0], 'Somatomotor': SP.EDGE['VIS2-SMN'], 'Cingulo-Opercular': SP.EDGE['VIS2-CON']}
RING_OTHER = '#D9D9D9'

# ------------------------------------------------------------------ data
SEG = pd.read_csv(L.SEG / 'subject_network_segregation.csv')
cov = pd.read_csv(L.COHORT / 'subject_covariates.csv')
assert (SEG.Subject.values == cov.participant_id.values).all()
grp = cov.Group.values
SST = pd.read_csv(L.SEG / 'stats_three_group/stats_Network_Segregation_3group.csv')
r_seg = SST[SST.Network == 'Visual2'].iloc[0]
PAIR = pd.read_csv(L.VIS2 / 'Visual2_all_partners_FC_3group.csv').set_index('Metric')
NBS = pd.read_csv(L.NBS / 'NBS_three_group_components_summary.csv')
E = {k: pd.read_csv(L.NBS / f'edges/NBS_{k}_edges_labelled.csv') for k in ['HC_gt_TC', 'HC_gt_BB']}
net = tg.network_of_parcels(); v2 = net == 'Visual2'
HEMI = tg.cabnp_mapping().Hemisphere.values
for e in E.values():
    e['i'] = e.Node_i - 1; e['j'] = e.Node_j - 1
    ab = pd.Series(net).map(tg.ABBR).values
    assert (ab[e.i] == e.NetA).all() and (ab[e.j] == e.NetB).all()
src = {'note': 'Fig. 4 source values (main analysis). Statistics are read from the result tables; nothing is re-tested.'}
fmtp = lambda p: f'{p:.2g}'

W, H = 180, 176
fig, ax_, _, _ = tg.newfig(W, H)
LAB = lambda x, y, s: fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight='bold', va='top')
TXT = lambda x, y, s, **k: fig.text(x / W, 1 - y / H, s, **{**dict(fontsize=6.5, color=SP.INK, va='top'), **k})


def stars(r, c):
    p = r[f'{c}_P_Holm']
    if r['Omnibus_Q_FDR'] < .05 and p < .05:
        return '***' if p < .001 else '**' if p < .01 else '*'
    return 'equiv.' if r[f'{c}_P_TOST'] < .05 else 'ns'


# ------------------------------------------------------------------ a VIS2 segregation (half-eye, as Fig. 2)
ax = ax_(12, 9, 30, 44)
rng = np.random.default_rng(4); col = 'Visual2_Segregation'; allv = SEG[col].values
summ = {}
for i, g in enumerate(G):
    v = SEG.loc[grp == g, col].values.astype(float)
    ys = np.linspace(v.min(), v.max(), 200); d = gaussian_kde(v, 0.35)(ys); d = d / d.max() * 0.40
    ax.fill_betweenx(ys, i, i + d, color=SP.GROUPCOL[g], alpha=0.55, lw=0, zorder=1)
    ax.plot(i + d, ys, color=SP.GROUPEDGE[g], lw=0.5, zorder=2)
    ax.scatter(i - 0.05 - rng.uniform(0, 0.2, len(v)), v, s=2.2, color=SP.GROUPCOL[g], ec=SP.GROUPEDGE[g], lw=0.2, zorder=3)
    q1, md, q3 = np.percentile(v, [25, 50, 75]); p5, p95 = np.percentile(v, [5, 95])
    ax.plot([i, i], [p5, p95], color=SP.INK, lw=0.6, zorder=4, solid_capstyle='butt')
    ax.plot([i, i], [q1, q3], color=SP.INK, lw=1.8, zorder=4, solid_capstyle='butt')
    ax.scatter(i, md, s=9, color='white', ec=SP.INK, lw=0.6, zorder=5)
    summ[g] = dict(n=int(len(v)), median=float(md), q25=float(q1), q75=float(q3), mean=float(v.mean()), sd=float(v.std(ddof=1)))
lo, hi = allv.min(), allv.max(); y0 = lo - 0.05 * (hi - lo); ax.set_ylim(y0, y0 + (hi - y0) / 0.72)
tr = ax.get_xaxis_transform(); lab = {}
for (c, a, b), yf in zip([('TCvsHC', 0, 1), ('TCvsBB', 1, 2), ('BBvsHC', 0, 2)], (0.77, 0.85, 0.93)):
    s = stars(r_seg, c); sig = s.startswith('*'); lab[c] = s
    ax.plot([a, a, b, b], [yf - 0.025, yf, yf, yf - 0.025], color=SP.INK, lw=0.5, transform=tr, clip_on=False)
    ax.text((a + b) / 2, yf - (0.02 if sig else -0.005), s, ha='center', va='bottom', fontsize=6.5 if sig else 5.5,
            transform=tr, clip_on=False)
ax.set_xticks(range(3)); ax.set_xticklabels(G); ax.tick_params(axis='x', length=0); ax.set_xlim(-0.35, 2.5)
ax.set_ylabel('VIS2 segregation'); ax.set_title('VIS2 segregation', fontsize=6.5, pad=2)
LAB(1.5, 2, 'a')
src['a_VIS2_segregation'] = {'groups': summ, 'omnibus_q_FDR': float(r_seg.Omnibus_Q_FDR), 'bracket': lab,
                             **{c: dict(g=float(r_seg[f'{c}_g']), P_Holm=float(r_seg[f'{c}_P_Holm']),
                                        P_TOST=float(r_seg[f'{c}_P_TOST'])) for c, _ in CONTR}}

# ------------------------------------------------------------------ b VIS2 network-pair FC
mets = [('Visual2_Visual2', 'VIS2 within'), ('Visual2_Visual1', 'VIS2–VIS1'), ('Visual2_Somatomotor', 'VIS2–SMN'),
        ('Visual2_Cingulo-Opercular', 'VIS2–CON'), ('Visual2_Auditory', 'VIS2–AUD')]
SHOWN = [m for m, _ in mets]
assert set(PAIR.index[PAIR.Omnibus_Q_FDR < .05]) == set(SHOWN), 'shown pairs = omnibus-significant members of the 12-pair family'
yy = [0, 1.4, 2.4, 3.4, 4.4]
BX, BWc, gapc = 78, 15, 2.2
for j, (c, ct) in enumerate(CONTR):
    colc, mk = SP.CONTRAST[c]
    ax = ax_(BX + j * (BWc + gapc), 13, BWc, 38)
    ax.axvspan(-0.5, 0.5, color=SP.EQUIV, lw=0, zorder=0)
    ax.axvline(0, color='#888888', lw=0.5, ls=':', zorder=1)
    for i, m in enumerate(SHOWN):
        r = PAIR.loc[m]; g, l_, h_ = r[f'{c}_g'], r[f'{c}_g_CI95_Lower'], r[f'{c}_g_CI95_Upper']
        sig = r['Omnibus_Q_FDR'] < .05 and r[f'{c}_P_Holm'] < .05
        ax.hlines(yy[i], l_, h_, color=colc, lw=0.9, zorder=2)
        ax.scatter(g, yy[i], s=15, marker=mk, color=colc if sig else 'white', ec=colc, lw=0.9, zorder=3)
    ax.set_ylim(max(yy) + 0.6, min(yy) - 0.6); ax.set_xlim(-1.25, 1.25)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_xticklabels(['−1', '', '0', '', '1'])
    ax.set_yticks(yy); ax.tick_params(axis='y', length=0, pad=1)
    ax.set_yticklabels([l for _, l in mets] if j == 0 else [])
    ax.spines['left'].set_visible(False); ax.set_title(ct, fontsize=6.5, pad=2)
    if j == 1: ax.set_xlabel('Hedges g [95% CI]')
TXT(BX + 1.5 * BWc + gapc, 4.5, 'VIS2 network-pair FC', ha='center')
LAB(52, 2, 'b')
src['b_VIS2_pairs'] = {m: {c: dict(g=float(PAIR.loc[m, f'{c}_g']), CI95=[float(PAIR.loc[m, f'{c}_g_CI95_Lower']), float(PAIR.loc[m, f'{c}_g_CI95_Upper'])],
                                   P_Holm=float(PAIR.loc[m, f'{c}_P_Holm']), sig=int(PAIR.loc[m, f'{c}_Sig'])) for c, _ in CONTR}
                          | {'omnibus_q_FDR': float(PAIR.loc[m, 'Omnibus_Q_FDR'])} for m in SHOWN}


# ------------------------------------------------------------------ c two segregation routes
def route(ax, grp_, c):
    XR = 34 / 22
    ax.set_xlim(0, XR); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.axis('off')
    pos = {'VIS2': (0.36, 0.48), 'SMN': (1.22, 0.84), 'VIS1': (1.22, 0.48), 'CON': (1.22, 0.12)}
    for partner, m in [('SMN', 'Visual2_Somatomotor'), ('VIS1', 'Visual2_Visual1'), ('CON', 'Visual2_Cingulo-Opercular')]:
        g = PAIR.loc[m, f'{c}_g']; sig = PAIR.loc[m, f'{c}_Sig'] == 1
        colr = SP.DIV(DNORM(np.clip(g, -0.8, 0.8))) if sig else '#C8C8C8'
        if sig: colr = tuple(np.array(mcolors.to_rgb(colr)) * 0.85)      # darker for thin lines on white
        (x0, y0), (x1, y1) = pos['VIS2'], pos[partner]
        ax.plot([x0, x1], [y0, y1], color=colr, lw=0.8 + 2.4 * abs(g) if sig else 0.8,
                ls=(0, (2.2, 1.4)) if g < 0 else '-', solid_capstyle='round', zorder=1)
        d = np.array([x1 - x0, y1 - y0]); nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        if nrm[1] * (y1 - 0.5) < 0 or (abs(y1 - y0) < 1e-6 and nrm[1] < 0): nrm = -nrm
        t = 0.66; xm, ym = x0 + t * (x1 - x0) + nrm[0] * 0.10, y0 + t * (y1 - y0) + nrm[1] * 0.10
        ax.text(xm, ym, tg.minus(f'{g:.2f}'), fontsize=5.5, color=SP.INK if sig else '#777777', ha='center', va='center')
    gw = PAIR.loc['Visual2_Visual2', f'{c}_g']; sigw = PAIR.loc['Visual2_Visual2', f'{c}_Sig'] == 1
    ring = SP.DIV(DNORM(np.clip(gw, -0.8, 0.8))) if sigw else '#C8C8C8'
    ax.add_patch(Circle(pos['VIS2'], 0.17, fill=False, ec=ring, lw=0.8 + 2.4 * abs(gw) if sigw else 0.8, zorder=2))
    ax.text(pos['VIS2'][0], pos['VIS2'][1] - 0.24, tg.minus(f'within {gw:.2f}'), fontsize=5.5, ha='center', va='top',
            color=SP.INK if sigw else '#777777')
    for k, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.12 if k == 'VIS2' else 0.07, color=SP.SIG if k == 'VIS2' else SP.NS, zorder=3))
        ax.text(x + (0.11 if k != 'VIS2' else 0), y, k, fontsize=5.5 if k == 'VIS2' else 6, ha='left' if k != 'VIS2' else 'center',
                va='center', color='white' if k == 'VIS2' else SP.INK, fontweight='bold', zorder=4)
    ax.text(0.0, 1.0, grp_, fontsize=6.5, fontweight='bold', ha='left', va='top')


route(ax_(140, 7, 36, 22), 'TC', 'TCvsHC')
route(ax_(140, 31, 36, 22), 'BB', 'BBvsHC')
LAB(136, 2, 'c')

# ------------------------------------------------------------------ d VIS2 partner maps
deg = {}
for k, e in E.items():
    m = v2[e.i] ^ v2[e.j]
    other = np.where(v2[e.i[m]], e.j[m], e.i[m])
    deg[k] = np.bincount(other, minlength=358)
vmax_d = max(d.max() for d in deg.values())
PNORM = mcolors.PowerNorm(0.5, vmin=1, vmax=vmax_d)
B = tg.Brain()
YD = 66
for r_, (k, gname) in enumerate([('HC_gt_TC', 'TC'), ('HC_gt_BB', 'BB')]):
    rgb = np.array([mcolors.to_rgb('#F4F4F4') if x == 0 else SP.MAG(0.12 + 0.88 * PNORM(x))[:3] for x in deg[k]])
    rgb[v2] = mcolors.to_rgb('#7A7A7A')
    yb = YD + r_ * 24
    B.row(ax_, 26, yb, 29, 21, rgb, outline=v2, titles=False)
    TXT(23, yb + 7, f'HC > {gname}', fontweight='bold', ha='right')
    TXT(23, yb + 11, f"{int((deg[k] > 0).sum())} partner parcels", fontsize=6, ha='right')
for c_, t in enumerate(['Left lateral', 'Left medial', 'Right medial', 'Right lateral']):
    TXT(26 + c_ * 29.4 + 14.5, YD - 1.5, t, ha='center', va='bottom')
cax = ax_(152, YD + 18, 22, 1.5)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=PNORM, cmap=mcolors.LinearSegmentedColormap.from_list('m', SP.MAG(np.linspace(0.12, 1, 256)))),
                  cax=cax, orientation='horizontal')
cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, width=0.5, length=2, pad=1); cb.set_ticks([1, 5, 10, 20])
cb.solids.set_rasterized(False); cb.solids.set_edgecolor('face')
cb.set_label('Edges to VIS2', fontsize=6.5, labelpad=1)
a_ = ax_(152, YD + 28, 2.8, 2.8); a_.add_patch(plt.Rectangle((0, 0), 1, 1, color='#7A7A7A')); a_.axis('off')
TXT(156, YD + 29.4, 'VIS2 (outlined)', fontsize=6, va='center')
LAB(1.5, YD - 4, 'd')
src['d_partner_parcels'] = {k: dict(n_parcels=int((deg[k] > 0).sum()), edges_to_VIS2_per_parcel=[int(x) for x in deg[k]],
                                    by_network=pd.Series(net[deg[k] > 0]).map(tg.ABBR).value_counts().to_dict()) for k in deg}

# ------------------------------------------------------------------ e NBS chord diagrams
YE = 114
order = []
for hemi in ['Left', 'Right']:
    idx = np.where(HEMI == hemi)[0]
    netord = pd.Series(net[idx]).map({n: i for i, n in enumerate(tg.NETS)}).values
    order.append(idx[np.lexsort((idx, netord))])
gap = 6; tot = len(order[0]) + len(order[1]) + 2 * gap
ang = np.zeros(358)
for p_, n_ in enumerate(order[0]): ang[n_] = np.deg2rad(90 + (gap / 2 + p_ + 0.5) / tot * 360)
for p_, n_ in enumerate(order[1]): ang[n_] = np.deg2rad(90 - (gap / 2 + p_ + 0.5) / tot * 360)
xy = np.c_[np.cos(ang), np.sin(ang)]; stepw = 360 / tot
tmin = 3.1; tmax = max(e.Edge_stat.max() for e in E.values())
STYLE = {'O': (SP.EDGE['other'], 0.6), 'V': (SP.EDGE['VIS2-other'], 0.5), 'C': (SP.EDGE['VIS2-CON'], 0.6), 'S': (SP.EDGE['VIS2-SMN'], 0.65)}
nb = NBS.set_index('Run')


def chord(ax, k, gname):
    e = E[k]
    ax.set_xlim(-1.42, 1.42); ax.set_ylim(-1.42, 1.42); ax.set_aspect('equal'); ax.axis('off')
    for n_ in range(358):
        a = np.rad2deg(ang[n_])
        ax.add_patch(Wedge((0, 0), 1.10, a - stepw / 2, a + stepw / 2, width=0.08, color=RING.get(net[n_], RING_OTHER), lw=0))
    for ordh in order:
        for nw in ['Visual2', 'Somatomotor', 'Cingulo-Opercular']:
            m = ordh[net[ordh] == nw]; a = np.mean(ang[m])
            rot = np.rad2deg(a) - 90 if np.sin(a) >= 0 else np.rad2deg(a) + 90
            ax.text(1.25 * np.cos(a), 1.25 * np.sin(a), tg.ABBR[nw], ha='center', va='center', fontsize=6, rotation=rot,
                    rotation_mode='anchor', color=SP.INK, fontweight='bold')
    ax.text(-0.15, -1.30, 'L', fontsize=6, ha='center'); ax.text(0.15, -1.30, 'R', fontsize=6, ha='center')
    partner = np.where(v2[e.i], net[e.j], np.where(v2[e.j], net[e.i], ''))
    cls = np.where(partner == 'Somatomotor', 'S', np.where(partner == 'Cingulo-Opercular', 'C', np.where(partner != '', 'V', 'O')))
    for c_ in ['O', 'V', 'C', 'S']:
        ee = e[cls == c_].sort_values('Edge_stat')
        for _, r in ee.iterrows():
            p0, p1 = xy[int(r.i)] * 0.975, xy[int(r.j)] * 0.975
            ctrl = (p0 + p1) / 2 * (0.55 - 0.125 * np.linalg.norm(p0 - p1))
            ax.add_patch(PathPatch(MPath([p0, ctrl, p1], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]), fc='none', ec=STYLE[c_][0],
                                   lw=0.25 + 0.7 * (r.Edge_stat - tmin) / (tmax - tmin), alpha=STYLE[c_][1], capstyle='round'))
    nS, nC = int((cls == 'S').sum()), int((cls == 'C').sum())
    ax.set_title(f"HC > {gname}: {int(nb.loc[k, 'Edges'])} edges, P = {fmtp(nb.loc[k, 'P_Holm_6dir'])}", fontsize=6.5, pad=0)
    ax.text(0, -1.50, f'VIS2–SMN {nS}    VIS2–CON {nC}', ha='center', va='top', fontsize=6)
    return dict(edges=int(nb.loc[k, 'Edges']), P_Holm=float(nb.loc[k, 'P_Holm_6dir']), VIS2_SMN=nS, VIS2_CON=nC,
                pair_counts=E[k].Pair.value_counts().to_dict())


src['e_NBS'] = {k: chord(ax_(x, YE + 4, 50, 50), k, gname) for k, gname, x in [('HC_gt_TC', 'TC', 14), ('HC_gt_BB', 'BB', 72)]}
LAB(1.5, YE, 'e')
h = [Line2D([], [], color=SP.EDGE['VIS2-SMN'], lw=1.4), Line2D([], [], color=SP.EDGE['VIS2-CON'], lw=1.4),
     Line2D([], [], color=SP.EDGE['VIS2-other'], lw=1.4), Line2D([], [], color=SP.EDGE['other'], lw=1.4)]
fig.legend(h, ['VIS2–SMN edge', 'VIS2–CON edge', 'Other VIS2 edge', 'Edge not involving VIS2'], loc='upper left',
           bbox_to_anchor=(130 / W, 1 - (YE + 16) / H), frameon=False, fontsize=6, handlelength=1.6, labelspacing=0.5)
hr = [plt.Rectangle((0, 0), 1, 1, color=RING[n]) for n in ['Visual2', 'Somatomotor', 'Cingulo-Opercular']] + [plt.Rectangle((0, 0), 1, 1, color=RING_OTHER)]
fig.legend(hr, ['VIS2', 'SMN', 'CON', 'Other networks'], loc='upper left', title='Outer ring', title_fontsize=6,
           bbox_to_anchor=(130 / W, 1 - (YE + 38) / H), frameon=False, fontsize=6, handlelength=1.0, labelspacing=0.4,
           alignment='left')

tg.save(fig, OUT, NAME)
(OUT / f'{NAME}_source_values.json').write_text(json.dumps(src, indent=1, ensure_ascii=False, default=float))
