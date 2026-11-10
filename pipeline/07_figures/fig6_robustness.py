"""Fig. 6 - robustness of the Tai Chi-specific findings.

a  specification curve (1,000 specifications, non-athlete reference): ranked Hedges' g of G2 span, three contrasts
b  specification curve: ranked standardised dose beta (G2 span ~ years) within TC and BB
c  alignment reference, atlas and denoising without GSR (32P): Hedges' g [95% CI] for G2 span, minor-axis variance and
   visual-network mean G2
d  signal-quality and anatomical covariates (incl. global signal amplitude): HC3 t of the three contrasts for G2 span
e  head motion: G2 span g [95% CI] across the motion subsets
Filled = omnibus P < 0.05 and Holm P < 0.05 (c-e); dark = significant positive (a, b). Grey band = +/-0.5 g.
Inputs: 06_robustness/robustness_summary/fig6_*.csv, 06_robustness/no_GSR_32P/fig6_extra_rows.csv, 06_robustness/multiverse/.
Usage: python pipeline/07_figures/fig6_robustness.py -> 07_figures/Fig6_robustness.pdf / .png
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import fig_common as tg
import sydnor_palette as SP

tg.setup()
plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'text.color': SP.INK, 'axes.labelcolor': SP.INK, 'xtick.color': SP.INK, 'ytick.color': SP.INK,
                     'axes.edgecolor': SP.INK})
OUT = tg.OUT
NAME = 'Fig6_robustness'
RS = tg.LAYOUT.SUMMARY
REF = pd.read_csv(RS / 'fig6_refs_atlas.csv'); COV = pd.read_csv(RS / 'fig6_covariates.csv'); MOT = pd.read_csv(RS / 'fig6_motion.csv')
EXTRA = pd.read_csv(tg.LAYOUT.NOGSR / 'fig6_extra_rows.csv')   # No GSR (32P) for c; + GSA for d
REF = pd.concat([REF, EXTRA[EXTRA.Setting == 'No GSR (32P)']], ignore_index=True)
COV = pd.concat([COV, EXTRA[EXTRA.Setting == '+ global signal amplitude']], ignore_index=True)
MV = pd.read_csv(tg.LAYOUT.MULTIVERSE / 'multiverse_specifications_HCref_3group.csv'); MV = MV[MV.status == 'ok']
MI = json.load(open(tg.LAYOUT.MULTIVERSE / 'multiverse_inference_HCref_3group.json'))
assert len(MV) == 1000 and (MV.alignment == 'HC reference').all()
prim = MV[MV.is_primary].iloc[0]
CC = {c: SP.CONTRAST[c][0] for c in ['TCvsHC', 'BBvsHC', 'TCvsBB']}
MK = {c: SP.CONTRAST[c][1] for c in CC}
CL = {'TCvsHC': 'TC − HC', 'BBvsHC': 'BB − HC', 'TCvsBB': 'TC − BB'}
GC = {'TC': SP.GROUPEDGE['TC'], 'BB': SP.GROUPCOL['BB']}          # TC yellow too faint as 0.5-pt dots: use its dark edge
light = lambda c, a=0.6: tuple(np.array(mcolors.to_rgb(c)) * (1 - a) + a)
src = {'note': 'Fig. 6 source values; read from robustness_summary/fig6_*.csv and the multiverse result files.'}

W, H = 180, 132
fig, ax_, _, _ = tg.newfig(W, H)
LAB = lambda x, y, s: fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight='bold', va='top')
TXT = lambda x, y, s, **k: fig.text(x / W, 1 - y / H, s, **{**dict(fontsize=6.5, color=SP.INK, va='top'), **k})


def spec_curve(ax, vals, pvals, primary, col, label=None):
    o = np.argsort(vals); v = vals[o]; sig = ((pvals < .05) & (vals > 0))[o]; x = np.linspace(0, 1, len(v))
    ax.scatter(x[~sig], v[~sig], s=0.6, color=light(col), lw=0, zorder=2, rasterized=False)
    ax.scatter(x[sig], v[sig], s=0.6, color=col, lw=0, zorder=3)
    k = np.searchsorted(v, primary); ax.scatter(x[min(k, len(x) - 1)], primary, marker='D', s=10, color=SP.INK, lw=0, zorder=5)
    ax.plot([], [], color=col, lw=1.4, label=label)


def fp(p):
    return f'{p:.3f}'


# ------------------------------------------------------------------ a multiverse G2 range
YA, HA = 6, 32
ax = ax_(13, YA, 95, HA)
for c in ['TCvsHC', 'TCvsBB', 'BBvsHC']:
    spec_curve(ax, MV[f'range_{c}_g'].values, MV[f'range_{c}_p'].values, prim[f'range_{c}_g'], CC[c],
               f"{CL[c]}: {MI['summary'][f'range_{c}_pct_sig_pos']:.0f}% P < 0.05, P$_{{\\mathrm{{perm}}}}$ = {fp(MI[f'range_{c}']['p_median_t'])}")
    src[f'a|{c}'] = dict(pct=MI['summary'][f'range_{c}_pct_sig_pos'], p_perm=MI[f'range_{c}']['p_median_t'],
                         p_n_sig=MI[f'range_{c}']['p_n_sig'], median_g=float(np.median(MV[f'range_{c}_g'])))
ax.axhspan(-0.5, 0.5, color=SP.EQUIV, lw=0, zorder=0); ax.axhline(0, color='#888888', lw=0.5)
ax.set_xlim(0, 1); ax.set_xticks([]); ax.set_ylim(-1.0, 1.5); ax.set_yticks([-1, -0.5, 0, 0.5, 1, 1.5])
ax.set_yticklabels(['−1', '−0.5', '0', '0.5', '1', '1.5'])
ax.set_xlabel('1,000 specifications, ranked'); ax.set_ylabel("Hedges' g, G2 range")
ax.legend(frameon=False, fontsize=6, loc='upper left', handlelength=1.2, borderaxespad=0.2, labelspacing=0.3)
ax.set_title('Multiverse: G2 range', fontsize=6.5, pad=2)
LAB(1, 1, 'a')

# ------------------------------------------------------------------ b multiverse dose
ax = ax_(126, YA, 51, HA)
for g in ['TC', 'BB']:
    pct = 100 * np.mean((MV[f'dose_{g}_p'] < .05) & (MV[f'dose_{g}_beta_std'] > 0))
    spec_curve(ax, MV[f'dose_{g}_beta_std'].values, MV[f'dose_{g}_p'].values, prim[f'dose_{g}_beta_std'], GC[g],
               f"{g}: {pct:.{0 if pct >= 10 else 1}f}% P < 0.05, P$_{{\\mathrm{{perm}}}}$ = {fp(MI[f'dose_{g}']['p_median_t'])}")
    src[f'b|{g}'] = dict(pct=float(pct), p_perm=MI[f'dose_{g}']['p_median_t'], median_beta=float(np.median(MV[f'dose_{g}_beta_std'])))
ax.axhline(0, color='#888888', lw=0.5)
ax.set_xlim(0, 1); ax.set_xticks([]); ax.set_ylim(-0.6, 1.0); ax.set_yticks([-0.5, 0, 0.5, 1]); ax.set_yticklabels(['−0.5', '0', '0.5', '1'])
ax.set_xlabel('1,000 specifications, ranked'); ax.set_ylabel('Standardised β, years')
ax.legend(frameon=False, fontsize=6, loc='upper left', handlelength=1.2, borderaxespad=0.2, labelspacing=0.3)
ax.set_title('Multiverse: training dose', fontsize=6.5, pad=2)
LAB(113, 1, 'b')

# ------------------------------------------------------------------ c references and atlases
YC, HC_ = 50, 36
SET = ['HC reference (primary)', 'External template', 'Schaefer-400', 'Schaefer-600', 'No GSR (32P)']
yy = [0, 1, 2.5, 3.5, 5]
OFF = {'TCvsHC': -0.24, 'BBvsHC': 0.0, 'TCvsBB': 0.24}


def mark(ax, x, y, c, filled, s=9):
    ax.scatter(x, y, s=s, marker=MK[c], color=CC[c] if filled else 'white', ec=CC[c], lw=0.7, zorder=3)


for j, (m, t) in enumerate([('G2 range', 'G2 range'), ('Minor-axis variance', 'Minor-axis variance'), ('VIS2 mean G2', 'Visual network G2')]):
    ax = ax_(31 + j * 23, YC, 20.5, HC_)
    ax.axvspan(-0.5, 0.5, color=SP.EQUIV, lw=0, zorder=0); ax.axvline(0, color='#888888', lw=0.5, ls=':')
    ax.axhline(1.75, color=SP.GRID, lw=0.5); ax.axhline(4.25, color=SP.GRID, lw=0.5)
    for i, s_ in enumerate(SET):
        r = REF[(REF.Setting == s_) & (REF.Metric == m)].iloc[0]
        for c in CC:
            ax.hlines(yy[i] + OFF[c], r[f'{c}_lo'], r[f'{c}_hi'], color=CC[c], lw=0.6, zorder=2)
            mark(ax, r[f'{c}_g'], yy[i] + OFF[c], c, r[f'{c}_sig'])
        src[f'c|{s_}|{m}'] = {c: [round(float(r[f'{c}_g']), 3), bool(r[f'{c}_sig'])] for c in CC}
    ax.set_ylim(5.5, -0.5); ax.set_yticks(yy); ax.set_yticklabels(SET if j == 0 else []); ax.tick_params(axis='y', length=0, pad=1)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(-0.9, 1.5); ax.set_xticks([-0.5, 0, 0.5, 1]); ax.set_xticklabels(['', '0', '', '1'])
    ax.set_title(t, fontsize=6.5, pad=2)
    if j == 1:
        ax.set_xlabel("Hedges' g [95% CI]")
TXT(31 + 23 * 1.5 - 1.25, YC - 6.5, 'Alignment reference, atlas and denoising', ha='center')
LAB(1, YC - 7, 'c')

# ------------------------------------------------------------------ d covariates
SETC = ['Primary', '+ intracranial volume', '+ VIS2 amplitude', '+ cortex amplitude', '+ DVARS', '+ censored volumes', '+ global signal amplitude']
ax = ax_(141, YC, 36, HC_)
ax.axvline(1.98, color='#888888', lw=0.5, ls='--'); ax.axvline(0, color='#888888', lw=0.5, ls=':')
for i, s_ in enumerate(SETC):
    r = COV[(COV.Setting == s_) & (COV.Metric == 'G2 range')].iloc[0]
    for c in CC:
        mark(ax, r[f'{c}_t'], i + OFF[c], c, r[f'{c}_sig'], s=11)
    src[f'd|{s_}'] = {c: round(float(r[f'{c}_t']), 2) for c in CC}
ax.set_ylim(len(SETC) - 0.5, -0.5); ax.set_yticks(range(len(SETC))); ax.set_yticklabels(SETC); ax.tick_params(axis='y', length=0, pad=1)
ax.spines['left'].set_visible(False); ax.set_xlim(-0.5, 4.8); ax.set_xticks([0, 2, 4])
ax.set_xlabel('Group t (HC3), G2 range'); ax.set_title('Covariates', fontsize=6.5, pad=2)
LAB(110, YC - 7, 'd')

# ------------------------------------------------------------------ e head motion
YE, HE = 102, 20
ax = ax_(13, YE, 115, HE)
subs = list(dict.fromkeys(MOT.Subset))
d = MOT[MOT.Metric == 'G2 range'].set_index('Subset').loc[subs]
for c in CC:
    x = np.arange(len(subs)) + {'TCvsHC': -0.12, 'BBvsHC': 0, 'TCvsBB': 0.12}[c]
    ax.vlines(x, d[f'{c}_lo'], d[f'{c}_hi'], color=CC[c], lw=0.6)
    ax.plot(x, d[f'{c}_g'], color=CC[c], lw=0.8)
    for xi, gi, si in zip(x, d[f'{c}_g'], d[f'{c}_sig']):
        mark(ax, xi, gi, c, si, s=11)
    src[f'e|{c}'] = {s_.replace(chr(10), ' '): [round(float(g_), 3), bool(si)] for s_, g_, si in zip(subs, d[f'{c}_g'], d[f'{c}_sig'])}
ax.set_xticks(range(len(subs)))
ax.set_xticklabels([f'{s_.replace("-" + chr(10), "-").replace(chr(10), " ")}\n(n = {n})' for s_, n in zip(subs, d.N.values)], fontsize=6)
ax.tick_params(axis='x', length=0)
ax.axhspan(-0.5, 0.5, color=SP.EQUIV, lw=0, zorder=0); ax.axhline(0, color='#888888', lw=0.5, ls=':')
ax.set_xlim(-0.5, len(subs) - 0.5); ax.set_ylim(-0.45, 1.3); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("Hedges' g [95% CI]")
ax.set_title('Head motion: G2 range', fontsize=6.5, pad=2)
LAB(1, YE - 5, 'e')

h = [Line2D([], [], color=CC[c], lw=0.8, marker=MK[c], ms=3.5) for c in CC] + \
    [Line2D([], [], color='#666666', lw=0, marker='o', ms=3.5, mfc='white'),
     plt.Rectangle((0, 0), 1, 1, color=SP.EQUIV, lw=0)]
fig.legend(h, [CL[c] for c in CC] + ['Not significant', '|g| < 0.5'], loc='upper left', bbox_to_anchor=(138 / W, 1 - (YE - 2) / H),
           frameon=False, fontsize=6, handlelength=1.8, labelspacing=0.5)

tg.save(fig, OUT, NAME)
(OUT / f'{NAME}_source_values.json').write_text(json.dumps(src, indent=1, ensure_ascii=False, default=float))
