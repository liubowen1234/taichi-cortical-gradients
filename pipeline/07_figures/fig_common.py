"""Shared colours, loaders and panel helpers for the main figures.

- Groups: HC grey, TC yellow, BB navy (see sydnor_palette.py for the colours used in the final figures).
- Subject-level comparisons: half-eye + dot strip with three brackets (half_eye, brackets).
- Cortical surfaces: fs_LR 32k (BrainSpace conte69 midthickness), Laplacian smoothing (150 iterations) for display,
  drawn with common/figlib.py; Glasser-358 parcels from the Cole-Anticevic label files.
All numbers are read from the result tables of the pipeline; nothing is re-tested here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, cm
from matplotlib.colors import LinearSegmentedColormap as LSC
from scipy.stats import gaussian_kde
from common.config import CFG, LAYOUT, mkdir
from common import figlib as fl
from common.data import cabnp_mapping, covariates, row_ids

FONT = CFG.path('font_dir')                                  # folder with Arial*.ttf (optional)
SURF = CFG.path('brainspace_matlab') / 'datasets/surfaces'
LAB = CFG.path('cabnp_dir') / 'SeparateHemispheres'
GRAD_A = LAYOUT.GRAD_A                                       # main analysis (non-athlete reference)
SA = GRAD_A / 'stats_three_group'
OUT = mkdir(LAYOUT.FIGURES)
MM = fl.MM

# ------------------------------------------------------------------ colours
GROUPS = ['HC', 'TC', 'BB']
GROUPCOL = {'HC': '#9A9A9A', 'TC': '#F5C451', 'BB': '#38387F'}
GROUPEDGE = {'HC': '#6E6E6E', 'TC': '#9C6F00', 'BB': '#24245A'}
GROUPTXT = {'HC': '#5F5F5F', 'TC': '#9C6F00', 'BB': '#38387F'}    # labels, text, thin lines
GROUPLINE = {'HC': '#9A9A9A', 'TC': '#9C6F00', 'BB': '#38387F'}   # curves on white
CONTRASTS = [('TCvsHC', 'TC − HC'), ('BBvsHC', 'BB − HC'), ('TCvsBB', 'TC − BB')]
DIFF = LSC.from_list('indigo_amber', ['#2D1E6B', '#9D90CB', '#F3F2EF', '#F2B66D', '#B4580A'], N=256)
POS, NEG = '#B4580A', '#5B4A9E'                                   # point colours for positive / negative effects
try:
    import cmcrameri.cm as cmc
    GRAD = LSC.from_list('lipari_light', cmc.lipari(np.linspace(0.22, 1.0, 256)))
except ImportError:
    GRAD = LSC.from_list('lipari_light', ['#45587a', '#665e77', '#85606f', '#aa6265', '#d46b5e',
                                          '#ea8b6a', '#e6ac81', '#ebcfa7', '#fdf5da'])
EQUIV = '#ECE8F4'
PATTERN = {'tc_up': '#C46A10', 'tc_down': '#5B4A9E', 'shared': '#1E9E5A', 'bb': '#1F8FD8', 'other': '#D9D3E6', 'ns': '#EDEDEB'}
MEDIAL, BOUNDARY = (0.84, 0.84, 0.83), (0.62, 0.62, 0.62)
INK2, GRID = '#555555', '#e3e3e3'
SMOOTH_ITERS = 150
# Cortical surfaces are written as vector polygons by default (editable and sharp at any zoom in the PDF).
# Pass --raster-surfaces to embed them as 600-dpi images instead (smaller files).
VECTOR_SURF = '--raster-surfaces' not in sys.argv

NETS = ['Visual1', 'Visual2', 'Somatomotor', 'Cingulo-Opercular', 'Dorsal-Attention', 'Language', 'Frontoparietal',
        'Auditory', 'Default', 'Posterior-Multimodal', 'Ventral-Multimodal', 'Orbito-Affective']
ABBR = dict(zip(NETS, ['VIS1', 'VIS2', 'SMN', 'CON', 'DAN', 'LAN', 'FPN', 'AUD', 'DMN', 'PMM', 'VMM', 'ORA']))
NETCOL = dict(zip(NETS, ['#0000ff', '#6400ff', '#00c8c8', '#990099', '#00b400', '#009b9b', '#8a8200', '#e03ee0',
                         '#ff0000', '#b15928', '#ff9c00', '#417c00']))
VIEWS = [('L', 'lateral_left', 'Left lateral'), ('L', 'medial_left', 'Left medial'),
         ('R', 'medial_right', 'Right medial'), ('R', 'lateral_right', 'Right lateral')]


def light(c, a):
    """Blend colour c towards white by fraction a."""
    c = np.array(mcolors.to_rgb(c)); return tuple(c + (1 - c) * a)


# ------------------------------------------------------------------ figure scaffolding
def setup():
    fl.set_style(FONT or Path('.'))      # registers Arial if font_dir is given; otherwise the default font is used


def newfig(W, H):
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax_ = lambda x, y, w, h: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    lab_ = lambda x, y, s: fig.text(x / W, 1 - y / H, s, fontsize=10, fontweight='bold', va='top')
    txt_ = lambda x, y, s, **k: fig.text(x / W, 1 - y / H, s, **{**dict(fontsize=8, color=INK2, va='top'), **k})
    return fig, ax_, lab_, txt_


def save(fig, outdir, name, png_dpi=600):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{name}.pdf', dpi=600)
    fig.savefig(outdir / f'{name}.png', dpi=png_dpi)
    plt.close(fig)
    print('saved', outdir / name)


def fmt_p(p):
    """Exact P: two significant digits; scientific with a true superscript below 0.001."""
    if p < 1e-3:
        e = int(np.floor(np.log10(p))); m = p / 10 ** e
        if round(m, 1) >= 10: m, e = m / 10, e + 1
        return f'{m:.1f} × $\\mathregular{{10^{{{e}}}}}$'.replace('-', '−')
    return f'{p:.3f}' if p < 0.01 else f'{p:.2f}'


def minus(s):
    return s.replace('-', '−')


# ------------------------------------------------------------------ data
def network_of_parcels():
    return cabnp_mapping().CABNP_Network.values


def subject_metrics():
    """Main-analysis subject metrics (Step 4) joined with lambda (step 9.1); rows in the analysis order (HC, TC, BB)."""
    a4 = pd.read_csv(GRAD_A / 'Step4_gradient_metrics/subject_gradient_metrics.csv')
    a91 = pd.read_csv(GRAD_A / 'step_9.1/step9_1_additional_subject_metrics.csv')
    cov = covariates()
    assert (a4.Subject.values == row_ids(len(cov))).all()
    assert (a91.Subject.values == cov.participant_id.values).all()
    assert (a4.Group.values == cov.Group.values).all() and (a91.Group.values == cov.Group.values).all()
    return a4.rename(columns={'Subject': 'RowID'}).join(a91.drop(columns=['Group']))


def parcel_long_matrix(col='G2'):
    L = pd.read_csv(GRAD_A / 'Step4_gradient_metrics/parcel_gradient_metrics_long.csv')
    sm = subject_metrics()
    return L.pivot(index='Subject', columns='Parcel', values=col).loc[sm.RowID].values


def stat_row(table, metric, key='Metric'):
    d = pd.read_csv(SA / table)
    r = d[d[key] == metric]
    assert len(r) == 1, (table, metric)
    return r.iloc[0]


# ------------------------------------------------------------------ cortex
class Brain:
    """fs_LR 32k surfaces + Glasser-358 vertex index (right hemisphere first, as in the FC arrays)."""

    def __init__(self):
        vL, fL = fl.load_surface(SURF / 'conte69_32k_left_hemisphere.gii')
        vR, fR = fl.load_surface(SURF / 'conte69_32k_right_hemisphere.gii')
        self.iL, self.iR = fl.glasser358_vertex_index(LAB / 'cortex_ColeAnticevic_NetPartition_GlasserParcels_L.label.gii',
                                                      LAB / 'cortex_ColeAnticevic_NetPartition_GlasserParcels_R.label.gii')
        self.fL, self.fR = fL, fR
        self.sL, self.sR = fl.smooth_surface(vL, fL, SMOOTH_ITERS), fl.smooth_surface(vR, fR, SMOOTH_ITERS)
        self.bL, self.bR = fl.parcel_boundary_faces(fL, self.iL), fl.parcel_boundary_faces(fR, self.iR)

    def row(self, ax_, x0, y, w, h, rgb, outline=None, titles=False, gap=0.4, views=VIEWS):
        for c_, (hemi, view, title) in enumerate(views):
            v, f, ii, b = (self.sL, self.fL, self.iL, self.bL) if hemi == 'L' else (self.sR, self.fR, self.iR, self.bR)
            o = None
            if outline is not None and np.any(outline):
                o = fl.set_boundary_faces(f, np.where(ii >= 0, outline[np.maximum(ii, 0)], False))
            ax = ax_(x0 + c_ * (w + gap), y, w, h)
            fl.render_surface(ax, v, f, fl.values_to_vertex_colors(ii, rgb, MEDIAL), view, boundary=b,
                              boundary_color=BOUNDARY, outline=o)
            if VECTOR_SURF:
                for c_ in ax.collections: c_.set_rasterized(False)
            if titles: ax.set_title(title, fontsize=8, color=INK2, pad=0)


def cmap_rgb(vals, cmap, vmin, vmax):
    return np.array(cmap(mcolors.Normalize(vmin, vmax)(np.clip(vals, vmin, vmax))))[:, :3]


def pattern_classes(p):
    """Three-group pattern of each parcel from a Step5 parcel table (hierarchical test: omnibus FDR, then Holm).
    Symmetric classification:
    tc_up / tc_down: TC differs from both HC and BB in the same direction, BB vs HC not significant (TC-specific).
    shared: TC and BB both differ from HC in the same direction (TC vs BB may or may not differ).
    bb: only BB differs from HC (BB vs HC significant, TC vs HC not significant).
    other: omnibus significant, any other pattern (e.g. only TC vs BB, or TC vs HC alone).  ns: omnibus not significant."""
    th, bh, tb = p.TCvsHC_Sig.values == 1, p.BBvsHC_Sig.values == 1, p.TCvsBB_Sig.values == 1
    same = np.sign(p.TCvsHC_g.values) == np.sign(p.TCvsBB_g.values)
    spec = th & tb & ~bh & same
    shared = th & bh & (np.sign(p.TCvsHC_g.values) == np.sign(p.BBvsHC_g.values))
    cat = np.where(p.Omnibus_Q_FDR.values < .05, 'other', 'ns').astype(object)
    cat[spec & (p.TCvsHC_g.values > 0)] = 'tc_up'
    cat[spec & (p.TCvsHC_g.values < 0)] = 'tc_down'
    cat[shared] = 'shared'
    cat[bh & ~th] = 'bb'
    return cat


# ------------------------------------------------------------------ V5 half-eye + dot strip
def xgroups(ax):
    ax.set_xticks(range(3)); ax.set_xticklabels(GROUPS)
    for t, g in zip(ax.get_xticklabels(), GROUPS):
        t.set_color(GROUPTXT[g]); t.set_fontweight('bold')
    ax.tick_params(axis='x', length=0)


def half_eye(ax, values_by_group, width=0.46, bw=0.35, seed=0, pad=0.08):
    """values_by_group: dict group -> 1-d array. Gradient-filled half density (right), median (white dot), IQR (thick),
    5-95% (thin) on the axis, participants as a jittered strip to the left."""
    rng = np.random.default_rng(seed)
    allv = np.concatenate([values_by_group[g] for g in GROUPS]); span = allv.max() - allv.min()
    for i, g in enumerate(GROUPS):
        v = values_by_group[g]; v = v[np.isfinite(v)]
        ys = np.linspace(v.min() - pad * span, v.max() + pad * span, 220)
        d = gaussian_kde(v, bw)(ys); d = d / d.max() * width
        lo_y, hi_y = ys.min(), ys.max()
        for k in range(len(ys) - 1):           # vertical gradient, light (bottom) to saturated (top)
            frac = (ys[k] - lo_y) / (hi_y - lo_y)
            c = light(GROUPCOL[g], 0.75 - 0.6 * frac)
            ax.fill([i, i + d[k], i + d[k + 1], i], [ys[k], ys[k], ys[k + 1], ys[k + 1]], color=c, ec=c, lw=0.3, zorder=1)
        ax.plot(i + d, ys, color=GROUPEDGE[g] if g != 'HC' else '#7A7A7A', lw=0.6, zorder=2)
        q1, med, q3 = np.percentile(v, [25, 50, 75]); p5, p95 = np.percentile(v, [5, 95])
        ax.plot([i, i], [p5, p95], color=GROUPTXT[g], lw=0.8, zorder=3, solid_capstyle='butt')
        ax.plot([i, i], [q1, q3], color=GROUPTXT[g], lw=2.6, zorder=3, solid_capstyle='round')
        ax.scatter(i, med, s=16, color='white', ec=GROUPTXT[g], lw=1.0, zorder=5)
        ax.scatter(i - 0.11 - rng.uniform(0, 0.17, len(v)), v, s=5, color=GROUPCOL[g], ec=GROUPEDGE[g], lw=0.3, zorder=4)
    xgroups(ax); ax.set_xlim(-0.42, 2.55)


def bracket_text(row, c):
    """Label for one bracket. Significant: g and Holm P. BB - HC not significant but TOST-equivalent: 'equivalent'."""
    g, p = row[f'{c}_g'], row[f'{c}_P_Holm']
    sig = (row['Omnibus_Q_FDR'] < .05) and (p < .05)
    if sig:
        return minus(f'g = {g:.2f}, P = ') + fmt_p(p), True
    if row[f'{c}_P_TOST'] < .05:
        return minus(f'g = {g:.2f}, equivalent'), False
    return minus(f'g = {g:.2f}, P = ') + fmt_p(p), False


def brackets(ax, row, y0, step, fontsize=6.5, tick=None):
    """Three brackets stacked above the data: TC-HC (0-1), TC-BB (1-2), BB-HC (0-2, top)."""
    tick = tick if tick is not None else step * 0.22
    for k, (a, b, c) in enumerate([(0, 1, 'TCvsHC'), (1, 2, 'TCvsBB'), (0, 2, 'BBvsHC')]):
        yy = y0 + k * step
        ax.plot([a, a, b, b], [yy - tick, yy, yy, yy - tick], color='#444444', lw=0.5, clip_on=False)
        t, sig = bracket_text(row, c)
        ax.text((a + b) / 2, yy + step * 0.06, t, ha='center', va='bottom', fontsize=fontsize,
                color='#1a1a1a' if sig else INK2, clip_on=False)


# ------------------------------------------------------------------ colour bars / keys
def hbar(fig, ax_, x, y, w, h, cmap, vmin, vmax, label=None, ticks=None, ticklabels=None, fontsize=7):
    cax = ax_(x, y, w, h)
    cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap), cax=cax, orientation='horizontal')
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=fontsize, width=0.5, length=2, pad=1)
    cb.solids.set_rasterized(False); cb.solids.set_edgecolor('face')   # vector colour bar, no hairline seams
    if ticks is not None: cb.set_ticks(ticks)
    if ticklabels is not None: cb.set_ticklabels(ticklabels)
    if label: cb.set_label(label, fontsize=fontsize, labelpad=1)
    return cb


# ------------------------------------------------------------------ forest plot: one mini-panel per contrast
def forest_cols(ax_, x0, y, w, h, rows, labels, gap=2.0, xlim=(-1.2, 1.2), xticks=(-1, -0.5, 0, 0.5, 1),
                xticklabels=('−1', '', '0', '', '1'), yy=None, label_colors=None, xlabel='Hedges g [95% CI]'):
    """rows: list of stat rows (pd.Series with <c>_g, <c>_g_CI95_Lower/Upper, <c>_P_Holm, Omnibus_Q_FDR).
    Filled = omnibus q < .05 and Holm P < .05; shaded band = +/-0.5 equivalence bounds. Returns the axes."""
    yy = list(range(len(rows))) if yy is None else yy
    axes = []
    for j, (c, ct) in enumerate(CONTRASTS):
        ax = ax_(x0 + j * (w + gap), y, w, h)
        ax.axvspan(-0.5, 0.5, color=EQUIV, lw=0, zorder=0)
        ax.axvline(0, color='#888888', lw=0.6, ls=':', zorder=1)
        for i, r in enumerate(rows):
            g, lo, hi = r[f'{c}_g'], r[f'{c}_g_CI95_Lower'], r[f'{c}_g_CI95_Upper']
            sig = r['Omnibus_Q_FDR'] < .05 and r[f'{c}_P_Holm'] < .05
            col = POS if g > 0 else NEG
            ax.hlines(yy[i], lo, hi, color=col, lw=0.9, zorder=2)
            ax.scatter(g, yy[i], s=16, color=col if sig else 'white', ec=col, lw=0.9, zorder=3)
        ax.set_ylim(max(yy) + 0.6, min(yy) - 0.6); ax.set_xlim(*xlim)
        ax.set_xticks(list(xticks)); ax.set_xticklabels(list(xticklabels))
        ax.set_yticks(yy); ax.tick_params(axis='y', length=0, pad=1)
        if j == 0:
            ax.set_yticklabels(labels)
            if label_colors is not None:
                for t, cc in zip(ax.get_yticklabels(), label_colors): t.set_color(cc)
        else:
            ax.set_yticklabels([])
        ax.spines['left'].set_visible(False)
        ax.set_title(ct, fontsize=8, pad=2)
        if j == 1 and xlabel: ax.set_xlabel(xlabel)
        axes.append(ax)
    return axes
