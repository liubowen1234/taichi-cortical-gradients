"""Shared plotting utilities for the paper figures (pure numpy / matplotlib; no nilearn / nibabel needed).

- GIFTI reader (GZipBase64Binary / Base64Binary / ASCII)
- Glasser-358 vertex labels on the conte69 32k (fs_LR) surface
- Lambert-shaded surface renderer (orthographic projection, back-face culling, painter's order),
  drawn as a rasterised PolyCollection so the surrounding figure stays vector.
- Arial font registration and journal style (8-10 pt).
"""
import base64, gzip, re, zlib, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection

MM = 1 / 25.4


# ------------------------------------------------------------------ style
def set_style(font_dir):
    for f in Path(font_dir).glob('Arial*.ttf'):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 600, 'figure.dpi': 150, 'axes.edgecolor': '#333333', 'axes.labelcolor': '#1a1a1a',
        'xtick.color': '#333333', 'ytick.color': '#333333', 'text.color': '#1a1a1a', 'mathtext.default': 'regular',
    })


def panel_label(ax, s, x=-0.02, y=1.02, fig=None):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right')


# ------------------------------------------------------------------ GIFTI
def read_gifti(path):
    txt = Path(path).read_text(errors='ignore')
    arrays = []
    for m in re.finditer(r'<DataArray(.*?)>(.*?)</DataArray>', txt, re.S):
        attr, body = m.group(1), m.group(2)
        a = dict(re.findall(r'(\w+)="([^"]*)"', attr))
        data = re.search(r'<Data>(.*?)</Data>', body, re.S).group(1).strip()
        dt = {'NIFTI_TYPE_FLOAT32': np.float32, 'NIFTI_TYPE_INT32': np.int32, 'NIFTI_TYPE_UINT8': np.uint8}[a['DataType']]
        dims = [int(a[f'Dim{i}']) for i in range(int(a['Dimensionality']))]
        if a['Encoding'] == 'GZipBase64Binary':
            raw = zlib.decompress(base64.b64decode(data))
            arr = np.frombuffer(raw, dt)
        elif a['Encoding'] == 'Base64Binary':
            arr = np.frombuffer(base64.b64decode(data), dt)
        else:
            arr = np.array(data.split(), dtype=dt)
        order = 'F' if a.get('ArrayIndexingOrder') == 'ColumnMajorOrder' else 'C'
        arrays.append((a.get('Intent'), arr.reshape(dims, order=order)))
    return arrays


def load_surface(path):
    arrs = read_gifti(path)
    v = [a for i, a in arrs if i == 'NIFTI_INTENT_POINTSET'][0].astype(float)
    f = [a for i, a in arrs if i == 'NIFTI_INTENT_TRIANGLE'][0].astype(int)
    return v, f


def smooth_surface(v, f, iters=40, lam=0.6):
    """Laplacian smoothing -> partially inflated surface (for display only)."""
    n = len(v)
    import scipy.sparse as sp
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel()
    W = sp.diags(1 / deg) @ A
    x = v.copy()
    c0 = v.mean(0)
    for _ in range(iters):
        x = x + lam * (W @ x - x)
    # keep roughly original size
    x = (x - x.mean(0)) * (np.std(v - c0) / np.std(x - x.mean(0))) + c0
    return x


def glasser358_vertex_index(label_L_path, label_R_path):
    """Return per-vertex 0-based index into the project's 358-parcel order (R first, H removed), -1 = medial wall."""
    L = [a for i, a in read_gifti(label_L_path)][0].ravel().astype(int)   # 1..180 = Left parcels (Glasser order)
    R = [a for i, a in read_gifti(label_R_path)][0].ravel().astype(int)   # 181..360 = Right parcels
    idx360_L = np.where(L > 0, L + 180, 0)       # project 360 order: Right 1..180, Left 181..360
    idx360_R = np.where(R > 0, R - 180, 0)
    keep = np.setdiff1d(np.arange(1, 361), [120, 300])
    map360_to_358 = np.full(361, -1); map360_to_358[keep] = np.arange(358)
    return map360_to_358[idx360_L], map360_to_358[idx360_R]


# ------------------------------------------------------------------ rendering
VIEWS = {  # (rotation that maps world -> camera; camera looks along -z)
    'lateral_left': lambda p: np.c_[-p[:, 1], p[:, 2], -p[:, 0]],
    'medial_left': lambda p: np.c_[p[:, 1], p[:, 2], p[:, 0]],
    'lateral_right': lambda p: np.c_[p[:, 1], p[:, 2], p[:, 0]],
    'medial_right': lambda p: np.c_[-p[:, 1], p[:, 2], -p[:, 0]],
    'dorsal': lambda p: np.c_[p[:, 0], p[:, 1], p[:, 2]],
    'ventral': lambda p: np.c_[p[:, 0], -p[:, 1], -p[:, 2]],
}


def render_surface(ax, v, f, vcolors, view, light=(0.3, 0.4, 1.0), ambient=0.38, boundary=None, boundary_color=(0.2, 0.2, 0.2),
                   outline=None, outline_color=(0.05, 0.05, 0.05)):
    """Draw a shaded surface on a 2D axes. vcolors: n_vertices x 3 (RGB 0-1)."""
    p = VIEWS[view](v)
    tri = p[f]                                  # T x 3 x 3
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    front = nrm[:, 2] > 0
    L = np.asarray(light, float); L /= np.linalg.norm(L)
    diff = np.clip(nrm @ L, 0, 1)
    shade = ambient + (1 - ambient) * diff
    fc = vcolors[f].mean(1)
    if boundary is not None:
        fc = np.where(boundary[:, None], np.asarray(boundary_color)[None, :] * 0.6 + fc * 0.4, fc)
    if outline is not None:                           # e.g. border of a network of interest
        fc = np.where(outline[:, None], np.asarray(outline_color)[None, :], fc)
    fc = np.clip(fc * shade[:, None], 0, 1)
    order = np.argsort(tri[:, :, 2].mean(1))
    order = order[front[order]]
    pc = PolyCollection(tri[order][:, :, :2], facecolors=fc[order], edgecolors='face', linewidths=0.15, rasterized=True)
    ax.add_collection(pc)
    xy = p[:, :2]
    ax.set_xlim(xy[:, 0].min() - 2, xy[:, 0].max() + 2); ax.set_ylim(xy[:, 1].min() - 2, xy[:, 1].max() + 2)
    ax.set_aspect('equal'); ax.axis('off')


def set_boundary_faces(f, vmask):
    """Faces straddling the border of a vertex set (e.g. all Visual2 vertices)."""
    m = vmask[f]
    return m.any(1) & ~m.all(1)


def parcel_boundary_faces(f, vidx):
    lab = vidx[f]
    return (lab[:, 0] != lab[:, 1]) | (lab[:, 1] != lab[:, 2]) | (lab[:, 0] != lab[:, 2])


def values_to_vertex_colors(vidx, parcel_rgb, medial_rgb=(0.82, 0.82, 0.80)):
    out = np.tile(np.asarray(medial_rgb, float), (len(vidx), 1))
    ok = vidx >= 0
    out[ok] = parcel_rgb[vidx[ok]]
    return out
