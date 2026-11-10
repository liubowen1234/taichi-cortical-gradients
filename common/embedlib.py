"""Python version of the BrainSpace (MATLAB) gradient embedding, extended to the alternative choices of the
specification-curve analysis.

kernel(): BrainSpace GradientMaps kernels - each seed's connectivity profile is thresholded at its sparsity percentile
          (MATLAB prctile equals numpy percentile with method='hazen'); cosine / normalised angle / Pearson / Spearman /
          Gaussian (gamma = 1/n) affinity; negative affinities set to 0; symmetrised.
dm():     diffusion map embedding (dense symmetric eigendecomposition; diffusion time 0).
le():     Laplacian eigenmaps (generalised eigenproblem L v = lambda D v, smallest eigenvalues).
pca():    principal component analysis of the affinity matrix (scores; lambda = % variance explained).
With the main settings the output equals the MATLAB step 1 to floating-point precision.
"""
import numpy as np
from scipy import linalg, stats


def kernel(A, kind='cosine', sparsity=90):
    A = np.asarray(A, float)
    X = A.T.copy()                                   # BrainSpace transposes the input; columns = seeds
    if sparsity > 0:
        thr = np.percentile(X, sparsity, axis=0, method='hazen')
        X[X < thr[None, :]] = 0
    if kind == 'cosine' or kind == 'angle':
        nrm = np.linalg.norm(X, axis=0); nrm[nrm == 0] = 1
        C = (X / nrm).T @ (X / nrm)
        C = np.clip(C, -1, 1)
        K = C if kind == 'cosine' else 1 - np.arccos(C) / np.pi
    elif kind == 'pearson':
        K = np.corrcoef(X.T)
    elif kind == 'spearman':
        K = np.corrcoef(stats.rankdata(X, axis=0).T)
    elif kind == 'gaussian':
        gamma = 1.0 / X.shape[0]
        sq = (X ** 2).sum(0); D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X.T @ X, 0)
        K = np.exp(-gamma * D2)
    else:
        raise ValueError(kind)
    K[K < 0] = 0
    if np.any(np.all(K == 0, axis=0)):
        raise RuntimeError('zero vector in kernel')
    K = np.tril(K) + np.tril(K, -1).T
    return K


def dm(K, n=10, alpha=0.5):
    d = K.sum(1) ** -alpha
    L = K * np.outer(d, d)
    dsi = L.sum(1) ** -0.5
    Ms = L * np.outer(dsi, dsi); Ms = (Ms + Ms.T) / 2
    w, V = np.linalg.eigh(Ms)
    o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    V = V * dsi[:, None]
    psi = V / V[:, [0]]
    lam = w[1:] / (1 - w[1:])
    return psi[:, 1:n + 1] * lam[:n][None, :], lam


def le(K, n=10):
    D = np.diag(K.sum(1)); L = D - K
    w, V = linalg.eigh(L, D)
    o = np.argsort(w); w, V = w[o], V[:, o]
    return V[:, 1:n + 1], w[1:]


def pca(K, n=10):
    Xc = K - K.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    score = U * s
    expl = 100 * s ** 2 / np.sum(s ** 2)
    return score[:, :n], expl


def embed(A, kind, sparsity, approach, n=10):
    K = kernel(A, kind, sparsity)
    if approach.startswith('dm'):
        alpha = {'dm': 0.5, 'dm_a0': 0.0, 'dm_a1': 1.0}[approach]
        return dm(K, n, alpha)
    if approach == 'le':
        return le(K, n)
    if approach == 'pca':
        return pca(K, n)
    raise ValueError(approach)
