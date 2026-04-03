import numpy as np
from scipy.linalg import svd, svdvals

def min_singular_triplet(M):
    U, s, Vh = svd(M, full_matrices=False)
    sigma_min = s[-1]
    u = U[:, -1]
    v = Vh.conj().T[:, -1]
    return sigma_min, u, v

def smin(A, z):
    n = A.shape[0]
    M = z * np.eye(n, dtype=complex) - A
    return svdvals(M)[-1]

def estimate_plot_region(A, eps, scale=2.0): 
    row_norms = np.sum(np.abs(A), axis=1)
    max_row = np.max(row_norms)
    lim = max(12.0, max_row * scale + eps * 6.0)
    return -lim, lim, -lim, lim
