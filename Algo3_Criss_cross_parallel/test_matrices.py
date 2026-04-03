import numpy as np
from scipy.linalg import block_diag

def vertical_bean_matrix(n_per_block=10, vertical_gap=5.0, coupling=0.0):
    B1 = np.zeros((n_per_block, n_per_block), dtype=complex)
    eigs1 = np.linspace(vertical_gap*1j - 1j, vertical_gap*1j + 1j, n_per_block)
    np.fill_diagonal(B1, eigs1)
    for i in range(n_per_block - 1):
        B1[i, i + 1] = 0.8
    B2 = np.zeros((n_per_block, n_per_block), dtype=complex)
    eigs2 = np.linspace(-1j, 1j, n_per_block)
    np.fill_diagonal(B2, eigs2)
    B3 = np.zeros((n_per_block, n_per_block), dtype=complex)
    eigs3 = np.linspace(-vertical_gap*1j - 1j, -vertical_gap*1j + 1j, n_per_block)
    np.fill_diagonal(B3, eigs3)
    for i in range(n_per_block - 1):
        B3[i, i + 1] = 2.0
    return block_diag(B1, B2, B3)

def random_matrix(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
