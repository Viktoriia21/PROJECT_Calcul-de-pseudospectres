import numpy as np

def random_matrix(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

def linear_eigenvalue_matrix(n, spacing=1.0, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(-2.0, 2.0, n)
    eigvals = t + 1j * (0.6 * t**2 - 1.5)
    
    D = np.diag(eigvals)
    for i in range(n - 1):
        D[i, i+1] = 2.5
    H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    U, _ = np.linalg.qr(H)
    A = U @ D @ U.conj().T
    return A