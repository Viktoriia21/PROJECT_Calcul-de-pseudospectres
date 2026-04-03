import numpy as np

def random_matrix(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
