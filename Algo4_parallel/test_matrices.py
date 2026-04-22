import numpy as np

def random_matrix(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

def malyshev_matrix(n=20):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i+1] = -1
    A[n-1, 0] = np.finfo(float).eps 
    return A
