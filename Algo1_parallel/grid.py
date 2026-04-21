import numpy as np
import scipy.linalg as la
from gershgorin import gershgorin_box
import multiprocessing as mp
import os

def compute_sigma_min(args):
    A, z, I = args
    return la.svdvals(A - z * I)[-1]

def pseudospectrum_grid(A, eps, nx=400, ny=400, num_processes=None):

    xmin, xmax, ymin, ymax = gershgorin_box(A, padding=eps)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    n = A.shape[0]
    I = np.eye(n)
    sigma_min = np.zeros_like(X)

    if num_processes is None:
        num_processes = max(1, os.cpu_count() - 1)

    print(f"Using {num_processes} processes")

    indices = [(i, j) for i in range(nx) for j in range(ny)]
    tasks = [(A, Z[j, i], I) for i, j in indices]

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(compute_sigma_min, tasks)

    # Fill the result array
    for idx, (i, j) in enumerate(indices):
        sigma_min[j, i] = results[idx]

    return X, Y, sigma_min
