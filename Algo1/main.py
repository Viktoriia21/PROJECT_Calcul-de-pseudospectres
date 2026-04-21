import time
import numpy as np
from test_matrices import random_matrix
from grid import pseudospectrum_grid
from plotting import plot_pseudospectrum

eps = 0.5
nx = ny = 400 

print("--- GRID ALGORITHM ---")

for n in [10, 20]:
    print(f"\n{'='*60}")
    print(f"GRID: n = {n} | ε = {eps}")
    print(f"{'='*60}")

    A = random_matrix(n, seed=42)
    eigvals = np.linalg.eigvals(A)

    start = time.time()
    X, Y, sigma_min = pseudospectrum_grid(A, eps, nx=nx, ny=ny)
    elapsed = time.time() - start

    print(f"Computation time: {elapsed:.2f} s")
    print(f"Grid size: {nx}×{ny} = {nx*ny:,} points")

    plot_pseudospectrum(X, Y, sigma_min, eps, n, eigvals)
