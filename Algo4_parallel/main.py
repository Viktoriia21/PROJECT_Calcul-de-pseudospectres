import numpy as np
from test_matrices import random_matrix, malyshev_matrix
from pseudospectrum_core import compute_grid_parallel
from plotting_utils import plot_componentwise_pseudospectrum

def run_experiment(A, n, epsilon, res, title, grid_range, cores):
    E = np.abs(A)
    print(f"\n" + "="*70)
    print(f" Parallel Experiment: n = {n}, cores = {cores}")
    print(f"="*70)
    
    X, Y, Z, exec_time = compute_grid_parallel(
        A, E, grid_range[0], grid_range[1], res=res, num_cores=cores
    )
    
    max_f = np.max(Z[np.isfinite(Z)])
    
    print(f"Grid: {res}x{res} ({res*res} points)")
    print(f"Execution Time: {exec_time:.4f} seconds")
    print(f"Max f(z): {max_f:.4e}")
    
    plot_componentwise_pseudospectrum(X, Y, Z, title=title)

def main():
    print("--- PARALLEL COMPONENTWISE ε-PSEUDOSPECTRUM ---")
    
    RESOLUTION = 200 
    CORES = 16 
    EPSILON = 0.1
    
    A10 = random_matrix(10, seed=42)
    evs10 = np.linalg.eigvals(A10)
    lim10 = max(np.abs(evs10.real).max(), np.abs(evs10.imag).max()) + 2.0
    
    run_experiment(
        A10, n=10, epsilon=EPSILON, res=RESOLUTION, cores=CORES,
        title="Parallel Pseudospectrum (10x10 Random)",
        grid_range=[(-lim10, lim10), (-lim10, lim10)]
    )

    A20 = malyshev_matrix(20)
    run_experiment(
        A20, n=20, epsilon=EPSILON, res=RESOLUTION, cores=CORES,
        title="Parallel Pseudospectrum (20x20 Malyshev)",
        grid_range=[(-0.3, 0.3), (-0.3, 0.3)]
    )

if __name__ == "__main__":
    main()
