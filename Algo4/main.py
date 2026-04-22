import numpy as np
from test_matrices import random_matrix, malyshev_matrix
from pseudospectrum_core import compute_grid
from plotting_utils import plot_componentwise_pseudospectrum
from plotting_2D import plot_pseudospectrum_contour

def run_experiment(A, n, epsilon, res, title, grid_range):
    E = np.abs(A)
    
    print(f"\n" + "="*70)
    print(f" Experiment: n = {n}, epsilon = {epsilon}")
    print(f"="*70)
    
    X, Y, Z, exec_time = compute_grid(A, E, grid_range[0], grid_range[1], res=res)
    
    max_f = np.max(Z[np.isfinite(Z)])
    target_threshold = 1.0 / epsilon
    
    print(f"Grid Resolution: {res}x{res}")
    print(f"Execution Time:  {exec_time:.4f} seconds")
    print(f"Max f(z) on grid: {max_f:.4e}")
    print(f"Target 1/epsilon: {target_threshold:.4e}")
    
    eigenvalues = np.linalg.eigvals(A)
    
    plot_componentwise_pseudospectrum(
        X, Y, Z, 
        title=f"Parallel Pseudospectrum ({n}x{n} {title.split()[-1]})"
    )
    
    plot_pseudospectrum_contour(
        X=X, 
        Y=Y, 
        Z=Z,
        eigenvalues=eigenvalues,
        eps=epsilon,
        n=n
    )

def main():
    print("--- COMPONENTWISE PSEUDOSPECTRUM (Malyshev-Sadkane 2004) ---")
    
    RESOLUTION = 200 
    EPSILON = 0.1    
    
    # 10x10 Random Matrix
    A10 = random_matrix(10, seed=42)
    evs10 = np.linalg.eigvals(A10)
    limit10 = max(np.abs(evs10.real).max(), np.abs(evs10.imag).max()) + 2.0
    
    run_experiment(
        A10, n=10, epsilon=EPSILON, res=RESOLUTION, 
        title="10×10 Random Matrix",
        grid_range=[(-limit10, limit10), (-limit10, limit10)]
    )

    # 20x20 Malyshev Matrix
    A20 = malyshev_matrix(20)
    run_experiment(
        A20, n=20, epsilon=EPSILON, res=RESOLUTION, 
        title="20×20 Malyshev Matrix",
        grid_range=[(-0.3, 0.3), (-0.3, 0.3)]
    )

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
