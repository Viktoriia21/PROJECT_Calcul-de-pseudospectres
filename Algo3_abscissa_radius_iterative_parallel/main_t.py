import sys
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from test_matrices import random_matrix
from utils import smin, estimate_plot_region, min_singular_triplet

# import required functions from algorithm.py
from algorithm import trace_boundary, trace_wrapper
from plotting import plot_pseudospectrum_boundary

eps = 0.5

# algorithm 2: calculate starting points
def find_starting_points(A, eps, eigvals, tol=1e-6):
    starts = []
    print(f"finding starting points for {len(eigvals)} eigenvalues...")
    
    n = A.shape[0]
    I = np.eye(n, dtype=complex)
    
    for i, lam in enumerate(eigvals):
        d0 = 1j
        z1 = lam + eps * d0 
        
        while True:
            sigma, u, v = min_singular_triplet(z1 * I - A)
            
            err = abs(sigma - eps) / eps
            if err <= tol:
                break
            
            denom = np.real(np.conj(d0) * np.vdot(v, u)) 
            if abs(denom) < 1e-10:
                break
            
            z1 = z1 - ((sigma - eps) / denom) * d0
            
       #print(f"  eigenvalue {i+1}: {lam:.2f} -> start: {z1:.4f} (error = {err:.3e})")
        starts.append(z1)

    return starts


if __name__ == '__main__':
    print("--- Brühl Curve Tracing ---")

    for n in [10, 20]:
        print("\n" + "=" * 80)
        print(f"   n = {n} | eps = {eps}")
        print("=" * 80)

        A = random_matrix(n, seed=42)
        
        # calculate eigenvalues
        eigvals = np.linalg.eigvals(A)

        xmin, xmax, ymin, ymax = estimate_plot_region(A, eps)
        #print(f"region: Re [{xmin:.1f}, {xmax:.1f}]   Im [{ymin:.1f}, {ymax:.1f}]")

        t0 = time.time()

        # get starting points using algorithm 2
        z_starts = find_starting_points(A, eps, eigvals)
    
        tasks = [(A, eps, z0, eigvals, i+1, len(z_starts)) for i, z0 in enumerate(z_starts)]
        
        # parallel tracing
        with ProcessPoolExecutor() as executor:
            raw_contours = list(executor.map(trace_wrapper, tasks))

        all_contours = []
        close_tol = 0.3

        # filter out overlapping contours
        for idx, contour in enumerate(raw_contours):
            if len(contour) <= 150:
                continue
                
            skip = False
            z0 = z_starts[idx]
            
            for existing_contour in all_contours:
                if np.min(np.abs(existing_contour - z0)) < close_tol:
                    skip = True
                    break
                    
            if skip:
                #print(f"skipping contour for start point {idx+1} — close to existing contour")
                continue

            all_contours.append(contour)

        elapsed = time.time() - t0
        total_points = sum(len(c) for c in all_contours)

        print(f"collected {len(all_contours)} contours")
        print(f"total boundary points: {total_points}")
        print(f"time: {elapsed:.2f} s")

        # --- PSEUDOSPECTRAL ABSCISSA AND RADIUS ---
        # calculate values based on the boundaries of the contours
        if all_contours:
            # combine all boundary points into one flat array
            all_points = np.concatenate(all_contours)
            
            # abscissa: max real part
            alpha_eps = np.max(np.real(all_points))
            
            # radius: max absolute value / modulus
            rho_eps = np.max(np.abs(all_points))
            
            print("-" * 40)
            print(f"pseudospectral abscissa (alpha_eps): {alpha_eps:.4f}")
            print(f"pseudospectral radius (rho_eps):   {rho_eps:.4f}")
            print("-" * 40)


        plot_pseudospectrum_boundary(
            all_contours,
            eigvals,
            eps=eps,
            n=n
        )