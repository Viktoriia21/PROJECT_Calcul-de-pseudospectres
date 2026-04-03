import time
import numpy as np
import sys
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from test_matrices import random_matrix, linear_eigenvalue_matrix
from utils import min_singular_triplet
from algorithm import criss_cross_abscissa, criss_cross_radius, trace_boundary

# --- Algo2 parallel ---
def find_starting_points(A, eps, eigvals, tol=1e-6):
    starts = []
    n = A.shape[0]
    I = np.eye(n, dtype=complex)
    for i, lam in enumerate(eigvals):
        d0 = 1j
        z1 = lam + eps * d0
        while True:
            sigma, u, v = min_singular_triplet(z1 * I - A)
            err = abs(sigma - eps) / eps
            if err <= tol: break
            denom = np.real(np.conj(d0) * np.vdot(v, u))
            if abs(denom) < 1e-10: break
            z1 = z1 - ((sigma - eps) / denom) * d0
        starts.append(z1)
    return starts

def trace_worker(args):
    A, eps, z0, eigvals = args
    return trace_boundary(A, eps, z0, eigvals)

# --- PLot ---
def plot_combined_steps(A, eps, alpha, rho, eigvals, pdata_abs, pdata_rad, n):
    fig, (ax_abs, ax_rad) = plt.subplots(1, 2, figsize=(24, 11))
    fig.suptitle(f"Criss-Cross Algorithm (N={n})", fontsize=16)
    plt.subplots_adjust(wspace=0.3)

    for ax in [ax_abs, ax_rad]:
        ax.axhline(0, color='grey', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('Real(z)')
        ax.set_ylabel('Imag(z)')
        ax.scatter(eigvals.real, eigvals.imag, color="pink", marker="x", s=60, linewidths=2.5, zorder=5)

    print(f"Tracing boundaries in PARALLEL for N={n} (like Algo2_parallel)...")
    starts = find_starting_points(A, eps, eigvals)
    
    args_list = [(A, eps, z0, eigvals) for z0 in starts]
    cores = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=cores) as pool:
        raw_contours = pool.map(trace_worker, args_list)
        
    all_contours = []
    for idx, contour in enumerate(raw_contours):
        if len(contour) <= 150: continue
        skip = False
        z0 = starts[idx]
        for existing_contour in all_contours:
            if np.min(np.abs(existing_contour - z0)) < 0.3:
                skip = True
                break
        if not skip:
            all_contours.append(contour)

    for ax in [ax_abs, ax_rad]:
        for contour in all_contours:
            ax.plot(contour.real, contour.imag, color='green', linewidth=2.0, alpha=0.7, zorder=2)

    # ==============================
    # 1. Plot Abscissa (Left)
    # ==============================
    ax_abs.set_title(f'Pseudospectral Abscissa: α₀.₅(A)={alpha:.5f}')

    for v_line in pdata_abs.get('vertical_lines', []):
        x_val, y_intervals = v_line
        for j in range(0, len(y_intervals) - 1, 2):
            ax_abs.plot([x_val, x_val], [y_intervals[j], y_intervals[j+1]], '-', color='gray', lw=1.5, zorder=3)

    for h_ray in pdata_abs.get('horizontal_rays', []):
        start_pt, end_pt = h_ray
        ax_abs.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], ':', color='gray', lw=1.5, marker='>', markersize=5, zorder=3)

    ax_abs.axvline(alpha, color='r', linestyle='--', lw=2.0, zorder=4)

    # ==============================
    # 2. Plot Radius (Right)
    # ==============================
    ax_rad.set_title(f'Pseudospectral Radius: ρ₀.₅(A)={rho:.5f}')

    for circle_r in pdata_rad.get('tangent_circles', []):
        circle = plt.Circle((0, 0), circle_r, color='gray', linestyle='--', fill=False, lw=1.5, zorder=3)
        ax_rad.add_artist(circle)

    for r_ray in pdata_rad.get('radial_rays', []):
        (r_s, t_s), (r_e, t_e) = r_ray
        x_s, y_s = r_s * np.cos(t_s), r_s * np.sin(t_s)
        x_e, y_e = r_e * np.cos(t_e), r_e * np.sin(t_e)
        ax_rad.plot([x_s, x_e], [y_s, y_e], ':', color='gray', lw=1.5, marker='>', markersize=5, zorder=3)

    final_circle = plt.Circle((0, 0), rho, color='r', linestyle='--', fill=False, lw=2.0, zorder=4)
    ax_rad.add_artist(final_circle)
    ax_rad.plot(0, 0, 'wo', markeredgecolor='black', markersize=7, zorder=6)
    
    if len(pdata_rad.get('radial_rays', [])) > 0:
        final_theta = pdata_rad['radial_rays'][-1][1][1]
        x_final = rho * np.cos(final_theta)
        y_final = rho * np.sin(final_theta)
        ax_rad.plot([0, x_final], [0, y_final], color='black', linestyle='-.', lw=1.5, alpha=0.7, zorder=2)

    all_x = [pt.real for c in all_contours for pt in c]
    all_y = [pt.imag for c in all_contours for pt in c]
    if all_x and all_y:
        x_min, x_max = min(all_x) - 1, max(all_x) + 1
        y_min, y_max = min(all_y) - 1, max(all_y) + 1
        ax_rad.set_xlim(x_min, x_max + rho/2)
        ax_rad.set_ylim(y_min, y_max + rho/2)
        ax_abs.set_xlim(x_min, x_max + 1)
        ax_abs.set_ylim(y_min, y_max)
        
        ax_abs.set_aspect('equal', adjustable='box')
        ax_rad.set_aspect('equal', adjustable='box')

    legend_elements = [
        mlines.Line2D([], [], color='pink', marker='x', linestyle='None', markersize=8, markeredgewidth=2.5, label='Eigenvalues σ(A)'),
        mlines.Line2D([], [], color='green', lw=1.5, label=f'∂σ({eps:.2f})(A) (Curve Tracing)'),
        mlines.Line2D([], [], color='gray', linestyle=':', marker='>', lw=1.5, label='Search Steps'),
        mlines.Line2D([], [], color='r', linestyle='--', lw=2.0, label='Final Estimate')
    ]
    ax_abs.legend(handles=legend_elements, loc='lower right', fontsize='small', framealpha=0.8)
    ax_rad.legend(handles=legend_elements, loc='lower right', fontsize='small', framealpha=0.8)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def run_benchmarks():
    eps = 0.5
    do_plot = len(sys.argv) > 1 and sys.argv[1].lower() == 'plot'

    tests_to_run = []
    
    for n in [10, 20]:
        tests_to_run.append((f"Random matrix(N={n})", random_matrix(n, seed=42)))
        
    A_linear = linear_eigenvalue_matrix(n=20, spacing=1.2)
    tests_to_run.append(("TEST with U*D*U^-1 (N=20) matrix", A_linear))

    for test_name, A in tests_to_run:
        n = A.shape[0]
        print("\n" + "=" * 80)
        print(f"   Pseudospectral Analysis: {test_name} (eps={eps})")
        print("=" * 80)

        # --- 1. PSEUDOSPECTRAL ABSCISSA ---
        print("\nExecuting Criss-Cross Abscissa Algorithm...")
        t0 = time.time()
        alpha, eigvals, plot_data_abs, history = criss_cross_abscissa(A, eps)
        t_alpha = time.time() - t0

        if do_plot:
            print("\nTable: Horizontal Search History (Abscissa)")
            print(f"{'Iter':<6} | {'Step Type':<15} | {'Alpha Estimate (Re(z))'}")
            print("-" * 50)
            for row in history:
                print(f"{row[0]:<6} | {row[1]:<15} | {row[2]:<22.10f}")
        
        print(f"\nFinal alpha_eps: {alpha:.4f}")
        print(f"Time for Abscissa: {t_alpha:.4f} seconds")
        print("-" * 60)

        # --- 2. PSEUDOSPECTRAL RADIUS ---
        print("\nExecuting Criss-Cross Radius Algorithm...")
        t0 = time.time()
        rho, history_radius, plot_data_rad = criss_cross_radius(A, eps)
        t_rho = time.time() - t0

        if do_plot:
            print("\nTable: Arc/Radial Search History (Radius)")
            print(f"{'Iter':<6} | {'Step Type':<18} | {'Radius Estimate (rho)'}")
            print("-" * 55)
            for row in history_radius:
                print(f"{row[0]:<6} | {row[1]:<18} | {row[2]:<22.10f}")

        print(f"\nFinal rho_eps: {rho:.4f}")
        print(f"Time for Radius: {t_rho:.4f} seconds")
        print("-" * 60)

        if do_plot:
            plot_combined_steps(A, eps, alpha, rho, eigvals, plot_data_abs, plot_data_rad, n)
            
    if do_plot:
        print("\nDisplaying all plots... Close the plot windows to exit.")
        plt.show()

if __name__ == "__main__":
    mp.freeze_support()
    run_benchmarks()