import numpy as np
import multiprocessing as mp
from scipy.linalg import eig
from utils import min_singular_triplet

# --- Task wrappers for parallel execution ---
num_cores = max(1, mp.cpu_count() - 1)

def _h_search_task(args):
    A, eps, mid_y, alpha_est, n = args
    H_h = np.block([
        [-mid_y * np.eye(n) + 1j * A.conj().T, -eps * np.eye(n)],
        [eps * np.eye(n), 1j * A + mid_y * np.eye(n)]
    ])
    h_res = eig(H_h, left=False, right=False)
    h_x_vals = h_res[np.abs(h_res.real) < 1e-2].imag
    pos_h_x = h_x_vals[h_x_vals > (alpha_est - 1e-8)]
    if len(pos_h_x) > 0:
        return np.max(pos_h_x), mid_y
    return None

def _r_search_task(args):
    A, eps, theta_mid, rho_est, n = args
    H_r = np.block([
        [1j * np.exp(1j * theta_mid) * A.conj().T, -eps * np.eye(n)],
        [eps * np.eye(n), 1j * np.exp(-1j * theta_mid) * A]
    ])
    try:
        valid_r_res = eig(H_r, left=False, right=False)
        valid_r = valid_r_res[np.abs(valid_r_res.real) < 1e-2].imag
        pos_valid_r = valid_r[valid_r > (rho_est - 1e-8)]
        if len(pos_valid_r) > 0:
            return np.max(pos_valid_r), theta_mid
    except:
        pass
    return None

# --- Algo 2 ---
def trace_boundary(A, eps, z_start, eigvals,
                   tau_init=0.1, max_steps=30000,
                   tol_close=0.08, correction_steps=18):
    """
    Traces the boundary of the epsilon-pseudospectrum using predictor-corrector logic.
    """
    n = A.shape[0]
    I = np.eye(n, dtype=complex)

    def trace_direction(z0, dir_sign=1.0):
        points = [z0]
        z = z0
        _, u, v = min_singular_triplet(z * I - A)

        for _ in range(max_steps):
            g = np.vdot(v, u)
            if abs(g) < 1e-10 or not np.isfinite(g):
                break

            tau = min(tau_init, 0.5 * np.min(np.abs(eigvals - z)))
            tangent = dir_sign * 1j * g / abs(g)
            z_pred = z + tau * tangent

            sigma_h, u_h, v_h = min_singular_triplet(z_pred * I - A)
            denom = np.vdot(u_h, v_h)

            if abs(denom) > 1e-10:
                z_new = z_pred - (sigma_h - eps) / denom
            else:
                z_new = z_pred

            points.append(z_new)
            z = z_new
            u, v = u_h, v_h

            if len(points) > 200 and np.min(np.abs(np.array(points[100:]) - z0)) < tol_close:
                break
        return np.array(points)

    forward = trace_direction(z_start, 1.0)
    backward = trace_direction(z_start, -1.0)
    return np.concatenate([forward[::-1], backward[1:]])

def get_abscissa_intersections(A, eps, x):
    n = A.shape[0]
    H = np.block([
        [x * np.eye(n) - A.conj().T, -eps * np.eye(n)],
        [eps * np.eye(n), A - x * np.eye(n)]
    ])
    eigs = eig(H, left=False, right=False)
    mask = np.abs(eigs.real) < 1e-2
    return np.sort(eigs[mask].imag)

def get_radius_intersections(A, eps, r):
    n = A.shape[0]
    M = np.block([
        [-eps * np.eye(n), A],
        [r * np.eye(n), np.zeros((n, n))]
    ])
    N = np.block([
        [np.zeros((n, n)), r * np.eye(n)],
        [A.conj().T, -eps * np.eye(n)]
    ])
    try:
        eigs = eig(M, N, left=False, right=False)
        mask = np.abs(np.abs(eigs) - 1.0) < 1e-2
        return np.sort(np.angle(eigs[mask]))
    except:
        return np.array([])

# =====================================================================
# ALGORITHM 3: CRISS-CROSS
# =====================================================================

def criss_cross_abscissa(A, eps, tol=1e-8):
    """
    Computes the epsilon-pseudospectral abscissa alpha_eps(A).
    """
    n = A.shape[0]
    history = []
    plot_data = {'vertical_lines': [], 'midpoints': [], 'horizontal_rays': []}

    # STEP 1: Find an eigenvalue lambda of A with the largest real part.
    # This gives us a valid starting point that is guaranteed to be inside the pseudospectrum.
    eigvals = np.linalg.eigvals(A)
    idx = np.argmax(eigvals.real)
    z_k = eigvals[idx]

    # STEP 2: Initial Horizontal Search.
    # From lambda, search horizontally to the right to find where this horizontal 
    # line intersects the boundary of the pseudospectrum.
    H_init = np.block([
        [-z_k.imag * np.eye(n) + 1j * A.conj().T, -eps * np.eye(n)],
        [eps * np.eye(n), 1j * A + z_k.imag * np.eye(n)]
    ])
    eigs_init = eig(H_init, left=False, right=False)
    x_vals = eigs_init[np.abs(eigs_init.real) < 1e-2].imag
    
    pos_x = x_vals[x_vals >= z_k.real - 1e-5]
    if len(pos_x) > 0:
        alpha_est = np.max(pos_x)
    else:
        alpha_est = z_k.real
        
    z_k = complex(alpha_est, z_k.imag)
    plot_data['horizontal_rays'].append(((eigvals[idx].real, eigvals[idx].imag), (z_k.real, z_k.imag)))

    for i in range(1, 20):
        alpha_old = alpha_est
        
        # STEP 3: Vertical Search.
        # Find all points where the vertical line Re(z) = alpha_est intersects the boundary.
        # This gives us a set of disjoint intervals along the vertical line.
        y_ints = get_abscissa_intersections(A, eps, alpha_est)
        plot_data['vertical_lines'].append((alpha_est, y_ints))
        
        new_alpha = alpha_old
        best_y = z_k.imag
        found_better = False

        # STEP 4: Horizontal Search from Midpoints.
        # For each vertical interval found in Step 3, compute its midpoint.
        # Then, perform a horizontal search from each midpoint to find a new, larger real part.
        if len(y_ints) >= 2:
            tasks = []
            for j in range(0, len(y_ints) - 1, 2):
                mid_y = (y_ints[j] + y_ints[j+1]) / 2.0
                plot_data['midpoints'].append((alpha_est, mid_y))
                # Setup rotated Hamiltonian to search horizontally along Im(z) = mid_y
                tasks.append((A, eps, mid_y, alpha_est, n))
            
            with mp.Pool(processes=num_cores) as pool:
                results = pool.map(_h_search_task, tasks)
            
            for res in results:
                if res:
                    current_max, m_y = res
                    # Update alpha_est if we found a point further to the right
                    if current_max > new_alpha:
                        new_alpha = current_max
                        best_y = m_y
                        found_better = True
        
        history.append([i, "Horiz Search", new_alpha])

        # Check for convergence
        if not found_better or abs(new_alpha - alpha_old) < tol:
            break
        
        alpha_est = new_alpha
        z_k = complex(alpha_est, best_y)
        plot_data['horizontal_rays'].append(((alpha_old, best_y), (z_k.real, z_k.imag)))

    return alpha_est, eigvals, plot_data, history

def criss_cross_radius(A, eps, tol=1e-8):
    """
    Computes the epsilon-pseudospectral radius rho_eps(A).
    Follows the 4-step Criss-Cross algorithm adapted for polar coordinates.
    """
    n = A.shape[0]
    history_radius = []
    plot_data_radius = {'radial_rays': [], 'arc_midpoints': [], 'tangent_circles': []}

    # STEP 1: Find an eigenvalue lambda of A with the largest absolute value (magnitude).
    eigvals = np.linalg.eigvals(A)
    idx = np.argmax(np.abs(eigvals))
    lam = eigvals[idx]

    # STEP 2: Initial Radial Search.
    # Search outwards from lambda along the ray passing through the origin and lambda
    # to find the intersection with the pseudospectrum boundary.
    theta_current = np.angle(lam)
    H_rad = np.block([
        [1j * np.exp(1j * theta_current) * A.conj().T, -eps * np.eye(n)],
        [eps * np.eye(n), 1j * np.exp(-1j * theta_current) * A]
    ])
    eigs_rad = eig(H_rad, left=False, right=False)
    r_vals = eigs_rad[np.abs(eigs_rad.real) < 1e-2].imag
    pos_r = r_vals[r_vals >= np.abs(lam) - 1e-5]
    
    if len(pos_r) > 0:
        rho_est = np.max(pos_r)
    else:
        rho_est = np.abs(lam)

    plot_data_radius['radial_rays'].append(((np.abs(eigvals[idx]), theta_current), (rho_est, theta_current)))

    for i in range(1, 20):
        rho_old = rho_est
        plot_data_radius['tangent_circles'].append(rho_est)
        
        # STEP 3: Arc Search.
        # Find all angles where the circle of radius rho_est intersects the boundary.
        # This yields a set of arcs on the circle that lie within the pseudospectrum.
        theta_ints = get_radius_intersections(A, eps, rho_est)
        
        new_rho = rho_old
        best_theta = theta_current
        found_better = False

        # STEP 4: Radial Search from Arc Midpoints.
        # For each arc found in Step 3, find its midpoint angle.
        # Search radially outwards from these midpoints to find a larger radius.
        if len(theta_ints) >= 2:
            tasks = []
            for j in range(len(theta_ints)):
                t1 = theta_ints[j]
                t2 = theta_ints[(j+1) % len(theta_ints)]
                
                # Handle angle wrap-around at pi / -pi
                if t2 <= t1:
                    t2 += 2 * np.pi
                theta_mid = (t1 + t2) / 2.0
                if theta_mid > np.pi:
                    theta_mid -= 2 * np.pi
                
                plot_data_radius['arc_midpoints'].append((rho_est, theta_mid))
                tasks.append((A, eps, theta_mid, rho_est, n))

            with mp.Pool(processes=num_cores) as pool:
                results = pool.map(_r_search_task, tasks)

            for res in results:
                if res:
                    current_max, t_mid = res
                    # Update rho_est if we found a point further out
                    if current_max > new_rho:
                        new_rho = current_max
                        best_theta = t_mid
                        found_better = True
        
        history_radius.append([i, "Arc/Radial Search", new_rho])

        # Check for convergence
        if not found_better or abs(new_rho - rho_old) < tol:
            break
        
        rho_est = new_rho
        theta_current = best_theta
        plot_data_radius['radial_rays'].append(((rho_old, best_theta), (new_rho, theta_current)))

    return rho_est, history_radius, plot_data_radius