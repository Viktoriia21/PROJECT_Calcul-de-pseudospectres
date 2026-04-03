import numpy as np
from utils import min_singular_triplet, smin

def trace_boundary(A, eps, z_start, eigvals,
                   tau_init=0.1,          
                   max_steps=30000,        
                   tol_close=0.08,         
                   correction_steps=18):   
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
    contour = np.concatenate((backward[::-1], forward[1:]))

    unique = [contour[0]]
    for p in contour[1:]:
        if np.abs(p - unique[-1]) > 1e-6:
            unique.append(p)

    return np.array(unique)

def compute_err(args):
    A, eps, x, y = args
    z = complex(x, y)
    return abs(smin(A, z) - eps), z

def trace_wrapper(args):
    A, eps, z0, eigvals, idx, total = args
    # print(f"Tracing from starting point {idx}/{total}...")
    return trace_boundary(A, eps, z0, eigvals)