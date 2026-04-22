import numpy as np
import time
from multiprocessing import Pool

def _worker(args):
    # function to compute f(lambda) for a single point.
    # Args: (A, E, lambda_val)
    A, E, lambda_val = args
    n = A.shape[0]
    try:
        # Resolvent
        inv_matrix = np.linalg.inv(A - lambda_val * np.eye(n))
        # rho(|inv| * E)
        Y = np.abs(inv_matrix) @ E
        return np.max(np.abs(np.linalg.eigvals(Y)))
    except np.linalg.LinAlgError:
        return np.inf

def compute_grid_parallel(A, E, real_range, imag_range, res=100, num_cores=16):
    start_time = time.time()
    
    x = np.linspace(real_range[0], real_range[1], res)
    y = np.linspace(imag_range[0], imag_range[1], res)
    X, Y = np.meshgrid(x, y)
    
    points_args = [
        (A, E, X[i, j] + 1j * Y[i, j]) 
        for i in range(res) for j in range(res)
    ]

    print(f"Starting parallel computation on {num_cores} cores...")
    with Pool(processes=num_cores) as pool:
        results = pool.map(_worker, points_args)
    
    # Reshape flat results back into a grid
    Z = np.array(results).reshape(res, res)
    
    execution_time = time.time() - start_time
    return X, Y, Z, execution_time
