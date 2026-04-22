import numpy as np
import time

def compute_f_lambda(A, E, lambda_val):
    #f(lambda) = rho(|(A - lambda*I)^-1| * E)

    n = A.shape[0]
    I = np.eye(n)
    try:
        # Compute Resolvent M = (A - lambda*I)^-1
        resolvent = np.linalg.inv(A - lambda_val * I)
        
        # Componentwise absolute value and multiplication by structure matrix E
        Y = np.abs(resolvent) @ E
        
        #Spectral radius rho(Y)
        eigvals = np.linalg.eigvals(Y)
        return np.max(np.abs(eigvals))
    except np.linalg.LinAlgError:
        return np.inf

def compute_grid(A, E, real_range, imag_range, res=100):
    start_time = time.time()
    
    x = np.linspace(real_range[0], real_range[1], res)
    y = np.linspace(imag_range[0], imag_range[1], res)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(res):
        for j in range(res):
            lambda_val = X[i, j] + 1j * Y[i, j]
            Z[i, j] = compute_f_lambda(A, E, lambda_val)
            
    execution_time = time.time() - start_time
    return X, Y, Z, execution_time
