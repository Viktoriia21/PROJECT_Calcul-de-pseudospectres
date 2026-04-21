import numpy as np
import scipy.linalg as la
from gershgorin import gershgorin_box

def pseudospectrum_grid(A, eps, nx=400, ny=400):
    xmin, xmax, ymin, ymax = gershgorin_box(A, padding=eps)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    n = A.shape[0]
    I = np.eye(n)
    sigma_min = np.zeros_like(X)

    for i in range(nx):
        for j in range(ny):
            z = Z[j, i]
            sigma_min[j, i] = la.svdvals(A - z * I)[-1]

    return X, Y, sigma_min
