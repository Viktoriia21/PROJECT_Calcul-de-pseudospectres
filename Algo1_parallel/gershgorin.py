import numpy as np

def gershgorin_disks(A):

    centers = np.diag(A)
    radii = np.sum(np.abs(A), axis=1) - np.abs(centers)
    return centers, radii

def gershgorin_box(A, padding=0.0):

    centers, radii = gershgorin_disks(A)
    xmin = np.min(centers.real - radii) - padding
    xmax = np.max(centers.real + radii) + padding
    ymin = np.min(centers.imag - radii) - padding
    ymax = np.max(centers.imag + radii) + padding
    return xmin, xmax, ymin, ymax
