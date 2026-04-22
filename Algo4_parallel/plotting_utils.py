import matplotlib.pyplot as plt
import numpy as np

def plot_componentwise_pseudospectrum(X, Y, Z, title="3D Componentwise Pseudospectrum"):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    Z_log = np.log10(Z + 1e-15)
    ax.plot_wireframe(X, Y, Z_log, color='black', linewidth=0.2, rstride=1, cstride=1)
    
    ax.view_init(elev=25, azim=-45)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(r'Re($\lambda$)')
    ax.set_ylabel(r'Im($\lambda$)')
    ax.set_zlabel(r'$\log_{10} \rho(\dots)$')
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    plt.tight_layout()
    plt.show()
