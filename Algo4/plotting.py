import matplotlib.pyplot as plt
import numpy as np

def plot_pseudospectrum_contour(X, Y, Z, eigenvalues=None, eps=0.1, n=10):
    plt.figure(figsize=(9.5, 7.5))

    target_level = 1.0 / eps
    
    max_z = np.max(Z[np.isfinite(Z)])
    if max_z > target_level * 40:
        target_level = np.percentile(Z[np.isfinite(Z)], 80.6)

    plt.contour(X, Y, Z,
                levels=[target_level],
                colors="red",
                linewidths=2.9)

    if eigenvalues is not None:
        plt.scatter(eigenvalues.real, eigenvalues.imag,
                    color="#009A45",
                    marker="x",
                    s=80,
                    linewidths=3.2,
                    label="Eigenvalues")

    plt.xlabel(r"$\mathrm{Re}(\lambda)$", fontsize=12)
    plt.ylabel(r"$\mathrm{Im}(\lambda)$", fontsize=12)
    plt.axis("equal")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)

    plt.title(
        rf"$\varepsilon$-pseudospectrum of a random matrix "
        rf"$A \in \mathbb{{C}}^{{{n}\times{n}}}$",
        fontsize=13, pad=16
    )

    plt.legend(loc="upper right", fontsize=10)
    
    sm = plt.contourf(X, Y, Z, levels=25, alpha=0.06, cmap='Reds')
    cbar = plt.colorbar(sm, label=r'$\rho(|(A-\lambda I)^{-1}| \cdot |A|)$')
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.show()
