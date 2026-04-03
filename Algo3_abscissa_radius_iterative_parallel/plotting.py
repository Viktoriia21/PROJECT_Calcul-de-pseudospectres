import matplotlib.pyplot as plt
import numpy as np

def plot_pseudospectrum_boundary(contours, eigenvalues, eps=0.5, n=None):
    plt.figure(figsize=(7, 7))
    
    all_points = np.concatenate(contours)
    alpha_eps = np.max(np.real(all_points))
    rho_eps = np.max(np.abs(all_points))

    for contour in contours:
        plt.plot(contour.real, contour.imag, color="red", linewidth=2)

    plt.axvline(x=alpha_eps, color="blue", linestyle="--", label=rf"$\alpha_\epsilon = {alpha_eps:.3f}$")

    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(rho_eps * np.cos(theta), rho_eps * np.sin(theta), 
             color="green", linestyle="--", label=rf"$\rho_\epsilon = {rho_eps:.3f}$")

    plt.scatter(eigenvalues.real, eigenvalues.imag, color="pink", marker="x", s=60, label="Eigenvalues")
    
    plt.xlabel(r"$\mathrm{Re}(z)$")
    plt.ylabel(r"$\mathrm{Im}(z)$")
    plt.axis("equal")
    plt.title(rf"$\varepsilon$-pseudospectrum ($n={n}, \varepsilon={eps}$)")
    plt.legend()
    plt.tight_layout()
    plt.show()