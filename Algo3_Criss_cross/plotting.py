import matplotlib.pyplot as plt
import numpy as np

def plot_pseudospectrum_boundary(contours, eigenvalues, eps=0.5, n=None):
    plt.figure(figsize=(6, 6))

    for contour in contours:
        plt.plot(contour.real,
                 contour.imag,
                 color="red",
                 linewidth=2)

    plt.scatter(eigenvalues.real,
                eigenvalues.imag,
                color="pink",
                marker="x",
                s=60,
                linewidths=2.5,
                label="Eigenvalues")

    plt.xlabel(r"$\mathrm{Re}(z)$")
    plt.ylabel(r"$\mathrm{Im}(z)$")
    plt.axis("equal")

    plt.title(
        rf"$\varepsilon$-pseudospectrum of a random matrix "
        rf"$A \in \mathbb{{C}}^{{{n}\times{n}}}$",
        fontsize=13
    )

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
