import matplotlib.pyplot as plt

def plot_pseudospectrum(X, Y, sigma_min, eps, n, eigvals):
    plt.figure(figsize=(6, 6))

    plt.contour(X, Y, sigma_min,
                levels=[eps],
                colors="red",
                linewidths=2)

    plt.scatter(eigvals.real, eigvals.imag,
                color="green",
                marker="x",
                s=60,
                linewidths=2.5,
                label="Eigenvalues")

    plt.xlabel(r"$\mathrm{Re}(z)$")
    plt.ylabel(r"$\mathrm{Im}(z)$")
    plt.axis("equal")

    plt.grid(True, which="both",
             linestyle="--",
             linewidth=0.5,
             alpha=0.4)

    plt.title(
        rf"$\varepsilon$-pseudospectrum of a random matrix "
        rf"$A \in \mathbb{{C}}^{{{n}\times{n}}}$",
        fontsize=13
    )

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
