from matplotlib.axes import Axes
from matplotlib import pyplot as plt
import numpy as np
import typing
from ....kernels import ExpQuadKernel


def simulate(X: np.ndarray, mu: float, kernel: ExpQuadKernel) -> dict:
    """
    Simulate data from a Gaussian process Poisson regression model with log link function.

    Args:
        X: Coordinates of observations.
        mu: Prior mean of the Gaussian process.
        kernel: Gaussian process kernel.
        k: Window size for nearest neighbors.

    Returns:
        sample: Sample from the process.
    """
    if np.ndim(X) < 2:
        X = X[:, None]
    cov = kernel(X)
    eta = np.random.multivariate_normal(mu * np.ones(X.shape[0]), cov)
    lam = np.exp(eta)
    y = np.random.poisson(lam)

    return {
        "X": X,
        "mu": mu,
        "num_nodes": X.shape[0],
        "num_dims": X.shape[1],
        "eta": eta,
        "lam": lam,
        "y": y,
        "alpha": kernel.alpha,
        "rho": kernel.rho,
        "epsilon": kernel.epsilon,
    }


def plot_realization_1d(realization: dict, ax: typing.Optional[Axes] = None) -> Axes:
    """
    Plot a realization of the model.

    Args:
        realization: Realization of the model, e.g., obtained from :func:`simulate`.
        ax: Axes used for plotting. Defaults to :func:`plt.gca`.

    Returns:
        ax: Axes used for plotting.
    """
    ax = ax or plt.gca()
    x = np.squeeze(realization["X"])
    ax.plot(x, realization["lam"], label=r"rate $\lambda$")
    ax.scatter(x, realization["y"], marker=".", color="black", label=r"counts $y$")
    ax.set_xlabel(r"Coordinate $x$")
    return ax
