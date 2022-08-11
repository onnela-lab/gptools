from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
import os
import typing


def get_include() -> str:
    """
    Get the include directory for the graph Gaussian process library.
    """
    return os.path.dirname(__file__)


def evaluate_squared_distance(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the squared distance between the Cartesian product of nodes, preserving batch shape.

    Args:
        x: Coordinates with shape `(..., n, p)`, where `...` is the batch shape, `n` is the number
            of nodes, and `p` is the number of dimensions of the embedding space.

    Returns:
        dist2: Squared distance Cartesian product of nodes with shape `(..., n, n)`.
    """
    return np.square(x[..., :, None, :] - x[..., None, :, :]).sum(axis=-1)


def plot_band(x: np.ndarray, ys: np.ndarray, *, p: float = 0.05, relative_alpha: float = 0.25,
              ax: typing.Optional[Axes] = None, **kwargs) -> tuple[Line2D, PolyCollection]:
    """
    Plot a credible band given posterior samples.

    Args:
        x: Coordinates of samples with shape `(n,)`, where `n` is the number of observations.
        ys: Samples with shape `(m, n)`, where `m` is the number of samples.
        p: Tail probability to exclude from the credible band such that the band spans the quantiles
            `[p / 2, 1 - p / 2]`.
        relative_alpha: Opacity of the band relative to the median line.
        ax: Axes to plot into.
        **kwargs: Keyword arguments passed to `ax.plot` for the median line.

    Returns:
        line: Median line.
        band: Credible band spanning the quantiles `[p / 2, 1 - p / 2]`.
    """
    ax = ax or plt.gca()
    l, m, u = np.quantile(ys, [p / 2, 0.5, 1.0 - p / 2], axis=0)
    line, = ax.plot(x, m, **kwargs)
    alpha = kwargs.get("alpha", 1.0) * relative_alpha
    return line, ax.fill_between(x, l, u, color=line.get_color(), alpha=alpha)
