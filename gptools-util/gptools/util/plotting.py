import matplotlib
import matplotlib.axes
import matplotlib.collections
import matplotlib.lines
import numpy as np
from typing import Optional


def plot_band(x: np.ndarray, ys: np.ndarray, *, p: float = 0.05, relative_alpha: float = 0.25,
              ax: Optional[matplotlib.axes.Axes] = None, **kwargs) \
        -> tuple[matplotlib.lines.Line2D, matplotlib.collections.PolyCollection]:
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
    from matplotlib import pyplot as plt
    ax = ax or plt.gca()
    l, m, u = np.quantile(ys, [p / 2, 0.5, 1.0 - p / 2], axis=0)
    line, = ax.plot(x, m, **kwargs)
    alpha = kwargs.get("alpha", 1.0) * relative_alpha
    return line, ax.fill_between(x, l, u, color=line.get_color(), alpha=alpha)
