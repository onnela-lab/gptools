from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numbers
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


def coordgrid(*xs: typing.Iterable[np.ndarray], ravel: bool = True, indexing:
              typing.Literal["ij", "xy"] = "ij") -> np.ndarray:
    """
    Obtain coordinates for all grid points induced by `xs`.

    Args:
        xs: Coordinates to construct the grid.
        ravel: Whether to reshape the leading dimensions.

    Returns:
        coord: Coordinates for all grid points with shape `(len(xs[0]), ..., len(xs[p - 1]), p)` if
            `ravel` is `False`, where `p = len(xs)` is the number of dimensions. If `ravel` is
            `True`, the shape is `(len(xs[0]) * ... * len(xs[p - 1]), p)`.
    """
    # Stack the coordinate matrices and move the coordinate dimension to the back.
    coords = np.moveaxis(np.stack(np.meshgrid(*xs, indexing=indexing)), 0, -1)
    if not ravel:
        return coords
    return coords.reshape((-1, len(xs)))


def spatial_neighborhoods(shape: tuple[int], ks: typing.Union[int, tuple[int]],
                          ravel: bool = True) -> np.ndarray:
    """
    Evaluate parental neighborhoods for tensors with finite window size.

    Args:
        shape: Shape of the tensor with Gaussian process distribution.
        ks: Sequence of window widths for each dimension. The same window width is used if `ks` is
            an integer.

    Returns:
        neighborhoods: Mapping from each element of the tensor to its neighborhood. If `ravel`, the
            shape is `(prod(shape), prod(ks))`. If `not ravel`, the shape is
            `(*shape, prod(ks), len(shape))`.
    """
    if isinstance(ks, numbers.Integral):
        ks = (ks,) * len(shape)
    if len(shape) != len(ks):
        raise ValueError("`shape` and `ks` must have matching length or `ks` must be an integer")
    for i, (n, k) in enumerate(zip(shape, ks)):
        if k > n:
            raise ValueError(f"window width {k} is larger than the tensor size {n} along dim {i}")
    # Get all possible combinations of indices and steps. We always ravel the steps.
    coords = coordgrid(*[np.arange(p) for p in shape], ravel=ravel)
    steps = coordgrid(*[np.arange(k) for k in ks], ravel=True)
    # Construct the vector neighborhoods.
    neighborhoods = coords[..., None, :] - steps
    if not ravel:
        return neighborhoods
    # Identify which indices are invalid so we can mark them as such after ravelling the indices.
    neighborhoods = np.moveaxis(neighborhoods, -1, 0)
    invalid = (neighborhoods < 0).any(axis=0)
    neighborhoods = np.ravel_multi_index(tuple(neighborhoods.clip(0)), shape)
    return np.where(invalid, -1, neighborhoods)


def neighborhood_to_edge_index(neighborhoods: np.ndarray) -> np.ndarray:
    """
    Convert a tensor of neighborhoods to an edgelist with self loops.

    Args:
        neighborhoods: Neighborhood matrix such that each row corresponds to the parental neighbors
        of the associated node. Negative node labels are omitted

    Returns:
        edge_index: Tuple of parent and child node labels.
    """
    if neighborhoods.ndim != 2:
        raise ValueError("neighborhoods must be a matrix")
    if (neighborhoods[:, 0] != np.arange(neighborhoods.shape[0])).any():
        raise ValueError("first element in the neighborhood must be the corresponding node")
    return np.transpose([(parent, child) for child, parents in enumerate(neighborhoods) for parent
                         in parents if parent >= 0])
