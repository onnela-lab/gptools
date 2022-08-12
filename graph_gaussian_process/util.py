import numpy as np
import typing
from .missing_module import MissingModule
try:
    import matplotlib.axes
    import matplotlib.collections
    import matplotlib.lines
    from matplotlib import pyplot as plt
except ModuleNotFoundError as ex:
    matplotlib = plt = MissingModule(ex)
try:
    import networkx as nx
except ModuleNotFoundError as ex:
    nx = MissingModule(ex)
try:
    import torch as th
except ModuleNotFoundError as ex:
    th = MissingModule(ex)


def evaluate_squared_distance(x: typing.Union[np.ndarray, "th.Tensor"]) \
        -> typing.Union[np.ndarray, "th.Tensor"]:
    """
    Evaluate the squared distance between the Cartesian product of nodes, preserving batch shape.

    Args:
        x: Coordinates with shape `(..., n, p)`, where `...` is the batch shape, `n` is the number
            of nodes, and `p` is the number of dimensions of the embedding space.

    Returns:
        dist2: Squared distance Cartesian product of nodes with shape `(..., n, n)`.
    """
    residuals = x[..., :, None, :] - x[..., None, :, :]
    return (residuals * residuals).sum(axis=-1)


def plot_band(x: np.ndarray, ys: np.ndarray, *, p: float = 0.05, relative_alpha: float = 0.25,
              ax: typing.Optional[matplotlib.axes.Axes] = None, **kwargs) \
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
    ax = ax or plt.gca()
    l, m, u = np.quantile(ys, [p / 2, 0.5, 1.0 - p / 2], axis=0)
    line, = ax.plot(x, m, **kwargs)
    alpha = kwargs.get("alpha", 1.0) * relative_alpha
    return line, ax.fill_between(x, l, u, color=line.get_color(), alpha=alpha)


def coordgrid(*xs: typing.Iterable[np.ndarray], ravel: bool = True,
              indexing: typing.Literal["ij", "xy"] = "ij") -> np.ndarray:
    """
    Obtain coordinates for all grid points induced by `xs`.

    Args:
        xs: Coordinates to construct the grid.
        ravel: Whether to reshape the leading dimensions.
        indexing: Whether to use Cartesian `xy` or matrix `ij` indexing (defaults to `ij`).

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


def lattice_neighborhoods(
        shape: tuple[int], k: typing.Union[int, tuple[int]], ravel: bool = True,
        bounds: typing.Literal["cuboid", "ellipsoid"] = "ellipsoid") -> np.ndarray:
    """
    Evaluate predecessor neighborhoods for nodes on a lattice with given window size.

    Args:
        shape: Shape of the tensor with Gaussian process distribution.
        k: Half window width or sequence of half window widths for each dimension. Each node will
            have a receptive field at most `k` to the "left" and "right" in each dimension.
        bounds: Boundary shape for the receptive field. "cuboid" results in a cuboid with dimensions
            `2 * k + 1`. "ellipsoid" results in a graph with ellipsoidal receptive field that is a
            strict subset of the cuboid. Using an ellipsoidal receptive field can reduce
            computational cost, especially in high dimensions. Ellipsoids can also attenuate
            artefacts that may arise from the anisotropy of the cuboidal receptive field.

    Returns:
        neighborhoods: Mapping from each element of the tensor to its neighborhood. If `ravel`, the
            shape is `(prod(shape), prod(ks))`. If `not ravel`, the shape is
            `(*shape, prod(ks), len(shape))`.
    """
    if bounds not in (expected := {"cuboid", "ellipsoid"}):
        raise ValueError(f"`bounds` must be one of {expected} but got {bounds}")
    # Convert shape and window widths to arrays and verify bounds.
    shape = np.asarray(shape)
    k = k * np.ones_like(shape)
    for i, (size, width) in enumerate(zip(shape, 2 * k + 1)):
        if width > size:
            raise ValueError(f"window width {width} exceeds the tensor size {size} along dim {i}")
    # Get all possible combinations of indices and steps.
    coords = coordgrid(*[np.arange(p) for p in shape], ravel=True)
    steps = coordgrid(*[np.roll(np.arange(- s, s + 1), - s) for s in k], ravel=True)
    # Construct the vector neighborhoods and a mask that removes indices outside the tensor bounds.
    neighborhoods = coords[..., None, :] - steps
    assert neighborhoods.shape == (shape.prod(), (2 * k + 1).prod(), len(shape))
    mask = ((neighborhoods >= 0) & (neighborhoods < shape)).all(axis=-1)
    # If we want an ellipsoidal neighborhood, we remove neighbors that are too far.
    if bounds == "ellipsoid":
        t = np.square(steps / k).sum(axis=-1)
        mask &= t <= 1
    # Mask invalid indices, ravel the coordinate indices, and mask again after ravelling.
    neighborhoods = np.where(mask[..., None], neighborhoods, 0)
    neighborhoods = np.ravel_multi_index(np.moveaxis(neighborhoods, -1, 0), shape)
    # Ensure all neighbors are predecessors of each node.
    mask &= neighborhoods <= np.arange(coords.shape[0])[:, None]
    neighborhoods = np.where(mask, neighborhoods, -1)
    if ravel:
        return neighborhoods
    return neighborhoods.reshape([*shape, (2 * k + 1).prod()])


def _check_indexing(indexing: typing.Literal["numpy", "stan"]) -> typing.Literal["numpy", "stan"]:
    if indexing not in (expected := {"numpy", "stan"}):
        raise ValueError(f"`indexing` must be one of {expected} but got {indexing}")
    return indexing


def neighborhood_to_edge_index(neighborhoods: np.ndarray,
                               indexing: typing.Literal["numpy", "stan"] = "stan") -> np.ndarray:
    """
    Convert a tensor of neighborhoods to an edgelist with self loops.

    Args:
        neighborhoods: Neighborhood matrix such that each row corresponds to the parental neighbors
            of the associated node. Negative node labels are omitted
        indexing: Whether to use zero-based indexing (`numpy`) or one-based indexing (`stan`).

    Returns:
        edge_index: Tuple of parent and child node labels.
    """
    if neighborhoods.ndim != 2:
        raise ValueError("neighborhoods must be a matrix")
    if (neighborhoods[:, 0] != np.arange(neighborhoods.shape[0])).any():
        raise ValueError("first element in the neighborhood must be the corresponding node")

    edge_index = np.transpose([(parent, child) for child, parents in enumerate(neighborhoods) for
                               parent in parents if parent >= 0])
    if _check_indexing(indexing) == "stan":
        return edge_index + 1
    return edge_index


def check_edge_index(edge_index: np.ndarray, indexing: typing.Literal["numpy", "stan"] = "stan") \
        -> np.ndarray:
    """
    Check the edge index, ensuring it has the right structure for the Stan implementation and
    corresponds to a directed acyclic graph.

    Args:
        edge_index: Tuple of parent and child node labels to check.
        indexing: Whether to use zero-based indexing (`numpy`) or one-based indexing (`stan`).

    Returns:
        edge_index: Edge index after checking.
    """
    # Check the tensor shape.
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge index must have shape (2, num_edges) but got {edge_index.shape}")

    # Check that child nodes start with the proper value.
    expected = 0 if _check_indexing(indexing) == "numpy" else 1
    parents, children = edge_index

    # Check that child labels are consecutive.
    unique, index = np.unique(children, return_index=True)
    if (np.diff(index) < 0).any() or (unique != np.arange(unique.size) + expected).any():
        raise ValueError(f"child node indices must be consecutive starting at {expected} for "
                         f"{indexing} indexing")

    # Check that the first edge of each node is a self-loop.
    if (children[index] != parents[index]).any():
        raise ValueError("the first edge of each child must be a self loop")

    # Check that there are no cycles after removing self-loops.
    graph = edge_index_to_graph(edge_index)
    try:
        cycle = nx.find_cycle(graph)
        raise ValueError(f"edge index induces a graph with the cycle: {cycle}")
    except nx.NetworkXNoCycle:
        pass

    return edge_index


def edge_index_to_graph(edge_index: np.ndarray, remove_self_loops: bool = True) -> "nx.DiGraph":
    """
    Convert edge indices to a directed graph.

    Args:
        edge_index: Tuple of parent and child node labels.
        remove_self_loops: Whether to remove self loops.

    Returns:
        graph: Directed graph induced by the edge indices.
    """
    graph = nx.DiGraph()
    graph.add_edges_from((u, v) for u, v in edge_index.T if u != v and remove_self_loops)
    return graph


def is_tensor(x: typing.Union[np.ndarray, "th.Tensor"]) -> bool:
    """
    Check if `x` is a torch tensor.
    """
    try:
        return isinstance(x, th.Tensor)
    except ModuleNotFoundError:
        return False
