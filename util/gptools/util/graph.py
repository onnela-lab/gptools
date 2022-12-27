
import enum
import numpy as np
from typing import Literal, Union
from . import coordgrid, FALSE


if FALSE:
    import networkx as nx


# Solutions to the Gauss circle problem (https://en.wikipedia.org/wiki/Gauss_circle_problem).
GAUSS_PROBLEM_SEQUENCE = np.asarray([1, 5, 13, 29, 49, 81, 113, 149, 197, 253, 317, 377, 441])


class LatticeBounds(enum.Enum):
    """
    Boundary shape for the receptive field on a lattice.

    - CUBE results in a hypercuboid with dimensions `2 * k + 1`.
    - DIAMOND results in the highest-volume hypercuboid whose vertices are axis-aligned that can fit
        into CUBE.
    - ELLIPSE results in the highest-volume ellipsoid that can fit into CUBE.
    """
    CUBE = "cube"
    DIAMOND = "diamond"
    ELLIPSE = "ellipse"


def lattice_predecessors(
        shape: tuple[int], k: Union[int, tuple[int]],
        bounds: LatticeBounds = LatticeBounds.ELLIPSE, compress: bool = True
        ) -> np.ndarray:
    """
    Evaluate predecessors for nodes on a lattice with given window size. Approximations will
    generally be poor if the half window width is smaller than the correlation length of the kernel.

    Args:
        shape: Shape of the tensor with Gaussian process distribution.
        k: Half window width or sequence of half window widths for each dimension. Each node will
            have a receptive field at most `k` to the "left" and "right" in each dimension.
        bounds: Bounds of the receptive field. See :class:`LatticeBounds` for details.
        compress: Whether to compress predecessors such that the number of colums is equal to the
            maximum degree.

    Returns:
        predecessors: Mapping from each element of the tensor to its predecessors with shape
            `(prod(shape), l)`, where `l` is the number of predecessors.
    """
    bounds = LatticeBounds(bounds)
    # Convert shape and window widths to arrays and verify bounds.
    shape = np.asarray(shape)
    k = k * np.ones_like(shape)
    for i, (size, width) in enumerate(zip(shape, 2 * k + 1)):
        if width > size:
            raise ValueError(f"window width {width} exceeds the tensor size {size} along dim {i}")
    # Get all possible combinations of indices and steps.
    coords = coordgrid(*[np.arange(p) for p in shape], ravel=True)
    steps = coordgrid(*[np.roll(np.arange(- s, s + 1), - s) for s in k], ravel=True)
    # Construct the vector predecessors and a mask that removes indices outside the tensor bounds.
    predecessors = coords[..., None, :] - steps
    assert predecessors.shape == (shape.prod(), (2 * k + 1).prod(), len(shape))
    mask = ((predecessors >= 0) & (predecessors < shape)).all(axis=-1)
    # If we want an ellipsoidal predecessors, we remove ones that are too far away.
    if bounds == LatticeBounds.ELLIPSE:
        t = np.square(steps / k).sum(axis=-1)
        mask &= t <= 1
    elif bounds == LatticeBounds.DIAMOND:
        t = np.abs(steps / k).sum(axis=-1)
        mask &= t <= 1
    # Mask invalid indices, ravel the coordinate indices, and mask again after ravelling.
    predecessors = np.where(mask[..., None], predecessors, 0)
    predecessors = np.ravel_multi_index(np.moveaxis(predecessors, -1, 0), shape)
    # Ensure all nodes are predecessors so the graph is acyclic.
    mask &= predecessors <= np.arange(coords.shape[0])[:, None]
    predecessors = np.where(mask, predecessors, -1)[:, 1:]
    if compress:
        predecessors = compress_predecessors(predecessors)
    return predecessors


def num_lattice_predecessors(k: int, bounds: LatticeBounds, p: int) -> int:
    """
    Evaluate the number of predecessors in a lattice graph.
    """
    bounds = LatticeBounds(bounds)
    if p == 1:
        return k
    elif p == 2 and bounds == LatticeBounds.CUBE:
        return 2 * k * (k + 1)
    elif p == 2 and bounds == LatticeBounds.DIAMOND:
        return k * (k + 1)
    elif p == 2 and bounds == LatticeBounds.ELLIPSE:
        return (GAUSS_PROBLEM_SEQUENCE[k] - 1) // 2
    else:
        raise NotImplementedError(f"k = {k}; bounds = {bounds}; p = {p}")


def compress_predecessors(predecessors: np.ndarray) -> np.ndarray:
    """
    Compress a predecessor matrix such that there is at least one predecessors that does not contain
    invalid indices. In other words, we remove as many columns as possible without discarding any
    information.

    Args:
        predecessors: Mapping from each element of the tensor to its predecessors.

    Returns:
        compressed: Predecessors after removing as many columns as possible.
    """
    if predecessors.ndim != 2:
        raise ValueError("predecessors must be a matrix")
    num_nodes, _ = predecessors.shape
    max_degree = (predecessors >= 0).sum(axis=1).max()
    compressed = - np.ones((num_nodes, max_degree), dtype=predecessors.dtype)
    for i, neighbors in enumerate(predecessors):
        neighbors = neighbors[neighbors >= 0]
        compressed[i, :len(neighbors)] = neighbors
    return compressed


def _check_indexing(indexing: Literal["numpy", "stan"]) -> Literal["numpy", "stan"]:
    if indexing not in (expected := {"numpy", "stan"}):
        raise ValueError(f"`indexing` must be one of {expected} but got {indexing}")
    return indexing


def predecessors_to_edge_index(predecessors: np.ndarray,
                               indexing: Literal["numpy", "stan"] = "stan") -> np.ndarray:
    """
    Convert a matrix of predecessors to an edgelist with self loops.

    Args:
        predecessors: Predecessor matrix such that each row corresponds to the predecessors of the
            associated node. Negative node labels are omitted.
        indexing: Whether to use zero-based indexing (`numpy`) or one-based indexing (`stan`).

    Returns:
        edge_index: Tuple of parent and child node labels.
    """
    if predecessors.ndim != 2:
        raise ValueError("predecessors must be a matrix")
    nodes = np.arange(predecessors.shape[0])
    if num_self_loops := (predecessors == nodes[:, None]).sum():
        raise ValueError(f"self-loops are not allowed; found {num_self_loops}")

    edge_index = np.transpose([(parent, child) for child, parents in enumerate(predecessors) for
                               parent in parents if parent >= 0])
    if _check_indexing(indexing) == "stan":
        return edge_index + 1
    return edge_index


def check_edge_index(edge_index: np.ndarray, indexing: Literal["numpy", "stan"] = "stan") \
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
    expected_min = 0 if _check_indexing(indexing) == "numpy" else 1
    if (actual_min := edge_index.min()) < expected_min:
        raise ValueError(f"expected indexing to start at {expected_min}; found {actual_min}")
    predecessors, successors = edge_index

    # Check that child labels are non-decreasing.
    if (np.diff(successors) < 0).any():
        raise ValueError("successor indices must be non-decreasing")

    # Check that there are no self-loops.
    if num_self_loops := (successors == predecessors).sum():
        raise ValueError(f"self-loops are not allowed; found {num_self_loops}")

    # Check that predecessors are less than successors.
    if (predecessors >= successors).any():
        raise ValueError("predecessors must be less than successors")

    return edge_index


def edge_index_to_graph(edge_index: np.ndarray) -> "nx.DiGraph":
    """
    Convert edge indices to a directed graph.

    Args:
        edge_index: Tuple of parent and child node labels.

    Returns:
        graph: Directed graph induced by the edge indices.
    """
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_edges_from(edge_index.T)
    return graph


def graph_to_edge_index(graph: "nx.Graph", return_mapping: bool = False,
                        indexing: Literal["numpy", "stan"] = "stan") -> np.ndarray:
    """
    Convert a graph to edge indices.

    Args:
        graph: Graph to convert.
        add_self_loops: Whether to include self loops.
        return_mapping: Whether to return the mapping from node labels to indices.
        indexing: Whether to use zero-based indexing (`numpy`) or one-based indexing (`stan`).

    Returns:
        edge_index: Tuple of parent and child node labels.
    """
    start = 1 if indexing == "stan" else 0
    mapping = {node: i for i, node in enumerate(sorted(graph), start=start)}
    edge_index = []
    for node in sorted(graph):
        neighbors = graph.neighbors(node)
        node = mapping[node]
        for neighbor in neighbors:
            neighbor = mapping[neighbor]
            if neighbor < node:
                edge_index.append((neighbor, node))

    edge_index = np.transpose(edge_index)
    if return_mapping:
        return edge_index, mapping
    return edge_index
