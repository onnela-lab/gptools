from gptools.util import graph
import networkx as nx
import numbers
import numpy as np
import pytest
import re
from typing import Literal, Optional, Union


@pytest.mark.parametrize("shape, ks", [
    ((23,), 5),
    ((19, 23), 7),
    ((19, 23), (5, 7)),
    ((7, 9, 11), 3),
    ((7, 9, 11), (3, 4, 5)),
])
@pytest.mark.parametrize("bounds", graph.LatticeBounds)
@pytest.mark.parametrize("compress", [False, True])
def test_lattice_predecessors(
        shape: tuple[int], ks: Union[int, tuple[int]], bounds: graph.LatticeBounds,
        compress: bool) -> None:
    predecessors = graph.lattice_predecessors(shape, ks, bounds, compress)

    # General shape check.
    rows, cols = predecessors.shape
    assert rows == np.prod(shape)
    expected_cols = np.prod(2 * np.asarray(ks) * np.ones_like(shape) + 1) - 1
    if compress:
        assert cols < expected_cols
    else:
        assert cols == expected_cols

    # Shape check for up to two dimensions if `k` is a scalar and the predecessors are compressed.
    if isinstance(ks, numbers.Number) and len(shape) < 3 and compress:
        expected_cols = graph.num_lattice_predecessors(ks, bounds, len(shape))
        assert cols == expected_cols


@pytest.mark.parametrize("shape, ks, bounds, match", [
    ((5, 6), (3, 7), "cube", "exceeds the tensor size"),
    ((10, 6), 2, "invalid-shape", "'invalid-shape' is not a valid LatticeBounds"),
])
def test_lattice_predecessors_invalid(
        shape: tuple[int], ks: tuple[int], bounds: graph.LatticeBounds, match: str) -> None:
    with pytest.raises(ValueError, match=re.escape(match)):
        graph.lattice_predecessors(shape, ks, bounds=bounds)


@pytest.mark.parametrize("indexing", ["numpy", "stan"])
def test_predecessors_to_edge_index(indexing: Literal["numpy", "stan"]) -> None:
    shape = (5, 7, 11)
    ks = (2, 3, 5)
    predecessors = graph.lattice_predecessors(shape, ks)
    edge_index = graph.predecessors_to_edge_index(predecessors, indexing)
    # The number of edges is the number of all combinations less the neighbors that fall outside the
    # bounds of the tensor. We just do a weak check here.
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] < predecessors.size
    # Check the validity of the edge index obtained from spatial predecessors.
    assert graph.check_edge_index(edge_index, indexing=indexing) is edge_index


@pytest.mark.parametrize("predecessors, match", [
    (np.asarray([[0], [0]]), "self-loops are not allowed; found 1"),
    (np.arange(3), "must be a matrix"),
])
def test_predecessors_to_edge_index_invalid(predecessors: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        graph.predecessors_to_edge_index(predecessors)


@pytest.mark.parametrize("edge_index, indexing, match", [
    (np.zeros(2), None, "edge index must have shape (2, num_edges) but got (2,)"),
    (np.zeros((3, 2)), None, "edge index must have shape (2, num_edges) but got (3, 2)"),
    (np.zeros((2, 3)), "foobar", "`indexing` must be one of"),
    (np.zeros((2, 3)), "stan", "expected indexing to start at 1"),
    (-np.ones((2, 3)), "numpy", "expected indexing to start at 0"),
    ([np.ones(2), np.ones(2)], "numpy", "self-loops are not allowed"),
    ([[0, 1, 2], [1, 2, 0]], "numpy", "successor indices must be non-decreasing"),
    ([[1], [0]], "numpy", "predecessors must be less than successors"),
])
def test_check_edge_index_invalid(
        edge_index: np.ndarray, indexing: Literal["numpy", "stan"],
        match: Optional[str]) -> None:
    with pytest.raises(ValueError, match=re.escape(match)):
        graph.check_edge_index(np.asarray(edge_index), indexing)


def test_compress_predecessors_invalid_shape():
    with pytest.raises(ValueError):
        graph.compress_predecessors(np.zeros(2))


@pytest.mark.parametrize("return_mapping", [False, True])
def test_edge_index_to_graph_roundtrip(return_mapping: bool) -> None:
    G = nx.erdos_renyi_graph(10, 0.1)
    edge_index = graph.graph_to_edge_index(G, indexing="numpy", return_mapping=return_mapping)
    if return_mapping:
        edge_index, mapping = edge_index
        assert isinstance(mapping, dict)
    H = graph.edge_index_to_graph(edge_index)
    assert set(map(tuple, G.edges)) == set(map(tuple, H.edges))
