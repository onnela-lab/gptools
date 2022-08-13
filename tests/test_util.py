from graph_gaussian_process import util
import numbers
import numpy as np
import pytest
import re
from scipy.spatial.distance import cdist
import typing


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("p", [1, 3, 5])
def test_evaluate_squared_distance(p: int) -> None:
    X = np.random.normal(0, 1, (57, p))
    np.testing.assert_allclose(util.evaluate_squared_distance(X), cdist(X, X) ** 2)


def test_plot_band() -> None:
    x = np.linspace(0, 1, 21)
    ys = np.random.normal(0, 1, (43, 21))
    line, band = util.plot_band(x, ys)


@pytest.mark.parametrize("shape", [(3,), (4, 5)])
@pytest.mark.parametrize("ravel", [False, True])
def test_coord_grid(shape: tuple[int], ravel: bool) -> None:
    xs = [np.arange(p) for p in shape]
    coords = util.coordgrid(*xs, ravel=ravel)
    if ravel:
        assert coords.shape == (np.prod(shape), len(shape),)
    else:
        assert coords.shape == shape + (len(shape),)


@pytest.mark.parametrize("shape, ks", [
    ((23,), 5),
    ((19, 23), 7),
    ((19, 23), (5, 7)),
    ((7, 9, 11), 3),
    ((7, 9, 11), (3, 4, 5)),
])
@pytest.mark.parametrize("bounds", util.LatticeBounds)
@pytest.mark.parametrize("compress", [False, True])
def test_lattice_predecessors(
        shape: tuple[int], ks: typing.Union[int, tuple[int]], bounds: util.LatticeBounds,
        compress: bool) -> None:
    predecessors = util.lattice_predecessors(shape, ks, bounds, compress)

    # General shape check.
    rows, cols = predecessors.shape
    assert rows == np.prod(shape)
    expected_cols = np.prod(2 * np.asarray(ks) * np.ones_like(shape) + 1)
    if compress:
        assert cols < expected_cols
    else:
        assert cols == expected_cols

    # Shape check for up to two dimensions if `k` is a scalar and the predecessors are compressed.
    if isinstance(ks, numbers.Number) and len(shape) < 3 and compress:
        expected_cols = util.num_lattice_predecessors(ks, bounds, len(shape))
        assert cols == expected_cols


@pytest.mark.parametrize("shape, ks, bounds, match", [
    ((5, 6), (3, 7), "cube", "exceeds the tensor size"),
    ((10, 6), 2, "invalid-shape", "'invalid-shape' is not a valid LatticeBounds"),
])
def test_lattice_predecessors_invalid(
        shape: tuple[int], ks: tuple[int], bounds: util.LatticeBounds, match: str) -> None:
    with pytest.raises(ValueError, match=re.escape(match)):
        util.lattice_predecessors(shape, ks, bounds=bounds)


@pytest.mark.parametrize("indexing", ["numpy", "stan"])
def test_predecessors_to_edge_index(indexing: typing.Literal["numpy", "stan"]) -> None:
    shape = (5, 7, 11)
    ks = (2, 3, 5)
    predecessors = util.lattice_predecessors(shape, ks)
    edge_index = util.predecessors_to_edge_index(predecessors, indexing)
    # The number of edges is the number of all combinations less the neighbors that fall outside the
    # bounds of the tensor. We just do a weak check here.
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] < predecessors.size
    # Check the validity of the edge index obtained from spatial predecessors.
    assert util.check_edge_index(edge_index, indexing=indexing) is edge_index


@pytest.mark.parametrize("predecessors, match", [
    (np.asarray([[1], [0]]), "first element in the predecessors must be the corresponding node"),
    (np.arange(3), "must be a matrix"),
])
def test_predecessors_to_edge_index_invalid(predecessors: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        util.predecessors_to_edge_index(predecessors)


@pytest.mark.parametrize("edge_index, indexing, match", [
    (np.zeros(2), None, "edge index must have shape (2, num_edges) but got (2,)"),
    (np.zeros((3, 2)), None, "edge index must have shape (2, num_edges) but got (3, 2)"),
    (np.zeros((2, 3)), "foobar", "`indexing` must be one of"),
    (np.zeros((2, 3)), "stan", "child node indices must be consecutive starting at 1"),
    (np.ones((2, 3)), "numpy", "child node indices must be consecutive starting at 0"),
    ([np.ones(2), np.zeros(2)], "numpy", "the first edge of each child must be a self loop"),
    ([[0, 2, 1], [0, 2, 1]], "numpy", "child node indices must be consecutive"),
    ([[0, 1, 1, 2, 2, 0], [0, 0, 1, 1, 2, 2]], "numpy", "edge index induces a graph with"),
])
def test_check_edge_index_invalid(
        edge_index: np.ndarray, indexing: typing.Literal["numpy", "stan"],
        match: typing.Optional[str]) -> None:
    with pytest.raises(ValueError, match=re.escape(match)):
        util.check_edge_index(np.asarray(edge_index), indexing)


def test_compress_predecessors_invalid_shape():
    with pytest.raises(ValueError):
        util.compress_predecessors(np.zeros(2))
