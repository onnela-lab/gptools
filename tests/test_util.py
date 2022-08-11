from graph_gaussian_process import util
import numbers
import numpy as np
import pathlib
import pytest
import re
from scipy.spatial.distance import cdist
import typing


def test_include() -> None:
    include = pathlib.Path(util.get_include()) / "graph_gaussian_process.stan"
    assert include.is_file()


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
@pytest.mark.parametrize("ravel", [False, True])
def test_spatial_neighborhoods(shape: tuple[int], ks: typing.Union[int, tuple[int]], ravel: bool) \
        -> None:
    neighborhoods = util.spatial_neighborhoods(shape, ks, ravel)
    if isinstance(ks, numbers.Integral):
        ks = (ks,) * len(shape)
    if ravel:
        neighborhoods.shape == (np.prod(shape), np.prod(ks))
    else:
        neighborhoods.shape == shape + ks


@pytest.mark.parametrize("shape, ks, match", [
    ((5, 6), (3,), "must have matching length"),
    ((5, 6), (3, 7), "is larger than the tensor size"),
])
def test_spatial_neighborhoods_invalid(shape: tuple[int], ks: tuple[int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        util.spatial_neighborhoods(shape, ks)


def test_neighborhood_to_edge_index() -> None:
    shape = (5, 7, 11)
    ks = (2, 3, 5)
    neighborhoods = util.spatial_neighborhoods(shape, ks)
    edge_index = util.neighborhood_to_edge_index(neighborhoods)
    # The number of edges is the number of all combinations less the neighbors that fall outside the
    # bounds of the tensor. We just do a weak check here.
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] < neighborhoods.size
    # Check the validity of the edge index obtained from spatial neighborhoods.
    assert util.check_edge_index(edge_index, indexing="numpy") is edge_index


@pytest.mark.parametrize("neighborhoods, match", [
    (np.asarray([[1], [0]]), "first element in the neighborhood must be the corresponding node"),
    (np.arange(3), "must be a matrix"),
])
def test_neighborhood_to_edge_index_invalid(neighborhoods: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        util.neighborhood_to_edge_index(neighborhoods)


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
