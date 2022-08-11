from graph_gaussian_process import util
import numpy as np
import pathlib
import pytest
from scipy.spatial.distance import cdist


def test_include() -> None:
    include = pathlib.Path(util.get_include()) / "graph_gaussian_process.stan"
    assert include.is_file()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("p", [1, 3, 5])
def test_evaluate_squared_distance(p: int) -> None:
    X = np.random.normal(0, 1, (57, p))
    np.testing.assert_allclose(util.evaluate_squared_distance(X), cdist(X, X) ** 2)


def test_plot_band():
    x = np.linspace(0, 1, 21)
    ys = np.random.normal(0, 1, (43, 21))
    line, band = util.plot_band(x, ys)
