import pathlib
from graph_gaussian_process import stan


def test_include() -> None:
    include = pathlib.Path(stan.get_include()) / "graph_gaussian_process.stan"
    assert include.is_file()
