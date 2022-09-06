from gptools import stan
import pathlib
import pytest


@pytest.mark.parametrize("filename", ["gptools_graph.stan", "gptools_fft.stan"])
def test_include(filename: str) -> None:
    include = pathlib.Path(stan.get_include()) / filename
    assert include.is_file()
