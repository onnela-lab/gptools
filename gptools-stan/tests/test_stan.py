import pathlib
from gptools import stan


def test_include() -> None:
    include = pathlib.Path(stan.get_include()) / "gptools.stan"
    assert include.is_file()
