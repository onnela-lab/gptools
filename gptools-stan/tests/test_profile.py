from doit_interface import dict2args
from gptools.stan.profile import PARAMETERIZATIONS
from gptools.stan.profile.__main__ import __main__
import pathlib
import pickle
import pytest


def _run_main(tmp_path: pathlib.Path, parameterization: str, noise_scale: float = 1.0, **kwargs):
    """
    Run the main process.
    """
    kwargs = {"iter_sampling": 2, "n": 7, "num_parents": 2} | kwargs
    path = tmp_path / "result.pkl"
    __main__([parameterization, str(noise_scale), str(path), "--show_diagnostics",
              *dict2args(**kwargs)])
    with path.open("rb") as fp:
        return pickle.load(fp), kwargs


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_profile(parameterization: str, tmp_path: pathlib.Path) -> None:
    result, kwargs = _run_main(tmp_path, parameterization)

    variables = result["fits"][0].stan_variables()
    assert variables["eta"].shape == (kwargs["iter_sampling"], kwargs["n"])


def test_profile_timeout(tmp_path: pathlib.Path) -> None:
    result, kwargs = _run_main(tmp_path, "standard_centered", iter_sampling=100_000, timeout=0.7)

    assert result["fits"][0] is None
    assert abs(result["durations"] - kwargs["timeout"]) < 0.1


def test_profile_max_chains(tmp_path: pathlib.Path) -> None:
    result, kwargs = _run_main(tmp_path, "standard_centered", max_chains=7)
    assert len(result["fits"]) == kwargs["max_chains"]
