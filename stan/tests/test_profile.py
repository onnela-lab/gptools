from doit_interface import dict2args
from gptools.stan.profile import PARAMETERIZATIONS
from gptools.stan.profile.__main__ import __main__
import numpy as np
import pathlib
import pickle
import pytest


def _run_main(tmp_path: pathlib.Path, method: str, parameterization: str, noise_scale: float = 1.0,
              **kwargs) -> tuple[dict, dict]:
    """
    Run the main process.
    """
    kwargs = {"iter_sampling": 3, "n": 7, "num_parents": 2} | kwargs
    path = tmp_path / "result.pkl"
    __main__([method, parameterization, str(noise_scale), str(path), "--show_diagnostics",
              "--ignore_converged", *dict2args(**kwargs)])
    with path.open("rb") as fp:
        return pickle.load(fp), kwargs


@pytest.mark.parametrize("method", ["sample", "variational"])
@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_profile(method: str, parameterization: str, tmp_path: pathlib.Path) -> None:
    result, kwargs = _run_main(tmp_path, method, parameterization)

    if method == "sample":
        variables = result["fits"][0].stan_variables()["eta"]
    elif method == "variational":
        variables = result["fits"][0].variational_sample.values[:, 3:]
        if "non_centered" in parameterization:
            _, variables = np.split(variables, 2, axis=1)
    else:
        raise ValueError(method)
    assert variables.shape == (kwargs["iter_sampling"], kwargs["n"])


def test_profile_timeout(tmp_path: pathlib.Path) -> None:
    result, kwargs = _run_main(tmp_path, "sample", "standard_centered", iter_sampling=100_000,
                               timeout=0.7)

    assert result["fits"][0] is None
    assert abs(result["durations"] - kwargs["timeout"]) < 0.1


def test_profile_max_chains(tmp_path: pathlib.Path) -> None:
    result, kwargs = _run_main(tmp_path, "sample", "standard_centered", max_chains=7)
    assert len(result["fits"]) == kwargs["max_chains"]
