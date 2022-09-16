from gptools.stan.profile.__main__ import __main__
import pathlib
import pickle
import pytest
import tempfile
import typing


@pytest.mark.parametrize("parametrization", ["graph_centered", "graph_non_centered"])
def test_profile(parametrization: typing.Literal["centered", "non_centered"]) -> None:
    iter_sampling = 2
    chains = 4  # cmdstanpy default.
    num_nodes = 7
    num_parents = 2
    with tempfile.TemporaryDirectory() as tempdir:
        filename = pathlib.Path(tempdir, "result.pkl")
        __main__([
            parametrization, "1.0", str(filename), f"--iter_sampling={iter_sampling}",
            f"--num_nodes={num_nodes}", "--show_diagnostics", f"--num_parents={num_parents}",
        ])
        with filename.open("rb") as fp:
            result = pickle.load(fp)

    variables = result["fit"].stan_variables()
    assert variables["eta"].shape == (chains * iter_sampling, num_nodes)
