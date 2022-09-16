from gptools import stan
import pathlib
import pytest
from unittest import mock


@pytest.mark.parametrize("filename", ["gptools_graph.stan", "gptools_fft.stan",
                                      "gptools_kernels.stan"])
def test_include(filename: str) -> None:
    include = pathlib.Path(stan.get_include()) / filename
    assert include.is_file()


def test_needs_compilation(tmp_path: pathlib.Path):
    # Create files.
    include_file = tmp_path / "include.stan"
    with include_file.open("w") as fp:
        fp.write("real x;")
    stan_file = tmp_path / "model.stan"
    with stan_file.open("w") as fp:
        fp.write("""parameters {
            #include include.stan
        }
        model { x ~ normal(0, 1); }
        """)

    # Check we compile as usual first.
    with mock.patch("cmdstanpy.CmdStanModel") as model:
        stan.compile_model(stan_file=stan_file)
    model.assert_called_once()
    args, _ = model.call_args
    assert args[3] is True

    # Actually build the model.
    stan.compile_model(stan_file=stan_file)

    # Check we still compile as usual.
    with mock.patch("cmdstanpy.CmdStanModel") as model:
        stan.compile_model(stan_file=stan_file)
    model.assert_called_once()
    args, _ = model.call_args
    assert args[3] is True

    # Touch the included file and check we compile with compile="force".
    include_file.touch()
    with mock.patch("cmdstanpy.CmdStanModel") as model:
        stan.compile_model(stan_file=stan_file)
    model.assert_called_once()
    args, _ = model.call_args
    assert args[3] == "force"

    # Remove the include file and check we get an error.
    include_file.unlink()
    with pytest.raises(FileNotFoundError, match="include.stan"):
        stan.compile_model(stan_file=stan_file)
