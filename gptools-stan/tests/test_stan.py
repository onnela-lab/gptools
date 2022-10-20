from gptools import stan
import pathlib
import pytest


@pytest.mark.parametrize("filename", ["gptools_graph.stan", "gptools_fft.stan",
                                      "gptools_kernels.stan"])
def test_include(filename: str) -> None:
    include = pathlib.Path(stan.get_include()) / filename
    assert include.is_file()


def test_needs_compilation(tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture):
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
    model = stan.compile_model(stan_file=stan_file)
    exe_file = pathlib.Path(model.exe_file)
    assert exe_file.exists()
    last_compiled = exe_file.stat().st_mtime

    # Check the model is not recompiled.
    stan.compile_model(stan_file=stan_file)
    assert exe_file.stat().st_mtime == last_compiled

    # Touch the included file and check we compile anew.
    include_file.touch()
    stan.compile_model(stan_file=stan_file)
    assert exe_file.stat().st_mtime > last_compiled

    # Remove the include file and check we get an error.
    include_file.unlink()
    with caplog.at_level("DEBUG"):
        stan.compile_model(stan_file=stan_file)
    assert any("--info" in record.message and "returned non-zero exit status 1" for record in
               caplog.records)
