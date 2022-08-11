import cmdstanpy
import os
import pathlib
import typing
from ..util import get_include


def sample_kwargs_from_env(**kwargs):
    """
    Extract keyword arguments for :meth:`cmdstanpy.CmdStanModel.sample` from environment variables.
    """
    kwargs.setdefault("chains", int(os.environ.get("STAN_CHAINS", 1)))
    kwargs.setdefault("iter_sampling", int(os.environ.get("STAN_ITER_SAMPLING", 500)))
    kwargs.setdefault("iter_warmup", int(os.environ.get("STAN_ITER_WARMUP", 500)))
    kwargs.setdefault("show_progress", int(os.environ.get("STAN_SHOW_PROGRESS", 0)))
    kwargs.setdefault("seed", int(os.environ.get("STAN_SEED", 42)))
    return kwargs


def compile(stan_file: pathlib.Path, stanc_options: typing.Optional[dict] = None, **kwargs) \
        -> cmdstanpy.CmdStanModel:
    """
    Compile a :class:`cmstanpy.CmdStanModel` model for examples.

    Args:
        stan_file: Path to Stan program file.
        stanc_options: Options for stanc compiler. The graph Gaussian process include path is
            automatically added.
        **kwargs: Keyword arguments passed to :class:`cmstanpy.CmdStanModel`.

    Returns:
        model: Compiled :class:`cmstanpy.CmdStanModel` model.
    """
    stanc_options = stanc_options or {}
    stanc_options.setdefault("include-paths", []).append(get_include())
    # Compile the model.
    return cmdstanpy.CmdStanModel(stan_file=stan_file, stanc_options=stanc_options, **kwargs)
