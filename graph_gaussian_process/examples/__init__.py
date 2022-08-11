import os


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
