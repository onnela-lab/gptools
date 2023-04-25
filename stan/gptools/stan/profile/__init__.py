import cmdstanpy
from gptools.stan import set_cmdstanpy_log_level
import numpy as np


SIZES = 16 * 2 ** np.arange(11)
FOURIER_ONLY_SIZE_THRESHOLD = 16 * 2 ** 9
LOG10_NOISE_SCALES = np.linspace(-1, 1, 7)
PARAMETERIZATIONS = ["graph_centered", "graph_non_centered", "fourier_centered",
                     "fourier_non_centered", "standard_centered", "standard_non_centered"]

# Make cmdstanpy less verbose.
set_cmdstanpy_log_level()


def sample_and_load_fit(model: cmdstanpy.CmdStanModel, **kwargs) -> cmdstanpy.CmdStanMCMC:  # noqa: E501, pragma: no cover
    """
    Wrapper function to sample and load the data so the posterior samples can be serialized.
    """
    fit = model.sample(**kwargs)
    fit.stan_variables()
    return fit
