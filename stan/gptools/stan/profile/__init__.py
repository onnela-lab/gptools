import cmdstanpy
import logging
import numpy as np


SIZES = 16 * 2 ** np.arange(9)
LOG10_NOISE_SCALES = np.linspace(-1, 1, 7)
PARAMETERIZATIONS = ["graph_centered", "graph_non_centered", "fourier_centered",
                     "fourier_non_centered", "standard_centered", "standard_non_centered"]

# Make cmdstanpy less verbose.
cmdstanpy_logger = cmdstanpy.utils.get_logger()
for handler in cmdstanpy_logger.handlers:
    handler.setLevel(logging.WARNING)


def sample_and_load_fit(model: cmdstanpy.CmdStanModel, **kwargs) -> cmdstanpy.CmdStanMCMC:  # noqa: E501, pragma: no cover
    """
    Wrapper function to sample and load the data so the posterior samples can be serialized.
    """
    fit = model.sample(**kwargs)
    fit.stan_variables()
    return fit
