import cmdstanpy
import numpy as np


SIZES = 16 * 2 ** np.arange(9)
LOG_NOISE_SCALES = 3 * np.linspace(-1, 1, 7)
PARAMETERIZATIONS = ["graph_centered", "graph_non_centered", "fourier_centered",
                     "fourier_non_centered", "standard_centered", "standard_non_centered"]


def sample_and_load_fit(model: cmdstanpy.CmdStanModel, **kwargs) -> cmdstanpy.CmdStanMCMC:  \
        # pragma: no cover
    """
    Wrapper function to sample and load the data so the posterior samples can be serialized.
    """
    fit = model.sample(**kwargs)
    fit.stan_variables()
    return fit
