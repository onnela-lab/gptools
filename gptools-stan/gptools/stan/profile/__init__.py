import numpy as np


SIZES = 16 * 2 ** np.arange(9)
LOG_NOISE_SCALES = 3 * np.linspace(-1, 1, 7)
PARAMETERIZATIONS = ["graph_centered", "graph_non_centered", "fourier_centered",
                     "fourier_non_centered", "standard_centered", "standard_non_centered"]
