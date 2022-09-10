from gptools.util import plotting
import numpy as np


def test_plot_band() -> None:
    x = np.linspace(0, 1, 21)
    ys = np.random.normal(0, 1, (43, 21))
    line, band = plotting.plot_band(x, ys)
