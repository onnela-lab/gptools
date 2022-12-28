import cmdstanpy
from gptools.util.conf import *  # noqa: F401, F403
from gptools.util.conf import extensions
import logging

project = "gptools-stan"
extensions.append("sphinxcontrib.stan")

cmdstanpy_logger = cmdstanpy.utils.get_logger()
for handler in cmdstanpy_logger.handlers:
    handler.setLevel(logging.WARNING)
