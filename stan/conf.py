import cmdstanpy
from gptools.util.conf import *  # noqa: F401, F403
from gptools.util.conf import extensions, intersphinx_mapping
import logging

project = "gptools-stan"
extensions.append("sphinxcontrib.stan")
intersphinx_mapping["cmdstanpy"] = \
    (f"https://cmdstanpy.readthedocs.io/en/v{cmdstanpy.__version__}", None)

cmdstanpy_logger = cmdstanpy.utils.get_logger()
for handler in cmdstanpy_logger.handlers:
    handler.setLevel(logging.WARNING)
