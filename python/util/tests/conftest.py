import numpy as np


# Raise errors for numerical issues.
np.seterr(divide="raise", over="raise", invalid="raise")


# Include fixtures that may be shared across different packages.
pytest_plugins = [
    "gptools.util.testing",
]
