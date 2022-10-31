import numpy as np
import pytest
import torch as th


# Raise errors for numerical issues.
np.seterr(divide="raise", over="raise", invalid="raise")
th.set_default_dtype(th.float64)


@pytest.fixture(params=[False, True], ids=["numpy", "torch"])
def use_torch(request: pytest.FixtureRequest) -> bool:
    return request.param


# Include fixtures that may be shared across different packages.
pytest_plugins = [
   "gptools.util.testing",
]
