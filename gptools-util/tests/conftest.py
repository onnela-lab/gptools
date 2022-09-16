import pytest
import torch as th

th.set_default_dtype(th.float64)


@pytest.fixture(params=[False, True], ids=["numpy", "torch"])
def use_torch(request: pytest.FixtureRequest) -> bool:
    return request.param


# Include fixtures that may be shared across different packages.
pytest_plugins = [
   "gptools.util.testing",
]
