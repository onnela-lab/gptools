from gptools import stan
from gptools.util import examples
import os
import pytest
from unittest import mock


@pytest.mark.parametrize("notebook", examples.discover_examples(stan))
def test_examples(notebook: str) -> None:
    env = {"STAN_ITER_WARMUP": "1", "STAN_ITER_SAMPLING": "1", "STAN_COMPILE": "force"}
    with mock.patch.dict(os.environ, env):
        examples.run_example(notebook)
