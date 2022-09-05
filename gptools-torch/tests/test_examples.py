from gptools import torch
from gptools.util import examples
import os
import pytest
from unittest import mock


@pytest.mark.parametrize("notebook", examples.discover_examples(torch))
def test_examples(notebook: str) -> None:
    env = {"MAX_NUM_STEPS": "1"}
    with mock.patch.dict(os.environ, env):
        examples.run_example(notebook)
