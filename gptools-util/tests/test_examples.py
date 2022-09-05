from gptools import util
from gptools.util import examples
import pytest


@pytest.mark.parametrize("notebook", examples.discover_examples(util))
def test_examples(notebook: str) -> None:
    examples.run_example(notebook)
