import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import pytest
from unittest import mock


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("filename", glob.glob("graph_gaussian_process/examples/*/*.ipynb"))
def test_example(filename: str) -> None:
    with open(filename) as fp:
        notebook = nbformat.read(fp, as_version=4)
    preprocessor = ExecutePreprocessor(timeout=60)
    with mock.patch.dict(os.environ, {"STAN_ITER_WARMUP": "1", "STAN_ITER_SAMPLING": "1"}):
        preprocessor.preprocess(notebook, {"metadata": {"path": os.path.dirname(filename)}})
