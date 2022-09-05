import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import pytest
from unittest import mock


filenames = glob.glob("gptools/*/examples/*/*.ipynb") + glob.glob("gptools/*/examples/*.ipynb")


@pytest.mark.parametrize("filename", filenames)
def test_example(filename: str) -> None:
    _, module, *_ = filename.split(os.path.sep)
    pytest.importorskip("cmdstanpy" if module == "stan" else module)
    with open(filename) as fp:
        notebook = nbformat.read(fp, as_version=4)
    preprocessor = ExecutePreprocessor(timeout=60)
    env = {
        "STAN_ITER_WARMUP": "1", "STAN_ITER_SAMPLING": "1", "STAN_COMPILE": "force",
        "MAX_NUM_STEPS": "1",
    }
    with mock.patch.dict(os.environ, env):
        preprocessor.preprocess(notebook, {"metadata": {"path": os.path.dirname(filename)}})
