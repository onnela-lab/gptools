from myst_parser.config.main import MdParserConfig
from myst_nb.core.config import NbParserConfig
from myst_nb.core.read import create_nb_reader, NbReader
from nbconvert.preprocessors import ExecutePreprocessor
import os
import pathlib
import pytest
from typing import Any


def run_myst_notebook(path: str) -> Any:
    """
    Run a myst example notebook.
    """
    timeout = 60 if "CI" in os.environ else None
    md_config = MdParserConfig()
    nb_config = NbParserConfig()
    with open(path) as fp:
        content = fp.read()
    reader: NbReader = create_nb_reader(path, md_config, nb_config, content)
    notebook = reader.read(content)
    preprocessor = ExecutePreprocessor(timeout=timeout)
    return preprocessor.preprocess(notebook, {"metadata": {"path": pathlib.Path(path).parent}})


notebooks = []
for package in ["stan", "torch", "util"]:
    notebooks.extend(str(path) for path in pathlib.Path(f"gptools-{package}/docs").glob("**/*.md")
                     if ".ipynb_checkpoints" not in path.parts)


@pytest.mark.parametrize("notebook", notebooks)
def test_notebooks(notebook: str) -> None:
    run_myst_notebook(notebook)
