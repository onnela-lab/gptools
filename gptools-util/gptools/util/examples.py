from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import pathlib
import typing


def discover_examples(package) -> typing.Iterable[str]:
    """
    Discover example notebooks within a package.
    """
    return [
        str(path.relative_to(pathlib.Path.cwd())) for path in
        pathlib.Path(package.__file__).parent.glob("**/*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    ]


def run_example(path: pathlib.Path, timeout: float = 60) -> typing.Any:
    """
    Execute an example notebook.
    """
    path = pathlib.Path(path)
    with open(path) as fp:
        notebook = nbformat.read(fp, as_version=4)
    preprocessor = ExecutePreprocessor(timeout=timeout)
    return preprocessor.preprocess(notebook, {"metadata": {"path": path.parent}})
