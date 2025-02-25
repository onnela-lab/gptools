# gptoolsStan [![gptools: Python](https://github.com/onnela-lab/gptools/actions/workflows/python.yml/badge.svg)](https://github.com/onnela-lab/gptools/actions/workflows/python.yml) [![](https://img.shields.io/pypi/v/gptools-stan)](https://pypi.org/project/gptools-stan)

*gptools* is a minimal package to publish Stan code for efficient Gaussian process inference. The package can be used with the [*cmdstanpy*](https://cmdstanpy.readthedocs.io/) interface for Stan in Python.

## Getting Started

1. Install *cmdstanpy* and *cmdstan* if you haven't already (see [here](https://cmdstanpy.readthedocs.io/en/v1.2.5/installation.html) for details).
2. Install *gptools* from [PyPI](https://pypi.org/project/gptools-stan/) by running `pip install gptools-stan` from the command line.
3. Compile your first model.

```python
from cmdstanpy import CmdStanModel
from gptools.stan import get_include

model = CmdStanModel(
  stan_file="path/to/your/model.stan",
  stanc_options={"include-paths": get_include()},
)
```

For an end-to-end example, see [this notebook](https://gptools-stan.readthedocs.io/docs/getting_started/getting_started.html).
