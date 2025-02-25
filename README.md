# *gptools*: Performant Gaussian Processes in [Stan](https://mc-stan.org) [![gptools-stan: Python](https://github.com/onnela-lab/gptools/actions/workflows/python.yml/badge.svg)](https://github.com/onnela-lab/gptools/actions/workflows/python.yml) [![gptools-stan: R](https://github.com/onnela-lab/gptools/actions/workflows/R.yml/badge.svg)](https://github.com/onnela-lab/gptools/actions/workflows/R.yml)

Gaussian processes (GPs) are powerful distributions for modeling functional data, but using them is computationally challenging except for small datasets. *gptools* implements two methods for performant GP inference in Stan.

1. A sparse approximation of the likelihood. This approach includes nearest neighbor Gaussian processes but also supports more general dependence structures, e.g., for periodic kernels.
2. An exact likelihood evaluation for data on regularly spaced lattices using fast Fourier transforms.

The implementation follows Stan’s design and exposes performant inference through a familiar interface. We provide interfaces in Python and R. See the accompanying publication [*Scalable Gaussian Process Inference with Stan*](https://doi.org/10.48550/arXiv.2301.08836) for details of the implementation. The comprehensive [documentation](http://gptools-stan.readthedocs.io/) includes many [examples](https://gptools-stan.readthedocs.io/docs/examples.html).

## Getting Started

You can use the *gptools* package by including it the `functions` block of your Stan program, e.g.,

```stan
functions {
    // Include utility functions, such as real fast Fourier transforms.
    #include gptools/util.stan
    // Include functions to evaluate GP likelihoods with Fourier methods.
    #include gptools/fft.stan
}
```

See [here](stan/gptools) for the list of all include options. You can download the [Stan files](stan) and use them with your favorite interface or use the provided Python and R interfaces.

### Python and *cmdstanpy*

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

### R and *cmdstanr*

1. Install *cmdstanr* and *cmdstan* if you haven't already (see [here](https://mc-stan.org/cmdstanr/#installation) for details).
2. Install *gptools* from [CRAN](https://cran.r-project.org/package=gptoolsStan) by running `install.packages("gptoolsStan")`.
3. Compile your first model.

```r
library(cmdstanr)
library(gptoolsStan)

model <- cmdstan_model(
  stan_file="path/to/your/model.stan",
  include_paths=gptools_include_path(),
)
```

> [!NOTE]
> Unfortunately, [*Rstan*](https://cran.r-project.org/package=rstan) is not supported because it [does not provide an option to specify include paths](https://discourse.mc-stan.org/t/specifying-include-paths-in-rstan/32182/2).

For an end-to-end example, see [this vignette](https://cran.r-project.org/web/packages/gptoolsStan/vignettes/getting_started.html).

## Contributing

Contributions to the package are always welcome! The repository structure is as follows:

- [`stan/gptools`](stan/gptools) contains the core implementation of *gptools*.
- [`python/stan`](python/stan) contains the Python wrapper for *gptools*.
- [`R`](R) contains the R wrapper for *gptools*.
- [`docs`](docs) contains the documentation hosted at https://gptools-stan.readthedocs.io.

## Replicating the Results

To replicate the results presented in the accompanying publication [*Scalable Gaussian Process Inference with Stan*](https://doi.org/10.48550/arXiv.2301.08836), please see the [dedicated repository of replication materials](https://github.com/onnela-lab/gptools-reproduction-material).
