# gptoolsStan [![gptoolsStan R package](https://github.com/onnela-lab/gptoolsStan/actions/workflows/main.yml/badge.svg)](https://github.com/onnela-lab/gptoolsStan/actions/workflows/main.yml) [![CRAN/METACRAN Version](https://img.shields.io/cran/v/gptoolsStan)](https://cran.r-project.org/package=gptoolsStan)

`gptoolsStan` is a minimal package to publish Stan code for efficient Gaussian process inference. The package can be used with the [`cmdstanr`](https://mc-stan.org/cmdstanr/) interface for Stan in R.

## Getting Started

1. Install `cmdstanr` if you haven't already (see [here](https://mc-stan.org/cmdstanr/#installation) for details).
2. Install this package by running `install.packages("gptoolsStan")`.
3. Compile your first model.
```r
library(cmdstanr)
library(gptoolsStan)

model <- cmdstan_model(
  stan_file="path/to/your/model.stan",
  include_paths=gptools_include_path(),
)
```

For an end-to-end example, see [this vignette](vignettes/getting_started.Rmd). More comprehensive [documentation](http://gptools-stan.readthedocs.io/), including many examples, is available although using the [`cmdstanpy`](https://mc-stan.org/cmdstanpy/) interface for Python.
