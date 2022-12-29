---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Getting started

```{code-cell} ipython3
import cmdstanpy
from gptools.util.kernels import ExpQuadKernel
from gptools.stan import compile_model
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os

mpl.style.use("../../../jss.mplstyle")
```

```{code-cell} ipython3
n = 100
x = np.arange(n)
data = {
    "n": n,
    "cov_rfft": ExpQuadKernel(1, n / 10, n).evaluate_rfft([n]) + 1e-9,
}
niter = 5 if "CI" in os.environ else 500
```

```{code-cell} ipython3
def fit_and_plot(stan_file: str) -> cmdstanpy.CmdStanModel:
    # Draw samples from the distribution.
    model = compile_model(stan_file=stan_file)
    fit = model.sample(data, chains=1, iter_warmup=niter, iter_sampling=niter)

    # Plot realizations.
    fig, ax = plt.subplots()
    idx = np.random.choice(fit.num_draws_sampling, min(10, fit.num_draws_sampling), replace=False)
    ax.plot(x, fit.f[idx].T,
            color="C0", alpha=0.25)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f$")
    ax.set_title(stan_file)
    fig.tight_layout()
    
fit_and_plot("getting_started.stan")
```

```{code-cell} ipython3
fit_and_plot("getting_started_non_centered.stan")
```
