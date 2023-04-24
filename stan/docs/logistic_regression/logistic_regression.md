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

# Logistic regression

This notebook illustrates the use of the `gptools-stan` package by fitting a univariate latent Gaussian process to synthetic binary outcomes. It comprises three steps:

1. generating synthetic data from the model.
2. defining the model in Stan using the Fourier methods provided by the package.
3. fitting the model to the synthetic data and analyzing the results.

```{code-cell} ipython3
from matplotlib import pyplot as plt
import numpy as np
from scipy import special


num_observations = 100
sigma = 1
length_scale = 10
eps = 1e-6
seed = 0

np.random.seed(seed)
x = np.arange(num_observations)
cov = sigma ** 2 * np.exp(- (x[:, None] - x[None, :]) ** 2 / (2 * length_scale ** 2)) \
    + eps * np.eye(num_observations)
z = np.random.multivariate_normal(np.zeros_like(x), cov)
proba = special.expit(z)
y = np.random.binomial(1, proba)


def plot_synthetic_data(x, y, proba, ax=None):
    ax = ax or plt.gca()
    ax.plot(x, proba, label="latent probability\n$p=\\mathrm{expit}\\left(z\\right)$")
    ax.scatter(x, y, label="binary outcome $y$", marker=".")
    ax.set_xlabel("covariate $x$")

fig, ax = plt.subplots()
plot_synthetic_data(x, y, proba)
ax.legend()
fig.tight_layout()
```

```{literalinclude} logistic_regression.stan
   :language: stan
```

```{code-cell} ipython3
from gptools.stan import compile_model


model = compile_model(stan_file="logistic_regression.stan")
data = {
    "num_observations": num_observations,
    "padding": 56,
    "y": y,
    "eps": eps,
}
fit = model.sample(data, seed=seed, show_progress=False, chains=1)
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 2])

plot_synthetic_data(x, y, proba, ax1)
center = np.mean(fit.proba, axis=0)
std = np.std(fit.proba, axis=0)
lower = center - std
upper = center + std
line, = ax1.plot(x, center, label="inferred latent\nprobability")
ax1.fill_between(x, lower, upper, color=line.get_color(), alpha=0.2)
ax1.legend(fontsize="small")

ax2.scatter(fit.length_scale, fit.sigma, marker=".", alpha=0.2)
ax2.scatter(length_scale, sigma, marker="X", color="k", edgecolor="w")
ax2.set_xlabel(r"length scale $\ell$")
ax2.set_ylabel(r"marginal scale $\sigma$")

fig.tight_layout()
```
