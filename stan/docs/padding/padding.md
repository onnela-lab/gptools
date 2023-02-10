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

# Effect and importance of padding

Contrary to most real-world scenarios, the fast Fourier transform and, consequently, {ref}`Fourier-methods` for Gaussian processes assume periodic boundary conditions. This peculiarity can be overcome by padding the domain. This example explores the effect and importance of padding for a simple Gaussian process model. We will draw a sample from a Gaussian process *without* periodic boundary conditions, $n$ observations, and squared exponential kernel with correlation length $\ell$. We will consider how well we can infer it using Fourier methods as a function of the amount of padding $q$ we use. Padding may, in principle, be introduced anywhere because of the periodic boundary conditions. However, this library has only been tested with padding on the right, as illustrated in the example below.

Before fitting GPs, let's visualize the kernel on finite domains with different paddings, i.e., we plot the correlation between a point at the origin and another position $x$.

```{code-cell} ipython3
from gptools.stan import compile_model
from gptools.util.kernels import ExpQuadKernel
from matplotlib import pyplot as plt
import numpy as np


# Declare the number of observations, correlation lengths, and the paddings we want to consider.
num_observations = 128
length_scale = 16
padding_factors = np.arange(5)

# Then show the kernels with periodic boundary conditions.
fig, ax = plt.subplots()
for factor in padding_factors:
    x = np.arange(num_observations + factor * length_scale)
    kernel = ExpQuadKernel(1, length_scale, period=x.size)
    ax.plot(x, kernel.evaluate(0, x[:, None]), label=fr"padding $q={factor}\times\ell$")
ax.axvline(num_observations, color="k", label="observation domain $n$", ls=":")
ax.legend(loc=(0.175, 0.625))
ax.set_xlabel("location $x$")
ax.set_ylabel("kernel $k(0, x)$")
fig.tight_layout()
```

The correlation between opposite sides of the domain is *very* strong without padding, appreciable for padding with one or two correlation lengths $\ell$, and becomes small when we pad with three or more correlation lengths.

We will draw a Gaussian process sample $f$ without periodic boundary conditions and use a normal observation model $y\sim\mathsf{Normal}\left(y,\kappa^2\right)$, where $\kappa^2$ is the observation variance. We fit the model using two methods: First, the exact but slow model without periodic boundary conditions. Second, a model using Fourier methods with different amounts of padding. We fit all models using the {meth}`cmdstanpy.CmdStanModel.optimize` method to avoid discrepancies between different fits that arise purely by chance as a result of drawing posterior samples.

```{code-cell} ipython3
# Pick a seed that generates a sample with different values at opposite sides of the domain.
np.random.seed(1)

# Generate a Gaussian process sample.
x = np.arange(num_observations)
kernel = ExpQuadKernel(1, length_scale)
cov = kernel.evaluate(x[:, None])
f = np.random.multivariate_normal(np.zeros_like(x), cov)

# Generate noisy observations.
kappa = 0.5
y = np.random.normal(f, kappa)

fig, ax = plt.subplots()
ax.plot(x, f, label=r"$f\sim\mathsf{GP}$", color="gray")
ax.scatter(x, y, marker=".", color="gray", label=r"$y\sim\mathsf{Normal}\left(f,\kappa^2\right)$")
ax.legend()
ax.set_xlabel("location $x$")
fig.tight_layout()
```

The model for the first method is as follows.

```{literalinclude} exact.stan
   :language: stan
```

```{code-cell} ipython3
exact_model = compile_model(stan_file="exact.stan")
data = {
    "num_observations": num_observations,
    "x": x,
    "y": y,
    "kappa": kappa,
    "length_scale": length_scale,
    "epsilon": 1e-9,
}
exact_fit = exact_model.optimize(data)

fig, ax = plt.subplots()
ax.plot(x, f, label="ground truth", color="gray")
ax.plot(x, exact_fit.f, label="MAP (exact)", color="k", ls="--")
ax.set_xlabel("location $x$")
ax.set_ylabel("Gaussian process $f$")
ax.legend()
fig.tight_layout()
```

The model for the second method is as follows.

```{literalinclude} padded.stan
   :language: stan
```

```{code-cell} ipython3
padded_model = compile_model(stan_file="padded.stan")
padded_fits = {factor: padded_model.optimize(data | {"padding": length_scale * factor}) for factor
               in padding_factors}

fig, ax = plt.subplots()
ax.plot(x, f, label="ground truth", color="gray", zorder=9)
ax.plot(x, exact_fit.f, label="MAP (exact)", color="k", ls="--", zorder=9)
for factor, fit in padded_fits.items():
    f_estimate = fit.f[:num_observations]
    ax.plot(x, f_estimate, label=fr"MAP ($q={factor}\times\ell)$")
    rmse = np.sqrt(np.square(exact_fit.f - f_estimate).mean())
    print(f"RMSE from exact MAP with padding q = {factor} * length_scale: {rmse:.5f}")
ax.set_xlabel("location $x$")
ax.set_ylabel("Gaussian process $f$")
ax.legend()
fig.tight_layout()
```

Ignoring the need to pad substantially affects the posterior inference, but we recover the exact MAP estimate well even if we pad with only two correlation lengths. In practice, we often don't know the correlation length ahead of time, and finding the "right" amount of padding that appropriately balances performance and the need for non-periodic boundary conditions may be an iterative process. For example, we can start with a small amount of padding and increase it until the posterior inference no longer changes.
