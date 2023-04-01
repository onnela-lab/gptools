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

# Simulation study to study the effect of padding for Fourier-based Gaussian processes

```{code-cell} ipython3
import cmdstanpy
from gptools.util.kernels import ExpQuadKernel, MaternKernel
from gptools.stan import compile_model
import logging
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from scipy.stats import gaussian_kde, kstest
from tqdm.notebook import tqdm

workspace = Path(os.environ.get("WORKSPACE", os.getcwd()))
fast = "CI" in os.environ
mpl.style.use("../jss.mplstyle")

# Disable cmdstan logging because we have a lot of fits.
cmdstanpy_logger = cmdstanpy.utils.get_logger()
for handler in cmdstanpy_logger.handlers:
    handler.setLevel(logging.WARNING)
```

```{raw-cell}
# Convert this cell to a code cell to remove cached pickle files.
!rm -f *.pkl
```

```{code-cell} ipython3
# Define hyperparameters and generate synthetic datasets.
np.random.seed(0)
m = 10 if fast else 100
iter_sampling = 10 if fast else 100
n = 128
sigma = 1
length_scale = 16
kappa = 1
epsilon = 1e-5
padding_factors = [0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5]


# Declare the kernels.
x = np.arange(n)
kernels = {
    "ExpQuadKernel": ExpQuadKernel(sigma, length_scale),
    "Matern32Kernel": MaternKernel(1.5, sigma, length_scale),
}

# Generate synthetic datasets.
def sample(n, cov, kappa):
    """
    Draw `n` samples from the generative model with covariance `cov` and observation noise `kappa`.
    """
    f = np.random.multivariate_normal(np.zeros(n), cov)
    y = np.random.normal(f, kappa)
    return f, y

fs = {}
ys = {}
for key, kernel in kernels.items():
    cov = kernel.evaluate(x[:, None]) + epsilon * np.eye(n)
    for _ in range(m):
        f, y = sample(n, cov, kappa)
        fs.setdefault(key, []).append(f)
        ys.setdefault(key, []).append(y)

fs = {key: np.asarray(value) for key, value in fs.items()}
ys = {key: np.asarray(value) for key, value in ys.items()}

# Visualize one of the samples for each kernel.
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
i = 0
for ax, key in zip(axes, fs):
    ax.plot(x, fs[key][i], label="Gaussian process $f$")
    ax.scatter(x, ys[key][i], label="target $y$", marker=".")
    ax.set_xlabel("covariate $x$")
ax.legend()
fig.tight_layout()
```

```{code-cell} ipython3
# Fit the standard non-centered model to each dataset independently.
root = Path("../stan/docs/padding")
standard_model = compile_model(stan_file=root / "exact.stan")
fourier_model = compile_model(stan_file=root / "padded.stan")


def get_fit(model, y, kernel, padding=0):
    if isinstance(kernel, ExpQuadKernel):
        kernel = 0
    elif isinstance(kernel, MaternKernel):
        kernel = 1
    else:
        raise ValueError(kernel)

    if padding != int(padding):
        raise ValueError("non-integer padding")

    data = {
        "x": x,
        "num_observations": n,
        "observe_first": n - 1,
        "y": y,
        "length_scale": length_scale,
        "sigma": sigma,
        "kappa": kappa,
        "epsilon": epsilon,
        "padding": int(padding),
        "kernel": kernel,
    }
    return model.sample(data, iter_warmup=5 * iter_sampling, iter_sampling=iter_sampling, chains=1,
                        show_progress=False, seed=0)

statistics_by_kernel = {}
for key, kernel in kernels.items():
    # Naive caching for log pdfs and ranks.
    filename = workspace / f"padding-cache-{key}-{m}.pkl"
    try:
        with open(filename, "rb") as fp:
            statistics_by_kernel[key] = pickle.load(fp)
        print(f"loaded results from {filename}")
    except FileNotFoundError:
        for y, f in tqdm(zip(ys[key], fs[key]), desc=key, total=m):
            # Get the fits for different models.
            fits = {"exact": get_fit(standard_model, y, kernel)}
            for factor in padding_factors:
                fits[factor] = get_fit(fourier_model, y, kernel, length_scale * factor)

            # Compute the posterior density at the held out data point and its rank for each fit.
            statistics_by_kernel.setdefault(key, []).append({
                "lpds": {
                    key: gaussian_kde(fit.f[:, n - 1]).logpdf(f[n - 1]).squeeze()
                    for key, fit in fits.items()
                },
                "ranks": {
                    key: np.sum(fit.f[:, n - 1] < f[n - 1])
                    for key, fit in fits.items()
                },
            })
        # Save the results.
        with open(filename, "wb") as fp:
            pickle.dump(statistics_by_kernel[key], fp)
```

```{code-cell} ipython3
# Transpose the data for plotting. This isn't pretty but gets the job done.
transposed = {}

for key, statistics in statistics_by_kernel.items():
    result = {}
    for record in statistics:
        for stat, values in record.items():
            result.setdefault(stat, []).append(list(values.values()))
    transposed[key] = {key: np.asarray(value) for key, value in result.items()}
```

```{code-cell} ipython3
# Show an example. We sample with a seed to align legends around the plot and pick an example
# that excacerbates the effect of periodic boundary conditions.
np.random.seed(1)
kernel = kernels["ExpQuadKernel"]
f, y = sample(n, kernel.evaluate(x[:, None]), kappa)
fits = {"exact": get_fit(standard_model, y, kernel)}
for factor in padding_factors:
    fits[factor] = get_fit(fourier_model, y, kernel, length_scale * factor)
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, sharex="col")

# Show an example.
ax = axes[0, 0]
ax.plot(x, f, color="k", label="latent GP $f$")
pts = ax.scatter(x, y, marker=".", color="k", label="data $y$")
plt.setp(ax.xaxis.get_ticklabels(), visible=False)
ax.legend(loc="upper right", fontsize="small")

ax = axes[1, 0]
ax.plot(x, f, color="k")
keys = ["exact", 0, 0.5, 1.0]
for i, key in enumerate(keys):
    color = f"C{i}"
    l = fits[key].f.mean(axis=0)
    label = key if key == "exact" else fr"$w/\ell={key:.2f}$"
    ax.plot(np.arange(l.size), l, color=color, label=label, alpha=0.7)
ax.set_xlabel("covariate $x$")
ax.legend(fontsize="small", ncol=2, loc="upper center")

for ax in axes[:, 0]:
    poly = ax.axvspan(n, 2 * n, facecolor="silver", alpha=0.2)
    poly.remove()
    ax.relim()
    ax.add_artist(poly)
    ax.text((n + ax.get_xlim()[1]) / 2, -1, "padding", ha="center", va="center",
            rotation=90, color="gray")
    ax.set_ylim(-2.5, 2.0)

# Show the evaluation for the two types of kernels.
for ax, (key, statistics) in zip(axes[:, 1], transposed.items()):
    lpds = statistics["lpds"]
    l = lpds.mean(axis=0)
    s = lpds.std(axis=0) / np.sqrt(m - 1)
    line = ax.axhline(l[0], label="exact")
    ax.axhspan(l[0] - s[0], l[0] + s[0], alpha=0.2, color=line.get_color())
    ax.errorbar(padding_factors, l[1:], s[1:], marker="o", markeredgecolor="w",
                color="gray", ls=":", label="padded Fourier")

    # Add the markers visualized in the example.
    if key == "ExpQuadKernel":
        for i, key in enumerate(keys[1:], 1):
            ax.scatter(key, l[padding_factors.index(key) + 1], zorder=9,
                       color=f"C{i}", edgecolor="w")

axes[1, 1].set_xlabel(r"padding factor $w/\ell$")
axes[1, 1].set_ylabel(r"log posterior density $\log p\left(f_n\mid y_{<n}\right)$", y=1.15)
axes[0, 1].legend(loc="center right", fontsize="small")

axes[0, 0].text(0.05, 0.05, "(a)", transform=axes[0, 0].transAxes)
axes[0, 1].text(0.95, 0.05, "(b)", transform=axes[0, 1].transAxes, ha="right")
axes[1, 0].text(0.05, 0.05, "(c)", transform=axes[1, 0].transAxes)
axes[1, 1].text(0.95, 0.05, "(d)", transform=axes[1, 1].transAxes, ha="right")

fig.tight_layout()
fig.savefig(workspace / "padding.pdf", bbox_inches="tight")
fig.savefig(workspace / "padding.png", bbox_inches="tight")
```

```{code-cell} ipython3
# For simulation-based calibration, we compare the ranks of the true value within the posterior
# samples with a uniform distribution using the Kolmogorov Smirnov test and report the pvalue.
pd.DataFrame({
    "padding_factors": ["exact"] + padding_factors,
} | {
    key: [kstest(rank, "randint", (0, iter_sampling + 1)).pvalue for rank in stats["ranks"].T]
    for key, stats in transposed.items()
}).round(3)
```
