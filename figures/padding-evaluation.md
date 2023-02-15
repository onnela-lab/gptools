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
from gptools.util.kernels import ExpQuadKernel
from gptools.stan import compile_model
import logging
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pickle
from scipy.stats import gaussian_kde
from tqdm.notebook import tqdm

mpl.style.use("../jss.mplstyle")

# Disable cmdstan logging because we have a lot of fits.
cmdstanpy_logger = cmdstanpy.utils.get_logger()
for handler in cmdstanpy_logger.handlers:
    handler.setLevel(logging.WARNING)
```

```{code-cell} ipython3
# Define hyperparameters and generate synthetic datasets.
np.random.seed(0)
m = 100
n = 128
sigma = 1
length_scale = 16
kappa = 1
epsilon = 1e-5
padding_factors = [0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5]

# Evaluate the kernel and covariance.
x = np.arange(n)
kernel = ExpQuadKernel(sigma, length_scale)
cov = kernel.evaluate(x[:, None]) + epsilon * np.eye(n)

def sample(n, cov, kappa):
    """
    Draw `n` saomples from the generative model with covariance `cov` and observation noise `kappa`.
    """
    f = np.random.multivariate_normal(np.zeros(n), cov)
    y = np.random.normal(f, kappa)
    return f, y

# Generate synthetic datasets.
fs = []
ys = []
for _ in range(m):
    f, y = sample(n, cov, kappa)
    fs.append(f)
    ys.append(y)
    
fs, ys = np.asarray([fs, ys])

# Visualize one of the samples.
fig, ax = plt.subplots()
i = 0
ax.plot(x, fs[i], label="Gaussian process $f$")
ax.scatter(x, ys[i], label="target $y$", marker=".")
ax.legend()
ax.set_xlabel("covariate $x$")
fig.tight_layout()
```

```{code-cell} ipython3
# Fit the standard non-centered model to each dataset independently. We reuse the model from the profiling scripts.
root = Path("../stan/docs/padding")
standard_model = compile_model(stan_file=root / "exact.stan")
fourier_model = compile_model(stan_file=root / "padded.stan")


def get_fits(y):
    # Fit the model (we, dirtily, take all the variables from the outer scope).
    sample_kwargs = {
        "iter_warmup": 500, 
        "iter_sampling": 100, 
        "chains": 1,
        "show_progress": False,
        "seed": 0,
    }
    data = {
        "x": x,
        "num_observations": n,
        "observe_first": n - 1,
        "y": y,
        "length_scale": length_scale,
        "sigma": sigma,
        "kappa": kappa,
        "epsilon": epsilon,
    }
    fits = {}
    for factor in padding_factors:
        padding = factor * length_scale
        if padding != int(padding):
            raise ValueError("non-integer padding")
        fits[factor] = fourier_model.sample(data | {"padding": int(padding)}, **sample_kwargs)
    fits["std"] = standard_model.sample(data, **sample_kwargs)
    return fits


try:  # Cheap caching.
    with open("padding-cache.pkl", "rb") as fp:
        logpds_by_padding = pickle.load(fp)
except FileNotFoundError:
    logpds_by_padding = {}
    for f, y in tqdm(zip(fs, ys), total=m):
        fits = get_fits(y)
        # Let's evaluate the log posterior density for the held out point using a KDE.
        for key, fit in fits.items():
            # Pick n - 1 rather than - 1 because padding changes the length.
            kde = gaussian_kde(fit.f[:, n - 1])
            logpds_by_padding.setdefault(key, []).append(float(kde.logpdf(f[-1])))

    logpds_by_padding = {key: np.asarray(value) for key, value in logpds_by_padding.items()}
    with open("padding-cache.pkl", "wb") as fp:
        pickle.dump(logpds_by_padding, fp)
```

```{code-cell} ipython3
# Show an example. We sample with a seed to align legends around the plot and pick an example
# that excacerbates the effect of periodic boundary conditions.
np.random.seed(1)
f, y = sample(n, cov, kappa)
fits = get_fits(y)
```

```{code-cell} ipython3
fig = plt.figure()
gs = fig.add_gridspec(2, 2, width_ratios=[3, 2])

ax = axeval = fig.add_subplot(gs[:, 1])
# Show the posterior density evaluation.

l = logpds_by_padding["std"].mean()
s = logpds_by_padding["std"].std() / np.sqrt(m - 1)
line = ax.axhline(l, color="C0", label="standard")
ax.axhspan(l - s, l + s, color=line.get_color(), alpha=0.2)

vals = np.asarray([value for key, value in logpds_by_padding.items() if key != "std"]).T
logpds = vals.mean(axis=0)
ax.errorbar(padding_factors, logpds, vals.std(axis=0) / np.sqrt(m - 1), 
            label="padded Fourier", color="#666666", ls=":", marker="o", markeredgecolor="w")

ax.set_xlabel(r"padding factor $w/\ell$")
ax.set_ylabel(r"posterior density $p\left(f_n\mid y_{<n}\right)$")
ax.legend(loc="center right", fontsize="small")

# Show an example.
ax = ax1 = fig.add_subplot(gs[0, 0])
ax.plot(x, f, color="k", label="latent GP $f$")
pts = ax.scatter(x, y, marker=".", color="k", label="data $y$")
plt.setp(ax.xaxis.get_ticklabels(), visible=False)
ax.legend(loc="upper right", fontsize="small")

ax = ax2 = fig.add_subplot(gs[1, 0], sharex=ax, sharey=ax)
ax.plot(x, f, color="k")
keys = ["std", 0, 0.5, 1.0]
for i, key in enumerate(keys):
    color = f"C{i}"
    l = fits[key].f.mean(axis=0)
    label = "exact" if key == "std" else fr"$w/\ell={key:.2f}$"
    ax.plot(np.arange(l.size), l, color=color, label=label, alpha=0.7)
    if key != "std":
        axeval.scatter(key, logpds[padding_factors.index(key)], color=color, 
                       edgecolor="w", zorder=9)
ax.set_xlabel("covariate $x$")
ax.legend(fontsize="small", ncol=2, loc="upper center")

for ax in [ax1, ax2]:
    poly = ax.axvspan(n, 2 * n, facecolor="silver", alpha=0.2)
    poly.remove()
    ax.relim()
    ax.add_artist(poly)
    ax.text((n + ax.get_xlim()[1]) / 2, -1, "padding", ha="center", va="center", 
            rotation=90, color="gray")
ax.set_ylim(top=2)

ax1.text(0.05, 0.05, "(a)", transform=ax1.transAxes)
ax2.text(0.05, 0.05, "(b)", transform=ax2.transAxes)
axeval.text(0.95, 0.05, "(c)", transform=axeval.transAxes, ha="right")

fig.tight_layout()
fig.savefig("padding.pdf", bbox_inches="tight")
```
