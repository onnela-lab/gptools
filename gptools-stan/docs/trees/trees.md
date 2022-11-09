---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
from gptools import util
from gptools.stan import compile_model
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import ndimage, stats
from scipy.fft import next_fast_len

mpl.style.use("../../../jss.mplstyle")
```

```{code-cell} ipython3
species = "tachve"
frequency = np.loadtxt(f"../../../data/{species}.csv", delimiter=",")
nrows, ncols = frequency.shape
# Twenty meters per quadrant but we want to measure in kilometers.
delta = 20 / 1000
extent = delta * (np.asarray([0, ncols, 0, nrows]) - 0.5)

fig, ax = plt.subplots()
im = ax.imshow(frequency, extent=extent)
ax.set_xlabel("easting (km)")
ax.set_ylabel("northing (km)")
fig.colorbar(im, ax=ax, location="top", label="tree density")
fig.tight_layout()
```

```{code-cell} ipython3
# Set up the shapes.
padding = 10
padded_rows = next_fast_len(nrows + padding)
padded_cols = next_fast_len(ncols + padding)

# Apply a random mask.
seed = 0
np.random.seed(seed)
test_fraction = 0.2
train_mask = np.random.uniform(size=frequency.shape) < (1 - test_fraction)

# Prepare the data for stan.
data = {
    "n": nrows,
    "np": padded_rows,
    "m": ncols,
    "mp": padded_cols,
    "frequency": np.where(train_mask, frequency, -1).astype(int),
    "epsilon": 1e-6,
}
```

```{code-cell} ipython3
# Compile and fit the model.
model = compile_model(stan_file="trees.stan")
niter = 3 if "CI" in os.environ else 500
fit = model.sample(data, chains=1, iter_warmup=niter, iter_sampling=niter, seed=seed)
print(fit.diagnose())
```

```{code-cell} ipython3
def evaluate_error(actual, prediction, error, num_bs=1000):
    """
    Evaluate bootstrapped errors between the actual data and model predictions.
    """
    idx = np.random.choice(actual.size, (num_bs, actual.size))
    bs_actual = actual[idx]
    bs_prediction = prediction[idx]
    if error == "mse":
        return np.square(bs_actual - bs_prediction).mean(axis=-1)
    elif error == "rmse":
        return np.square(bs_actual - bs_prediction).mean(axis=-1) ** 0.5
    elif error == "scaled_mse":
        return (np.square(bs_actual - bs_prediction) / np.maximum(bs_actual, 1)).mean(axis=-1)
    else:
        raise ValueError(error)


# Evaluate predictions and errors using Gaussian filtering.
error = "scaled_mse"
test_mask = ~train_mask
smoothed_errors = []
sigmas = np.logspace(-0.8, 1)
for sigma in sigmas:
    smoothed_mask = ndimage.gaussian_filter(train_mask.astype(float), sigma)
    smoothed_masked_frequency = ndimage.gaussian_filter(np.where(train_mask, frequency, 0), sigma)
    smoothed_prediction = smoothed_masked_frequency / smoothed_mask
    smoothed_errors.append(evaluate_error(frequency[test_mask], smoothed_prediction[test_mask], error))
smoothed_errors = np.asarray(smoothed_errors)


# Also evaluate the errors for the Gaussian process rates.
rates = np.exp(fit.stan_variable("eta")[:, :nrows, :ncols])
gp_errors = evaluate_error(frequency[test_mask], np.median(rates, axis=0)[test_mask], error)

def plot_errors(smoothed_errors, gp_errors, ax = None):
    ax = ax or plt.gca()
    smoothed_loc = smoothed_errors.mean(axis=-1)
    smoothed_scale = smoothed_errors.std(axis=-1)
    line, = ax.plot(sigmas, smoothed_loc, label="Gaussian filter", color="C1")
    ax.fill_between(sigmas, smoothed_loc - smoothed_scale, smoothed_loc + smoothed_scale,
                    alpha=0.5, color=line.get_color())

    gp_loc = gp_errors.mean()
    gp_scale = gp_errors.std()
    line = ax.axhline(gp_loc, label="Gaussian process")
    ax.axhspan(gp_loc - gp_scale, gp_loc + gp_scale, alpha=0.5, color=line.get_color())

    ax.set_xscale("log")

plot_errors(smoothed_errors, gp_errors)
```

```{code-cell} ipython3
# Show the summary figure for the paper.
fig = plt.figure(figsize=(6.4, 4.4))
wspace = 0.15
gs_main = mpl.gridspec.GridSpec(2, 2, height_ratios=[0.03, 1], figure=fig, wspace=wspace,
                                hspace=0.05)
gs = mpl.gridspec.GridSpecFromSubplotSpec(2, 2, gs_main[1, :], wspace=wspace, hspace=0.4)

rate = np.median(rates, axis=0)
cmap = mpl.cm.viridis.copy()
cmap.set_under("silver")
kwargs = {
    "origin": "lower",
    "extent": extent,
    "cmap": cmap,
    "norm": mpl.colors.Normalize(0, rate.max()),
}
label_offset = -0.03

ax = ax1 = fig.add_subplot(gs[0, 0])
im = ax.imshow(frequency, **kwargs)
ax.set_ylabel("northing (km)")
ax.set_xlabel("easting (km)")
ax.xaxis.set_ticks([0, 0.5, 1])
ax.yaxis.set_ticks([0, 0.3])
ax.text(label_offset, 1, "(a)", transform=ax.transAxes, ha="right", va="center")

ax = fig.add_subplot(gs[0, 1], sharex=ax, sharey=ax)
im = ax.imshow(data["frequency"], **kwargs)
plt.setp(ax.get_yticklabels(), visible=False)
ax.set_xlabel("easting (km)")
ax.text(label_offset, 1, "(b)", transform=ax.transAxes, ha="right", va="center")

ax = fig.add_subplot(gs[1, 0], sharex=ax, sharey=ax)
im = ax.imshow(rate, **kwargs)
ax.set_ylabel("northing (km)")
ax.set_xlabel("easting (km)")
ax.text(label_offset, 1, "(c)", transform=ax.transAxes, ha="right", va="center")

cax = fig.add_subplot(gs_main[0])
fig.colorbar(im, cax=cax, orientation="horizontal", label="tree density", extend="max")
cax.xaxis.set_ticks_position("top")
cax.xaxis.set_label_position("top")

ax = fig.add_subplot(gs[1, 1])
plot_errors(smoothed_errors, gp_errors, ax)

ax.set_xscale("log")
ax.set_xlabel("smoothing scale")
ax.set_ylabel("scaled mean\nsquared error")
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.legend(fontsize="small")
ax.text(label_offset, 1, "(d)", transform=ax.transAxes, ha="right", va="center")

fig.savefig("trees.pdf", bbox_inches="tight")
```
