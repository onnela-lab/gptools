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

# Passengers on the London Underground network

The [London Underground](https://en.wikipedia.org/wiki/London_Underground), nicknamed "The Tube", transports millions of people each day. Which factors might affect passenger numbers at different stations? Here, we use the transport network to build a sparse Gaussian process model and fit it to daily passenger numbers. We first load the prepared data, including features such as the [transport zone](https://en.wikipedia.org/wiki/London_fare_zones), number of interchanges, and location, for each station. Detailes on the data preparation can be found [here](https://github.com/onnela-lab/gptools/blob/main/data/prepare_tube_data.py).

We next compile the model and draw posterior samples. The model is shown below.

```{literalinclude} tube.stan
   :language: stan
```

```{code-cell} ipython3
:tags: []

from gptools.stan import compile_model
import json
import numpy as np
import os


with open("../../data/tube-stan.json") as fp:
    data = json.load(fp)

# Sample a training mask and update the data for Stan.
seed = 0
train_frac = 0.8
np.random.seed(seed)
train_mask = np.random.uniform(0, 1, data["num_stations"]) < train_frac

# Apply the training mask and include degree and zone effects.
y = np.asarray(data["passengers"])
data.update({
    "include_zone_effect": 1,
    "include_degree_effect": 1,
    # We use -1 for held-out data.
    "passengers": np.where(train_mask, y, -1),
})


if "CI" in os.environ:
    niter = 10
elif "READTHEDOCS" in os.environ:
    niter = 200
else:
    niter = None
model_with_gp = compile_model(stan_file="tube.stan")
chains = 1 if "READTHEDOCS" in os.environ or "CI" in os.environ else 4
fit_with_gp = model_with_gp.sample(data, chains=chains, iter_warmup=niter, iter_sampling=niter,
                                   seed=seed, adapt_delta=0.9, show_progress=False)
print(fit_with_gp.diagnose())
```

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib as mpl
from matplotlib import pyplot as plt

# Show some basic diagnostics plots.
fig, (ax1, ax2) = plt.subplots(1, 2)
rate = np.exp(fit_with_gp.stan_variable("log_mean").mean(axis=0))
ax1.scatter(y[train_mask], rate[train_mask], marker=".", label="train")
ax1.scatter(y[~train_mask], rate[~train_mask], marker=".", label="test")
mm = y.min(), y.max()
ax1.plot(mm, mm, color="k", ls=":")
ax1.set_xlabel("actual")
ax1.set_ylabel("predicted")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_aspect("equal")
ax1.legend()

a = fit_with_gp.stan_variable("length_scale")
b = fit_with_gp.stan_variable("sigma")
pts = ax2.scatter(a, b, marker=".", c=fit_with_gp.method_variables()["lp__"])
d = fit_with_gp.method_variables()["divergent__"].astype(bool).ravel()
ax2.scatter(a[d], b[d], color="C1", marker="x")
fig.colorbar(pts, ax=ax2, label="log probability", location="top")
ax2.set_xscale("log")
ax2.set_xlabel(r"length scale $\ell$")
ax2.set_ylabel(r"marginal scale $\sigma$")
fig.tight_layout()
```

We construct a figure that shows the data, effects of zone and degree, and the residual effects captured by the Gaussian process.

```{code-cell} ipython3
from pathlib import Path

fig, axes = plt.subplots(2, 2, gridspec_kw={"width_ratios": [2, 1]}, figsize=(6, 6))
ax1, ax2 = axes[:, 0]
kwargs = {"marker": "o", "s": 10}


X = np.asarray(data["station_locations"])
ax1.scatter(*X[~train_mask].T, facecolor="w", edgecolor="gray", **kwargs)
pts1 = ax1.scatter(*X[train_mask].T, c=y[train_mask], norm=mpl.colors.LogNorm(vmin=np.min(y)),
                   **kwargs)

c = fit_with_gp.stan_variable("f").mean(axis=0)
vmax = np.abs(c).max()
pts2 = ax2.scatter(*X.T, c=c, vmin=-vmax, vmax=vmax,
                   **kwargs, cmap="coolwarm")
ax1.set_aspect("equal")
ax2.set_aspect("equal")

ax2.annotate("Canary Wharf", X[np.argmax(c)], (20, -12), ha="center",
             va="center", fontsize="small",
             arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-0.5"})
ax2.annotate("Hainault\nLoop", X[np.argmin(c)], (31, 13), ha="right",
             va="center", fontsize="small",
             arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.2",
                         "patchA": None, "shrinkA": 10})

ax1.set_axis_off()
ax2.set_axis_off()
location = "top"
fraction = 0.05
cb1 = fig.colorbar(pts1, ax=ax1, label="entries & exits", location=location, fraction=fraction)
cb2 = fig.colorbar(pts2, ax=ax2, label="Gaussian process effect", location=location, fraction=fraction)

for ax in [ax1, ax2]:
    segments = []
    for u, v in np.transpose(data["edge_index"]):
        segments.append([X[u - 1], X[v - 1]])
    collection = mpl.collections.LineCollection(segments, zorder=0, color="silver")
    ax.add_collection(collection)

ax1.set_ylabel("northing (km)")
ax1.set_xlabel("easting (km)")
ax2.set_ylabel("northing (km)")
ax2.set_xlabel("easting (km)")
ax3, ax4 = axes[:, 1]

effect = fit_with_gp.stan_variable("degree_effect")
effect -= effect.mean(axis=1, keepdims=True)
line, *_ = ax3.errorbar(np.arange(effect.shape[1]) + 1, effect.mean(axis=0), effect.std(axis=0),
                        marker="o")
line.set_markeredgecolor("w")
ax3.set_ylabel("degree effect")
ax3.set_xlabel("degree")
ax3.set_xticks([1, 3, data["num_degrees"]])
ax3.set_xticklabels(["1", "3", f"{data['num_degrees']}+"])
ax3.axhline(0, color="k", ls=":")


effect = fit_with_gp.stan_variable("zone_effect")
effect -= effect.mean(axis=1, keepdims=True)
line, *_ = ax4.errorbar(np.arange(effect.shape[1]) + 1, effect.mean(axis=0), effect.std(axis=0),
                        marker="o")
line.set_markeredgecolor("w")
ax4.set_ylabel("zone effect")
ax4.set_xlabel("zone")
ax4.set_xticks([2, 4, data["num_zones"]])
ax4.set_xticklabels(["2", "4", f"{data['num_zones']}+"])
ax4.axhline(0, color="k", ls=":")

ax1.text(0.025, 0.05, "(a)", transform=ax1.transAxes)
ax2.text(0.025, 0.05, "(c)", transform=ax2.transAxes)
ax3.text(0.05, 0.95, "(b)", transform=ax3.transAxes, va="top")
ax4.text(0.95, 0.95, "(d)", transform=ax4.transAxes, va="top", ha="right")


fig.tight_layout()

workspace = Path(os.environ.get("WORKSPACE", os.getcwd()))
fig.savefig(workspace / "tube.pdf", bbox_inches="tight")
fig.savefig(workspace / "tube.png", bbox_inches="tight")
```

On the one hand, the three northern stations of the [Hainault Loop](https://en.wikipedia.org/wiki/Hainault_Loop) ([Roding Valley](https://en.wikipedia.org/wiki/Roding_Valley_tube_station), [Chigwell](https://en.wikipedia.org/wiki/Chigwell_tube_station), and [Grange Hill](https://en.wikipedia.org/wiki/Grange_Hill_tube_station)) are underused because they are serviced by only three trains an hour whereas nearby stations (such as [Hainault](https://en.wikipedia.org/wiki/Hainault_tube_station), [Woodford](https://en.wikipedia.org/wiki/Woodford_tube_station), and [Buckhurst Hill](https://en.wikipedia.org/wiki/Buckhurst_Hill_tube_station)) are serviced by twelve trains an hour. On the other hand, [Canary Wharf](https://en.wikipedia.org/wiki/Canary_Wharf_tube_station) at the heart of the financial district has much higher use than would be expected for a station that only serves a single line in zone 2.
