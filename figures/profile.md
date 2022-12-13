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

# Profiling of different methods and parameterizations

```{code-cell} ipython3
import cmdstanpy
import itertools as it
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pickle
from gptools.stan.profile import PARAMETERIZATIONS, LOG10_NOISE_SCALES, SIZES
import pandas as pd
import re
from scipy import stats
import types

mpl.style.use("../jss.mplstyle")
fig_width, fig_height = mpl.rcParams["figure.figsize"]

NOISE_SCALES = 10 ** LOG10_NOISE_SCALES
```

```{code-cell} ipython3
# Load the runtimes for the sampler.
durations = {}
product = it.product(PARAMETERIZATIONS, LOG10_NOISE_SCALES, SIZES)
for parameterization, log_noise_scale, size in product:
    with open(f"../workspace/profile/sample/{parameterization}/"
              f"log10_noise_scale-{log_noise_scale:.3f}_size-{size}.pkl", "rb") as fp:
        result = pickle.load(fp)
        fltr = ~result["timeouts"]
        duration = result["durations"][fltr].mean() if fltr.all() else np.nan
        durations.setdefault(parameterization, []).append(duration)

shape = (len(LOG10_NOISE_SCALES), len(SIZES))
durations = {key: np.reshape(value, shape) for key, value in durations.items()}
```

```{code-cell} ipython3
def load_lps(filename: str) -> np.ndarray:
    """
    Load a configuration and compute the log probability for held-out data.
    """
    with open(filename, "rb") as fp:
        result = pickle.load(fp)
    errors = []
    for fit, eta, data in zip(result["fits"], result["etas"], result["data"]):
        if fit is None:
            mses.append(np.nan)
            continue
        test_idx = np.setdiff1d(1 + np.arange(size), data["observed_idx"]) - 1
        if not test_idx.size:
            raise ValueError("there are no test values")
        test_eta = eta[test_idx]

        if isinstance(fit, cmdstanpy.CmdStanVB):
            df = pd.DataFrame(fit.variational_sample.values, columns=fit.column_names)
            df = df[[column for column in df if re.fullmatch(r"eta\[\d+\]", column)]]
            eta_samples = df.values
        elif isinstance(fit, cmdstanpy.CmdStanMCMC):
            eta_samples = fit.stan_variable("eta")
            if fit.method_variables()["divergent__"].sum():
                print("divergent transitions")
        else:
            raise TypeError(fit)


        error = 0
        for i in test_idx:
            # assert False
            kde = stats.gaussian_kde(eta_samples[:, i])
            error += kde.logpdf(eta[i]).squeeze()
        errors.append(error)
    return np.asarray(errors)
```

```{code-cell} ipython3
size = 1024
methods = ["sample", "variational"]
parameterizations = ["fourier_non_centered", "fourier_centered"]
product = it.product(methods, parameterizations, LOG10_NOISE_SCALES)
lps = {}
for method, parameterization, log_noise_scale in product:
    suffix = "-train-test"  if method == "sample" else ""
    filename = f"../workspace/profile/{method}/{parameterization}/" \
        f"log10_noise_scale-{log_noise_scale:.3f}_size-{size}{suffix}.pkl"
    lps.setdefault((method, parameterization), []).append(load_lps(filename))

num_bootstrap = 1000
mean_lps = {
    key: np.asarray([
        np.random.dirichlet(np.ones_like(value), num_bootstrap) @ value
        for value in values
    ]).T for key, values in lps.items()
}

deltas = {
    method: mean_lps[(method, "fourier_non_centered")] - mean_lps[(method, "fourier_centered")]
    for method in methods
}
```

```{code-cell} ipython3
color_by_method = {
    "standard": "C0",
    "graph": "C1",
    "fourier": "C2",
}
marker_by_parameterization = {
    "centered": "o",
    "non_centered": "s",
}
ls_by_parameterization = {
    "centered": "-",
    "non_centered": "--",
}

fig = plt.figure(layout="constrained")
# First ratio is manually fiddled to match the heights of ax1 and ax3
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[1, 0], sharey=ax1)

gs4 = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, gs[1, 1], hspace=0,
                                           wspace=0,
                                           height_ratios=[2, 1])
ax4t = fig.add_subplot(gs4[0])
ax4b = fig.add_subplot(gs4[1])

ax = ax1
ax.set_xlabel(r"size $n$")
ax.set_ylabel(r"runtime (seconds)")
ax = ax2
ax.set_xlabel(r"size $n$")
ax.set_ylabel(r"runtime (seconds)")

for i, ax in [(0, ax1), (-1, ax2)]:
    for key, value in durations.items():
        method, parameterization = key.split("_", 1)
        line, = ax.plot(
            SIZES, value[i], color=color_by_method[method],
            marker=marker_by_parameterization[parameterization],
            ls=ls_by_parameterization[parameterization],
        )
        line.set_markeredgecolor("w")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.05, 0.95, fr"({'ab'[i]}) $\kappa=10^{{{LOG10_NOISE_SCALES[i]:.0f}}}$",
            transform=ax.transAxes, va="top")
    ax.set_xlabel("size $n$")
    ax.set_ylabel("duration (seconds)")


ax = ax3
ax.set_xlabel(r"noise scale $\kappa$")
ax.set_ylabel(r"runtime (seconds)")

ax.set_xlabel(r"noise scale $\kappa$")
ax.set_ylabel("duration (seconds)")
ax.set_xscale("log")
mappable = mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(SIZES.min(), SIZES.max()),
                                 cmap="viridis")
method = "fourier"
for parameterization in ["non_centered", "centered"]:
    for size, y in zip(SIZES, durations[f"{method}_{parameterization}"].T):
        ax.plot(NOISE_SCALES, y, color=mappable.to_rgba(size),
                ls=ls_by_parameterization[parameterization])
ax.text(0.05, 0.95, "(c)", transform=ax.transAxes, va="top")
fig.colorbar(mappable, ax=ax, label="size $n$")

ax = ax4b
ax.set_xlabel(r"noise scale $\kappa$")
ax.spines["top"].set_visible(False)

# Monkeypatch for fixed order of magnitude.
def _set_order_of_magnitude(self):
    self.orderOfMagnitude = 3
ax4b.yaxis.major.formatter._set_order_of_magnitude = types.MethodType(
    _set_order_of_magnitude, ax4b.yaxis.major.formatter,
)
ax4t.yaxis.major.formatter._set_order_of_magnitude = types.MethodType(
    _set_order_of_magnitude, ax4t.yaxis.major.formatter,
)

for ax in [ax4b, ax4t]:
    for key, value in deltas.items():
        marker = "o" if key == "sample" else "s"
        line, *_ = ax.errorbar(NOISE_SCALES, value.mean(axis=0), value.std(axis=0),
                               label=key, marker=marker, markeredgecolor="w", ls="none", zorder=9)
        ax.plot(NOISE_SCALES, value.mean(axis=0), color="silver", zorder=0)
    ax.axhline(0, color="k", ls=":", zorder=1)

    ax.set_xscale("log")
    ax.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
ax4b.legend(loc="lower right")
ax4b.set_ylim(-10e3, -7e3)
ax4t.set_ylim(-1300, 600)
ax4b.yaxis.get_offset_text().set_visible(False)
ax4b.set_yticks([-10e3, -8e3])

# Visibility adjustment must happen after plotting.
ax4t.set_ylabel(r"log p.p.d. difference $\Delta$", y=0.2)
ax4t.set_xticklabels([])
ax4t.spines["bottom"].set_visible(False)
plt.setp(ax.xaxis.get_majorticklines(), visible=False)
plt.setp(ax.xaxis.get_minorticklines(), visible=False)
ax4t.text(0.05, 0.95, "(d)", transform=ax.transAxes, va="top")

# Disable automatic layout.
fig.get_layout_engine().set(rect=[0, 0, 1, 0.93])
fig.draw_without_rendering()
fig.set_layout_engine(None)

# Add the broken axis markers.
angle = np.deg2rad(30)
scale = 0.01
ax = ax4t
pm = np.asarray([-1, 1])
for x in [0, 1]:
    for ax, y in [(ax4t, 0), (ax4b, 1)]:
        pos = ax.get_position()
        line = mpl.lines.Line2D(
            x + scale * np.cos(angle) * pm / pos.width,
            y + scale * np.sin(angle) * pm / pos.height,
            transform=ax.transAxes, clip_on=False, color="k",
            lw=mpl.rcParams["axes.linewidth"], in_layout=False,
        )
        ax.add_line(line)

# Add the figure-level legend.
handles_labels = [
    (
        mpl.lines.Line2D([], [], linestyle=ls_by_parameterization["centered"],
                         marker=marker_by_parameterization["centered"], color="gray",
                         markeredgecolor="w"),
        "centered",
    ),
    (
        mpl.lines.Line2D([], [], linestyle=":",
                         marker=marker_by_parameterization["non_centered"], color="gray",
                         markeredgecolor="w"),
        "non-centered",
    ),
]
for method, color in color_by_method.items():
    handles_labels.append((
        mpl.lines.Line2D([], [], color=color),
        method,
    ))
bbox1 = ax1.get_position()
bbox2 = ax2.get_position()
bbox_to_anchor = [bbox1.xmin, 0, bbox2.xmax - bbox1.xmin, 1]
legend = fig.legend(*zip(*handles_labels), fontsize="small", loc="upper center",
                    ncol=5, bbox_to_anchor=bbox_to_anchor)

fig.savefig("scaling.pdf", bbox_inches="tight")
fig.savefig("scaling.png", bbox_inches="tight")
```
