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

# Passengers on the London Underground network

```{code-cell} ipython3
from gptools.stan import compile_model
from gptools.util import encode_one_hot
from gptools.util.graph import graph_to_edge_index, check_edge_index
import json
import matplotlib as mpl
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import os
from scipy.spatial.distance import pdist
from scipy import stats
```

```{code-cell} ipython3
with open("../../../data/tube.json") as fp:
    data = json.load(fp)


graph = nx.Graph()
graph.add_nodes_from(data["nodes"].items())
graph.add_edges_from(data["edges"])

# Remove the new stations that don't have data yet (should just be the two new Northern Line stations).
stations_to_remove = [node for node, data in graph.nodes(data=True) if data["entries"] is None]
graph.remove_nodes_from(stations_to_remove)
# Remove Kensington Olympia because it's hardly used in regular transit.
graph.remove_node("940GZZLUKOY")
print(f"loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

# Convert the graph to an edge index with self loops to pass to Stan.
edge_index, mapping = graph_to_edge_index(graph, return_mapping=True)
check_edge_index(edge_index)
inverse = {b: a for a, b in mapping.items()}
```

```{code-cell} ipython3
seed = 0
train_frac = 0.8
max_zone = 6
max_degree = 5

def get_node_attribute(graph: nx.Graph, key: str) -> np.ndarray:
    """
    Get a node attribute with consistent order.
    """
    return np.asarray([data[key] for _, data in sorted(graph.nodes(data=True))])

y = get_node_attribute(graph, "entries") + get_node_attribute(graph, "exits")
X = np.transpose([get_node_attribute(graph, "x"), get_node_attribute(graph, "y")]) / 1000
X = X - np.mean(X, axis=0)

# One-hot encode nodes and zones.
zones = get_node_attribute(graph, "zone")
one_hot_zones = encode_one_hot(zones.clip(max=max_zone) - 1)
degrees = np.asarray([graph.degree[node] for node in sorted(graph)])
one_hot_degrees = encode_one_hot(degrees.clip(max=max_degree) - 1)

# Sample a training mask.
np.random.seed(seed)
train_mask = np.random.uniform(0, 1, y.size) < train_frac

distances = pdist(X)
data = {
    "num_stations": graph.number_of_nodes(),
    "num_edges": edge_index.shape[1],
    "edge_index": edge_index,
    "one_hot_zones": one_hot_zones,
    "num_zones": one_hot_zones.shape[1],
    "one_hot_degrees": one_hot_degrees,
    "num_degrees": one_hot_degrees.shape[1],
    "passengers": np.where(train_mask, y, -1),
    "station_locations": X,
    "epsilon": 0,
    "include_zone_effect": 1,
    "include_degree_effect": 1,
}
distances.min(), distances.max()
```

```{code-cell} ipython3
niter = 3 if "CI" in os.environ else 1000
model_with_gp = compile_model(stan_file="tube.stan")
fit_with_gp = model_with_gp.sample(data, chains=1, iter_warmup=niter, iter_sampling=niter,
                                   seed=seed, adapt_delta=0.9)
print(fit_with_gp.diagnose())
```

```{code-cell} ipython3
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
d = fit_with_gp.method_variables()["divergent__"].astype(bool).squeeze()
ax2.scatter(a[d], b[d], color="C1", marker="x")
fig.colorbar(pts, ax=ax2, label="log probability", location="top")
ax2.set_xscale("log")
ax2.set_xlabel(r"length scale $\ell$")
ax2.set_ylabel(r"marginal scale $\sigma$")
fig.tight_layout()
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, gridspec_kw={"width_ratios": [2, 1]}, figsize=(6, 6))
ax1, ax2 = axes[:, 0]
cmap = mpl.cm.viridis.copy()
cmap.set_under("gray")
kwargs = {"marker": "o", "s": 10}

pts1 = ax1.scatter(*X.T, c=np.where(train_mask, y, 0.1), norm=mpl.colors.LogNorm(vmin=np.min(y)),
                   **kwargs, cmap=cmap)

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
    for u, v in graph.edges():
        segments.append([X[mapping[u] - 1], X[mapping[v] - 1]])
    collection = mpl.collections.LineCollection(segments, zorder=0, color="silver")
    ax.add_collection(collection)

ax1.set_ylabel("northing (km)")
ax1.set_xlabel("easting (km)")
ax2.set_ylabel("northing (km)")
ax2.set_xlabel("easting (km)")
ax3, ax4 = axes[:, 1]

effect = fit_with_gp.stan_variable("degree_effect")
line, *_ = ax3.errorbar(np.arange(effect.shape[1]) + 1, effect.mean(axis=0), effect.std(axis=0),
                        marker="o")
line.set_markeredgecolor("w")
ax3.set_ylabel("degree effect")
ax3.set_xlabel("degree")
ax3.set_xticks([1, 3, max_degree])
ax3.set_xticklabels(["1", "3", f"{max_degree}+"])
ax3.axhline(0, color="k", ls=":")


effect = fit_with_gp.stan_variable("zone_effect")
line, *_ = ax4.errorbar(np.arange(effect.shape[1]) + 1, effect.mean(axis=0), effect.std(axis=0),
                        marker="o")
line.set_markeredgecolor("w")
ax4.set_ylabel("zone effect")
ax4.set_xlabel("zone")
ax4.set_xticks([2, 4, max_zone])
ax4.set_xticklabels(["2", "4", f"{max_zone}+"])
ax4.axhline(0, color="k", ls=":")

ax1.text(0.025, 0.05, "(a)", transform=ax1.transAxes)
ax2.text(0.025, 0.05, "(c)", transform=ax2.transAxes)
ax3.text(0.05, 0.95, "(b)", transform=ax3.transAxes, va="top")
ax4.text(0.95, 0.95, "(d)", transform=ax4.transAxes, va="top", ha="right")


fig.tight_layout()
fig.savefig("tube.pdf", bbox_inches="tight")
fig.savefig("tube.png", bbox_inches="tight")
```

On the one hand, the three northern stations of the [Hainault Loop](https://en.wikipedia.org/wiki/Hainault_Loop) ([Roding Valley](https://en.wikipedia.org/wiki/Roding_Valley_tube_station), [Chigwell](https://en.wikipedia.org/wiki/Chigwell_tube_station), and [Grange Hill](https://en.wikipedia.org/wiki/Grange_Hill_tube_station)) are underused because they are serviced by only three trains an hour whereas nearby stations (such as [Hainault](https://en.wikipedia.org/wiki/Hainault_tube_station), [Woodford](https://en.wikipedia.org/wiki/Woodford_tube_station), and [Buckhurst Hill](https://en.wikipedia.org/wiki/Buckhurst_Hill_tube_station)) are serviced by twelve trains an hour. On the other hand, [Canary Wharf](https://en.wikipedia.org/wiki/Canary_Wharf_tube_station) at the heart of the financial district has much higher use than would be expected for a station that only serves a single line in zone 2.

Let's compare the predictive performance on the held out data with and without the Gaussian process.

```{code-cell} ipython3
model_without_gp = compile_model(stan_file="tube_without_gp.stan")
niter = 3 if "CI" in os.environ else 1000
fit_without_gp = model_without_gp.sample(data, chains=1, iter_warmup=niter,
                                         iter_sampling=niter, seed=seed, adapt_delta=0.9)
print(fit_without_gp.diagnose())
```

```{code-cell} ipython3
fig, ax = plt.subplots()

test = np.log(y)[~train_mask]
mm = test.min(), test.max()
ax.plot(mm, mm, color="k", ls=":")

bs_samples = []

for label, fit in [("with GP", fit_with_gp), ("without GP", fit_without_gp)]:
    log_mean = fit.stan_variable("log_mean")[..., ~train_mask]
    kappa = fit.stan_variable("kappa")

    # Evaluate the score and bootstrapped error.
    log_score = stats.norm(log_mean, kappa[:, None]).logpdf(test).mean(axis=0)
    x = np.random.dirichlet(np.ones(log_score.shape[0]), 1000) @ log_score
    print(f"{label}: {log_score.mean():.3f} +- {np.std(x):.3f}")
    bs_samples.append(x)

    ax.errorbar(test, log_mean.mean(axis=0), log_mean.std(axis=0), ls="none", marker=".", label=label)

ax.legend(fontsize="small")
ax.set_aspect("equal")
fig.tight_layout()

delta = np.diff(bs_samples, axis=0).squeeze()
print(f"difference: {delta.mean():.3f} +- {delta.std():.3f}")
```
