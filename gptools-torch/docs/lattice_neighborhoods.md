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

# Lattice neighborhoods

Considering Gaussian processes on directed acyclic graphs is a general approach to reduce the computational burden for inference. However, often, we are interested in graph structures that capture the spatial relationships between nodes. The function `lattice_predecessors` does just that. It constructs a directed graph for nodes on a lattice, ensuring that the graph is acyclic. Two options are supported: Cuboidal and ellipsoidal receptive fields. The latter are preferable because they preserve the isotropy of the space we are trying to approximate. They are also more efficient because the volume of a hyperellipsoid is strictly smaller than a hypercube with the same diameter, reducing the number of nodes in the receptive field. For example, in three dimensions, the relative volume of a sphere and cube is
$$
\frac{V_\mathrm{sphere}}{V_\mathrm{cube}}=\frac{4\pi r^3 / 3}{\left(2 r\right)^3}\approx 0.52.
$$
The number of nodes is reduced by a factor of two. But, because matrix inversion required to evaluate the likelihood scales approximately cubicly with the number of nodes, we can reduce the computational cost by almost an order of magnitude. 

The cell below illustrates the receptive fields in two dimensions and shows both predecessors (that the example node depends on in the likelihood) and successors (that depend on the example node). The union of predecessors and successors is the receptive field of the node.

```{code-cell} ipython3
from gptools.torch.graph import GraphGaussianProcess
from gptools.util import graph
from gptools.util.kernels import DiagonalKernel, ExpQuadKernel
from gptools.util import coordgrid
import matplotlib as mpl
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats
from tabulate import tabulate
import torch as th


mpl.rcParams["figure.dpi"] = 144
```

```{code-cell} ipython3
# Define the lattice size, receptive field size, and choose example node positions.
width, height = 10, 15
k = (4, 3)
x, y = 5, 7

shape = (width, height)
node = np.ravel_multi_index((x, y), shape)
coords = coordgrid(np.arange(width), np.arange(height))

# For each of the two receptive field methods, ...
fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
for ax, bounds in zip(axes, graph.LatticeBounds):
    # Plot the lattice and example node.
    ax.scatter(*coords.T, color="silver", marker=".")
    ax.scatter(*coords[node], label="example node", zorder=99).set_edgecolor("w")
    
    # Get the lattice predecessors and construct a directed graph.
    predecessors = graph.lattice_predecessors((width, height), k, bounds=bounds)
    edge_index = graph.predecessors_to_edge_index(predecessors, indexing="numpy")
    G = graph.edge_index_to_graph(edge_index)
    
    # Show predecessors (that the example node depends on in the likelihood) and successors 
    # (that depend on the example node in the likelihood).
    for label in ["predecessors", "successors"]:
        # Get the nodes and ensure they satisfy the ordering constraint.
        nodes = np.asarray(list(getattr(G, label)(node)))
        if label == "predecessors":
            assert (nodes <= node).all()
        else:
            assert (nodes >= node).all()
        
        # Plot the nodes.
        zorder = 10 if label == "predecessors" else 8
        ax.scatter(*coords[nodes].T, label=label, zorder=zorder).set_edgecolor("w")
        edges = G.out_edges(node) if label == "successors" else G.in_edges(node)
        edges = nx.draw_networkx_edges(G, coords, edges, ax=ax, node_size=0, alpha=0.5, 
                                       arrowsize=7)
        for edge in edges:
            edge.set_zorder(9)
            
    ax.set_aspect("equal")
    ax.set_title(f"{bounds.value} bounds")
    ax.set_xlabel("Coordinate $x_1$")
        
axes[0].set_ylabel("Coordinate $x_2$")
axes[0].yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
axes[0].legend(loc="upper right")
fig.tight_layout()
```

We can also compare how samples from the true Gaussian process and our nearest-neighbor approximation compare.

```{code-cell} ipython3
# Set up parameters for a Gaussian process.
seed = 0
n = 201
x = th.linspace(0, 1, n)
X = x[:, None]
kernel = ExpQuadKernel(1.2, 0.1) + DiagonalKernel(1e-3)
ks = [2, 10, 20, 30]

# Construct the exact Gaussian process and draw a realization.
dist = th.distributions.MultivariateNormal(th.zeros(n), kernel.evaluate(X))
th.manual_seed(seed)
y = dist.sample()
log_prob_rows = [("exact", dist.log_prob(y))]

# Draw realizations from graph Gaussian processes with different sizes.
ys = []
for k in ks:
    predecessors = graph.lattice_predecessors(x.shape, k)
    gdist = GraphGaussianProcess(dist.loc, X, predecessors, kernel)
    th.manual_seed(seed)
    ys.append(gdist.sample())
    log_prob_rows.append((f"k = {k}", gdist.log_prob(y)))

# Compare the realizations.
fig, ax = plt.subplots()
for k, y_ in zip(ks, ys):
    ax.plot(x, y_, label=f"$k={k}$")
ax.plot(x, y, color="k", ls="--", label="exact")
ax.set_xlabel("Cooordinate $x$")
ax.set_ylabel("Function $y(x)$")
ax.legend()
fig.tight_layout()

# Show the log probabilities under different approximations.
print(tabulate(log_prob_rows))
```

Let's have a look at the same idea in two dimensions.

```{code-cell} ipython3
seed = 0
shape = (70, 70)
kernel = ExpQuadKernel(1.1, 4) + DiagonalKernel(1e-3)
k = 4

width, height = shape
x = th.arange(width)
y = th.arange(height)
coords = th.as_tensor(coordgrid(x, y))
dist = th.distributions.MultivariateNormal(th.zeros(coords.shape[0]), kernel.evaluate(coords))
th.manual_seed(seed)
eta = dist.sample()
print(f"reference log_prob = {dist.log_prob(eta)}")

etas = {
    "exact": eta,
}

ref = graph.num_lattice_predecessors(k, "cube", 2)

# Add different predecessor shapes.
for bounds in graph.LatticeBounds:
    # Pick the scale that's closest in terms of number of predecessors in the graph.
    num_predecessors = graph.num_lattice_predecessors(np.arange(10), bounds, 2)
    l = np.argmin(np.abs(num_predecessors - ref))
    
    predecessors = graph.lattice_predecessors(shape, l, bounds)
    gdist = GraphGaussianProcess(dist.loc, coords, predecessors, kernel)
    print(f"{bounds}: # predecessors = {predecessors.shape[1]}; log_prob = {gdist.log_prob(eta)}")
    th.manual_seed(seed)
    etas[bounds.value] = gdist.sample()
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for ax, (label, eta) in zip(axes.ravel(), etas.items()):
    ax.pcolormesh(x, y, eta.reshape(shape).T)
    ax.set_aspect("equal")
    ax.set_title(label)
    
    pearsonr, _ = stats.pearsonr(eta, etas['exact'])
    print(f"{label} corr with ground truth: {pearsonr}")
```
