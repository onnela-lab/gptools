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

# Variational inference of a spatial Poisson process

Here, we consider a two-dimensional Poisson count model with latent log rate that follows a Gaussian process. Inference proceeds by minimizing a Monte Carlo estimate of the evidence lower bound under a factorized posterior approximation.

```{code-cell} ipython3
from gptools.util.kernels import DiagonalKernel, ExpQuadKernel
from gptools.torch.graph import GraphGaussianProcess
from gptools.torch.util import ParameterizedDistribution, TerminateOnPlateau, VariationalModel
from gptools.util import coordgrid
from gptools.util.graph import lattice_predecessors, LatticeBounds, num_lattice_predecessors
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import numpy as np
from scipy import special
import torch as th
from tqdm.notebook import tqdm

mpl.rcParams["figure.dpi"] = 144
```

```{code-cell} ipython3
# Set up parameter values and sample from the Gaussian process.
width = 80
height = 90
kernel = ExpQuadKernel(1.1, 4) + DiagonalKernel(1e-3)
k = 5
mu = -1
seed = 0
bounds = LatticeBounds.ELLIPSE

x = th.arange(width)
y = th.arange(height)
coords = th.as_tensor(coordgrid(x, y))

shape = (width, height)
predecessors = lattice_predecessors(shape, k, bounds=bounds)
dist = GraphGaussianProcess(mu * th.ones(width * height), coords, predecessors, kernel)

th.manual_seed(seed)
eta = dist.sample().reshape(shape)
lam = eta.exp()
counts = th.distributions.Poisson(lam).sample()

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
vmin = min(lam.min(), counts.min())
vmax = max(lam.max(), counts.max())
ax1.pcolormesh(x, y, lam.T, vmin=vmin, vmax=vmax)
ax2.pcolormesh(x, y, counts.T, vmin=vmin, vmax=vmax)
ax1.set_title(r"Density $\lambda=\exp\eta$")
ax2.set_title(r"Counts $y \sim\mathrm{Poisson}\left(\lambda\right)$")
ax1.set_aspect("equal")
ax2.set_aspect("equal")
fig.tight_layout()

# Ravel the counts for the rest of the notebook.
counts = counts.ravel()
```

```{code-cell} ipython3
# Set up a variational model.
class SpatialCountModel(VariationalModel):
    def __init__(self, approximations, counts):
        super().__init__(approximations)
        self.counts = counts
        
    def log_prob(self, parameters):
        lam = parameters["eta"].exp()
        mu = parameters["mu"]
        dist = GraphGaussianProcess(mu[..., None] * th.ones_like(self.counts), coords, predecessors, kernel)
        return dist.log_prob(parameters["eta"]) \
            + th.distributions.Poisson(lam).log_prob(self.counts).sum(axis=-1)
     

# Initialize using Laplace approximation with flat prior.
eta_loc = (counts + 0.1).log()
eta_scale = (- eta_loc / 2).exp()
model = SpatialCountModel({
    "eta": ParameterizedDistribution(th.distributions.Normal, loc=eta_loc, scale=eta_scale),
    "mu": ParameterizedDistribution(th.distributions.Normal, loc=0.0, scale=1.0),
}, counts)
model.check_log_prob_shape()
```

```{code-cell} ipython3
# Fit the model ...
batch_size = 10
optim = th.optim.Adam(model.parameters(), lr=0.1)
scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=20, verbose=True)

losses = []
terminator = TerminateOnPlateau(30, max_num_steps=1 if "CI" in os.environ else None)
with tqdm() as progress:
    while terminator:
        optim.zero_grad()
        loss = - model.batch_elbo_estimate((batch_size,)).mean()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        scheduler.step(loss)
        terminator.step(loss)
        progress.update()
        progress.set_description(
            f"loss: {loss.item():.3f}; "
            f"termination plateau: {terminator.elapsed} / {terminator.patience}"
        )
    

# ... and show the losses from the last batch.
fig, ax = plt.subplots()
ax.plot(losses)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
fig.tight_layout()
```

```{code-cell} ipython3
# Compare the true density with the inferred density.
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
posterior = model.distributions()["eta"]
estimate = posterior.mean.detach().reshape(shape).T.exp()
vmin = min(lam.min(), estimate.min())
vmax = max(lam.max(), estimate.max())

ax1.pcolormesh(x, y, lam.T, vmin=vmin, vmax=vmax)
ax2.pcolormesh(x, y, estimate, vmin=vmin, vmax=vmax)
ax1.set_title(r"Density $\lambda=\exp\eta$")
ax2.set_title(r"Inferred density")
ax1.set_aspect("equal")
ax2.set_aspect("equal")
fig.tight_layout()
```

```{code-cell} ipython3
# Show the inferred parameters.
fig, ax = plt.subplots()
dist = model.distributions()["mu"]
with th.no_grad():
    lin = dist.mean + 3 * dist.scale * th.linspace(-1, 1, 100)
    ax.plot(lin, dist.log_prob(lin).exp())
ax.axvline(mu, color="k", ls="--")
ax.set_xlabel(r"Mean $\mu$")
ax.set_ylabel("Posterior density")
fig.tight_layout()
```
