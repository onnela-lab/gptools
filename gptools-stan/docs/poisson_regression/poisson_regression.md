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

# Gaussian process Poisson regression

Poisson regression is suitable for estimating the latent rate of count data, such as the number of buses arriving at a stop within an hour or the number of car accidents per day. Here, we estimate the rate using a Gaussian process given synthetic count data. First, we draw the log rate $\eta$ from a Gaussian process with squared exponential kernel $k$ observed at discrete positions $x$, i.e. the covariance of two observations is
$$
\mathrm{cov}\left(\eta(x),\eta(y)\right)=\sigma^2\exp\left(-\frac{d(x, y)^2}{2\ell^2}\right) + \epsilon\delta(x - y),
$$
where $d\left(\cdot,\cdot\right)$ is a distance measure, $\sigma$ is the marginal scale of the process, and $\ell$ is its correlation length. The "nugget" variance $\epsilon$ ensures the covariance matrix is numerically positive semi-definite. Second, we sample counts $y$ from a Poisson distribution with rate $\lambda = \exp\eta$. Let's generate a sample with $n$ regularly-spaced observations and visualize it.

```{code-cell} ipython3
from gptools.util.kernels import DiagonalKernel, ExpQuadKernel, Kernel
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


def simulate(x: np.ndarray, kernel: Kernel, mu: float = 0) -> dict:
    """Simulate a Gaussian process Poisson model with mean mu observed at x."""
    # Add an extra dimension to the vector x because the kernel expects an array
    # with shape (num_data_points, num_dimensions).
    X = x[:, None]
    cov = kernel.evaluate(X)
    eta = np.random.multivariate_normal(mu * np.ones_like(x), cov)
    rate = np.exp(eta)
    y = np.random.poisson(rate)
    # Return the results as a dictionary, including input arguments (we need to extract kernel 
    # parameters from the `CompositeKernel`).
    return {"x": x, "X": X, "mu": mu, "y": y, "eta": eta, "y": y, "rate": rate, "n": x.size,
            "sigma": kernel.a.sigma, "length_scale": kernel.a.length_scale, 
            "epsilon": kernel.b.epsilon}


def plot_sample(sample: dict, ax: mpl.axes.Axes = None) -> mpl.axes.Axes:
    """Visualize a sample of a Gaussian process Poisson model."""
    ax = ax or plt.gca()
    ax.plot(sample["x"], sample["rate"], label=r"rate $\lambda$")
    ax.scatter(sample["x"], sample["y"], marker=".", color="k", label="counts $y$",
               alpha=0.5)
    ax.set_xlabel("Location $x$")
    return ax
    

np.random.seed(0)
n = 64
x = np.arange(n)
kernel = ExpQuadKernel(sigma=1.2, length_scale=5, period=n) + DiagonalKernel(1e-3, n)
sample = simulate(x, kernel)
plot_sample(sample).legend()
```

We used a kernel with periodic boundary conditions by passing `period=n` to the kernel. The Gaussian process is thus defined on a ring with circumference $n$.

To learn the latent rate using stan, we first define the data block of the program, i.e., the information we provide to the inference algorithm. Here, we assume that the marginal scale $\sigma$, correlation length $\ell$, and nugget variance $\epsilon$ are known.

```{literalinclude} data.stan
   :language: stan
```

The model is simple: a multivariate normal prior for the log rate $\eta$ and a Poisson observation model.

```{literalinclude} poisson_regression_centered.stan
   :language: stan
```

Let's compile the model, draw posterior samples, and visualize the result.

```{code-cell} ipython3
from cmdstanpy import CmdStanMCMC
from gptools.stan import compile_model
from gptools.util import Timer
from gptools.util.plotting import plot_band
import os


def sample_and_plot(stan_file: str, data: dict, return_fit: bool = False, **kwargs) -> CmdStanMCMC:
    """Draw samples from the posterior and visualize them."""
    # Set default parameters. We use a small number of samples during testing.
    niter = 1 if os.environ.get("CI") else 100
    kwargs = {"iter_warmup": niter, "iter_sampling": niter, "chains": 1,
              "refresh": niter // 10 or None} | kwargs
    
    # Compile the model and draw posterior samples.
    model = compile_model(stan_file=stan_file)
    with Timer(f"sampled using {stan_file}"):
        fit = model.sample(data, **kwargs)
        
    # Visualize the result.
    ax = plot_sample(sample)
    plot_band(x, np.exp(fit.stan_variable("eta")), label="inferred rate", ax=ax)
    ax.legend()
    if return_fit:
        return fit
    
    
centered_fit = sample_and_plot("poisson_regression_centered.stan", sample, return_fit=True)
```

Stan's Hamiltonian Monte Carlo sampler explores the posterior distribution well and recovers the underlying rate. But it takes its time about it–especially given how small the dataset is. The exploration is slow because adjacenct elements of the log rate are highly correlated under the posterior distribution as shown in the scatter plot below. The mean correlation between adjacent elements is 0.95. The posterior is highly correlated because the Gaussian process prior demands that adjacent points are close: if one moves up, the other must follow to keep the log rate $\eta$ smooth. If the data are "strong", i.e., there is more information in the likelihood than in the prior, the data can decorrelate the posterior: each log rate is primarily informed by the corresponding count rather than the prior.

```{code-cell} ipython3
fig, ax = plt.subplots()
etas = centered_fit.stan_variable("eta")
idx = 20
ax.scatter(etas[:, idx - 1], etas[:, idx]).set_edgecolor("w")
ax.set_aspect("equal")
ax.set_xlabel(fr"Log rate $\eta_{{{idx}}}$")
ax.set_ylabel(fr"Log rate $\eta_{{{idx + 1}}}$")
fig.tight_layout()
```

In this example, the data are relatively weak (recall that the variance of the Poisson distribution is equal to its mean such that, for small counts, the likelihood is not particularly informative). We can alleviate this problem by switching to a *non-centered* parameterization such that the parameters are uncorrelated under the prior distribution. In particular, we use a parameter $z\sim\text{Normal}\left(0, 1\right)$ and transform the white noise such that it is a draw from the Gaussian process prior. This approach is a higher-dimensional analogue of sampling a standard normal distribution and multiplying by the standard deviation $\sigma$ as opposed to sampling from a normal distribution with standard deviation $\sigma$ directly. The log rate is
$$
\eta = Lz,
$$
where $L$ is the [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) of the covariance matrix $\Sigma$ of the kernel. The Cholesky decomposition of the covariance matrix is the higher-dimensional analogue of taking the square route of the variance. Because the data are weak, they can only slightly correlate the parameters $z$. The Stan program is shown below.

```{literalinclude} poisson_regression_non_centered.stan
   :language: stan
```

```{code-cell} ipython3
non_centered_fit = sample_and_plot("poisson_regression_non_centered.stan", sample, return_fit=True)
```

Sampling from the posterior is substantially faster. While the log rates $\eta$ remain highly correlated (see left panel below), the posterior for the parameters $z$ the Hamiltonian sampler is exploring are virtually uncorrelated (see right panel below). Takeaway: if the data are strong, use a centered parameterization. If the data are weak, use a non-centered parameterization.

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2)
etas = non_centered_fit.stan_variable("eta")
zs = non_centered_fit.stan_variable("z")
idx = 20
ax1.scatter(etas[:, idx - 1], etas[:, idx]).set_edgecolor("w")
ax1.set_aspect("equal")
ax1.set_xlabel(fr"Log rate $\eta_{{{idx}}}$")
ax1.set_ylabel(fr"Log rate $\eta_{{{idx + 1}}}$")

ax2.scatter(zs[:, idx - 1], zs[:, idx]).set_edgecolor("w")
ax2.set_aspect("equal")
ax2.set_xlabel(fr"Log rate $z_{{{idx}}}$")
ax2.set_ylabel(fr"Log rate $z_{{{idx + 1}}}$")

fig.tight_layout()
```

## Nearest-neighbor Gaussian processes

Despite the non-centered parameterization being faster, its performance scales poorly with increasing sample size. Evaluating the Cholesky decomposition scales with $n^3$ such that this approach is only feasible for relatively small datasets. However, intuitively, only local correlations are important for the Gaussian process (at least for the squared exponential kernel we used above). Points separated by a large distance are virtually uncorrelated but these negligible correlations nevertheless slow down our calculations. Fortunately, we can factorize the joint distribution to obtain a product of conditional distributions.
$$
p\left(\eta\mid \Sigma\right)=\prod_{i=1}^n p\left(\eta_{i}\mid\eta_{<i},\Sigma\right),
$$
where $\eta_{<i}$ denotes all parameters $\left\{\eta_1, \ldots \eta_{i-1}\right\}$ preceeding $\eta_i$. In a nearest neighbor Gaussian process, we approximate the full conditional by only conditioning on at most the $k$ preceeding parameters, i.e., we neglect all correlations between parameters that are further apart than the $k^\text{th}$ nearest neighbor. We thus need to decompose $n$ square matrices each having $k + 1$ rows. The computational cost scales as $nk^3$–a substantial saving if $n$ is large compared with $k$.

More generally, we can construct a Gaussian process on a directed acyclic graph using the factorization into conditionals. Here, we employ this general formulation and represent the nearest neighbor Gaussian process as a graph. The conditioning structure is illustrated below.

```{code-cell} ipython3
from gptools.util.graph import lattice_predecessors, predecessors_to_edge_index

k = 5
predecessors = lattice_predecessors((n,), k)

idx = 20
fig, ax = plt.subplots()
ax.scatter(predecessors[idx, -1] - np.arange(3) - 1, np.zeros(3), color="silver", 
           label="ignored nodes")
ax.scatter(idx + np.arange(3) + 1, np.zeros(3), color="silver")
ax.scatter(predecessors[idx, 1:], np.zeros(k), label="conditioning nodes")
ax.scatter(idx, 0, label="target node")
ax.legend()
for j in predecessors[idx, :-1]:
    ax.annotate("", xy=(j, 0), xytext=(idx, 0), 
                arrowprops={"arrowstyle": "-|>", "connectionstyle": "arc3,rad=.5"})
ax.set_ylim(-.25, .25)
ax.set_axis_off()
fig.tight_layout()
```

The function `lattice_predecessors` constructs a nearest neighbor matrix `predecessors` with shape `(n, k + 1)` such that the first `k` elements of the $i^\text{th}$ row are the predecessors of node $i$ and the last element is the node itself. The corresponding Stan program is shown below. The distribution `graph_gp` encodes the Gaussian process.

```{literalinclude} poisson_regression_graph_centered.stan
   :language: stan
```

```{code-cell} ipython3
edges = predecessors_to_edge_index(predecessors)
sample |= {"edge_index": edges, "num_edges": edges.shape[1]}
sample_and_plot("poisson_regression_graph_centered.stan", sample)
```

The centered graph Gaussian process is indeed faster than the standard implementation. However, it suffers from the same challenges: the posterior for $\eta$ is highly correlated. Fortunately, we can also construct a non-centered parameterization.

```{literalinclude} poisson_regression_graph_non_centered.stan
   :language: stan
```

```{code-cell} ipython3
edges = predecessors_to_edge_index(predecessors)
sample |= {"edge_index": edges, "num_edges": edges.shape[1]}
sample_and_plot("poisson_regression_graph_non_centered.stan", sample)
```

The non-centered parameterization of the graph Gaussian process is even faster, and we have reduced the runtime by more than an order of magnitude for this small dataset.

+++

## Gaussian process using fast Fourier transforms

If, as in this example, observations are regularly spaced, we can evaluate the likelihood using the fast Fourier transform. Similarly, we can construct a non-centered parameterization by transforming Fourier coefficients with {stan:func}`gp_transform_irfft`.

```{literalinclude} poisson_regression_fourier_centered.stan
   :language: stan
```

```{code-cell} ipython3
sample_and_plot("poisson_regression_fourier_centered.stan", sample)
```

```{code-cell} ipython3
sample_and_plot("poisson_regression_fourier_non_centered.stan", sample)
```

Given the substantial performance improvements, we can readily increase the sample size as illustrated below for a dataset with more than a thousand observations.

```{code-cell} ipython3
x = np.arange(1024)
kernel = ExpQuadKernel(sigma=1.2, length_scale=15, period=x.size) + DiagonalKernel(1e-3, x.size)
sample = simulate(x, kernel)
plot_sample(sample).legend()
```

```{code-cell} ipython3
sample_and_plot("poisson_regression_fourier_non_centered.stan", sample)
```
