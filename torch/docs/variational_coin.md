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

# Variational inference of coin bias

This example illustrates variational Bayesian inference of the bias of a coin. The model is
$$\begin{align}
p&\sim\text{Beta}\left(10, 10\right)\\
x&\sim\text{Bernoulli}\left(p\right),
\end{align}$$
where $p$ is the bias of the coin, and $x$ are Bernoulli trials. We use Monte Carlo estimates of the evidence lower bound to learn the parameters of the variational approximation.

```{code-cell} ipython3
from gptools.torch.util import ParameterizedDistribution, VariationalModel
from matplotlib import pyplot as plt
import os
import torch as th
from typing import Dict
```

```{code-cell} ipython3
class CoinModel(VariationalModel):
    """
    Simple model for inferring the bias of a coin (see
    https://pyro.ai/examples/svi_part_i.html#A-simple-example for details).
    """
    def __init__(self, approximations, x) -> None:
        super().__init__(approximations)
        self.x = x
        self.prior = th.distributions.Beta(10, 10)

    def log_prob(self, parameters: Dict[str, th.Tensor]) -> th.Tensor:
        # Evaluate the probability of the bias `p` under the prior and the likelihood.
        return self.prior.log_prob(parameters["p"]) \
            + th.distributions.Bernoulli(parameters["p"][..., None]).log_prob(self.x).sum(axis=-1)


# Generate some data and initialize the model with an uninformative posterior. Instatiating with
# the prior could also be an option.
x = (th.arange(10) < 6).to(float)
model = CoinModel({
    "p": ParameterizedDistribution(th.distributions.Beta, concentration0=1, concentration1=1),
}, x)
model.check_log_prob_shape()
```

```{code-cell} ipython3
# Train the variational approximation.
batch_size = 100
optim = th.optim.Adam(model.parameters(), lr=0.1)

losses = []
for _ in range(1 if "CI" in os.environ else 2000):
    optim.zero_grad()
    loss = - model.batch_elbo_estimate((batch_size,)).mean()
    loss.backward()
    optim.step()
    losses.append(loss.item())
```

```{code-cell} ipython3
# Visualize loss and posterior.
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(losses)
ax1.set_ylabel("Negative ELBO")
ax1.set_xlabel("Iteration")

lin = th.linspace(0, 1, 100)
ax2.plot(lin, model.prior.log_prob(lin).exp(), label="prior")
ax2.plot(lin, model.distributions()["p"].log_prob(lin).exp().detach(), label="approx. posterior")
posterior = th.distributions.Beta(model.prior.concentration0 + (x == 1).sum(),
                                  model.prior.concentration1 + (x == 0).sum())
ax2.plot(lin, posterior.log_prob(lin).exp(), label="exact posterior")
ax2.axvline(x.mean(), color="k", ls="--", label="MLE")
ax2.set_xlabel("Coin bias $p$")
ax2.set_ylabel("Posterior density")
ax2.set_ylim(top=6.5)
ax2.legend(fontsize="small")
fig.tight_layout()
```
