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

# Stan introduction

This notebook is a minimal introduction to Stan using linear regression as an example.

```{literalinclude} stan_intro.stan
    :language: stan
```

```{code-cell} ipython3
import numpy as np

np.random.seed(0)
n = 100
p = 3
X = np.random.normal(0, 1, (n, p))
theta = np.random.normal(0, 1, p)
sigma = np.random.gamma(2, 2)
y = np.random.normal(X @ theta, sigma)

print(f"coefficients: {theta.round(3)}")
print(f"observation noise scale: {sigma:.3f}")
```

```{code-cell} ipython3
import cmdstanpy

model = cmdstanpy.CmdStanModel(stan_file="stan_intro.stan")
fit = model.sample(data={"n": n, "p": p, "X": X, "y": y}, seed=0)
print(fit.diagnose())
print(fit.summary()[["5%", "50%", "95%"]].round(3))
```
