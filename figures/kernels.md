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

```{code-cell} ipython3
from gptools.util.kernels import ExpQuadKernel, MaternKernel
from gptools.util.fft.fft1 import transform_irfft, evaluate_rfft_scale
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path

workspace = Path(os.environ.get("WORKSPACE", os.getcwd()))


mpl.style.use("../jss.mplstyle")
```

```{code-cell} ipython3
np.random.seed(9)  # Seed picked for good legend positioning. Works for any though.
fig, axes = plt.subplots(2, 2)
length_scale = 0.2
kernels = {
    "squared exp.": lambda period: ExpQuadKernel(1, length_scale, period),
    "Matern ³⁄₂": lambda period: MaternKernel(1.5, 1, length_scale, period),
}

x = np.linspace(0, 1, 101, endpoint=False)
z = np.random.normal(0, 1, x.size)

for ax, (key, kernel) in zip(axes[1], kernels.items()):
    value = kernel(None).evaluate(0, x[:, None])
    line, = axes[0, 0].plot(x, value, ls="--")
    rfft = kernel(1).evaluate_rfft([x.size])
    value = np.fft.irfft(rfft, x.size)
    axes[0, 1].plot(rfft, label=key)
    axes[0, 0].plot(x, value, color=line.get_color())
    
    for maxf, ls in [(x.size // 2 + 1, "-"), (5, "--"), (3, ":")]:
        rfft_scale = evaluate_rfft_scale(cov_rfft=rfft, size=x.size)
        rfft_scale[maxf:] = 0
        f = transform_irfft(z, np.zeros_like(z), rfft_scale=rfft_scale)
        ax.plot(x, f, ls=ls, color=line.get_color(), label=fr"$\xi_\max={maxf}$")
        
    ax.set_xlabel("position $x$")
    ax.set_ylabel(f"{key} GP $f$")
    
ax = axes[0, 0]
ax.set_ylabel("kernel $k(0,x)$")
ax.set_xlabel("position $x$")
ax.legend([
    mpl.lines.Line2D([], [], ls="--", color="gray"),
    mpl.lines.Line2D([], [], ls="-", color="gray"),
], ["standard", "periodic"], fontsize="small")
ax.text(0.05, 0.05, "(a)", transform=ax.transAxes)
ax.yaxis.set_ticks([0, 0.5, 1])
    
ax = axes[0, 1]
ax.set_yscale("log")
ax.set_ylim(1e-5, x.size)
ax.set_xlabel(r"frequency $\xi$")
ax.set_ylabel(r"Fourier kernel $\tilde k=\phi\left(k\right)$")
ax.legend(fontsize="small", loc="center right")
ax.text(0.95, 0.95, "(b)", transform=ax.transAxes, ha="right", va="top")

ax = axes[1, 0]
ax.legend(fontsize="small", loc="lower center")
ax.text(0.95, 0.95, "(c)", transform=ax.transAxes, ha="right", va="top")

ax = axes[1, 1]
ax.legend(fontsize="small", loc="lower center")
ax.sharey(axes[1, 0])
ax.text(0.95, 0.95, "(d)", transform=ax.transAxes, ha="right", va="top")

for ax in [axes[0, 0], *axes[1]]:
    ax.xaxis.set_ticks([0, 0.5, 1])

fig.tight_layout()
fig.savefig(workspace / "kernels.pdf", bbox_inches="tight")
fig.savefig(workspace / "kernels.png", bbox_inches="tight")
```
