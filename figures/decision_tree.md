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
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


mpl.style.use("../jss.mplstyle")
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.set_aspect(1)

offset = 2
xy = {
    (0, 0): ("how strong are\nthe data?", {"question": True}),
    (-1, -1): ("non-centered\nparameterization", {"rotate": True}),
    (0, -1): ("centered\nparameterization", {"rotate": True}),
    (1, -1): ("other method?", {"rotate": True}),

    (offset + 0.5, 0): ("is the GP on a\nregular grid?", {"question": True}),
    (offset, -1): ("Fourier\nmethods", {"rotate": True}),
    (offset + 1, -1): ("are the data\n\"dense\"?", {"question": True}),
    (offset + 0.5, -2): ("Fourier inducing\npoints", {"rotate": True}),
    (offset + 1.5, -2): ("structured\ndependencies", {"rotate": True}),
}
ax.scatter(*np.transpose(list(xy)))
for (x, y), (text, kwargs) in xy.items():
    kwargs = kwargs or {}
    rotate = kwargs.pop("rotate", False)
    question = kwargs.pop("question", False)
    kwargs = {
        "fontsize": "small",
        "ha": "center",
        "va": "top" if rotate else "center",
        "rotation": 90 if rotate else 0,
        "bbox": {
            "edgecolor": "none" if question else "k",
            "boxstyle": "round,pad=.5",
            "facecolor": "k" if question else "w",
        },
        "color": "w" if question else "k",
    } | kwargs
    ax.text(x, y, text, **kwargs)


def connect(xy1, xy2, frac=0.25, color="k", label=None):
    x1, y1 = xy1
    x2, y2 = xy2
    xy = [xy1, (x1, y1 - frac), (x2, y1 - frac), xy2]
    ax.plot(*np.transpose(xy), color=color, zorder=0, linewidth=1)
    if label:
        kwargs = {
            "fontsize": "small",
            "rotation": 90,
            "ha": "center",
            "va": "center",
            "bbox": {
                "edgecolor": "none",
                "facecolor": "w",
            }
        }
        ax.text(x2, (y1 - frac + y2) / 2, label, **kwargs)
    
connect((0, 0), (-1, -1), label="weak")
connect((0, 0), (0, -1), label="strong")
connect((0, 0), (1, -1), label="very\nstrong")

connect((offset + 0.5, 0), (offset, -1), label="yes")
connect((offset + 0.5, 0), (offset + 1, -1), label="no")
connect((offset + 1, -1), (offset + 0.5, -2), label="yes")
connect((offset + 1, -1), (offset + 1.5, -2), label="no")

ax.set_xlim(-1.25, 3.75)
ax.set_ylim(-3, 0.25)
ax.set_axis_off()
ax.text(-1, 0, "(a)", ha="center", va="center")
ax.text(1.8, 0, "(b)", ha="center", va="center")
fig.tight_layout()
fig.savefig("decision_tree.pdf", bbox_inches="tight")
fig.savefig("decision_tree.png", bbox_inches="tight")
```
