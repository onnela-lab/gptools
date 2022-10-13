from . import ArrayOrTensor, ArrayOrTensorDispatch, OptionalArrayOrTensor


dispatch = ArrayOrTensorDispatch()


def evaluate_residuals(x: ArrayOrTensor, y: OptionalArrayOrTensor = None,
                       period: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Evaluate the residuals between points respecting periodic boundary conditions.

    If `period is not None` and boundary conditions apply, residuals have the correct "local"
    behavior, i.e., points to the left have a negative residual and points to the right have a
    positive residual. This leads to a discontinuity a distance `period / 2` from any reference
    point. The discontinuity is immaterial for even kernel functions.

    Args:
        x: Coordinates with shape `(..., p)`, where `...` is the batch shape and `p` is the number
            of dimensions of the embedding space.
        y: Coordinates with shape `(..., p)` which must be broadcastable to `x`. If not given, the
            distance between the Cartesian product of `x` will be evaluated.
        period: Period of circular boundary conditions.

    Returns:
        dist2: Squared distance between `x` and `y`.

    Example:
      .. plot::

        from gptools.util.kernels import evaluate_residuals
        from matplotlib import pyplot as plt
        import numpy as np

        width = 3  # Width of the domain.
        step_seq = [6, 7]  # Number of grid points in the domain.
        boundary_color = "silver"  # Color for domain boundaries.

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

        for ax, steps in zip(axes, step_seq):
            # Plot distances for two locations.
            x = np.linspace(-width, 2 * width, 3 * steps + 1, endpoint=True)
            ys = [x[steps + 2], x[2 * steps - 1]]
            for y in ys:
                line, = ax.plot(x, evaluate_residuals(x, y, width), marker=".")
                ax.scatter(y, 0, color=line.get_color(), label=f"$y={y:.1f}$")

            # Plot boundary indicators.
            for i in range(-1, 3):
                ax.axvline(i * width, color=boundary_color, ls="--")
            ax.axhline(width / 2, color=boundary_color, ls="--")
            ax.axhline(-width / 2, color=boundary_color, ls="--", label="domain boundaries")
            ax.axhline(0, color="silver", ls=":")

            ax.set_aspect("equal")
            ax.set_ylabel(r"residual $x - y$")
            ax.set_title(fr"$n={steps}$")

        axes[1].set_xlabel("position $x$")

        # Adjust boundaries.
        factor = 0.2
        ax.set_ylim(-(1 + factor) * width / 2, (1 + factor) * width / 2)
        ax.set_xlim(-(1 + factor / 2) * width, (2 + factor / 2) * width)
        fig.tight_layout()
    """
    # Expand the shape so we get the Cartesian product of elements in x (while keeping the batch
    # shape).
    if y is None:
        x, y = x[..., :, None, :], x[..., None, :, :]
    residuals = x - y
    if period is not None:
        residuals = residuals - period * dispatch.round(residuals / period)
    return residuals


def evaluate_squared_distance(x: ArrayOrTensor, y: OptionalArrayOrTensor = None,
                              period: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    r"""
    Evaluate the squared distance between points respecting periodic boundary conditions.

    Args:
        x: Coordinates with shape `(..., p)`, where `...` is the batch shape and `p` is the number
            of dimensions of the embedding space.
        y: Coordinates with shape `(..., p)` which must be broadcastable to `x`. If not given, the
            distance between the Cartesian product of `x` will be evaluated.
        period: Period of circular boundary conditions.

    Returns:
        dist2: Squared distance between `x` and `y`.

    Example:
      .. plot::

        from gptools.util import coordgrid
        from gptools.util.kernels import evaluate_squared_distance
        from matplotlib import pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        height = 40
        width = 50
        shape = (height, width)
        xs = coordgrid(np.arange(height), np.arange(width))

        idx = len(xs) // 4 + width // 4
        periods = [None, shape]
        dists = [np.sqrt(evaluate_squared_distance(xs[idx], xs, period=period))
                for period in periods]

        vmax = np.sqrt(width ** 2 + height ** 2)
        for ax, period, dist in zip(axes, periods, dists):
            dist = dist.reshape(shape)
            im = ax.imshow(dist, vmax=vmax, origin="lower")
            colorbar = fig.colorbar(im, ax=ax, location="top")
            label = r"distance $d\left(x,y\right)$"
            if period:
                label += " (periodic boundaries)"
            colorbar.set_label(label)
            cs = ax.contour(dist, colors="w", levels=[10, 20, 30], linestyles=["-", "--", ":"])
            plt.clabel(cs)
            ax.scatter(*np.unravel_index(idx, shape)[::-1], color="C1").set_edgecolor("w")
            ax.set_xlabel("position $x$")

        axes[0].set_ylabel("position $y$")
        fig.tight_layout()
    """
    residuals = evaluate_residuals(x, y, period)
    return (residuals * residuals).sum(axis=-1)


class Kernel:
    """
    Base class for covariance kernels.

    Args:
        epsilon: Diagonal "nugget" variance.
        period: Period for circular boundary conditions.
    """
    def __init__(self, epsilon: float = 0, period: OptionalArrayOrTensor = None):
        self.epsilon = epsilon
        self.period = period

    def __call__(self, x: ArrayOrTensor, y: OptionalArrayOrTensor = None) -> ArrayOrTensor:
        cov = self._evaluate(x, y)
        if self.epsilon and y is None:
            return cov + self.epsilon * dispatch[cov].eye(cov.shape[-1])
        return cov

    def _evaluate(self, x: ArrayOrTensor, y: OptionalArrayOrTensor = None) -> ArrayOrTensor:
        raise NotImplementedError

    @property
    def is_periodic(self):
        return self.period is not None


class ExpQuadKernel(Kernel):
    r"""
    Exponentiated quadratic kernel.

    .. math::

        \text{cov}\left(x, y\right) = \alpha^2 \exp\left(-\frac{\left(x-y\right)^2}{2\rho^2}\right)
        + \delta\left(x - y\right)

    Args:
        alpha: Scale of the covariance.
        rho: Correlation length.
        epsilon: Diagonal "nugget" variance.
        period: Period for circular boundary conditions.
    """
    def __init__(self, alpha: float, rho: float, epsilon: float = 0,
                 period: OptionalArrayOrTensor = None) -> None:
        super().__init__(epsilon, period)
        self.alpha = alpha
        self.rho = rho

    def _evaluate(self, x: ArrayOrTensor, y: OptionalArrayOrTensor = None) -> ArrayOrTensor:
        residuals = evaluate_residuals(x, y, self.period) / self.rho
        exponent = - dispatch.square(residuals).sum(axis=-1) / 2
        return self.alpha * self.alpha * dispatch.exp(exponent)
