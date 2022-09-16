import numpy as np
import pytest
import typing
from .kernels import ExpQuadKernel, Kernel
from . import coordgrid, ArrayOrTensor


class KernelConfiguration:
    """
    Kernel configuration for testing.

    Args:
        dims: Sequence of dimension domains. `None` indicates an unbounded domain.
        kernel_cls: Class of the kernel to create.
        **kwargs: Keyword arguments passed to the kernel.
    """
    def __init__(self, dims: tuple, kernel_cls: typing.Type[Kernel], **kwargs) -> None:
        self.dims = dims
        self.kernel_cls = kernel_cls
        self.kwargs = kwargs
        if all(self.dims):
            self.kwargs.setdefault("period", self.dims)

    def __call__(self) -> Kernel:
        return self.kernel_cls(**self.kwargs)

    def sample_locations(self, size: tuple = None) -> ArrayOrTensor:
        """
        Sample locations consistent with the domain on which to apply the kernel.
        """
        locations = []
        for dim in self.dims:
            if dim is None:
                locations.append(np.random.normal(0, 1, size))
            else:
                domain = dim
                locations.append(np.random.uniform(0, domain, size))
        locations = np.asarray(locations)
        return np.moveaxis(locations, 0, -1)

    def coordgrid(self, shape) -> ArrayOrTensor:
        """
        Create a coordinate grid on which to evaluate the kernel.
        """
        lins = []
        for dim, n in zip(self.dims, shape):
            if dim is None:  # pragma: no cover
                raise ValueError("cannot create coordinate grid for unbounded support")
            lins.append(np.linspace(0, dim, n, endpoint=False))
        return coordgrid(*lins)


_kernel_configurations = [
    KernelConfiguration([None], ExpQuadKernel, alpha=1.3, rho=0.2),
    KernelConfiguration([None, None, None], ExpQuadKernel, alpha=4, rho=0.2, epsilon=0.1),
    KernelConfiguration([2, 3], ExpQuadKernel, alpha=1.7, rho=0.3, epsilon=0.2),
    KernelConfiguration([1.5], ExpQuadKernel, alpha=1.5, rho=0.1),
    KernelConfiguration([2, 3, 4], ExpQuadKernel, alpha=2.1, rho=np.asarray([0.1, 0.15, 0.2])),
]


@pytest.fixture(params=_kernel_configurations)
def kernel_configuration(request: pytest.FixtureRequest) -> KernelConfiguration:
    return request.param
