import numpy as np
import pytest
from typing import Type
from .kernels import ExpQuadKernel, Kernel, MaternKernel
from . import coordgrid, ArrayOrTensor


class KernelConfiguration:
    """
    Kernel configuration for testing.

    Args:
        dims: Sequence of dimension domains. `None` indicates an unbounded domain.
        kernel_cls: Class of the kernel to create.
        **kwargs: Keyword arguments passed to the kernel.
    """
    def __init__(self, dims: tuple, kernel_cls: Type[Kernel], **kwargs) -> None:
        if all(dims):
            dims = np.asarray(dims)
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
    KernelConfiguration([None], ExpQuadKernel, sigma=1.3, length_scale=0.2),
    KernelConfiguration([None, None, None], ExpQuadKernel, sigma=4, length_scale=0.2),
    KernelConfiguration([2, 3], ExpQuadKernel, sigma=1.7, length_scale=0.3),
    KernelConfiguration([1.5], ExpQuadKernel, sigma=1.5, length_scale=0.1),
    KernelConfiguration([2, 3, 4], ExpQuadKernel, sigma=2.1,
                        length_scale=np.asarray([0.1, 0.15, 0.2])),
    KernelConfiguration([None], MaternKernel, dof=3 / 2, sigma=1.3, length_scale=0.2),
    KernelConfiguration([None, None, None], MaternKernel, dof=3 / 2, sigma=4, length_scale=0.2),
    KernelConfiguration([None], MaternKernel, dof=5 / 2, sigma=1.3, length_scale=0.2),
    KernelConfiguration([None, None, None], MaternKernel, dof=5 / 2, sigma=4, length_scale=0.2),
]


@pytest.fixture(params=_kernel_configurations)
def kernel_configuration(request: pytest.FixtureRequest) -> KernelConfiguration:
    return request.param
