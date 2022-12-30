import math
from typing import Optional
from .. import ArrayOrTensor, ArrayOrTensorDispatch, mutually_exclusive_kwargs, \
    OptionalArrayOrTensor
from .fft1 import pack_rfft, unpack_rfft
from .util import log2, log_prob_stdnorm


dispatch = ArrayOrTensorDispatch()


def _get_rfft2_scale(cov_rfft2: OptionalArrayOrTensor, cov: OptionalArrayOrTensor,
                     rfft2_scale: OptionalArrayOrTensor, width: Optional[int]) -> ArrayOrTensor:
    num_given = sum([cov_rfft2 is not None, cov is not None, rfft2_scale is not None])
    if num_given != 1:  # pragma: no cover
        raise ValueError("exactly one of `cov_rfft2`, `cov`, or `rfft2_scale` must be given")
    if rfft2_scale is not None:
        return rfft2_scale
    return evaluate_rfft2_scale(cov_rfft2=cov_rfft2, cov=cov, width=width)


@mutually_exclusive_kwargs("cov", "cov_rfft2")
def evaluate_rfft2_scale(*, cov_rfft2: OptionalArrayOrTensor = None,
                         cov: OptionalArrayOrTensor = None, width: Optional[int] = None) \
        -> ArrayOrTensor:
    """
    Evaluate the scale of Fourier coefficients.

    Args:
        cov_rfft2: Precomputed real fast Fourier transform of the kernel with shape
            `(..., height, width // 2 + 1)`.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., height, width)`.
        width: Number of columns of the signal (cannot be inferred from the Fourier coefficients).

    Returns:
        scale: Scale of Fourier coefficients with shape `(..., height, width // 2 + 1)`.
    """
    if cov is not None:
        *_, height, width = cov.shape
        cov_rfft2 = dispatch[cov].fft.rfft2(cov).real
    else:
        *_, height, _ = cov_rfft2.shape
    size = width * height
    rfft2_scale = size * cov_rfft2 / 2

    # Recall how the two-dimensional RFFT is computed. We first take an RFFT of rows of the matrix.
    # This leaves us with a real first column (zero frequency term) and a real last column if the
    # number of columns is even (Nyqvist frequency term). Second, we take a *full* FFT of the
    # columns. The first column will have a real coefficient in the first row (zero frequency in the
    # "row-dimension"). All elements in rows beyond n // 2 + 1 are irrelevant because the column was
    # real. The same applies to the last column if there is a Nyqvist frequency term. Finally, we
    # will also have a real-only Nyqvist frequency term in the first and last column if the number
    # of rows is even.

    # The zero-frequency term in both dimensions which must always be real.
    rfft2_scale[..., 0, 0] *= 2

    # If the number of colums is even, the last row will be real after the row-wise RFFT.
    # Consequently, the first element is real after the column-wise FFT.
    if width % 2 == 0:
        rfft2_scale[..., 0, width // 2] *= 2

    # If the number of rows is even, we have a Nyqvist frequency term in the first column.
    if height % 2 == 0:
        rfft2_scale[..., height // 2, 0] *= 2

    # If the number of rows and columns is even, we also have a Nyqvist frequency term in the last
    # column.
    if height % 2 == 0 and width % 2 == 0:
        rfft2_scale[..., height // 2, width // 2] *= 2

    return dispatch.sqrt(rfft2_scale)


def unpack_rfft2(z: ArrayOrTensor, shape: tuple[int]) -> ArrayOrTensor:
    """
    Unpack the Fourier coefficients of a two-dimensional real Fourier transform with shape
    `(..., height, width // 2 + 1)` to a batch of matrices with shape `(..., height, width)`.

    TODO: add details on packing structure.

    Args:
        z: Two-dimensional real Fourier transform coefficients.
        shape: Shape of the real signal. Necessary because the number of columns cannot be inferred
            from `rfft2`.

    Returns:
        z: Unpacked matrices with shape `(..., height, width)`.
    """
    *_, height, width = shape
    ncomplex = (width - 1) // 2
    parts = [
        # First column is always real.
        unpack_rfft(z[..., :height // 2 + 1, 0], height)[..., None],
        # Real and imaginary parts of complex coefficients.
        z[..., 1:ncomplex + 1].real,
        z[..., 1:ncomplex + 1].imag,
    ]
    if width % 2 == 0:  # Nyqvist frequency terms if the number of columns is even.
        parts.append(unpack_rfft(z[..., :height // 2 + 1, width // 2], height)[..., None])
    return dispatch.concatenate(parts, axis=-1)


def pack_rfft2(z: ArrayOrTensor) -> ArrayOrTensor:
    """
    Transform a batch of real matrices with shape `(..., height, width)` to a batch of complex
    Fourier coefficients with shape `(..., height, width // 2 + 1)` ready for inverse real fast
    Fourier transformation in two dimensions.

    Args:
        z: Unpacked matrices with shape `(..., height, width)`.

    Returns:
        rfft: Two-dimensional real Fourier transform coefficients.
    """
    *batch_shape, height, width = z.shape
    ncomplex = (width - 1) // 2
    rfft2 = dispatch[z].empty((*batch_shape, height, width // 2 + 1),
                              dtype=dispatch.get_complex_dtype(z))
    # Real FFT in the first column due to zero-frequency terms for the row-wise Fourier transform.
    rfft2[..., 0] = pack_rfft(z[..., 0], full_fft=True)
    # Complex Fourier coefficients.
    rfft2[..., 1:ncomplex + 1] = z[..., 1:ncomplex + 1] + 1j * z[..., ncomplex + 1:2 * ncomplex + 1]
    # Real FFT in the last column due to the Nyqvist frequency terms for the row-wise Fourier
    # transform if the number of columns is even.
    if width % 2 == 0:
        rfft2[..., width // 2] = pack_rfft(z[..., width - 1], full_fft=True)
    return rfft2


def transform_irfft2(z: ArrayOrTensor, loc: ArrayOrTensor, *,
                     cov_rfft2: OptionalArrayOrTensor = None, cov: OptionalArrayOrTensor = None,
                     rfft2_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Transform white noise in the Fourier domain to a Gaussian process realization.

    Args:
        z: Unpacked matrices with shape `(..., height, width)`. See :func:`unpack_rfft2` for
            details.
        loc: Mean of the Gaussian process with shape `(..., height, width)`.
        cov_rfft2: Precomputed real fast Fourier transform of the kernel with shape
            `(..., height, width // 2 + 1)`.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., height, width)`.
        rfft2_scale: Optional precomputed scale of Fourier coefficients with shape
            `(..., height, width // 2 + 1)`.

    Returns:
        y: Realization of the Gaussian process.
    """
    rfft2 = pack_rfft2(z) * _get_rfft2_scale(cov_rfft2, cov, rfft2_scale, z.shape[-1])
    return dispatch[rfft2].fft.irfft2(rfft2, z.shape[-2:]) + loc


def transform_rfft2(y: ArrayOrTensor, loc: ArrayOrTensor, *,
                    cov_rfft2: OptionalArrayOrTensor = None, cov: OptionalArrayOrTensor = None,
                    rfft2_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Transform a Gaussian process realization to white noise in the Fourier domain.

    Args:
        y: Realization of the Gaussian process.
        loc: Mean of the Gaussian process with shape `(..., height, width)`.
        cov_rfft2: Precomputed real fast Fourier transform of the kernel with shape
            `(..., height, width // 2 + 1)`.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., height, width)`.
        rfft2_scale: Optional precomputed scale of Fourier coefficients with shape
            `(..., height, width // 2 + 1)`.

    Returns:
        z: Unpacked matrices with shape `(..., height, width)`. See :func:`unpack_rfft2` for
            details.
    """
    rfft2_scale = _get_rfft2_scale(cov_rfft2, cov, rfft2_scale, y.shape[-1])
    return unpack_rfft2(dispatch[y].fft.rfft2(y - loc) / rfft2_scale, y.shape)


def evaluate_rfft2_log_abs_det_jacobian(width: int, *, cov_rfft2: OptionalArrayOrTensor = None,
                                        cov: OptionalArrayOrTensor = None,
                                        rfft2_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Evaluate the log absolute determinant of the Jacobian associated with :func:`transform_rfft2`.

    Args:
        width: Number of columns of the signal (cannot be inferred from the Fourier coefficients).
        cov_rfft2: Precomputed real fast Fourier transform of the kernel with shape
            `(..., height, width // 2 + 1)`.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., height, width)`.
        rfft2_scale: Optional precomputed scale of Fourier coefficients with shape
            `(..., height, width // 2 + 1)`.

    Returns:
        log_abs_det_jacobian: Log absolute determinant of the Jacobian.
    """
    rfft2_scale = _get_rfft2_scale(cov_rfft2, cov, rfft2_scale, width)
    height = rfft2_scale.shape[-2]
    assert rfft2_scale.shape[-1] == width // 2 + 1
    ncomplex_horizontal = (width - 1) // 2
    ncomplex_vertical = (height - 1) // 2
    parts = [
        # Real part of the first-column RFFT.
        - dispatch.log(rfft2_scale[..., :height // 2 + 1, 0]).sum(axis=-1),
        # Imaginary part of the first-column RFFT.
        - dispatch.log(rfft2_scale[..., 1:ncomplex_vertical + 1, 0]).sum(axis=-1),
        # Complex coefficients in subsequent columns.
        - 2 * dispatch.log(rfft2_scale[..., 1:1 + ncomplex_horizontal]).sum(axis=(-2, -1))
    ]
    # Account for Nyqvist frequencies in the last column if the number of columns is even.
    if width % 2 == 0:
        parts.extend([
            # Real part of the last-column RFFT.
            - dispatch.log(rfft2_scale[..., :height // 2 + 1, width // 2]).sum(axis=-1),
            # Imaginary part of the last-column RFFT.
            - dispatch.log(rfft2_scale[..., 1:ncomplex_vertical + 1, width // 2]).sum(axis=-1),
        ])
    size = width * height
    nterms = (size - 1) // 2
    if height % 2 == 0 and width % 2 == 0:
        nterms -= 1
    return sum(parts) - log2 * nterms + size * math.log(size) / 2


def evaluate_log_prob_rfft2(y: ArrayOrTensor, loc: ArrayOrTensor, *,
                            cov_rfft2: OptionalArrayOrTensor = None,
                            cov: OptionalArrayOrTensor = None,
                            rfft2_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Evaluate the log probability of a two-dimensional Gaussian process realization in Fourier space.

    Args:
        y: Realization of a Gaussian process with shape `(..., height, width)`, where `...` is the
            batch shape, `height` is the number of rows, and `width` is the number of columns.
        loc: Mean of the Gaussian process with shape `(..., height, width)`.
        cov_rfft2: Precomputed real fast Fourier transform of the kernel with shape
            `(..., height, width // 2 + 1)`.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., height, width)`.
        rfft2_scale: Optional precomputed scale of Fourier coefficients with shape
            `(..., height, width // 2 + 1)`.

    Returns:
        log_prob: Log probability of the Gaussian process realization with shape `(...)`.
    """
    rfft2_scale = _get_rfft2_scale(cov_rfft2, cov, rfft2_scale, y.shape[-1])
    rfft2 = transform_rfft2(y, loc, rfft2_scale=rfft2_scale)
    return log_prob_stdnorm(rfft2).sum(axis=(-2, -1)) \
        + evaluate_rfft2_log_abs_det_jacobian(y.shape[-1], rfft2_scale=rfft2_scale)
