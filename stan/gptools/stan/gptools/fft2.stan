// IMPORTANT: stan uses the questionable R indexing which is one-based and inclusive on both ends.
// I.e., x[1:3] includes x[1] to x[3]. More generally, x[i:j] comprises j - i + 1 elements. It could
// at least have been exclusive on the right...

/**
Evaluate the scale of Fourier coefficients.

:param cov_rfft2: Precomputed real fast Fourier transform of the kernel with shape
    :code:`(height, width %/% 2 + 1)`.
:param width: Number of columns of the signal (cannot be inferred from the Fourier coefficients).
:returns: Scale of Fourier coefficients with shape :code:`(height, width %/% 2 + 1)`.
*/
matrix gp_evaluate_rfft2_scale(matrix cov_rfft2, int width) {
    int height = rows(cov_rfft2);
    int n = width * height;
    int fftwidth = width %/% 2 + 1;
    int fftheight = height %/% 2 + 1;
    matrix[height, fftwidth] fftscale = n * cov_rfft2 / 2;
    // Check positive-definiteness.
    real minval = min(fftscale);
    if (minval < 0) {
        reject("covariance matrix is not positive-definite (minimum eigenvalue is ", minval, ")");
    }

    // Adjust the scale for the zero-frequency (and Nyqvist) terms in the first column.
    fftscale[1, 1] *= 2;
    if (height % 2 == 0) {
        fftscale[fftheight, 1] *= 2;
    }
    // If the width is even, the last column has the same structure as the first column.
    if (width % 2 == 0) {
        fftscale[1, fftwidth] *= 2;
    }
    // If the number of rows and columns is even, we also have a Nyqvist frequency term in the last
    // column.
    if (width % 2 == 0 && height % 2 == 0) {
        fftscale[fftheight, fftwidth] *= 2;
    }
    return sqrt(fftscale);
}


/**
Unpack the complex Fourier coefficients of a two-dimensional real Fourier transform with shape to a
real matrix with shape :code:`(height, width)`.

TODO: add details on packing structure.

:param z: Two-dimensional real Fourier transform coefficients with shape
    :code:`(height, width %/% 2 + 1)`.
:param width: Number of columns of the signal (cannot be inferred from the Fourier coefficients
    :code:`z`).
:returns: Unpacked matrix with shape :code:`(height, width)`.
*/
matrix gp_unpack_rfft2(complex_matrix z, int m) {
    int height = rows(z);
    int n = m * height;
    int fftwidth = m %/% 2 + 1;
    int fftheight = height %/% 2 + 1;
    int wcomplex = (m - 1) %/% 2;

    matrix[height, m] result;

    // First column is always real.
    result[:, 1] = gp_unpack_rfft(z[:fftheight, 1], height);
    // Real and imaginary parts of complex coefficients.
    result[:, 2:wcomplex + 1] = get_real(z[:, 2:wcomplex + 1]);
    result[:, 2 + wcomplex:2 * wcomplex + 1] = get_imag(z[:, 2:wcomplex + 1]);
    // Nyqvist frequency if the number of columns is even.
    if (m % 2 == 0) {
        result[:, m] = gp_unpack_rfft(z[:fftheight, fftwidth], height);
    }
    return result;
}


/**
Transform a real matrix with shape :code:`(height, width)` to a matrix of complex Fourier
coefficients with shape :code:`(height, width %/% 2 + 1)` ready for inverse real fast Fourier
transformation in two dimensions.

:param z: Unpacked matrices with shape :code:`(height, width)`.
:returns: Two-dimensional real Fourier transform coefficients.
*/
complex_matrix gp_pack_rfft2(matrix z) {
    int height = rows(z);
    int width = cols(z);
    int ncomplex = (width - 1) %/% 2;
    complex_matrix[height, width %/% 2 + 1] result;
    // Real FFT in the first column due to zero-frequency terms for the row-wise Fourier transform.
    result[:, 1] = expand_rfft(gp_pack_rfft(z[:, 1]), height);
    // Complex Fourier coefficients.
    result[:, 2:ncomplex + 1] = z[:, 2:ncomplex + 1] + 1.0i * z[:, ncomplex + 2:2 * ncomplex + 1];
    // Real FFT in the last column due to the Nyqvist frequency terms for the row-wise Fourier
    // transform if the number of columns is even.
    if (width % 2 == 0) {
        result[:, width %/% 2 + 1] = expand_rfft(gp_pack_rfft(z[:, width]), height);
    }
    return result;
}


/**
Transform a Gaussian process realization to white noise in the Fourier domain.

:param y: Realization of the Gaussian process with shape :code:`(height, width)`.
:param loc: Mean of the Gaussian process with shape :code:`(height, width)`.
:param cov_rfft2: Precomputed real fast Fourier transform of the kernel with shape
    :code:`(height, width %/% 2 + 1)`.
:returns: Unpacked matrix with shape :code:`(height, width)`.
*/
matrix gp_rfft2(matrix y, matrix loc, matrix cov_rfft2) {
    return gp_unpack_rfft2(rfft2(y - loc) ./ gp_evaluate_rfft2_scale(cov_rfft2, cols(y)), cols(y));
}


/**
Transform white noise in the Fourier domain to a Gaussian process realization.

:param z: Unpacked matrix with shape :code:`(height, width)`.
:param loc: Mean of the Gaussian process with shape :code:`(height, width)`.
:param cov_rfft2: Precomputed real fast Fourier transform of the kernel with shape
    :code:`(height, width %/% 2 + 1)`.
:returns: Realization of the Gaussian process.
*/
matrix gp_inv_rfft2(matrix z, matrix loc, matrix cov_rfft2) {
    complex_matrix[rows(z), cols(z) %/% 2 + 1] y = gp_pack_rfft2(z)
        .* gp_evaluate_rfft2_scale(cov_rfft2, cols(z));
    return inv_rfft2(y, cols(z)) + loc;
}


/**
Evaluate the log absolute determinant of the Jacobian associated with
:stan:func:`gp_rfft2`.

:param cov_rfft2: Precomputed real fast Fourier transform of the kernel with shape
    :code:`(height, width %/% 2 + 1)`.
:param width: Number of columns of the signal (cannot be inferred from the precomputed kernel
    Fourier transform :code:`cov_rfft2`).
:returns: Log absolute determinant of the Jacobian.
*/
real gp_rfft2_log_abs_det_jacobian(matrix cov_rfft2, int width) {
    int height = rows(cov_rfft2);
    int n = width * height;
    int fftwidth = width %/% 2 + 1;
    int fftheight = height %/% 2 + 1;
    matrix[height, fftwidth] log_rfft2_scale = log(gp_evaluate_rfft2_scale(cov_rfft2, width));
    real ladj = 0;

    // For the real part, we always use the full height of the non-redundant part. For the imaginary
    // part, we discard the last element if the number of rows is even because it's the real Nyqvist
    // frequency.
    int idx = (height % 2) ? fftheight : fftheight - 1;
    ladj -= sum(log_rfft2_scale[:fftheight, 1]) + sum(log_rfft2_scale[2:idx, 1]);

    // Evaluate the "bulk" likelihood that needs no adjustment.
    ladj -= 2 * sum(to_vector(log_rfft2_scale[:, 2:fftwidth - 1]));

    if (width % 2) {
        // If the width is odd, the last column comprises all-independent terms.
        ladj -= 2 * sum(log_rfft2_scale[:, fftwidth]);
    } else {
        ladj -= sum(log_rfft2_scale[:fftheight, fftwidth]) + sum(log_rfft2_scale[2:idx, fftwidth]);
    }
    // Correction terms from the transform that only depend on the shape.
    int nterms = (n - 1) %/% 2;
    if (height % 2 == 0 && width % 2 == 0) {
        nterms -=1;
    }
    ladj += - log2() * nterms + n * log(n) / 2;
    return ladj;
}


/**
Evaluate the log probability of a two-dimensional Gaussian process realization in Fourier space.

:param y: Realization of a Gaussian process with shape :code:`(height, width)`, where :code:`height`
    is the number of rows, and :code:`width` is the number of columns.
:param loc: Mean of the Gaussian process with shape :code:`(height, width)`.
:param cov_rfft2: Precomputed real fast Fourier transform of the kernel with shape
    :code:`(height, width %/% 2 + 1)`.
:returns: Log probability of the Gaussian process realization.
*/
real gp_rfft2_lpdf(matrix y, matrix loc, matrix cov_rfft2) {
    return std_normal_lpdf(to_vector(gp_rfft2(y, loc, cov_rfft2)))
        + gp_rfft2_log_abs_det_jacobian(cov_rfft2, cols(y));
}


/**
Evaluate the two-dimensional real fast Fourier transform of the periodic squared exponential kernel.

:param height: Number of grid rows.
:param width: Number of grid columns.
:param sigma: Scale of the covariance.
:param length_scale: Correlation lengths in each dimension.
:param period: Periods for circular boundary conditions in each dimension.
:returns: Fourier transform of the periodic squared exponential kernel with shape
    :code:`(height, width %/% 2 + 1`).
*/
matrix gp_periodic_exp_quad_cov_rfft2(int height, int width, real sigma, vector length_scale,
                                      vector period) {
    vector[height %/% 2 + 1] rfftm = gp_periodic_exp_quad_cov_rfft(height, sigma, length_scale[1],
                                                                   period[1]);
    vector[width %/% 2 + 1] rfftn = gp_periodic_exp_quad_cov_rfft(width, 1, length_scale[2],
                                                                  period[2]);
    return get_real(expand_rfft(rfftm, height)) * rfftn';
}


/**
Evaluate the two-dimensional real fast Fourier transform of the periodic Matern kernel.

:param nu: Smoothness parameter (1 / 2, 3 / 2, and 5 / 2 are typical values).
:param height: Number of grid rows.
:param width: Number of grid columns.
:param sigma: Scale of the covariance.
:param length_scale: Correlation lengths in each dimension.
:param period: Periods for circular boundary conditions in each dimension.
:returns: Fourier transform of the periodic Matern kernel with shape
    :code:`(height, width %/% 2 + 1`).
*/
matrix gp_periodic_matern_cov_rfft2(real nu, int height, int width, real sigma, vector length_scale,
                                    vector period) {
    int nrfft = width %/% 2 + 1;
    matrix[height, nrfft] result;
    real ndim = 2;
    row_vector[nrfft] col_part = (linspaced_row_vector(nrfft, 0, nrfft - 1) * length_scale[2]
                                  / period[2]) ^ 2;
    // We only iterate up to height %/% 2 + 1 because the kernel is symmetric in positive and
    // negative frequencies.
    for (i in 1:height %/% 2 + 1) {
        int krow = i - 1;
        result[i] = 1 + 2 / nu * pi() ^ 2 * ((krow * length_scale[1] / period[1]) ^ 2 + col_part);
        if (i > 1) {
            result[height - i + 2] = result[i];
        }
    }
    return sigma ^ 2 * height * width * 2 ^ ndim * (pi() / (2 * nu)) ^ (ndim / 2)
        * tgamma(nu + ndim / 2) / tgamma(nu)
        * result .^ -(nu + ndim / 2) * prod(to_array_1d(length_scale ./ period));
}
