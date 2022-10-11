// IMPORTANT: stan uses the questionable R indexing which is one-based and inclusive on both ends.
// I.e., x[1:3] includes x[1] to x[3]. More generally, x[i:j] comprises j - i + 1 elements. It could
// at least have been exclusive on the right...

/**
Evaluate the scale of Fourier coefficients.
*/
matrix gp_evaluate_rfft2_scale(matrix cov) {
    int height = rows(cov);
    int width = cols(cov);
    int n = width * height;
    int fftwidth = width %/% 2 + 1;
    int fftheight = height %/% 2 + 1;
    matrix[height, fftwidth] fftscale = n * get_real(rfft2(cov)) / 2;

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
Transform a Gaussian process realization to white noise in the Fourier domain.
*/
matrix gp_transform_rfft2(matrix y, matrix loc, matrix cov, matrix rfft2_scale) {
    int height = rows(y);
    int width = cols(y);
    int n = width * height;
    int fftwidth = width %/% 2 + 1;
    int fftheight = height %/% 2 + 1;
    int wcomplex = (width - 1) %/% 2;

    complex_matrix[height, fftwidth] ffty = rfft2(y - loc) ./ rfft2_scale;
    matrix[height, width] z;

    // First column is always real.
    z[:, 1] = gp_unpack_rfft(ffty[:fftheight, 1], height);
    // Real and imaginary parts of complex coefficients.
    z[:, 2:wcomplex + 1] = get_real(ffty[:, 2:wcomplex + 1]);
    z[:, 2 + wcomplex:2 * wcomplex + 1] = get_imag(ffty[:, 2:wcomplex + 1]);
    // Nyqvist frequency if the number of columns is even.
    if (width % 2 == 0) {
        z[:, width] = gp_unpack_rfft(ffty[:fftheight, fftwidth], height);
    }
    return z;
}


/**
Transform a Gaussian process realization to white noise in the Fourier domain.
*/
matrix gp_transform_rfft2(matrix y, matrix loc, matrix cov) {
    return gp_transform_rfft2(y, loc, cov, gp_evaluate_rfft2_scale(cov));
}


/**
Evaluate the log absolute determinant of the Jacobian associated with :cpp:func:`gp_transform_rfft`.
*/
real gp_rfft2_log_abs_det_jacobian(matrix cov, matrix fftscale) {
    int height = rows(cov);
    int width = cols(cov);
    int n = width * height;
    int fftwidth = width %/% 2 + 1;
    int fftheight = height %/% 2 + 1;
    matrix[height, fftwidth] logfftscale = log(fftscale);
    real ladj = 0;

    // For the real part, we always use the full height of the non-redundant part. For the imaginary
    // part, we discard the last element if the number of rows is even because it's the real Nyqvist
    // frequency.
    int idx = (height % 2) ? fftheight : fftheight - 1;
    ladj -= sum(logfftscale[:fftheight, 1]) + sum(logfftscale[2:idx, 1]);

    // Evaluate the "bulk" likelihood that needs no adjustment.
    ladj -= 2 * sum(to_vector(logfftscale[:, 2:fftwidth - 1]));

    if (width % 2) {
        // If the width is odd, the last column comprises all-independent terms.
        ladj -= 2 * sum(logfftscale[:, fftwidth]);
    } else {
        ladj -= sum(logfftscale[:fftheight, fftwidth]) + sum(logfftscale[2:idx, fftwidth]);
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
Evaluate the log absolute determinant of the Jacobian associated with :cpp:func:`gp_transform_rfft`.
*/
real gp_rfft2_log_abs_det_jacobian(matrix cov) {
    return gp_rfft2_log_abs_det_jacobian(cov, gp_evaluate_rfft2_scale(cov));
}


/**
Evaluate the log probability of a two-dimensional Gaussian process with zero mean in Fourier space.

:param y: Random variable whose likelihood to evaluate.
:param loc: Mean of the Gaussian process.
:param cov: First row of the covariance matrix.

:returns: Log probability of the Gaussian process.
*/
real gp_rfft2_lpdf(matrix y, matrix loc, matrix cov) {
    matrix[rows(cov), cols(cov) %/% 2 + 1] rfft2_scale = gp_evaluate_rfft2_scale(cov);
    return std_normal_lpdf(gp_transform_rfft2(y, loc, cov, rfft2_scale))
        + gp_rfft2_log_abs_det_jacobian(cov, rfft2_scale);
}
