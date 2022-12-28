// IMPORTANT: stan uses the questionable R indexing which is one-based and inclusive on both ends.
// I.e., x[1:3] includes x[1] to x[3]. More generally, x[i:j] comprises j - i + 1 elements. It could
// at least have been exclusive on the right...

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


complex_matrix gp_pack_rfft2(matrix z) {
    int height = rows(z);
    int width = cols(z);
    int ncomplex = (width - 1) %/% 2;
    complex_matrix[height, width %/% 2 + 1] result;
    // Real FFT in the first column due to zero-frequency terms for the row-wise Fourier transform.
    result[:, 1] = expand_rfft(gp_pack_rfft(z[:, 1]), height);
    # Complex Fourier coefficients.
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
*/
matrix gp_transform_rfft2(matrix y, matrix loc, matrix cov_rfft2) {
    return gp_unpack_rfft2(rfft2(y - loc) ./ gp_evaluate_rfft2_scale(cov_rfft2, cols(y)), cols(y));
}


/**
Transform white noise in the Fourier domain to a Gaussian process realization.
*/
matrix gp_transform_inv_rfft2(matrix z, matrix loc, matrix cov_rfft2) {
    complex_matrix[rows(z), cols(z) %/% 2 + 1] y = gp_pack_rfft2(z)
        .* gp_evaluate_rfft2_scale(cov_rfft2, cols(z));
    return inv_rfft2(y, cols(z)) + loc;
}


/**
Evaluate the log absolute determinant of the Jacobian associated with :stan:func:`gp_transform_rfft`.
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
Evaluate the log probability of a two-dimensional Gaussian process with zero mean in Fourier space.

:param y: Random variable whose likelihood to evaluate.
:param loc: Mean of the Gaussian process.
:param cov_rfft2: Real fast Fourier transform of the covariance kernel.

:returns: Log probability of the Gaussian process.
*/
real gp_rfft2_lpdf(matrix y, matrix loc, matrix cov_rfft2) {
    return std_normal_lpdf(gp_transform_rfft2(y, loc, cov_rfft2))
        + gp_rfft2_log_abs_det_jacobian(cov_rfft2, cols(y));
}
