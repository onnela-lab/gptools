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
    // For the real part, we always use the full height of the non-redundant part. For the imaginary
    // part, we discard the last element if the number of rows is even because it's the real Nyqvist
    // frequency.
    int idx = (height % 2) ? fftheight : fftheight - 1;

    if (width % 2 == 0) {
        // If the width is even, the last column has the same structure as the first column.
        fftscale[1, fftwidth] *= 2;
        if (height % 2 == 0) {
            fftscale[fftheight, fftwidth] *= 2;
        }
    }
    return sqrt(fftscale);
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
    int height = rows(y);
    int width = cols(y);
    int n = width * height;
    int fftwidth = width %/% 2 + 1;
    int fftheight = height %/% 2 + 1;

    // Evaluate the Fourier coefficients and their scale. We divide the latter by two to account for
    // real and imaginary parts.
    matrix[height, fftwidth] fftscale = gp_evaluate_rfft2_scale(cov);
    complex_matrix[height, fftwidth] ffty = rfft2(y - loc) ./ fftscale;
    matrix[height, fftwidth] fftreal = get_real(ffty);
    matrix[height, fftwidth] fftimag = get_imag(ffty);


    // For the real part, we always use the full height of the non-redundant part. For the imaginary
    // part, we discard the last element if the number of rows is even because it's the real Nyqvist
    // frequency.
    int idx = (height % 2) ? fftheight : fftheight - 1;
    real log_prob = std_normal_lpdf(fftreal[:fftheight, 1]) + std_normal_lpdf(fftimag[2:idx, 1]);

    // Evaluate the "bulk" likelihood that needs no adjustment.
    log_prob += std_normal_lpdf(to_vector(ffty[:, 2:fftwidth - 1]));

    if (width % 2) {
        // If the width is odd, the last column comprises all-independent terms.
        log_prob += std_normal_lpdf(ffty[:, fftwidth]);
    } else {
        log_prob += std_normal_lpdf(fftreal[:fftheight, fftwidth])
            + std_normal_lpdf(fftimag[2:idx, fftwidth]);
    }

    return log_prob + gp_rfft2_log_abs_det_jacobian(cov, fftscale);
}
