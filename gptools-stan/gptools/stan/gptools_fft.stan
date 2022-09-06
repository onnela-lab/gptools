/**
* Evaluate the log probability of a Gaussian process in Fourier space.
*
* @param y Random variable whose likelihood to evaluate.
* @param cov First row of the covariance matrix.
*
* @return Log probability of the Gaussian process.
*/
real fft_gp_lpdf(vector y, vector cov) {
    int n = size(y);
    int m = n %/% 2 + 1;
    // The last index of imaginary components to consider. This is necessary to distinguish between
    // the odd case (without Nyqvist frequency) and even (with Nyqvist frequency).
    int idx;
    // Evaluate the scale of Fourier coefficients.
    vector[m] fft_scale = sqrt(n * abs(fft(cov)[:m]) / 2);
    // The first element has larger scale because it only has a real part but must still have the
    // right variance. The same applies to the last element if the number of elements is even
    // (Nyqvist frequency).
    fft_scale[1] *= sqrt(2);
    if (n % 2 == 0) {
        fft_scale[m] *= sqrt(2);
        idx = m - 1;
    } else {
        idx = m;
    }
    complex_vector[m] fft = fft(y)[:m];
    return normal_lpdf(get_real(fft) | 0, fft_scale)
        + normal_lpdf(get_imag(fft[2:idx]) | 0, fft_scale[2:idx])
        - log(2) * ((n - 1) %/% 2) + n * log(n) / 2;
}


/**
* Evaluate the squared distance between two vectors, respecting circular boundary conditions.
*
* @param x First vector.
* @param y Second vector.
* @param period Period of circular boundary conditions for each dimension.
*
* @return Squared distance between x and y.
*/
real sqdist(vector x, vector y, vector period) {
        vector[size(x)] residual = abs(x - y);
        residual = fmin(residual, period - residual);
        return residual' * residual;
}


/**
* Evaluate the Cartesian product of squared distances between two arrays of coordinates.
*
* @param x First matrix with n rows and p columns.
* @param y Second matrix with m rows and p columns.
* @param period Period of circular boundary conditions for each dimension.
*
* @return Cartesian product of squared distances with n rows and m columns.
*/
matrix sqdist(array [] vector x, array [] vector y, vector period) {
    matrix[size(x), size(y)] result;
    for (i in 1:size(x)) {
        for (j in 1:size(y)) {
            result[i, j] = sqdist(x[i], y[j], period);
        }
    }
    return result;
}

/**
* Evaluate the squared exponential kernel with periodic boundary conditions.
*
* @param x First matrix with n rows and p columns.
* @param y Second matrix with m rows and p columns.
* @param alpha Amplitude of the covariance matrix.
* @param rho Correlation scale of the covariance matrix.
* @param period Period of circular boundary conditions for each dimension.
*
* @return Cartesian product of squared distances with n rows and m columns.
*/
matrix gp_exp_quad_cov(array [] vector x, array [] vector y, real alpha, real rho, vector period) {
    // TODO: should return cov_matrix instead of matrix.
    return alpha ^ 2 * exp(- sqdist(x, y, period) / (2 * rho ^ 2));
}
