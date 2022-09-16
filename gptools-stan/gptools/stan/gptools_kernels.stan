/**
* Evaluate the squared distance between two vectors, respecting circular boundary conditions.
*
* @param x First vector.
* @param y Second vector.
* @param period Period of circular boundary conditions for each dimension.
*
* @return Squared distance between x and y.
*/
real dist2(vector x, vector y, vector period, vector scale) {
        vector[size(x)] residual = abs(x - y);
        residual = fmin(residual, period - residual) ./ scale;
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
matrix dist2(array [] vector x, array [] vector y, vector period, vector scale) {
    matrix[size(x), size(y)] result;
    for (i in 1:size(x)) {
        for (j in 1:size(y)) {
            result[i, j] = dist2(x[i], y[j], period, scale);
        }
    }
    return result;
}

/**
* Evaluate the Cartesian product of squared distances between elements of an array of coordinates.
*
* @param x Matrix with n rows and p columns.
* @param period Period of circular boundary conditions for each dimension.
*
* @return Cartesian product of squared distances with n rows and n columns.
*/
matrix dist2(array [] vector x, vector period, vector scale) {
    matrix[size(x), size(x)] result;
    for (i in 1:size(x)) {
        result[i, i] = 0;
        for (j in i + 1:size(x)) {
            real z = dist2(x[i], x[j], period, scale);
            result[i, j] = z;
            result[j, i] = z;
        }
    }
    return result;
}

/**
* Evaluate the squared exponential kernel with periodic boundary conditions.
*
* @param x1 First matrix with n rows and p columns.
* @param x2 Second matrix with m rows and p columns.
* @param sigma Amplitude of the covariance matrix.
* @param length_scale Correlation scale of the covariance matrix.
* @param period Period of circular boundary conditions for each dimension.
*
* @return Cartesian product of squared distances with n rows and m columns.
*/
matrix gp_periodic_exp_quad_cov(array [] vector x1, array [] vector x2, real sigma,
                                vector length_scale, vector period) {
    return sigma * sigma * exp(- dist2(x1, x2, period, length_scale) / 2);
}

/**
* Evaluate the squared exponential kernel with periodic boundary conditions.
*
* @param x Matrix with n rows and p columns.
* @param sigma Amplitude of the covariance matrix.
* @param length_scale Correlation scale of the covariance matrix.
* @param period Period of circular boundary conditions for each dimension.
*
* @return Cartesian product of squared distances with n rows and n columns.
*/
matrix gp_periodic_exp_quad_cov(array [] vector x1, real sigma, vector length_scale,
                                vector period) {
    return sigma * sigma * exp(- dist2(x1, period, length_scale) / 2);
}
