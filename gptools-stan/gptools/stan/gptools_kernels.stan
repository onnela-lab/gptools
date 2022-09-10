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
    return alpha ^ 2 * exp(- sqdist(x, y, period) / (2 * rho ^ 2));
}
