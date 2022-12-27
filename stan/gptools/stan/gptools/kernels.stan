/**
Evaluate the residuals between two vectors, respecting circular boundary conditions.
*/
vector evaluate_residuals(vector x, vector y, vector period, vector scale) {
    if (min(scale) <= 0) {
        reject("distance scale factor must be positive");
    }
    if (min(period) <= 0) {
        reject("period for circular boundary conditions must be positive");
    }
    vector[size(x)] residual = abs(x - y);
    return fmin(residual, period - residual) ./ scale;
}

/**
Evaluate the squared distance between two vectors, respecting circular boundary conditions.

The squared distance is evaluated as

.. math::

    \Delta^2 = \sum_{i=1}^p \delta_i^2,

where :math:`\delta_i = \min\left(\left|x_i - y_i\right|, u_i - \left|x_i - y_i\right|\right)` is
the distance between :math:`x` and :math:`y` in the :math:`i^\text{th}` dimension after accounting
for periodic boundary conditions on the domain of size :math:`u_i`.

:param x: First vector with :math:`p` elements.
:param y: Second vector with :math:`p` elements.
:param period: Period of circular boundary conditions :math:`u` for each of :math:`p` dimension.

:returns: Squared distance between :math:`x` and :math:`y`.
*/
real dist2(vector x, vector y, vector period, vector scale) {
    vector[size(x)] residual = evaluate_residuals(x, y, period, scale);
    return residual' * residual;
}

/**
Evaluate the Cartesian product of squared distances between two arrays of coordinates.

:param x: First matrix with :math:`n` rows and :math:`p` columns.
:param y: Second matrix with :math:`m` rows and :math:`p` columns.
:param period: Period of circular boundary conditions for each dimension.

:returns: Cartesian product of squared distances with :math:`n` rows and :math:`m` columns.
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
Evaluate the Cartesian product of squared distances between elements of an array of coordinates.

:param x: Matrix with :math:`n` rows and :math:`p` columns.
:param period: Period of circular boundary conditions for each dimension.

:returns: Cartesian product of squared distances with :math:`n` rows and :math:`n` columns.
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
Evaluate the periodic squared exponential kernel.
*/
matrix gp_periodic_exp_quad_cov(array [] vector x1, array [] vector x2, real sigma,
                                vector length_scale, vector period) {
    int m = size(x1);
    int n = size(x2);
    return sigma * sigma * exp(-dist2(x1, x2, period, length_scale) / 2);
}

/**
Evaluate the real fast Fourier transform of the periodic squared exponential kernel.
*/
vector gp_periodic_exp_quad_cov_rfft(int n, real sigma, real length_scale, real period) {
    int nrfft = n %/% 2 + 1;
    return n * sigma ^ 2 * length_scale / period * sqrt(2 * pi())
        * exp(-2 * (pi() * linspaced_vector(nrfft, 0, nrfft - 1) * length_scale / period) ^ 2);
}

/**
Evaluate the two-dimensional real fast Fourier transform of the periodic squared exponential kernel.
*/
matrix gp_periodic_exp_quad_cov_rfft2(int m, int n, real sigma, vector length_scale, vector period) {
    vector[m %/% 2 + 1] rfftm = gp_periodic_exp_quad_cov_rfft(m, sigma, length_scale[1], period[1]);
    vector[n %/% 2 + 1] rfftn = gp_periodic_exp_quad_cov_rfft(n, 1, length_scale[2], period[2]);
    return get_real(expand_rfft(rfftm, m)) * rfftn';
}

/**
Evaluate the real fast Fourier transform of the periodic Matern kernel.
*/
vector gp_periodic_matern_cov_rfft(real dof, int n, real sigma, real length_scale, real period) {
    int nrfft = n %/% 2 + 1;
    vector[nrfft] k = linspaced_vector(nrfft, 0, nrfft - 1);
    return sigma ^ 2 * n * sqrt(2 * pi() / dof) * tgamma(dof + 0.5) / tgamma(dof)
        * (1 + 2 / dof * (pi() * length_scale / period * k) ^ 2) ^ -(dof + 0.5) * length_scale
        / period;
}


/**
Evaluate the real fast Fourier transform of the two-dimensional periodic Matern kernel.
*/
matrix gp_periodic_matern_cov_rfft2(real dof, int m, int n, real sigma, vector length_scale,
                                    vector period) {
    int nrfft = n %/% 2 + 1;
    matrix[m, nrfft] result;
    real ndim = 2;
    row_vector[nrfft] col_part = (linspaced_row_vector(nrfft, 0, nrfft - 1) * length_scale[2]
                                  / period[2]) ^ 2;
    // We only iterate up to m %/% 2 + 1 because the kernel is symmetric in positive and negative
    // frequencies.
    for (i in 1:m %/% 2 + 1) {
        int krow = i - 1;
        result[i] = 1 + 2 / dof * pi() ^ 2 * ((krow * length_scale[1] / period[1]) ^ 2 + col_part);
        if (i > 1) {
            result[m - i + 2] = result[i];
        }
    }
    return sigma ^ 2 * m * n * 2 ^ ndim * (pi() / (2 * dof)) ^ (ndim / 2)
        * tgamma(dof + ndim / 2) / tgamma(dof)
        * result .^ -(dof + ndim / 2) * prod(to_array_1d(length_scale ./ period));
}

matrix gp_periodic_matern_cov_rfft2(real dof, int m, int n, real sigma, real length_scale,
                                    vector period) {
    return gp_periodic_matern_cov_rfft2(dof, m, n, sigma, [length_scale, length_scale]', period);
}
