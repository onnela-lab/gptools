functions {
    #include gptools_util.stan
    #include gptools_fft.stan
    #include gptools_kernels.stan
}

data {
    // Number of rows and columns of the frequency map and the padded number of rows and columns. We
    // pad to overcome the periodic boundary conditions inherent in fast Fourier transform methods.
    int num_rows, num_cols, num_rows_padded, num_cols_padded;
    // Number of trees in each quadrant. Masked quadrants are indicated by a negative value.
    array [num_rows, num_cols] int frequency;
    // Nugget variance.
    real<lower=0> epsilon;
}

parameters {
    // "Raw" parameter for the non-centered parameterization.
    matrix[num_rows_padded, num_cols_padded] z;
    // Mean log rate for the trees.
    real mu;
    // Kernel parameters and averdispersion parameter for the negative binomial distribution.
    real<lower=0> sigma, length_scale, kappa;
}

transformed parameters {
    // Evaluate the RFFT of the Matern 3/2 kernel on the padded grid.
    matrix[num_rows_padded, num_cols_padded %/% 2 + 1] rfft2_cov = epsilon +
        gp_periodic_matern_cov_rfft2(1.5, num_rows_padded, num_cols_padded, sigma,
        length_scale, [num_rows_padded, num_cols_padded]');
    // Transform from white-noise to a Gaussian process realization.
    matrix[num_rows_padded, num_cols_padded] f = gp_transform_inv_rfft2(
        z, rep_matrix(mu, num_rows_padded, num_cols_padded), rfft2_cov);
}

model {
    // Observation model with masking for negative values.
    for (i in 1:num_rows) {
        for (j in 1:num_cols) {
            if (frequency[i, j] >= 0) {
                frequency[i, j] ~ neg_binomial_2(exp(f[i, j]), 1 / kappa);
            }
        }
    }
    // Implies that eta ~ gp_rfft(mu, rfft2_cov) is a realization of the Gaussian process.
    z ~ std_normal();
    // We bound the correlation length below because the likelihood is flat and place an upper bound
    // of 200m correlation which is broad enough to avoid the posterior samples hitting the
    // boundary.
    length_scale ~ uniform(1, 10);
    // Prior on the overall density based on Gelman's default prior paper (although we here use
    // count rather than binary regression).
    mu ~ cauchy(0, 2.5);
    // We want to allow a reasonable amount of variation but restrict excessively variable
    // functions. So we use slightly lighter tail than the Cauchy to constrain the sampler.
    sigma ~ student_t(2, 0, 1);
    // The overdispersion parameter again has a weak prior.
    kappa ~ cauchy(0, 1);
}
