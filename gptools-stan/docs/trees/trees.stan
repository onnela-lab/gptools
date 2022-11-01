functions {
    #include gptools_util.stan
    #include gptools_fft.stan
    #include gptools_kernels.stan
}

data {
    // Number of rows and columns of the frequency map and the padded number of rows and columns. We
    // pad to overcome the periodic boundary conditions inherent in fast Fourier transform methods.
    int n, m, np, mp;
    // Number of trees in each quadrant. Masked quadrants are indicated by a negative value.
    array [n, m] int frequency;
    // Nugget variance.
    real<lower=0> epsilon;
}

parameters {
    // "Raw" parameter for the non-centered parameterization.
    matrix[np, mp] eta_;
    // Mean log rate for the trees.
    real mu;
    // Overdispersion parameter for the negative binomial distribution.
    real<lower=0> phi;
    // Kernel parameters.
    real<lower=0> sigma;
    real<lower=0> length_scale;
}

transformed parameters {
    // Evaluate the RFFT of the Matern 3/2 kernel on the padded grid.
    matrix[np, mp %/% 2 + 1] rfft2_cov = epsilon +
        gp_periodic_matern_cov_rfft2(1.5, np, mp, sigma, ones_vector(2) * length_scale, [np, mp]');
    // Transform from white-noise to a Gaussian process realization.
    matrix[np, mp] eta = gp_transform_irfft2(eta_, mu + zeros_matrix(np, mp),
                                             gp_evaluate_rfft2_scale(rfft2_cov, mp));
}

model {
    // Observation model with masking for negative values.
    for (i in 1:n) {
        for (j in 1:m) {
            if (frequency[i, j] >= 0) {
                frequency[i, j] ~ neg_binomial_2(exp(eta[i, j]), 1 / phi);
            }
        }
    }
    // Implies that eta ~ gp_rfft(mu, rfft2_cov) is a realization of the Gaussian process.
    eta_ ~ std_normal();
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
    phi ~ cauchy(0, 1);
}
