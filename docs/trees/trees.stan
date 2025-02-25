functions {
    #include gptools/util.stan
    #include gptools/fft.stan
}

data {
    // Number of rows and columns of the frequency map and the padded number of rows and columns. We
    // pad to overcome the periodic boundary conditions inherent in fast Fourier transform methods.
    int num_rows, num_cols, num_rows_padded, num_cols_padded;
    // Number of trees in each quadrant. Masked quadrants are indicated by a negative value.
    array [num_rows, num_cols] int frequency;
}

parameters {
    // "Raw" parameter for the non-centered parameterization.
    matrix[num_rows_padded, num_cols_padded] z;
    // Mean log rate for the trees.
    real loc;
    // Kernel parameters and averdispersion parameter for the negative binomial distribution.
    real<lower=0> sigma, kappa;
    real<lower=log(2), upper=log(28)> log_length_scale;
}

transformed parameters {
    real<lower=0> length_scale = exp(log_length_scale);
    // Evaluate the RFFT of the Matern 3/2 kernel on the padded grid.
    matrix[num_rows_padded, num_cols_padded %/% 2 + 1] rfft2_cov =
        gp_periodic_matern_cov_rfft2(1.5, num_rows_padded, num_cols_padded, sigma,
        [length_scale, length_scale]', [num_rows_padded, num_cols_padded]');
    // Transform from white-noise to a Gaussian process realization.
    matrix[num_rows_padded, num_cols_padded] f = gp_inv_rfft2(
        z, rep_matrix(loc, num_rows_padded, num_cols_padded), rfft2_cov);
}

model {
    // Implies that eta ~ gp_rfft(loc, rfft2_cov) is a realization of the Gaussian process.
    to_vector(z) ~ std_normal();
    // Weakish priors on all other parameters.
    loc ~ student_t(2, 0, 1);
    sigma ~ student_t(2, 0, 1);
    kappa ~ student_t(2, 0, 1);
    // Observation model with masking for negative values.
    for (i in 1:num_rows) {
        for (j in 1:num_cols) {
            if (frequency[i, j] >= 0) {
                frequency[i, j] ~ neg_binomial_2(exp(f[i, j]), 1 / kappa);
            }
        }
    }
}
