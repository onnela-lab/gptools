// Graph gaussian process with log link for Poisson observations.

functions {
    #include gptools_util.stan
    #include gptools_fft1.stan
    #include gptools_kernels.stan
}

data {
    #include data.stan
}

parameters {
    vector[n] z;
}

transformed parameters {
    vector[n] eta;
    vector[n %/% 2 + 1] cov_rfft = gp_periodic_exp_quad_cov_rfft(n, sigma, length_scale, n, 10)
        + epsilon;
    eta = gp_transform_irfft(z, zeros_vector(n), gp_evaluate_rfft_scale(cov_rfft, n));
}

model {
    // White noise prior (implies eta ~ gp_rfft(...) and observation model.
    z ~ normal(0, 1);
    y ~ poisson_log(eta);
}
