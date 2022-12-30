// Graph gaussian process with log link for Poisson observations.

functions {
    #include gptools/util.stan
    #include gptools/fft1.stan
}

data {
    #include data.stan
}

parameters {
    vector[n] z;
}

transformed parameters {
    vector[n] eta;
    vector[n %/% 2 + 1] cov_rfft = gp_periodic_exp_quad_cov_rfft(n, sigma, length_scale, n)
        + epsilon;
    eta = gp_transform_inv_rfft(z, zeros_vector(n), cov_rfft);
}

model {
    // White noise prior (implies eta ~ gp_rfft(...) and observation model.
    z ~ normal(0, 1);
    y ~ poisson_log(eta);
}
