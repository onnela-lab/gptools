// Graph gaussian process with log link for Poisson observations.

functions {
    #include gptools/util.stan
    #include gptools/fft1.stan
    #include gptools/kernels.stan
}

data {
    #include data.stan
}

parameters {
    vector[n] eta;
}

transformed parameters {
    // Evaluate covariance of the point at zero with everything else.
    vector[n %/% 2 + 1] cov_rfft = gp_periodic_exp_quad_cov_rfft(n, sigma, length_scale, n)
        + epsilon;
}

model {
    // Fourier Gaussian process and observation model.
    eta ~ gp_rfft(zeros_vector(n), cov_rfft);
    y ~ poisson_log(eta);
}
