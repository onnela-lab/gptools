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
    vector[n] eta;
}

transformed parameters {
    // Evaluate covariance of the point at zero with everything else.
    vector[n] cov = gp_periodic_exp_quad_cov(zeros(1), X, sigma, length_scale, n);
    cov[1] += epsilon;
}

model {
    // Fourier Gaussian process and observation model.
    eta ~ gp_fft(zeros(n), cov);
    y ~ poisson_log(eta);
}
