// Graph gaussian process with log link for Poisson observations.

functions {
    #include gptools_fft.stan
    #include gptools_kernels.stan
    #include gptools_util.stan
}

#include data.stan

parameters {
    vector[n] eta_;
}

transformed parameters {
    // Evaluate covariance of the point at zero with everything else.
    vector[n] cov = gp_periodic_exp_quad_cov(zeros(1), X, alpha, rho, n);
    vector[n] eta;
    cov[1] += epsilon;
    eta = fft_gp_transform(eta_, cov);
}

model {
    eta_ ~ normal(0, 1);
    y ~ poisson_log(eta);
}
