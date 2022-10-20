// Standard centered Gaussian process.

functions {
    #include gptools_util.stan
}

#include data.stan

parameters {
    vector[n] eta_;
}

transformed parameters {
    vector[n] eta;
    {
        matrix[n, n] chol = cholesky_decompose(gp_exp_quad_cov(X, sigma, length_scale));
        eta = chol * eta_;
    }
}

model {
    eta_ ~ normal(0, 1);
    y ~ normal(eta, noise_scale);
}
