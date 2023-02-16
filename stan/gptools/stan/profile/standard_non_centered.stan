// Standard centered Gaussian process.

functions {
    #include gptools/util.stan
}

#include data.stan

parameters {
    vector[n] eta_;
}

transformed parameters {
    vector[n] eta;
    {
        matrix[n, n] chol = cholesky_decompose(
            add_diag(gp_exp_quad_cov(X, sigma, length_scale), epsilon));
        eta = chol * eta_;
    }
}

model {
    eta_ ~ normal(0, 1);
    y[observed_idx] ~ normal(eta[observed_idx], noise_scale);
}
