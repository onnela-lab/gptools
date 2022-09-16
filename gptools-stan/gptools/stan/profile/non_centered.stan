// Standard centered Gaussian process.

#include data.stan

parameters {
    vector[num_nodes] eta_;
}

transformed parameters {
    vector[num_nodes] eta;
    {
        matrix[num_nodes, num_nodes] chol = cholesky_decompose(gp_exp_quad_cov(X, alpha, rho));
        eta = chol * eta_;
    }
}

model {
    eta_ ~ normal(0, 1);
    y ~ normal(eta, noise_scale);
}
