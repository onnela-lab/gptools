// Standard centered Gaussian process.

#include data.stan

parameters {
    vector[num_nodes] eta;
}

model {
    matrix[num_nodes, num_nodes] cov = gp_exp_quad_cov(X, alpha, rho);
    eta ~ multi_normal(rep_vector(0, num_nodes), cov);
    y ~ normal(eta, noise_scale);
}
