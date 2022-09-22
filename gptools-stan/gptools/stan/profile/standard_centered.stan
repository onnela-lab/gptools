// Standard centered Gaussian process.

#include data.stan

parameters {
    vector[n] eta;
}

model {
    matrix[n, n] cov = gp_exp_quad_cov(X, alpha, rho);
    eta ~ multi_normal(rep_vector(0, n), cov);
    y ~ normal(eta, noise_scale);
}
