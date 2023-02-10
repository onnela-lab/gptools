// Standard centered Gaussian process.

functions {
    #include gptools/util.stan
}

#include data.stan

parameters {
    vector[n] eta;
}

model {
    matrix[n, n] cov = add_diag(gp_exp_quad_cov(X, sigma, length_scale),
                                epsilon);
    eta ~ multi_normal(zeros_vector(n), cov);
    y[observed_idx] ~ normal(eta[observed_idx], noise_scale);
}
