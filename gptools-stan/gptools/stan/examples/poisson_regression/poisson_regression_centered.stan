// Gaussian process with log link for Poisson observations.

functions {
    #include gptools_kernels.stan
}

#include data.stan

parameters {
    vector[n] eta;
}

model {
    matrix[n, n] cov = add_diag(gp_periodic_exp_quad_cov(X, alpha, rho, n), epsilon);
    eta ~ multi_normal(rep_vector(0, n), cov);
    y ~ poisson_log(eta);
}
