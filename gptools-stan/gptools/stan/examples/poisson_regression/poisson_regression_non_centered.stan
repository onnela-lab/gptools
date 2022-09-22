// Gaussian process with log link for Poisson observations.

functions {
    #include gptools_kernels.stan
}

#include data.stan

parameters {
    vector[n] eta_;
}

transformed parameters {
    vector[n] eta;
    {
        matrix[n, n] cov = add_diag(gp_periodic_exp_quad_cov(X, alpha, rho, n), epsilon);
        eta = cholesky_decompose(cov) * eta_;
    }
}

model {
    eta_ ~ normal(0, 1);
    y ~ poisson_log(eta);
}
