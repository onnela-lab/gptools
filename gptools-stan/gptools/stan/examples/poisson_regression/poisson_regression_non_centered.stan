// Gaussian process with log link for Poisson observations.

functions {
    #include gptools_kernels.stan
}

#include data.stan

parameters {
    vector[num_nodes] eta_;
}

transformed parameters {
    vector[num_nodes] eta;
    {
        matrix[num_nodes, num_nodes] cov = add_diag(gp_periodic_exp_quad_cov(X, alpha, rep_vector(rho, 1), rep_vector(num_nodes, 1)), epsilon);
        eta = cholesky_decompose(cov) * eta_;
    }
}

model {
    eta_ ~ normal(0, 1);
    y ~ poisson_log(eta);
}
