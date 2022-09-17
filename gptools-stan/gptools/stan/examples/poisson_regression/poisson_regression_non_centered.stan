// Gaussian process with log link for Poisson observations.

functions {
    #include gptools_kernels.stan
}

data {
    // Information about nodes.
    int num_nodes;
    int num_dims;
    array [num_nodes] int y;
    array [num_nodes] vector[num_dims] X;

    // Kernel parameters.
    real<lower=0> alpha, rho, epsilon;
}

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
