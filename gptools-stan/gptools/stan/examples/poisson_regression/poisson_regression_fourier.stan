// Graph gaussian process with log link for Poisson observations.

functions {
    #include gptools_fft.stan
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
    vector[num_nodes] eta;
}

transformed parameters {
    // Evaluate covariance of the point at zero with everything else.
    vector[num_nodes] cov = gp_exp_quad_cov(X, rep_array(rep_vector(0, 1), 1), alpha, rho, rep_vector(num_nodes, 1))[:, 1];
    cov[1] += epsilon;
}

model {
    eta ~ fft_gp(cov);
    y ~ poisson_log(eta);
}

generated quantities {
    matrix[num_nodes, num_nodes] full_cov = add_diag(gp_exp_quad_cov(X, X, alpha, rho, rep_vector(num_nodes, 1)), epsilon);
    real lpdf_fft = fft_gp_lpdf(eta | cov);
    real lpdf_multi_normal = multi_normal_lpdf(eta | rep_vector(0, num_nodes), full_cov);
}
