// Gaussian process with log link for Poisson observations.

functions {
    #include gptools_util.stan
    #include gptools_kernels.stan
}

data {
    #include data.stan
}

parameters {
    vector[n] z;
}

transformed parameters {
    vector[n] eta;
    {
        // Evaluate the covariance and apply the Cholesky decomposition to the white noise z. We
        // wrap the evaluation in braces because Stan only writes top-level variables to the output
        // CSV files, and we don't need to store the entire covariance matrix.
        matrix[n, n] cov = add_diag(gp_periodic_exp_quad_cov(
            X, X, sigma, rep_vector(length_scale, 1), rep_vector(n, 1)), epsilon);
        eta = cholesky_decompose(cov) * z;
    }
}

model {
    // White noise prior (implies eta ~ multi_normal(zeros_vector(n), cov)) and observation model.
    z ~ normal(0, 1);
    y ~ poisson_log(eta);
}
