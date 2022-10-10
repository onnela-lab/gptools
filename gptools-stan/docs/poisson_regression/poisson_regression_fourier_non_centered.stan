// Graph gaussian process with log link for Poisson observations.

functions {
    #include gptools_util.stan
    #include gptools_fft1.stan
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
        // Evaluate covariance of the point at zero with everything else and transform the white
        // noise. We wrap the evaluation in braces because Stan only writes top-level variables to
        // the output CSV files, and we don't need to store the entire covariance matrix.
        vector[n] cov = gp_periodic_exp_quad_cov(zeros(1), X, sigma, length_scale, n);
        cov[1] += epsilon;
        eta = gp_transform_irfft(z, zeros(n), cov);
    }
}

model {
    // White noise prior (implies eta ~ gp_fft(...) and observation model.
    z ~ normal(0, 1);
    y ~ poisson_log(eta);
}
