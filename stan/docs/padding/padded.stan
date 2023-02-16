functions {
    #include gptools/util.stan
    #include gptools/fft1.stan
}

data {
    int num_observations, padding, observe_first;
    int kernel;
    vector[num_observations] y;
    real<lower=0> kappa, length_scale, epsilon, sigma;
}

transformed data {
    int num = num_observations + padding;
    vector[num %/% 2 + 1] cov_rfft;
    if (kernel == 0) {
        cov_rfft = gp_periodic_exp_quad_cov_rfft(num, sigma, length_scale, num);
    } else if (kernel == 1) {
        cov_rfft = gp_periodic_matern_cov_rfft(1.5, num, sigma, length_scale, num);
    }
    cov_rfft += epsilon;
}

parameters {
    vector[num] z;
}

transformed parameters {
    vector[num] f = gp_inv_rfft(z, zeros_vector(num), cov_rfft);
}

model {
    z ~ normal(0, 1);
    y[:observe_first] ~ normal(f[:observe_first], kappa);
}
