functions {
    #include gptools/util.stan
    #include gptools/fft1.stan
}

data {
    int n, p;
    array [n] int y;
    real<lower=0> epsilon;
}

transformed data {
    int m = n + p;
}

parameters {
    real<lower=0> sigma;
    real<lower=log(2), upper=log(n / 2.0)> log_length_scale;
    vector[m] raw;
}

transformed parameters {
    real length_scale = exp(log_length_scale);
    // Evaluate the covariance kernel in the Fourier domain. We add the nugget
    // variance because the Fourier transform of a delta function is a constant.
    vector[m %/% 2 + 1] cov_rfft = gp_periodic_exp_quad_cov_rfft(
        m, sigma, length_scale, m) + epsilon;
    // Transform the "raw" parameters to the latent log odds ratio.
    vector[m] z = gp_inv_rfft(raw, zeros_vector(m), cov_rfft);
}

model {
    // White noise implies that z ~ GaussianProcess(...).
    raw ~ normal(0, 1);
    y ~ bernoulli_logit(z[:n]);
    sigma ~ normal(0, 3);
    // Implicit log uniform prior on the length scale.
}

generated quantities {
    vector[n] proba = inv_logit(z[:n]);
}
