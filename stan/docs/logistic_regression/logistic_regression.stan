// Include the library. No further setup is required if you use the provided
// `gptools.stan.compile_model` wrapper for the cmdstanpy interface. If you use
// a different interface, the library path needs to be added to the
// `include_paths` of the interface (see
// https://mc-stan.org/docs/stan-users-guide/stanc-args.html for details). The
// library path can be obtained by running `python -m gptools.stan` from the
// command line.
functions {
    #include gptools/util.stan
    #include gptools/fft1.stan
}

data {
    int num_observations;
    array [num_observations] int y;
    // Padding to attenuate the effect of periodic boundary conditions inherent
    // to Fourier methods due to the use of fast Fourier transforms (see
    // https://gptools-stan.readthedocs.io/en/latest/docs/padding/padding.html
    // for details).
    int padding;
    // Nugget variance to ensure the kernel is positive-definite.
    real<lower=0> eps;
}

transformed data {
    // Number of grid points including the observations and padding.
    int n = num_observations + padding;
}

parameters {
    // Marginal scale and correlation length of the squared exponential kernel.
    real<lower=0> sigma;
    // "Raw" parameters used for a non-centered parameterization (see
    // https://mc-stan.org/docs/stan-users-guide/reparameterization.html for
    // details).
    vector[n] raw;
    // Bounded length scale to suppress scales less than the grid spacing and
    // larger than the domain. Parameters outside these bounds are not
    // identifiable (see
    // https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#322_Containment_Prior_Model
    // for details). Alternative approaches include inverse gamma priors, but
    // the hard bounds work for this simple example.
    real<lower=log(2), upper=log(num_observations / 2.0)> log_length_scale;
}

transformed parameters {
    real length_scale = exp(log_length_scale);
    // Evaluate the covariance kernel in the Fourier domain. We add the nugget
    // variance because the Fourier transform of a delta function is a constant.
    vector[n %/% 2 + 1] cov_rfft = gp_periodic_exp_quad_cov_rfft(
        n, sigma, length_scale, n) + eps;
    // Transform the "raw" parameters to the latent log odds ratio.
    vector[n] z = gp_inv_rfft(raw, zeros_vector(n), cov_rfft);
}

model {
    // White noise implies that z ~ GaussianProcess(...).
    raw ~ normal(0, 1);
    // Observation model.
    y ~ bernoulli_logit(z[:num_observations]);
    // Heavy-tailed half-Cauchy prior for the marginal scale.
    sigma ~ cauchy(0, 1);
    // Implicit log uniform prior on the length scale.
}

generated quantities {
    // Return the inferred probability curve.
    vector[num_observations] proba = inv_logit(z[:num_observations]);
}
