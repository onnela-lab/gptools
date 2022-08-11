// Graph gaussian process with normal noise and non-centered parametrization.

functions {
    #include graph_gaussian_process.stan
}

#include data.stan

parameters {
    vector[num_nodes] eta_;
}

transformed parameters {
   vector[num_nodes] eta = to_ggp(eta_, X, alpha, rho, epsilon, edge_index, degrees);
}

model {
    eta_ ~ normal(0, 1);
    y ~ normal(eta, noise_scale);
}
