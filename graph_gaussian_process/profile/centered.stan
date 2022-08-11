// Graph gaussian process with normal noise and centered parametrization.

functions {
    #include graph_gaussian_process.stan
}

#include data.stan

parameters {
    vector[num_nodes] eta;
}

model {
    eta ~ ggp(X, alpha, rho, epsilon, edge_index, degrees);
    y ~ normal(eta, noise_scale);
}
