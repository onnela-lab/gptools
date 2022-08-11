// Graph gaussian process with log link for Poisson observations.

functions {
    #include graph_gaussian_process.stan
}

#include data.stan

parameters {
    vector[num_obs] eta;
}

model {
    eta ~ graph_gp(X, alpha, rho, eps, edge_index, degrees);
    y ~ poisson_log(eta);
}
