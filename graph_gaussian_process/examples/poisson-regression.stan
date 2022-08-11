// Graph gaussian process with log link for Poisson observations.

functions {
    #include graph_gaussian_process.stan
}

data {
    // Information about nodes.
    int num_obs;
    int num_dims;
    array [num_obs] int y;
    array [num_obs] vector[num_dims] X;

    // Information about the graph.
    int num_edges;
    array [2, num_edges] int edge_index;

    // Kernel parameters.
    real<lower=0> alpha, rho, eps;
}

transformed data {
    array [num_obs] int degrees = in_degrees(num_obs, edge_index);
}

parameters {
    vector[num_obs] eta;
}

model {
    eta ~ ggp(X, alpha, rho, eps, edge_index, degrees);
    y ~ poisson_log(eta);
}
