// Shared data definition for Poisson regression models.

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
