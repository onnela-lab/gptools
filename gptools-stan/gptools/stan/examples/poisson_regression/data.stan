// Shared data definition for Poisson regression models.

data {
    // Information about nodes.
    int num_nodes;
    int num_dims;
    array [num_nodes] int y;
    array [num_nodes] vector[num_dims] X;

    // Information about the graph.
    int num_edges;
    array [2, num_edges] int edge_index;

    // Kernel parameters.
    real<lower=0> alpha, rho, epsilon;
}
