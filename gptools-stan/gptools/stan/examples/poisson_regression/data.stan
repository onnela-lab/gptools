// Shared data definition for Poisson regression models.

data {
    // Information about nodes.
    int n;
    int num_dims;
    array [n] int y;
    array [n] vector[num_dims] X;

    // Information about the graph.
    int num_edges;
    array [2, num_edges] int edge_index;

    // Kernel parameters.
    real<lower=0> alpha, rho, epsilon;
}
