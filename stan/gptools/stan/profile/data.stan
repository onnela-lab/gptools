data {
    // Basic parameters.
    int n;
    int num_dims;
    array [n] vector[num_dims] X;
    vector[n] y;
    real<lower=0> sigma, length_scale, epsilon, noise_scale;

    // Only needed for graph-based Gaussian processes.
    int num_edges;
    array [2, num_edges] int edge_index;

    // Filtering training data.
    int num_observed;
    array [num_observed] int observed_idx;
}
