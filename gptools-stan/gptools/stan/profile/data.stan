data {
    int num_nodes;
    int num_dims;
    array [num_nodes] vector[num_dims] X;
    vector[num_nodes] y;
    real<lower=0> alpha, rho, epsilon, noise_scale;
    int num_edges;
    array [2, num_edges] int edge_index;
}
