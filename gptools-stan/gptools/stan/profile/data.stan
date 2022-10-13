data {
    int n;
    int num_dims;
    array [n] vector[num_dims] X;
    vector[n] y;
    real<lower=0> sigma, length_scale, epsilon, noise_scale;
    int num_edges;
    array [2, num_edges] int edge_index;
}
