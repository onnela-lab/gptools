/**
* Evaluate the in-degree of nodes in the directed acyclic graph induced by edges. The node labels of
* children must be ordered and each node must at least have a self loop. The function does not check
* whether the graph is actually acyclic.
*
* @param n Number of nodes.
* @param edge_index Directed edges as a tuple of parent and child indices, i.e. the node labelled
*   `edge_index[1, i]` is the parent of the node labelled `edge_index[2, i]`.
*
* @return In-degree of each node.
*/
array [] int in_degrees(int n, array [,] int edge_index) {
    array [n] int count = rep_array(0, n);
    int previous = 0;
    for (i in 1:size(edge_index[2])) {
        int current = edge_index[2, i];
        if (previous > current) {
            reject("nodes are not ordered: ", previous, " > ", current);
        }
        if (previous != current && edge_index[1, i] != current) {
            reject("first edge of node ", i, " is not a self loop");
        }
        if (previous > 0 && count[previous] < 1) {
            reject("node ", i, " has no edges");
        }
        count[current] += 1;
        previous = current;
    }
    if (previous != n) {
        reject("expected ", n, " nodes but found ", previous);
    }
    return count;
}

/**
* Evaluate the location and scale for a node given its parents.
*/
vector conditional_loc_scale(vector y, array[] vector x, real alpha, real rho, real epsilon,
                             array [] int predecessors) {
    vector[2] loc_scale;
    if (size(predecessors) == 1) {
        loc_scale[1] = 0;
        loc_scale[2] = sqrt(alpha ^ 2 + epsilon);
    } else {
        // Evaluate the local covariance.
        matrix[size(predecessors), size(predecessors)] cov = add_diag(
            gp_exp_quad_cov(x[predecessors], alpha, rho), epsilon);
        vector[size(predecessors) - 1] v = mdivide_left_spd(cov[2:, 2:], cov[2:, 1]);
        loc_scale[1] = dot_product(v, y[predecessors[2:]]);
        loc_scale[2] = sqrt(cov[1, 1] - dot_product(v, cov[2:, 1]));
    }
    return loc_scale;
}

/**
* Evaluate the log probability of a graph Gaussian process with zero mean. If a non-zero mean is
* required, it can be subtracted from `y`.
*
* @param y State of each node.
* @param x Position of each node.
* @param alpha Scale parameter for the covariance.
* @param rho Correlation length.
* @param epsilon Additional diagonal variance.
* @param edges Directed edges between nodes constituting a directed acyclic graph. Edges are stored
*   as a matrix with shape `(2, m)`, where `m` is the number of edges. The first row comprises
*   parents of children in the second row. The first row can have arbitrary order (except the first
*   edge of each node must be a self loop), but the second row must be sorted.
* @param degrees In-degree of each node.
*
* @return Log probability of the graph Gaussian process.
*/
real graph_gp_lpdf(vector y, array [] vector x, real alpha, real rho, real epsilon,
                   array [,] int edges, array[] int degrees) {
    real lpdf = 0;
    int offset_ = 1;
    for (i in 1:size(x)) {
        int in_degree = degrees[i];
        vector[2] loc_scale = conditional_loc_scale(y, x, alpha, rho, epsilon,
                                                    segment(edges[1], offset_, in_degree));
        lpdf += normal_lpdf(y[i] | loc_scale[1], loc_scale[2]);
        offset_ += in_degree;
    }
    return lpdf;
}


/**
* Transform white noise to a sample from a graph Gaussian process with zero mean. A mean can be
* added.
*
* @param z White noise for each node.
* @param x Position of each node.
* @param alpha Scale parameter for the covariance.
* @param rho Correlation length.
* @param epsilon Additional diagonal variance.
* @param edges Directed edges between nodes constituting a directed acyclic graph. Edges are stored
*   as a matrix with shape `(2, m)`, where `m` is the number of edges. The first row comprises
*   parents of children in the second row. The first row can have arbitrary order (except the first
*   edge of each node must be a self loop), but the second row must be sorted.
* @param degrees In-degree of each node.
*
* @return Sample from the Graph gaussian process.
*/
vector graph_gp_transform(vector z, array [] vector x, real alpha, real rho, real epsilon,
                          array [,] int edges, array [] int degrees) {
    vector[size(z)] y;
    int offset_ = 1;
    for (i in 1:size(x)) {
        int in_degree = degrees[i];
        vector[2] loc_scale = conditional_loc_scale(y, x, alpha, rho, epsilon,
                                                 segment(edges[1], offset_, in_degree));
        y[i] = loc_scale[1] + loc_scale[2] * z[i];
        offset_ += in_degree;
    }
    return y;
}

/**
* Evaluate the log probability of a Gaussian process in Fourier space.
*/
real fft_gp_lpdf(vector y, vector cov) {
    int n = size(y);
    int m = n %/% 2 + 1;
    // The last index of imaginary components to consider. This is necessary to distinguish between
    // the odd case (without Nyqvist frequency) and even (with Nyqvist frequency).
    int idx;
    // Evaluate the scale of Fourier coefficients.
    vector[m] fft_scale = sqrt(n * abs(fft(cov)[:m]) / 2);
    // The first element has larger scale because it only has a real part but must still have the
    // right variance. The same applies to the last element if the number of elements is even
    // (Nyqvist frequency).
    fft_scale[1] *= sqrt(2);
    if (n % 2 == 0) {
        fft_scale[m] *= sqrt(2);
        idx = m - 1;
    } else {
        idx = m;
    }
    complex_vector[m] fft = fft(y)[:m];
    return normal_lpdf(get_real(fft) | 0, fft_scale)
        + normal_lpdf(get_imag(fft[2:idx]) | 0, fft_scale[2:idx])
        - log(2) * ((n - 1) %/% 2) + n * log(n) / 2;
}
