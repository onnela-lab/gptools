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
                             array [] int neighborhood) {
    vector[2] loc_scale;
    if (size(neighborhood) == 1) {
        loc_scale[1] = 0;
        loc_scale[2] = sqrt(alpha ^ 2 + epsilon);
    } else {
        // Evaluate the local covariance.
        matrix[size(neighborhood), size(neighborhood)] cov = add_diag(
            gp_exp_quad_cov(x[neighborhood], alpha, rho), epsilon);
        vector[size(neighborhood) - 1] v = mdivide_left_spd(cov[2:, 2:], cov[2:, 1]);
        loc_scale[1] = dot_product(v, y[neighborhood[2:]]);
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
real ggp_lpdf(vector y, array [] vector x, real alpha, real rho, real epsilon, array [,] int edges,
              array[] int degrees) {
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
vector to_ggp(vector z, array [] vector x, real alpha, real rho, real epsilon, array [,] int edges,
              array [] int degrees) {
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
