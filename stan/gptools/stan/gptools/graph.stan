/**
Evaluate the out-degree of nodes in the directed acyclic graph induced by :code:`edge_index`. The
node labels of successors must be ordered and predecessors must have index less than successors.

:param n: Number of nodes.
:param edge_index: Directed edges as a tuple of predecessor and successor indices, i.e. the node
    labelled :code:`edge_index[1, i]` is the predecessor of the node labelled
    :code:`edge_index[2, i]`.
:returns: In-degree of each node.
*/
array [] int out_degrees(int n, array [,] int edge_index) {
    array [n] int count = rep_array(0, n);
    int previous = 0;
    for (i in 1:size(edge_index[2])) {
        int current = edge_index[2, i];
        if (previous > current) {
            reject("nodes are not ordered: ", previous, " > ", current, " at ", i);
        }
        if (edge_index[1, i] == current) {
            reject("self-loops are not allowed: ", current, " at ", i);
        }
        if (edge_index[1, i] > current) {
            reject("predecessor is greater than successor: ", edge_index[1, i], " > ", current,
                   " at ", i);
        }
        count[current] += 1;
        previous = current;
    }
    return count;
}


/**
Evaluate the location and scale for a node given its parents.

:param kernel: The kernel to use (0 for squared exponential, 1 for Matern 3/2, 2 for Matern 5/2).
:param y: State of :math:`k` parents.
:param x: Array of :math:`k + 1` locations in :math:`p` dimensions. The first row corresponds to
  the target node. Subsequent rows correspond to parents of the target node.
:param sigma: Amplitude of the covariance matrix.
:param length_scale: Correlation scale of the covariance matrix.
:param epsilon: Nugget variance.
:param predecessors: Labels of :math:`k` predecessors of the target node.
:returns: Location and scale parameters for the distribution of the node at :math:`x_1` given
  nodes :math:`j > 1`.
*/
vector gp_graph_conditional_loc_scale(int kernel, vector y, array[] vector x, real sigma, real
                                      length_scale, int node, array [] int predecessors,
                                      real epsilon) {
    int k = size(predecessors);
    matrix[1, k] cov12;
    matrix[k, k] cov22;
    if (kernel == 0) {
        cov12 = gp_exp_quad_cov({x[node]}, x[predecessors], sigma, length_scale);
        cov22 = gp_exp_quad_cov(x[predecessors], sigma, length_scale);
    } else if (kernel == 1) {
        cov12 = gp_matern32_cov({x[node]}, x[predecessors], sigma, length_scale);
        cov22 = gp_matern32_cov(x[predecessors], sigma, length_scale);
    } else if (kernel == 2) {
        cov12 = gp_matern52_cov({x[node]}, x[predecessors], sigma, length_scale);
        cov22 = gp_matern52_cov(x[predecessors], sigma, length_scale);
    } else {
        reject("invalid kernel indicator: ", kernel);
    }
    return gp_conditional_loc_scale(y[predecessors], sigma ^ 2 + epsilon, to_vector(cov12),
                                    add_diag(cov22, epsilon));
}


/**
Evaluate the log probability of a graph Gaussian process.

:param kernel: The kernel to use (0 for squared exponential, 1 for Matern 3/2, 2 for Matern 5/2).
:param y: State of each node.
:param mu: Mean for each node.
:param x: Position of each node.
:param sigma: Scale parameter for the covariance.
:param length_scale: Correlation length.
:param epsilon: Additional diagonal variance.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
  as a matrix with shape `(2, m)`, where `m` is the number of edges. The first row comprises
  parents of children in the second row. The first row can have arbitrary order (except the first
  edge of each node must be a self loop), but the second row must be sorted.
:param degrees: In-degree of each node.

:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_lpdf(vector y, int kernel, vector mu, array [] vector x, real sigma,
                   real length_scale, array [,] int edges, array[] int degrees, real epsilon) {
    real lpdf = 0;
    int offset_ = 1;
    vector[size(y)] z = y - mu;
    for (i in 1:size(x)) {
        vector[2] loc_scale = gp_graph_conditional_loc_scale(
            kernel, z, x, sigma, length_scale, i, segment(edges[1], offset_, degrees[i]), epsilon);
        lpdf += normal_lpdf(z[i] | loc_scale[1], loc_scale[2]);
        offset_ += degrees[i];
    }
    return lpdf;
}

/**
Evaluate the log probability of a graph Gaussian process with zero mean.

:param y: State of each node.
:param mu: Mean for each node.
:param x: Position of each node.
:param sigma: Scale parameter for the covariance.
:param length_scale: Correlation length.
:param epsilon: Additional diagonal variance.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
  as a matrix with shape `(2, m)`, where `m` is the number of edges. The first row comprises
  parents of children in the second row. The first row can have arbitrary order (except the first
  edge of each node must be a self loop), but the second row must be sorted.
:param degrees: In-degree of each node.

:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_exp_quad_cov_lpdf(vector y, vector mu, array [] vector x, real sigma,
                                real length_scale, array [,] int edges, array[] int degrees,
                                real epsilon) {
    return gp_graph_lpdf(y | 0, mu, x, sigma, length_scale, edges, degrees, epsilon);
}


/**
Transform white noise to a sample from a graph Gaussian process with zero mean.

:param z: White noise for each node.
:param mu: Mean for each node.
:param x: Position of each node.
:param sigma: Scale parameter for the covariance.
:param length_scale: Correlation length.
:param epsilon: Additional diagonal variance.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
  as a matrix with shape `(2, m)`, where `m` is the number of edges. The first row comprises
  parents of children in the second row. The first row can have arbitrary order (except the first
  edge of each node must be a self loop), but the second row must be sorted.
:param degrees: In-degree of each node.

:returns: Sample from the Graph gaussian process.
*/
vector gp_transform_inv_graph(vector z, int kernel, vector mu, array [] vector x, real sigma,
                                           real length_scale, array [,] int edges,
                                           array [] int degrees, real epsilon) {
    vector[size(z)] y;
    int offset_ = 1;
    for (i in 1:size(x)) {
        vector[2] loc_scale = gp_graph_conditional_loc_scale(
            kernel, y, x, sigma, length_scale, i, segment(edges[1], offset_, degrees[i]), epsilon);
        y[i] = loc_scale[1] + loc_scale[2] * z[i];
        offset_ += degrees[i];
    }
    return y + mu;
}

/**
Transform white noise to a sample from a graph Gaussian process with zero mean.

:param z: White noise for each node.
:param mu: Mean for each node.
:param x: Position of each node.
:param sigma: Scale parameter for the covariance.
:param length_scale: Correlation length.
:param epsilon: Additional diagonal variance.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
  as a matrix with shape `(2, m)`, where `m` is the number of edges. The first row comprises
  parents of children in the second row. The first row can have arbitrary order (except the first
  edge of each node must be a self loop), but the second row must be sorted.
:param degrees: Out-degree of each node.

:returns: Sample from the Graph gaussian process.
*/
vector gp_transform_inv_graph_exp_quad_cov(vector z, vector mu, array [] vector x, real sigma,
                                           real length_scale, array [,] int edges,
                                           array [] int degrees, real epsilon) {
    return gp_transform_inv_graph(z, 0, mu, x, sigma, length_scale, edges, degrees, epsilon);
}


vector gp_transform_inv_graph_exp_quad_cov(vector z, vector mu, array [] vector x, real sigma,
                                           real length_scale, array [,] int edges) {
    return gp_transform_inv_graph_exp_quad_cov(z, mu, x, sigma, length_scale, edges,
                                               out_degrees(size(z), edges), 0);
}
