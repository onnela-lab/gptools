/**
Evaluate the in-degree of nodes in the directed acyclic graph induced by :code:`edges`. The node
labels of successors must be ordered and predecessors must have index less than successors.

:param n: Number of nodes.
:param edge_index: Directed edges as a tuple of predecessor and successor indices, i.e. the node
    labelled :code:`edge_index[1, i]` is the predecessor of the node labelled
    :code:`edge_index[2, i]`.

:returns: In-degree of each node.
*/
array [] int in_degrees(int n, array [,] int edge_index) {
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
Evaluate the conditional location and scale parameter of a univariate normal random variable given
correlated observations from a multivariate normal distribution.

:param y: Observation to condition on.
:param cov11: Marginal variance of the target random variable.
:param cov21: Covariance between :math:`y` and the target random variable.
:param cov22: Covariance amongst the elements of :math:`y`.

:returns: Location and scale as a vector.
*/
vector gp_graph_cls(vector y, real cov11, vector cov21, matrix cov22) {
    if (size(y) == 0) {
        return [0, sqrt(cov11)]';
    }
    vector[size(y)] v = mdivide_left_spd(cov22, cov21);
    return [dot_product(v, y), sqrt(cov11 - dot_product(v, cov21))]';
}

/**
Evaluate the conditional location and scale parameter of a univariate normal random variable given
correlated observations from a multivariate normal distribution. See
:stan:func:`gp_graph_cls(vector, real, vector, matrix)` for details.
*/
vector gp_graph_cls(vector y, real cov11, matrix cov21, matrix cov22) {
    return gp_graph_cls(y, cov11, to_vector(cov21), cov22);
}

/**
Evaluate the location and scale for a node given its parents.

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
vector gp_graph_exp_quad_cov_cls(vector y, array[] vector x, real sigma, real length_scale,
                                 int node, array [] int predecessors, real epsilon) {
    return gp_graph_cls(
        y[predecessors], sigma ^ 2 + epsilon,
        gp_exp_quad_cov({x[node]}, x[predecessors], sigma, length_scale),
        add_diag(gp_exp_quad_cov(x[predecessors], sigma, length_scale), epsilon)
    );
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
    real lpdf = 0;
    int offset_ = 1;
    vector[size(y)] z = y - mu;
    for (i in 1:size(x)) {
        vector[2] loc_scale = gp_graph_exp_quad_cov_cls(
            z, x, sigma, length_scale, i, segment(edges[1], offset_, degrees[i]), epsilon);
        lpdf += normal_lpdf(z[i] | loc_scale[1], loc_scale[2]);
        offset_ += degrees[i];
    }
    return lpdf;
}


/**
Transform white noise to a sample from a graph Gaussian process with zero mean. A mean can be
added after the transformation.

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
vector gp_transform_inv_graph_exp_quad_cov(vector z, vector mu, array [] vector x, real sigma,
                                           real length_scale, array [,] int edges,
                                           array [] int degrees, real epsilon) {
    vector[size(z)] y;
    int offset_ = 1;
    for (i in 1:size(x)) {
        vector[2] loc_scale = gp_graph_exp_quad_cov_cls(
            y, x, sigma, length_scale, i, segment(edges[1], offset_, degrees[i]), epsilon);
        y[i] = loc_scale[1] + loc_scale[2] * z[i];
        offset_ += degrees[i];
    }
    return y + mu;
}
