// General functions for graph Gaussian processes --------------------------------------------------

/**
Evaluate the out-degree of nodes in the directed acyclic graph induced by :code:`edge_index`. The
node labels of successors must be ordered and predecessors must have index less than successors.

:param n: Number of nodes.
:param edge_index: Directed edges as a tuple of predecessor and successor indices, i.e., the node
    labelled :code:`edge_index[1, i]` is the predecessor of the node labelled
    :code:`edge_index[2, i]`.
:returns: Out-degree of each node.
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
Evaluate the location and scale for a node given its predecessors, assuming zero mean.

:param y: State of each node.
:param x: Array of :code:`k + 1` locations in :code:`p` dimensions.
:param kernel: Kernel to use (0 for squared exponential, 1 for Matern 3/2, 2 for Matern 5/2).
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param predecessors: Labels of :code:`k` predecessors of the target node.
:param epsilon: Nugget variance for numerical stability.
:returns: Location and scale parameters for the distribution of the node given its predecessors.
*/
vector gp_graph_conditional_loc_scale(
    vector y, array[] vector x, int kernel, real sigma, array [] real length_scale,
    int node, array [] int predecessors, real epsilon
) {
    int k = size(predecessors);

    // If there are no predecessors, we simply use the marginal distribution.
    if (k == 0) {
        return [0, sqrt(sigma ^ 2 + epsilon)]';
    }

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

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param kernel: Kernel to use (0 for squared exponential, 1 for Matern 3/2, 2 for Matern 5/2).
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_lpdf(vector y, vector loc, array [] vector x, int kernel, real sigma,
    array [] real length_scale, array [,] int edges, array[] int degrees, real epsilon
) {
    real lpdf = 0;
    int offset_ = 1;
    vector[size(y)] z = y - loc;
    for (i in 1:size(x)) {
        vector[2] loc_scale = gp_graph_conditional_loc_scale(
            z, x, kernel, sigma, length_scale, i, segment(edges[1], offset_, degrees[i]), epsilon);
        lpdf += normal_lpdf(z[i] | loc_scale[1], loc_scale[2]);
        offset_ += degrees[i];
    }
    return lpdf;
}


/**
Evaluate the log probability of a graph Gaussian process.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param kernel: Kernel to use (0 for squared exponential, 1 for Matern 3/2, 2 for Matern 5/2).
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_lpdf(
    vector y,
    vector loc,
    array [] vector x,
    int kernel,
    real sigma,
    real length_scale,
    array [,] int edges,
    array[] int degrees,
    real epsilon
) {
    int p = size(x[1]);
    return gp_graph_lpdf(y | loc, x, kernel, sigma, rep_array(length_scale, p), edges, degrees, epsilon);
}


/**
Evaluate the log probability of a graph Gaussian process.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param kernel: Kernel to use (0 for squared exponential, 1 for Matern 3/2, 2 for Matern 5/2).
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_lpdf(vector y, vector loc, array [] vector x, int kernel, real sigma,
                   real length_scale, array [,] int edges) {
    return gp_graph_lpdf(y | loc, x, kernel, sigma, length_scale, edges,
                         out_degrees(size(y), edges), 1e-12);
}


/**
Evaluate the log probability of a graph Gaussian process.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param kernel: Kernel to use (0 for squared exponential, 1 for Matern 3/2, 2 for Matern 5/2).
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_lpdf(vector y, vector loc, array [] vector x, int kernel, real sigma,
                   array [] real length_scale, array [,] int edges) {
    return gp_graph_lpdf(y | loc, x, kernel, sigma, length_scale, edges,
                         out_degrees(size(y), edges), 1e-12);
}


/**
Transform white noise to a sample from a graph Gaussian process

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param kernel: Kernel to use (0 for squared exponential, 1 for Matern 3/2, 2 for Matern 5/2).
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph(vector z, vector loc, array [] vector x, int kernel, real sigma,
    array [] real length_scale, array [,] int edges, array [] int degrees, real epsilon
) {
    vector[size(z)] y;
    int offset_ = 1;
    for (i in 1:size(x)) {
        vector[2] loc_scale = gp_graph_conditional_loc_scale(
            y, x, kernel, sigma, length_scale, i, segment(edges[1], offset_, degrees[i]), epsilon);
        y[i] = loc_scale[1] + loc_scale[2] * z[i];
        offset_ += degrees[i];
    }
    return y + loc;
}


/**
Transform white noise to a sample from a graph Gaussian process

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param kernel: Kernel to use (0 for squared exponential, 1 for Matern 3/2, 2 for Matern 5/2).
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph(vector z, vector loc, array [] vector x, int kernel, real sigma,
                    real length_scale, array [,] int edges, array [] int degrees, real epsilon
) {
    int p = size(x[1]);
    return gp_inv_graph(z, loc, x, kernel, sigma, rep_array(length_scale, p), edges, degrees, epsilon);
}


/**
Transform white noise to a sample from a graph Gaussian process

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param kernel: Kernel to use (0 for squared exponential, 1 for Matern 3/2, 2 for Matern 5/2).
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph(vector z, vector loc, array [] vector x, int kernel, real sigma,
                    array [] real length_scale, array [,] int edges) {
    return gp_inv_graph(z, loc, x, kernel, sigma, length_scale, edges,
                                  out_degrees(size(z), edges), 1e-12);
}


/**
Transform white noise to a sample from a graph Gaussian process

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param kernel: Kernel to use (0 for squared exponential, 1 for Matern 3/2, 2 for Matern 5/2).
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph(vector z, vector loc, array [] vector x, int kernel, real sigma,
                    real length_scale, array [,] int edges) {
    int p = size(x[1]);
    return gp_inv_graph(z, loc, x, kernel, sigma, rep_array(length_scale, p), edges,
                                  out_degrees(size(z), edges), 1e-12);
}

// Functions for graph Gaussian processes with squared exponential kernel --------------------------

/**
Evaluate the log probability of a graph Gaussian with squared exponential kernel.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_exp_quad_cov_lpdf(vector y, vector loc, array [] vector x, real sigma,
                                real length_scale, array [,] int edges, array[] int degrees,
                                real epsilon) {
    return gp_graph_lpdf(y | loc, x, 0, sigma, length_scale, edges, degrees, epsilon);
}


/**
Evaluate the log probability of a graph Gaussian with squared exponential kernel.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_exp_quad_cov_lpdf(vector y, vector loc, array [] vector x, real sigma,
                                array [] real length_scale, array [,] int edges, array[] int degrees, real epsilon
) {
    return gp_graph_lpdf(y | loc, x, 0, sigma, length_scale, edges, degrees, epsilon);
}


/**
Evaluate the log probability of a graph Gaussian with squared exponential kernel.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_exp_quad_cov_lpdf(vector y, vector loc, array [] vector x, real sigma,
                                real length_scale, array [,] int edges) {
    return gp_graph_lpdf(y | loc, x, 0, sigma, length_scale, edges);
}


/**
Transform white noise to a sample from a graph Gaussian process with squared exponential kernel.

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph_exp_quad_cov(vector z, vector loc, array [] vector x, real sigma,
                                 real length_scale, array [,] int edges, array [] int degrees,
                                 real epsilon) {
    int p = size(x[1]);
    return gp_inv_graph(z, loc, x, 0, sigma, rep_array(length_scale, p), edges, degrees, epsilon);
}


/**
Transform white noise to a sample from a graph Gaussian process with squared
exponential kernel and different length scales along each feature dimension.

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation lengths of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic
    graph. Edges are stored as a matrix with shape :code:`(2, m)`, where
    :code:`m` is the number of edges. The first row comprises parents of
    children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph_exp_quad_cov(vector z, vector loc, array [] vector x, real sigma,
                                 array [] real length_scale, array [,] int edges, array [] int degrees, real epsilon
) {
    return gp_inv_graph(z, loc, x, 0, sigma, length_scale, edges, degrees, epsilon);
}


/**
Transform white noise to a sample from a graph Gaussian process with squared exponential kernel.

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph_exp_quad_cov(vector z, vector loc, array [] vector x, real sigma,
                                 real length_scale, array [,] int edges) {
    return gp_inv_graph(z, loc, x, 0, sigma, length_scale, edges);
}


/**
Transform white noise to a sample from a graph Gaussian process with squared exponential kernel.

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph_exp_quad_cov(vector z, vector loc, array [] vector x, real sigma,
                                 array [] real length_scale, array [,] int edges) {
    return gp_inv_graph(z, loc, x, 0, sigma, length_scale, edges);
}


// Functions for graph Gaussian processes with Matern 3 / 2 kernel ---------------------------------

/**
Evaluate the log probability of a graph Gaussian with Matern 3 / 2 kernel.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_matern32_cov_lpdf(vector y, vector loc, array [] vector x, real sigma,
                                real length_scale, array [,] int edges, array[] int degrees,
                                real epsilon) {
    return gp_graph_lpdf(y | loc, x, 1, sigma, length_scale, edges, degrees, epsilon);
}


/**
Evaluate the log probability of a graph Gaussian with Matern 3 / 2 kernel.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_matern32_cov_lpdf(vector y, vector loc, array [] vector x, real sigma,
                                real length_scale, array [,] int edges) {
    return gp_graph_lpdf(y | loc, x, 1, sigma, length_scale, edges);
}


/**
Evaluate the log probability of a graph Gaussian with Matern 3 / 2 kernel.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_matern32_cov_lpdf(vector y, vector loc, array [] vector x, real sigma,
                                array [] real length_scale, array [,] int edges, array[] int degrees,
                                real epsilon) {
    return gp_graph_lpdf(y | loc, x, 1, sigma, length_scale, edges, degrees, epsilon);
}


/**
Evaluate the log probability of a graph Gaussian with Matern 3 / 2 kernel.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_matern32_cov_lpdf(vector y, vector loc, array [] vector x, real sigma,
                                array [] real length_scale, array [,] int edges) {
    return gp_graph_lpdf(y | loc, x, 1, sigma, length_scale, edges);
}


/**
Transform white noise to a sample from a graph Gaussian process with Matern 3 / 2 kernel.

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph_matern32_cov(vector z, vector loc, array [] vector x, real sigma,
                                 real length_scale, array [,] int edges, array [] int degrees,
                                 real epsilon) {
    return gp_inv_graph(z, loc, x, 1, sigma, length_scale, edges, degrees, epsilon);
}


/**
Transform white noise to a sample from a graph Gaussian process with Matern 3 / 2 kernel.

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph_matern32_cov(vector z, vector loc, array [] vector x, real sigma,
                                 real length_scale, array [,] int edges) {
    return gp_inv_graph(z, loc, x, 1, sigma, length_scale, edges);
}


/**
Transform white noise to a sample from a graph Gaussian process with Matern 3 / 2 kernel.

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph_matern32_cov(vector z, vector loc, array [] vector x, real sigma,
                                 array [] real length_scale, array [,] int edges, array [] int degrees,
                                 real epsilon) {
    return gp_inv_graph(z, loc, x, 1, sigma, length_scale, edges, degrees, epsilon);
}


/**
Transform white noise to a sample from a graph Gaussian process with Matern 3 / 2 kernel.

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph_matern32_cov(vector z, vector loc, array [] vector x, real sigma,
                                 array [] real length_scale, array [,] int edges) {
    return gp_inv_graph(z, loc, x, 1, sigma, length_scale, edges);
}


// Functions for graph Gaussian processes with Matern 5 / 2 kernel ---------------------------------

/**
Evaluate the log probability of a graph Gaussian with Matern 5 / 2 kernel.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_matern52_cov_lpdf(vector y, vector loc, array [] vector x, real sigma,
                                real length_scale, array [,] int edges, array[] int degrees,
                                real epsilon) {
    return gp_graph_lpdf(y | loc, x, 2, sigma, length_scale, edges, degrees, epsilon);
}


/**
Evaluate the log probability of a graph Gaussian with Matern 5 / 2 kernel.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_matern52_cov_lpdf(vector y, vector loc, array [] vector x, real sigma,
                                real length_scale, array [,] int edges) {
    return gp_graph_lpdf(y | loc, x, 2, sigma, length_scale, edges);
}


/**
Evaluate the log probability of a graph Gaussian with Matern 5 / 2 kernel.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_matern52_cov_lpdf(vector y, vector loc, array [] vector x, real sigma,
                                array [] real length_scale, array [,] int edges, array[] int degrees,
                                real epsilon) {
    return gp_graph_lpdf(y | loc, x, 2, sigma, length_scale, edges, degrees, epsilon);
}


/**
Evaluate the log probability of a graph Gaussian with Matern 5 / 2 kernel.

:param y: State of each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Log probability of the graph Gaussian process.
*/
real gp_graph_matern52_cov_lpdf(vector y, vector loc, array [] vector x, real sigma,
                                array [] real length_scale, array [,] int edges) {
    return gp_graph_lpdf(y | loc, x, 2, sigma, length_scale, edges);
}


/**
Transform white noise to a sample from a graph Gaussian process with Matern 5 / 2 kernel.

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph_matern52_cov(vector z, vector loc, array [] vector x, real sigma,
                                 real length_scale, array [,] int edges, array [] int degrees,
                                 real epsilon) {
    return gp_inv_graph(z, loc, x, 2, sigma, length_scale, edges, degrees, epsilon);
}


/**
Transform white noise to a sample from a graph Gaussian process with Matern 5 / 2 kernel.

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph_matern52_cov(vector z, vector loc, array [] vector x, real sigma,
                                 real length_scale, array [,] int edges) {
    return gp_inv_graph(z, loc, x, 2, sigma, length_scale, edges);
}


/**
Transform white noise to a sample from a graph Gaussian process with Matern 5 / 2 kernel.

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:param degrees: Out-degree of each node.
:param epsilon: Nugget variance for numerical stability.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph_matern52_cov(vector z, vector loc, array [] vector x, real sigma,
                                 array [] real length_scale, array [,] int edges, array [] int degrees,
                                 real epsilon) {
    return gp_inv_graph(z, loc, x, 2, sigma, length_scale, edges, degrees, epsilon);
}


/**
Transform white noise to a sample from a graph Gaussian process with Matern 5 / 2 kernel.

:param z: White noise for each node.
:param loc: Mean of each node.
:param x: Position of each node.
:param sigma: Marginal scale of the kernel.
:param length_scale: Correlation length of the kernel for each dimension.
:param edges: Directed edges between nodes constituting a directed acyclic graph. Edges are stored
    as a matrix with shape :code:`(2, m)`, where :code:`m` is the number of edges. The first row
    comprises parents of children in the second row. The first row can have arbitrary order, but the
    second row must be sorted.
:returns: Sample from the Graph gaussian process.
*/
vector gp_inv_graph_matern52_cov(vector z, vector loc, array [] vector x, real sigma,
                                 array [] real length_scale, array [,] int edges) {
    return gp_inv_graph(z, loc, x, 2, sigma, length_scale, edges);
}
