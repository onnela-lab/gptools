// Graph gaussian process with log link for Poisson observations.

functions {
    #include gptools.stan

    real sqdist(vector x, vector y, vector period) {
        vector[size(x)] residual = abs(x - y);
        residual = fmin(residual, period - residual);
        return residual' * residual;
    }


    matrix sqdist(array [] vector x, array [] vector y, vector period) {
        matrix[size(x), size(y)] result;
        for (i in 1:size(x)) {
            for (j in 1:size(y)) {
                result[i, j] = sqdist(x[i], y[j], period);
            }
        }
        return result;
    }


    matrix gp_exp_quad_cov(array [] vector x, array [] vector y, real alpha, real rho, vector period) {
        // TODO: should return cov_matrix instead of matrix.
        return alpha ^ 2 * exp(- sqdist(x, y, period) / (2 * rho ^ 2));
    }
}

#include data.stan

parameters {
    vector[num_nodes] eta;
}

transformed parameters {
    // Evaluate covariance of the point at zero with everything else.
    vector[num_nodes] cov = gp_exp_quad_cov(X, rep_array(rep_vector(0, 1), 1), alpha, rho, rep_vector(num_nodes, 1))[:, 1];
    cov[1] += epsilon;
}

model {
    eta ~ fft_gp(cov);
    y ~ poisson_log(eta);
}

generated quantities {
    matrix[num_nodes, num_nodes] full_cov = add_diag(gp_exp_quad_cov(X, X, alpha, rho, rep_vector(num_nodes, 1)), epsilon);
    real lpdf_fft = fft_gp_lpdf(eta | cov);
    real lpdf_multi_normal = multi_normal_lpdf(eta | rep_vector(0, num_nodes), full_cov);
}
