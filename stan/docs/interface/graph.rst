Gaussian Processes on Graphs
============================

The module implements methods to approximate Gaussian processes using :ref:`sparse-approximation`\ s. The functions of immediate use to practitioners are :stan:func:`gp_graph_exp_quad_cov_lpdf(vector, vector, array [] vector, real, real, array [,] int)` to evaluate the likelihood of a one-dimensional signal and :stan:func:`gp_transform_inv_graph_exp_quad_cov(vector, vector, array [] vector, real, real, array[,] int)` to construct a :ref:`non-centered parameterization <parameterizations>` using white noise. Analogues are implemented for Matérn kernels with :math:`\nu=3/2` and :math:`\nu=5/2`.

.. stan:autodoc:: ../../gptools/stan/gptools/graph.stan
