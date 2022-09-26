🚀 gptools-stan
===============

.. toctree::
    :hidden:

    docs/poisson_regression/poisson_regression

The interface definitions below provide a comprehensive overview of the functionality offered by the Stan library. Please see the example :doc:`docs/poisson_regression/poisson_regression` for an illustration of how to use the library.

🔌 Interface
------------

The Stan library is organized as multiple files each comprising related functionality, such as utility functions or kernels.

⚙️ Utility functions
^^^^^^^^^^^^^^^^^^^^

.. standoc:: gptools/stan/gptools_util.stan

⏩ Likelihood evaluations based on fast Fourier transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. standoc:: gptools/stan/gptools_fft.stan

📏 Kernel functions with periodic boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. standoc:: gptools/stan/gptools_kernels.stan

🕸️ Gaussian processes on graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. standoc:: gptools/stan/gptools_graph.stan
