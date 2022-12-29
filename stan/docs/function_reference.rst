Function Reference
------------------

🐍 Python interface
^^^^^^^^^^^^^^^^^^^

.. automodule:: gptools.stan
    :members:

The Stan library is organized as multiple files each comprising related functionality, such as utility functions or kernels.

⚙️ Utility functions
^^^^^^^^^^^^^^^^^^^^

.. stan:autodoc:: ../gptools/stan/gptools/util.stan

⏩ Likelihood evaluations based on fast Fourier transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. stan:autodoc:: ../gptools/stan/gptools/fft1.stan
.. stan:autodoc:: ../gptools/stan/gptools/fft2.stan

📏 Kernel functions with periodic boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. stan:autodoc:: ../gptools/stan/gptools/kernels.stan

🕸️ Gaussian processes on graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. stan:autodoc:: ../gptools/stan/gptools/graph.stan
