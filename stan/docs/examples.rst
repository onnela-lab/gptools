Examples
========

.. toctree::
    :hidden:

    getting_started/getting_started
    logistic_regression/logistic_regression
    padding/padding
    poisson_regression/poisson_regression
    trees/trees
    tube/tube

- :doc:`getting_started/getting_started` expands on the :ref:`getting-started` section on the homepage and illustrates the use of `cmdstanpy <https://github.com/stan-dev/cmdstanpy>`_ to draw samples of Gaussian processes using Stan. The example considers centered and non-centered :ref:`parameterizations` for :ref:`Fourier-methods`.
- :doc:`logistic_regression/logistic_regression` illustrates the use of Fourier methods to fit a Gaussian process to binary outcome data.
- :doc:`padding/padding` explores the effect and importance of padding for :ref:`Fourier-methods` when the process does not have periodic boundary conditions.
- :doc:`poisson_regression/poisson_regression` uses a latent GP with Poisson likelihood for count data. The example considers three different approaches (standard approach using the full covariance matrix, :ref:`sparse-approximation`, and :ref:`Fourier-methods`) with both centered and non-centered :ref:`parameterizations` for a total of six models.
- :doc:`tube/tube` uses the :ref:`sparse-approximation` to explore daily passenger numbers on the London Underground network. It illustrates how to combine fixed effects with a latent GP to infer a smooth function that accounts for residual passenger number variability after accounting for the number of interchanges and transport zone of each station on the network.
- :doc:`trees/trees` uses :ref:`Fourier-methods` to model the density of trees on a 50 hectar plot in Panama. The example highlights the importance of padding the data to attenuate the effects of periodic boundary conditions inherent to the discrete Fourier transform.
