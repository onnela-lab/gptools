Scalable Gaussian process inference with *Stan*
===============================================

.. toctree::
    :hidden:

    docs/getting_started/getting_started
    docs/poisson_regression/poisson_regression
    docs/trees/trees
    docs/tube/tube
    docs/fourier_inducing_points/fourier_inducing_points
    docs/function_reference

Gaussian processes (GPs) are flexible distributions to model functional data. Whilst theoretically appealing, they are computationally cumbersome except for small datasets. This package implements two methods for scaling GP inference in *Stan*:

1. a `sparse approximation`_ of the likelihood that is generally applicable.
2. an exact method for regularly spaced data modeled by stationary kernels using fast `Fourier methods`_.

The implementation follows *Stan*'s design and exposes performant inference through a familiar interface.

Getting Started
---------------

The library is loaded with *Stan*'s :code:`#include` statement, and methods to evaluate or approximate the likelihood of a GP use the declarative :code:`~` sampling syntax. The following brief example uses `Fourier methods`_ to sample GP realizations.

.. literalinclude:: docs/getting_started/getting_started.stan
    :language: stan

You can learn more by following the `Examples`_ or delving into the :doc:`docs/function_reference`. The `Background`_ section offers a deeper explanation of the methods used to evaluate likelihoods and the pros and cons of different parameterizations. See the `accompanying publication "Scalable Gaussian process inference with Stan" <https://arxiv.org/abs/xxxx.xxxxx>`_ for further details.

Installation
^^^^^^^^^^^^

If you have a recent python installation, the library can be installed by running

.. code-block:: bash

    pip install gptools-stan

from the command line. The library exposes a function :func:`gptools.stan.compile_model` for compiling :class:`cmdstanpy.CmdStanModel`\ s with the correct include paths. For example, the example above can be compiled using the following snippet.

.. testsetup::

    stan_file = "stan/docs/getting_started/getting_started.stan"

.. doctest::

    >>> from gptools.stan import compile_model
    >>>
    >>> # stan_file = path/to/getting_started.stan
    >>> model = compile_model(stan_file=stan_file)
    >>> model.name
    'getting_started'

If you use `cmdstanr <https://mc-stan.org/cmdstanr/>`__ or another *Stan* `interface <https://mc-stan.org/users/interfaces/>`__, you can download the `library files from GitHub <https://github.com/tillahoffmann/gptools/tree/main/stan/gptools/stan/gptools>`__. Then add the library location to the compiler :code:`include_paths` as `described in the manual <https://mc-stan.org/docs/stan-users-guide/stanc-args.html>`__ (see `here <https://mc-stan.org/cmdstanr/reference/model-method-compile.html>`__ for cmdstanr instructions).

Examples
--------

- :doc:`docs/getting_started/getting_started` expands on the `getting started`_ example above and illustrates the use of `cmdstanpy <https://github.com/stan-dev/cmdstanpy>`_ to draw samples. The example considers centered and non-centered `parameterizations`_ for `Fourier methods`_.
- :doc:`docs/poisson_regression/poisson_regression` uses a latent GP with Poisson likelihood for count data. The example considers three different approaches (standard approach using the full covariance matrix, `sparse approximation`_, and `Fourier methods`_) with both centered and non-centered `parameterizations`_ for a total of six models.
- :doc:`docs/tube/tube` uses the `sparse approximation`_ to explore daily passenger numbers on the London Underground network. It illustrates how to combine fixed effects with a latent GP to infer a smooth function that accounts for residual passenger number variability after accounting for the number of interchanges and transport zone of each station on the network.
- :doc:`docs/trees/trees` uses `Fourier methods`_ to model the density of trees on a 50 hectar plot in Panama. The example highlights the importance of padding the data to attenuate the effects of periodic boundary conditions inherent to the discrete Fourier transform.

Background
----------

A GP is a distribution over functions :math:`f\parenth{x}` such that any finite set of :math:`n` values :math:`\vec{f}=f\parenth{\vec{x}}` evaluated at :math:`\vec{x}=\braces{x_1,\ldots,x_n}` follows a multivariate normal distribution. The distribution is thus fully specified by its mean :math:`\mu\parenth{x}=\E{f\parenth{x}}` and covariance (kernel) :math:`k\parenth{x,x'}=\cov\parenth{f\parenth{x},f\parenth{x'}}`. Evaluating the likelihood requires inverting the covariance matrix :math:`\mat{K}` obtained by evaluating the kernel :math:`k` at all pairs of observed locations :math:`\vec{x}`. Unfortunately, the computational cost of inverting :math:`\mat{K}` scales as :math:`\BigO\parenth{n^3}`, making GPs prohibitively expensive save for relatively small datasets.

Sparse approximation
^^^^^^^^^^^^^^^^^^^^

The joint distribution of observations :math:`\vec{f}` may be expressed as the product of conditional distributions

.. math::

    \proba{\vec{f}}=\proba{f_1}\prod_{j=2}^n \proba{f_j\mid f_{j-1}, \ldots, f_1}.

The conditional structure can be encoded by a directed acyclic graph (DAG) whose nodes represent observations such that a directed edge exists from a node :math:`j` to each of its predecessors :math:`\pred_j=\braces{j-1,\ldots,1}`, where the ordering is arbitrary. If two observations do not depend on one another, the corresponding edge can be removed from the DAG to reduce the computational cost. In particular, evaluating each factor above requires inverting a matrix with size equal to the number of predecessors of the corresponding node---a substantial saving if the graph is sparse. For example, nearest-neighbor methods, a special case, reduce the asymptotic runtime to :math:`\BigO\parenth{n q^3}` by retaining only edges from each node to at most :math:`q` of its nearest predecessors.

.. note::

    :stan:func:`gp_graph_exp_quad_cov_lpdf` evaluates the likelihood of a GP on a DAG. :code:`gp_graph_matern32_cov_lpdf` and :code:`gp_graph_matern52_cov_lpdf` evaluate the corresponding likelihoods for Matérn kernels. Transformations for non-centered `parameterizations`_ are implemented as :stan:func:`gp_transform_inv_graph_exp_quad_cov(vector, vector, array [] vector, real, real, array [,] int)`, :code:`gp_transform_inv_graph_matern32_cov`, and :code:`gp_transform_inv_graph_matern52_cov`.

Fourier methods
^^^^^^^^^^^^^^^

If the observation points :math:`\vec{x}` form a regular grid and the kernel is stationary, i.e., :math:`k\parenth{x,x'}=k\parenth{x-x'}`, we can use the fast Fourier transform (FFT) to evaluate the likelihood exactly in :math:`\BigO\parenth{n\log n}` time. As the Fourier transform is a linear operator and :math:`f` is (infinite-dimensional) multivariate normal, the Fourier coefficients

.. math::

    \tilde f\parenth{\xi}=\int_{-\infty}^\infty dx\,\exp\parenth{-2\pi\imag\xi x} f\parenth{x}

are also multivariate normal. Assuming :math:`\mu\parenth{x}=0` for simplicity, the mean of Fourier coefficients is zero and their expected complex-conjugate product at two different frequencies :math:`\xi` and :math:`\xi'` is

.. math::

    \E{\tilde f\parenth{\xi}\overline{\tilde f\parenth{\xi'}}}&=\int_{-\infty}^\infty dx\,dx'\,\exp\parenth{-2\pi\imag\parenth{x\xi-x'\xi'}}k\parenth{x-x'}\\
    % &=\int d\Delta\,dx'\,\exp\parenth{-2\pi\imag\parenth{\Delta\xi +x'\parenth{\xi-\xi'}}}k\parenth{\Delta}\\
    &=\int_{-\infty}^\infty dx'\, \exp\parenth{-2\pi\imag x'\parenth{\xi-\xi'}}
    \int_{-\infty}^\infty d\Delta\,\exp\parenth{-2\pi\imag \Delta\xi} k\parenth{\Delta}

where we changed variables to :math:`x=\Delta + x'` and factorized the integrals in the second line. The first integral is a representation of the Dirac delta function :math:`\delta\parenth{\xi-\xi'}`, and the second integral is the Fourier transform of the kernel. Fourier coefficients are thus uncorrelated, and, subject to careful bookkeeping, we can evaluate the GP likelihood exactly. In other words, the Fourier transform diagonalizes stationary kernels.

.. note::

    :stan:func:`gp_rfft_lpdf` and :stan:func:`gp_rfft2_lpdf` evaluate the likelihood of a GP on one- and two-dimensional grids, respectively. Transformations for non-centered `parameterizations`_ are implemented as :stan:func:`gp_transform_inv_rfft` and :stan:func:`gp_transform_inv_rfft2`.

Parameterizations
^^^^^^^^^^^^^^^^^

Consider a simple model with latent GP and normal observation noise with variance :math:`kappa^2`

.. math::

    \vec{f}&\dist\mathsf{MultivariateNormal}\parenth{0, \mat{K}}\\
    \vec{y}&\dist\mathsf{Normal}\parenth{\vec{f}, \kappa^2}.

The model employs the natural *centered* parameterization, i.e., each observation :math:`y_i` is independent given the corresponding latent :math:`f_i`. This parameterization works well if the data are informative (small :math:`\kappa`) because each observation :math:`y_i` constrains the corresponding latent parameter :math:`f_i`. The elements of :math:`\vec{f}` are thus relatively uncorrelated under the posterior, and the Hamiltonian sampler can explore the distribution efficiently.

However, if the data are weak (large :math:`\kappa`), they cannot independently constrain each element of :math:`\vec{f}` and the GP prior dominates the posterior. The resulting correlation among elements of :math:`\vec{f}` frustrates the sampler, especially if the correlation length is large. We can overcome this challenge by employing a *non-centered* parameterization such that the parameters of the model are uncorrelated under the prior. Here, we reparameterize the model in terms of a white noise vector :math:`\vec{z}` of the same size as :math:`\vec{f}` and obtain realizations of the GP :math:`\vec{f}=\phi^{-1}\parenth{\vec{z}}` using an inverse transform :math:`\phi^{-1}` which must be selected carefully to ensure :math:`\vec{f}` follows the desired distribution. The reparameterized model is

.. math::

    \vec{z}&\dist\mathsf{Normal}\parenth{0, 1}\\
    \vec{f}&=\phi^{-1}\parenth{\vec{z}, 0, \mat{K}}\\
    \vec{y}&\dist\mathsf{Normal}\parenth{\vec{f}, \kappa^2}.
