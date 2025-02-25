Scalable Gaussian process inference with *Stan*
===============================================

.. image:: https://github.com/onnela-lab/gptools/actions/workflows/python.yml/badge.svg
    :target: https://github.com/onnela-lab/gptools/actions/workflows/python.yml
.. image:: https://github.com/onnela-lab/gptools/actions/workflows/R.yml/badge.svg
    :target: https://github.com/onnela-lab/gptools/actions/workflows/R.yml
.. image:: https://readthedocs.org/projects/gptools-stan/badge/?version=latest
    :target: https://gptools-stan.readthedocs.io/en/latest/?badge=latest
.. image:: https://img.shields.io/pypi/v/gptools-stan
    :target: https://pypi.org/project/gptools-stan
.. image:: https://img.shields.io/cran/v/gptoolsStan
    :target: https://cran.r-project.org/package=gptoolsStan
.. image:: https://img.shields.io/static/v1?label=&message=GitHub&color=gray&logo=github
    :target: https://github.com/onnela-lab/gptools

.. toctree::
    :hidden:

    docs/background
    docs/examples
    docs/interface

Gaussian processes (GPs) are powerful distributions for modeling functional data, but using them is computationally challenging except for small datasets. *gptools* implements two methods for performant GP inference in Stan.

1. A :ref:`sparse-approximation` of the likelihood. This approach includes nearest neighbor Gaussian processes but also supports more general dependence structures, e.g., for periodic kernels.
2. An exact likelihood evaluation for data on regularly spaced lattices using fast :ref:`Fourier-methods`.

The implementation follows Stan's design and exposes performant inference through a familiar interface. We provide interfaces in Python and R. See the accompanying publication `Scalable Gaussian Process Inference with Stan <https://doi.org/10.48550/arXiv.2301.08836>`__ for details of the implementation.

.. _getting-started:

Getting Started
---------------

The library is loaded with *Stan*'s :code:`#include` statement, and methods to evaluate or approximate the likelihood of a GP use the declarative :code:`~` sampling syntax. The following brief example uses :ref:`Fourier-methods` to sample GP realizations.

.. literalinclude:: docs/getting_started/getting_started.stan
    :language: stan

You can learn more by following the :doc:`docs/examples` or delving into the :doc:`docs/interface`. The :doc:`docs/background` section offers a deeper explanation of the methods used to evaluate likelihoods and the pros and cons of different parameterizations.

Getting Started in Python
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Install *cmdstanpy* and *cmdstan* if you haven't already (see `here <https://cmdstanpy.readthedocs.io/en/v1.2.5/installation.html>`__ for details).
2. Install *gptools* from `PyPI <https://pypi.org/project/gptools-stan/>`__ by running :code:`pip install gptools-stan` from the command line.
3. Compile your first model. The library exposes a function :func:`gptools.stan.compile_model` for compiling :class:`cmdstanpy.CmdStanModel`\ s with the correct include paths. For example, the example above can be compiled using the following snippet.

.. doctest::

    >>> from gptools.stan import compile_model
    >>>
    >>> stan_file = "docs/getting_started/getting_started.stan"
    >>> model = compile_model(stan_file=stan_file)
    >>> model.name
    'getting_started'

Getting Started in R
^^^^^^^^^^^^^^^^^^^^

1. Install *cmdstanr* and *cmdstan* if you haven't already (see `here <https://mc-stan.org/cmdstanr/#installation>`__ for details).
2. Install *gptools* from `CRAN <https://cran.r-project.org/package=gptoolsStan>`__ by running :code:`install.packages("gptoolsStan")`.
3. Compile your first model.

.. code-block:: R

    library(cmdstanr)
    library(gptoolsStan)

    model <- cmdstan_model(
    stan_file="docs/getting_started/getting_started.stan",
    include_paths=gptools_include_path(),
    )

If you use another *Stan* `interface <https://mc-stan.org/users/interfaces/>`__, you can download the `library files from GitHub <https://github.com/onnela-lab/gptools/tree/main/stan>`__. Then add the library location to the compiler :code:`include_paths` as `described in the manual <https://mc-stan.org/docs/stan-users-guide/stanc-args.html>`__.

Reproducing Results From the Accompanying Publication
-----------------------------------------------------

The `accompanying publication "Scalable Gaussian process inference with Stan" <https://arxiv.org/abs/2301.08836>`__ provides theoretical background and a technical description of the methods. All results and figures can be reproduced by following the instructions in the `repository of reproduction materials <https://github.com/onnela-lab/gptools-reproduction-material>`__.
