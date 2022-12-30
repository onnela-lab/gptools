Scalable Gaussian process inference with *Stan*
===============================================

.. toctree::
    :hidden:

    docs/background
    docs/examples
    docs/interface

Gaussian processes (GPs) are flexible distributions to model functional data. Whilst theoretically appealing, they are computationally cumbersome except for small datasets. This package implements two methods for scaling GP inference in *Stan*:

1. a :ref:`sparse-approximation` of the likelihood that is generally applicable.
2. an exact method for regularly spaced data modeled by stationary kernels using fast :ref:`Fourier-methods`.

The implementation follows *Stan*'s design and exposes performant inference through a familiar interface.

.. _getting-started:

Getting Started
---------------

The library is loaded with *Stan*'s :code:`#include` statement, and methods to evaluate or approximate the likelihood of a GP use the declarative :code:`~` sampling syntax. The following brief example uses :ref:`Fourier-methods` to sample GP realizations.

.. literalinclude:: docs/getting_started/getting_started.stan
    :language: stan

You can learn more by following the :doc:`docs/examples` or delving into the :doc:`docs/interface`. The :doc:`docs/background` section offers a deeper explanation of the methods used to evaluate likelihoods and the pros and cons of different parameterizations. See the `accompanying publication "Scalable Gaussian process inference with Stan" <https://arxiv.org/abs/xxxx.xxxxx>`_ for further details.

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
