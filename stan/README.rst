Scalable Gaussian process inference with *Stan*
===============================================

.. image:: https://github.com/onnela-lab/gptools/actions/workflows/main.yml/badge.svg
    :target: https://github.com/onnela-lab/gptools/actions/workflows/main.yml
.. image:: https://readthedocs.org/projects/gptools-stan/badge/?version=latest
    :target: https://gptools-stan.readthedocs.io/en/latest/?badge=latest
.. image:: https://img.shields.io/pypi/v/gptools-stan
    :target: https://pypi.org/project/gptools-stan
.. image:: https://img.shields.io/static/v1?label=&message=GitHub&color=gray&logo=github
    :target: https://github.com/onnela-lab/gptools

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

You can learn more by following the :doc:`docs/examples` or delving into the :doc:`docs/interface`. The :doc:`docs/background` section offers a deeper explanation of the methods used to evaluate likelihoods and the pros and cons of different parameterizations. See the `accompanying publication "Scalable Gaussian process inference with Stan" <https://arxiv.org/abs/2301.08836>`__ for further details.

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

If you use `cmdstanr <https://mc-stan.org/cmdstanr/>`__ or another *Stan* `interface <https://mc-stan.org/users/interfaces/>`__, you can download the `library files from GitHub <https://github.com/onnela-lab/gptools/tree/main/stan/gptools/stan/gptools>`__. Then add the library location to the compiler :code:`include_paths` as `described in the manual <https://mc-stan.org/docs/stan-users-guide/stanc-args.html>`__ (see `here <https://mc-stan.org/cmdstanr/reference/model-method-compile.html>`__ for cmdstanr instructions).

Reproducing results from the accompanying publication
-----------------------------------------------------

The `accompanying publication "Scalable Gaussian process inference with Stan" <https://arxiv.org/abs/2301.08836>`__ provides theoretical background and a technical description of the methods. All results and figures can be reproduced using one of the approaches below.

Docker runtime
^^^^^^^^^^^^^^

`Docker <https://www.docker.com>`__ can run software in isolated containers. If you have docker installed, you can reproduce the results by running

.. code:: bash

    docker run --rm tillahoffmann/gptools -v /path/to/output/directory:/workspace doit --db-file=/workspace/.doit.db results:stan

This command will download a prebuilt docker image and execute the steps required to generate all figures in the publication. Results will be placed in the specified output directory; make sure the directory exists before executing the command and that the specified path is an absolute, e.g., :code:`/path/to/...` instead of :code:`../path/to/...`. You do not need to install any other software or download the source code. Intermediate results are cached if the process is interrupted, and the process can pick up where it left off when invoked using the same command. Your timing results are likely to differ from the results reported in the publication because runtimes vary substantially between different machines. All results reported in the manuscript were obtained on a 2020 Macbook Pro with M1 Apple Silicon chip and 16 GB of memory. Cross-architecture images are built following `this guide <https://blog.jaimyn.dev/how-to-build-multi-architecture-docker-images-on-an-m1-mac/>`__.

If you would rather build the docker image from scratch, run :code:`docker build -t my-image-name .` from the root directory of this repository. You can then reproduce the results using the command above, replacing :code:`tillahoffmann/gptools` with :code:`my-image-name`. Optionally, run :code:`docker run --rm gptools doit tests:stan` to ensure the image runs as expected; this takes about ten to fifteen minutes on a Macbook.

Local runtime
^^^^^^^^^^^^^
You can reproduce the results using your local computing environment (rather than an isolated container runtime) as follows.

1. Ensure a recent python version is installed (the code was tested with python 3.10 on Ubuntu 22.04.2 and macOS 13.2.1).
2. Install all dependencies by running :code:`pip install -r dev_requirements.txt`.
3. Install :code:`cmdstan` by running :code:`install_cmdstan --version=2.31.0`.
4. Optionally, run :code:`doit tests:stan` to test the installation.
5. Run the command :code:`doit results:stan` to reproduce the results.
