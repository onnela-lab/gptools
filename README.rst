📈 Tools for Gaussian processes
================================

.. image:: https://github.com/tillahoffmann/gp-tools/actions/workflows/main.yml/badge.svg
  :target: https://github.com/tillahoffmann/gp-tools/actions/workflows/main.yml

This collection of packages provides tools for inference using Gaussian processes with a focus on:

- Gaussian processes on graphs to approximate the likelihood with a sparse precision matrix. Nearest-neighbor Gaussian processes are a special case of Gaussian processes on graphs.
- Evaluating the likelihood exactly using Fourier techniques for Gaussian processes on regular grids. Observations between grid points can be approximated by conditioning on the local neighborhood, i.e., using local inducing points.

The collection includes three packages:

- :doc:`gptools-stan/README` comprises an implementation using the probabilistic programming framework `stan <https://mc-stan.org>`__ for fast posterior sampling with Hamiltonian Monte Carlo.
- :doc:`gptools-torch/README` comprises an implementation using the machine learning library `torch <https://pytorch.org>`__ which is compatible with `pyro <https://pyro.ai>`__.
- :doc:`gptools-util/README` comprises an implementation in `numpy <https://numpy.org>`__ and common utility functions, such as constructing directed acyclic dependency graphs from nearest neighbors or evaluating covariance kernels.

.. toctree::
  :hidden:

  gptools-stan/README
  gptools-torch/README
  gptools-util/README

▶️ Installation
---------------

Each package can be installed using pip by running :code:`pip install gptools-[package name]` from the command line.
