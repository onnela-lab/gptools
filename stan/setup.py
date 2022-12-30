import re
from setuptools import find_namespace_packages, setup

with open("README.rst") as fp:
    long_description = fp.read()
long_description = re.sub(r".. (literalinclude|testsetup|toctree)", "..", long_description)
long_description = re.sub(".. doctest::", ".. code-block::", long_description)
long_description = re.sub(":(doc|class|func|ref):", ":code:", long_description)

setup(
    name="gptools-stan",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_namespace_packages(),
    version="0.1.0",
    install_requires=[
        # Required because of a bug in how complex numbers are handled (see
        # https://github.com/stan-dev/cmdstanpy/pull/612).
        "cmdstanpy>=1.0.7",
        "gptools-util",
        "numpy",
    ],
    extras_require={
        "docs": [
            "myst-nb",
            "networkx",
            "openpyxl",
            "pyproj",
            "sphinx",
            "sphinx-multiproject",
            "sphinx_rtd_theme",
            "sphinx-stan>=0.1.5",
        ],
        "tests": [
            "doit-interface",
            "flake8",
            "jupyter",
            "matplotlib",
            "pytest",
            "pytest-cov",
            "scipy",
            "tabulate",
            "twine",
        ],
    }
)
