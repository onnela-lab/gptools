from setuptools import find_namespace_packages, setup

with open("README.rst") as fp:
    long_description = fp.read()
long_description = long_description.replace(".. stan:autodoc:: ", ".. ")\
    .replace(".. toctree::", "..").replace(":doc:", ":code:").replace(":ref:", ":code:")\
    .replace(".. literalinclude::", "..").replace(":func:", ":code:").replace(":class:", ":code:")\
    .replace(".. testsetup::", "..").replace(".. doctest::", ".. code-block::")

setup(
    name="gp-tools-stan",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_namespace_packages(),
    version="0.1.0",
    install_requires=[
        # Required because of a bug in how complex numbers are handled (see
        # https://github.com/stan-dev/cmdstanpy/pull/612).
        "cmdstanpy>=1.0.7",
        "gp-tools-util",
        "numpy",
    ],
    extras_require={
        "docs": [
            "myst-nb",
            "networkx",
            "openpyxl",
            "pyproj",
            "sphinx",
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
