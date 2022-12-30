from setuptools import find_namespace_packages, setup

with open("README.rst") as fp:
    long_description = fp.read().replace(".. toctree::", "..").replace(":doc:", ":code:")

setup(
    name="gptools-torch",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_namespace_packages(),
    version="0.1.0",
    install_requires=[
        "gptools-util",
        "numpy",
        "torch",
    ],
    extras_require={
        "docs": [
            "myst-nb",
            "sphinx",
            "sphinx-multiproject",
            "sphinx_rtd_theme",
        ],
        "tests": [
            "doit-interface",
            "flake8",
            "jupyter",
            "matplotlib",
            "networkx",
            "pytest",
            "pytest-cov",
            "scipy",
            "tabulate",
            "tqdm",
            "twine",
        ],
    }
)
