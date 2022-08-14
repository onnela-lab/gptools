from setuptools import find_packages, setup


setup(
    name="graph_gaussian_process",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "numpy",
    ],
    extras_require={
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
        ],
        "docs": [
            "sphinx",
        ],
        "cmdstanpy": [
            "cmdstanpy",
        ],
        "torch": [
            "torch",
        ],
    }
)
