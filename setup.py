from setuptools import find_packages, setup


setup(
    name="graph_gaussian_process",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "cmdstanpy",
        "matplotlib",
        "numpy",
        "torch",
    ],
    extras_require={
        "tests": [
            "doit-interface",
            "flake8",
            "ipykernel",
            "nbconvert",
            "networkx",
            "pytest",
            "pytest-cov",
            "scipy",
            "tabulate",
        ],
        "docs": [
            "sphinx",
        ]
    }
)
