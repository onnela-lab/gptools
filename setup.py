from setuptools import find_packages, setup


setup(
    name="graph_gaussian_process",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "cmdstanpy",
        "matplotlib",
        "numpy",
    ],
    extras_require={
        "tests": [
            "doit-interface",
            "flake8",
            "nbconvert",
            "pytest",
            "pytest-cov",
            "scipy",
        ],
        "docs": [
            "sphinx",
        ]
    }
)
