from setuptools import find_namespace_packages, setup


setup(
    name="gp-tools-stan",
    packages=find_namespace_packages(),
    version="0.1.0",
    install_requires=[
        "cmdstanpy>=1.0.7",
        "gp-tools-util",
        "numpy",
    ],
    extras_require={
        "tests": [
            "doit-interface",
            "flake8",
            "jupyter",
            "matplotlib",
            "pytest",
            "pytest-cov",
            "scipy",
            "tabulate",
        ],
    }
)
