from setuptools import find_namespace_packages, setup

with open("README.rst") as fp:
    long_description = fp.read()

print(long_description)

setup(
    name="gp-tools-stan",
    long_description=long_description,
    long_description_content_type="text/x-rst",
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
