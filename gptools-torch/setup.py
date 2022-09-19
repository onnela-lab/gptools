from setuptools import find_namespace_packages, setup


setup(
    name="gp-tools-torch",
    packages=find_namespace_packages(),
    version="0.1.0",
    install_requires=[
        "gp-tools-util",
        "numpy",
        "torch",
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
    }
)
