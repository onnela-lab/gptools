from setuptools import find_namespace_packages, setup


setup(
    name="gptools-util",
    packages=find_namespace_packages(),
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
            "torch",
        ],
        "docs": [
            "sphinx",
        ],
    }
)
