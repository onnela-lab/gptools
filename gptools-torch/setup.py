from setuptools import find_namespace_packages, setup


setup(
    name="gptools-torch",
    packages=find_namespace_packages(),
    version="0.1.0",
    install_requires=[
        "gptools-util",
        "numpy",
        "torch",
    ],
    extras_require={
        "tests": [
            "doit-interface",
            "flake8",
            "pytest",
            "pytest-cov",
        ],
        "docs": [
            "sphinx",
        ],
    }
)
