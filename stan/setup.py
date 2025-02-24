import re
from setuptools import find_namespace_packages, setup

with open("README.rst") as fp:
    long_description = fp.read()
long_description = re.sub(
    r".. (literalinclude|testsetup|toctree)::", "..\n    comment", long_description
)
long_description = re.sub(".. doctest::", ".. code-block::", long_description)
long_description = re.sub(":(doc|class|func|ref):", ":code:", long_description)

setup(
    name="gptools-stan",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_namespace_packages(),
    include_package_data=True,
    version="0.2.1",
    python_requires=">=3.8",
    install_requires=[
        # Required because of a bug in how complex numbers are handled (see
        # https://github.com/stan-dev/cmdstanpy/pull/612).
        "cmdstanpy>=1.0.7",
        "numpy",
    ],
)
