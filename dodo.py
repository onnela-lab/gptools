import doit_interface as di
import itertools as it
import numpy as np
import pathlib


manager = di.Manager.get_instance()

modules = ["stan", "torch", "util"]
requirements_txt = []
for module in modules:
    # Generate requirement files.
    prefix = pathlib.Path(f"gptools-{module}")
    target = prefix / "test_requirements.txt"
    requirements_in = prefix / "test_requirements.in"
    manager(
        basename="requirements", name=module, targets=[target],
        file_dep=[prefix / "setup.py", requirements_in],
        actions=[f"pip-compile -v -o {target} {requirements_in}"]
    )
    requirements_txt.append(target)

    # Tasks for linting, tests, and building a distribution.
    manager(basename="lint", name=module, actions=[["flake8", prefix]])
    action = ["pytest", "-v", f"--cov=gptools.{module}", "--cov-report=term-missing",
              "--cov-fail-under=100", "--durations=5", prefix]
    manager(basename="tests", name=module, actions=[action])
    manager(basename="package", name=module, actions=[
        di.actions.SubprocessAction("python setup.py sdist", cwd=prefix),
        f"twine check {prefix / 'dist/*.tar.gz'}",
    ])

# Generate dev and doc requirements.
target = "doc_requirements.txt"
requirements_in = "doc_requirements.in"
manager(
    basename="requirements", name="doc", targets=[target],
    file_dep=[requirements_in] + [f"gptools-{module}/setup.py" for module in modules],
    actions=[f"pip-compile -v -o {target} {requirements_in}"]
)

target = "dev_requirements.txt"
requirements_in = "dev_requirements.in"
manager(
    basename="requirements", name="dev", targets=[target],
    file_dep=[requirements_in, "doc_requirements.txt", *requirements_txt],
    actions=[f"pip-compile -v -o {target} {requirements_in}"]
)

# Build documentation at the root level (we don't have namespace-package-level documentation).
with di.defaults(basename="docs"):
    manager(name="html", actions=["sphinx-build . docs/_build"])
    manager(name="tests", actions=["sphinx-build -b doctest . docs/_build"])

# Compile example notebooks to create html reports.
for path in pathlib.Path.cwd().glob("gptools-*/**/*.ipynb"):
    if ".ipynb_checkpoints" in path.parts:
        continue
    target = path.with_suffix(".html")
    manager(basename="compile_example", name=path.with_suffix("").name, file_dep=[path],
            targets=[target], actions=[f"jupyter nbconvert --execute --to=html {path}"])

# Run different profiling configurations. We expect the centered parameterization to be better for
# strong data and the non-centered parameterization to be better for weak data.
log_sigmas = np.linspace(-3, 3, 11)
parameterizations = ["standard_centered", "standard_non_centered", "graph_centered",
                     "graph_non_centered", "fourier_centered", "fourier_non_centered"]
sizes = np.logspace(1.5, 3.5, 13).astype(int)

for parameterization, log_sigma, size in it.product(parameterizations, log_sigmas, sizes):
    name = f"log_sigma-{log_sigma:.3f}_size-{size}"
    target = f"workspace/profile/{parameterization}/{name}.pkl"
    args = ["python", "-m", "gptools.stan.profile", parameterization, np.exp(log_sigma), target,
            "--iter_sampling=100", f"--num_nodes={size}", "--max_chains=-1", "--timeout=30"]
    file_dep = [
        "profile/__main__.py",
        "gptools_fft.stan",
        "gptools_graph.stan",
        "gptools_kernels.stan",
        "gptools_util.stan",
        "profile/data.stan",
        f"profile/{parameterization}.stan",
    ]
    prefix = pathlib.Path("gptools-stan/gptools/stan")
    manager(basename=f"profile/{parameterization}", name=name, actions=[args], targets=[target],
            file_dep=[prefix / x for x in file_dep])
