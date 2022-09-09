import doit_interface as di
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

    # Tasks for linting and tests.
    manager(basename="lint", name=module, actions=[["flake8", prefix]])
    action = ["pytest", "-v", f"--cov=gptools.{module}", "--cov-report=term-missing",
              "--cov-fail-under=100", prefix]
    manager(basename="tests", name=module, actions=[action])

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

# Run different profiling configurations. We expect the centered parametrization to be better for
# strong data and the non-centered parametrization to be better for weak data.
profile_configurations = {
    "centered-weak": ("centered", 1.0),
    "centered-strong": ("centered", 0.1),
    "non_centered-weak": ("non_centered", 1.0),
    "non_centered-strong": ("non_centered", 0.1),
}
for name, (parametrization, noise_scale) in profile_configurations.items():
    target = f"workspace/{name}.pkl"
    args = ["python", "-m", "gptools.stan.profile", parametrization, noise_scale, target,
            "--iter_sampling=100"]
    file_dep = [
        "gptools/stan/gptools_graph.stan",
        "gptools/stan/profile/__main__.py",
        "gptools/stan/profile/data.stan",
        f"gptools/stan/profile/{parametrization}.stan",
    ]
    manager(basename="profile", name=name, actions=[args], targets=[target],
            file_dep=file_dep)
