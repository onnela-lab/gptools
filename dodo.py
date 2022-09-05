import doit_interface as di
import pathlib


manager = di.Manager.get_instance()

requirement_configs = [
    ("test_stan", ["setup.py"]),
    ("test_torch", ["setup.py"]),
    ("dev", ["test_stan_requirements.txt", "test_torch_requirements.txt"]),
]
for name, file_dep in requirement_configs:
    requirements_in = f"{name}_requirements.in"
    target = f"{name}_requirements.txt"
    manager(
        basename="requirements", name=name, targets=[target], file_dep=[requirements_in, *file_dep],
        actions=[f"pip-compile -v -o {target} {requirements_in}"],
    )
manager(basename="requirements", name="sync", file_dep=["dev_requirements.txt"],
        actions=["pip-sync dev_requirements.txt"])

with di.defaults(basename="docs"):
    manager(name="html", actions=["sphinx-build . docs/_build"])
    manager(name="tests", actions=["sphinx-build -b doctest . docs/_build"])

manager(basename="lint", actions=["flake8"])
manager(basename="tests", actions=["pytest --cov-fail-under=100 --cov=gptools "
                                   "--cov-report=term-missing --cov-report=html"])


for path in pathlib.Path("gptools/examples").glob("*/*.ipynb"):
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
    args = ["python", "-m", "gptools.profile", parametrization, noise_scale, target,
            "--iter_sampling=100"]
    file_dep = [
        "gptools/gptools.stan",
        "gptools/profile/__main__.py",
        "gptools/profile/data.stan",
        f"gptools/profile/{parametrization}.stan",
    ]
    manager(basename="profile", name=name, actions=[args], targets=[target],
            file_dep=file_dep)
