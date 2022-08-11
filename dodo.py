import doit_interface as di

manager = di.Manager.get_instance()

for name, file_dep in [("test", ["setup.py"]), ("dev", ["test_requirements.txt"])]:
    requirements_in = f"{name}_requirements.in"
    target = f"{name}_requirements.txt"
    manager(
        basename="requirements", name=name, targets=[target], file_dep=[requirements_in, *file_dep],
        actions=[f"pip-compile -v -o {target} {requirements_in}"],
    )

with di.defaults(basename="docs"):
    manager(name="html", actions=["sphinx-build . docs/_build"])
    manager(name="tests", actions=["sphinx-build -b doctest . docs/_build"])

manager(basename="lint", actions=["flake8"])
manager(basename="tests", actions=["pytest --cov-fail-under=100"])
