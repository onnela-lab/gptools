import doit_interface as di
import itertools as it
import numpy as np
import pathlib


manager = di.Manager.get_instance()

# Prevent each process from parallelizing which can lead to competition across processes.
di.SubprocessAction.set_global_env({
    "NUMEXPR_NUM_THREADS": 1,
    "OPENBLAS_NUM_THREADS": 1,
    "OMP_NUM_THREADS": 1,
    "MKL_NUM_THREADS": 1,
})

modules = ["stan", "torch", "util"]
requirements_txt = []
for module in modules:
    # Generate requirement files.
    prefix = pathlib.Path(f"gptools-{module}")
    target = prefix / "test_requirements.txt"
    requirements_in = prefix / "test_requirements.in"
    manager(
        basename="requirements", name=module, targets=[target],
        file_dep=[prefix / "setup.py", requirements_in, "shared_requirements.in"],
        actions=[f"pip-compile -v -o {target} {requirements_in}"]
    )
    requirements_txt.append(target)

    # Tasks for linting, tests, and building a distribution.
    manager(basename="lint", name=module, actions=[["flake8", prefix]])
    action = ["pytest", "-v", f"--cov=gptools.{module}", "--cov-report=term-missing",
              "--cov-report=html", "--cov-fail-under=100", "--durations=5", prefix]
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
manager(basename="requirements", name="sync", file_dep=[target], actions=[["pip-sync", target]])

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
try:
    from gptools.stan.profile import LOG_NOISE_SCALES, PARAMETERIZATIONS, SIZES
    with di.group_tasks("profile"):
        product = it.product(["sample", "variational"], PARAMETERIZATIONS, LOG_NOISE_SCALES, SIZES)
        for method, parameterization, log_sigma, size in product:
            name = f"method-{method}_log_noise_scale-{log_sigma:.3f}_size-{size}"
            target = f"workspace/profile/{method}/{parameterization}/{name}.pkl"
            args = [
                "python", "-m", "gptools.stan.profile", method, parameterization, np.exp(log_sigma),
                target, "--iter_sampling=100", f"--n={size}", "--max_chains=-1", "--timeout=30",
            ]
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
            manager(basename=f"profile/{parameterization}", name=name, actions=[args],
                    targets=[target], file_dep=[prefix / x for x in file_dep])
except ModuleNotFoundError:
    pass


# Tree data from https://datadryad.org/stash/dataset/doi:10.15146/5xcp-0d46.
with di.defaults(basename="trees"):
    # Download elevation data.
    target = "data/elevation.tsv"
    url = "https://datadryad.org/stash/downloads/file_stream/148941"
    manager(name="elevation", targets=[target], actions=[["curl", "-L", "-o", "$@", url]],
            uptodate=[True])

    # Download the tree archive.
    archive = "data/bci.tree.zip"
    url = "https://datadryad.org/stash/downloads/file_stream/148942"
    manager(name="zip", targets=[archive], actions=[["curl", "-L", "-o", "$@", url]],
            uptodate=[True])
    # Extract the rdata from the archive.
    rdata_target = "data/bci.tree8.rdata"
    manager(name="rdata", targets=[rdata_target], file_dep=[archive], actions=[
        ["unzip", "-jo", archive, "home/fullplotdata/tree/bci.tree8.rdata", "-d", "data"]
    ])
    # Convert to CSV.
    csv_target = "data/bci.tree8.csv"
    manager(name="csv", targets=[csv_target], file_dep=[rdata_target], actions=[
        ["Rscript", "--vanilla", "data/rda2csv.r", rdata_target, "bci.tree8", csv_target],
    ])
    # Aggregate the trees to get a smaller dataset.
    for species in ["tachve"]:
        target = f"data/{species}.csv"
        script = "data/aggregate_trees.py"
        manager(name=species, targets=[target], file_dep=[csv_target, script],
                actions=[["$!", script, csv_target, species, target]])


with di.defaults(basename="tube"):
    url = "http://crowding.data.tfl.gov.uk/Annual%20Station%20Counts/2019/" \
        "AnnualisedEntryExit_2019.xlsx"
    entry_exit_target = "data/AnnualisedEntryExit_2019.xlsx"
    manager(name="entry-exit-data", targets=[entry_exit_target],
            actions=[["curl", "-L", "-o", "$@", url]], uptodate=[True])
    target = "data/tube.json"
    manager(name="graph", targets=[target], file_dep=[entry_exit_target],
            actions=[["$!", "data/construct_tube_network.py", entry_exit_target, target]])
