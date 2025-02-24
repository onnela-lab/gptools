import doit_interface as di
from doit_interface.actions import SubprocessAction
import itertools as it
import os
from pathlib import Path

workspace = Path(os.environ.get("WORKSPACE", "workspace")).resolve()
fast = "CI" in os.environ
manager = di.Manager.get_instance()

# Prevent each process from parallelizing which can lead to competition across processes.
SubprocessAction.set_global_env(
    {
        "NUMEXPR_NUM_THREADS": 1,
        "OPENBLAS_NUM_THREADS": 1,
        "OMP_NUM_THREADS": 1,
        "MKL_NUM_THREADS": 1,
    }
)

modules = ["stan", "util"]
for module in modules:
    # Generate requirement files.
    prefix = Path(module)
    # Tasks for linting, tests, building a distribution, and project-specific documentation.
    manager(
        basename="lint",
        name=module,
        actions=[["flake8", prefix], ["black", "--check", prefix]],
    )
    action = [
        "pytest",
        "-v",
        f"--cov=gptools.{module}",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-fail-under=100",
        "--durations=5",
        prefix,
    ]
    manager(basename="tests", name=module, actions=[action])
    manager(
        basename="package",
        name=module,
        actions=[
            SubprocessAction("python setup.py sdist", cwd=prefix),
            f"twine check {prefix / 'dist/*.tar.gz'}",
        ],
    )
    # Documentation and doctests.
    rm_build_action = f"rm -rf {module}/docs/_build"
    action = SubprocessAction(
        f"sphinx-build -n -W . {workspace}/docs/{module}", env={"PROJECT": module}
    )
    manager(basename="docs", name=module, actions=[rm_build_action, action])
    action = SubprocessAction(
        f"sphinx-build -b doctest . {workspace}/docs/{module}", env={"PROJECT": module}
    )
    manager(basename="doctest", name=module, actions=[rm_build_action, action])

    # Compile example notebooks to create html reports.
    for path in Path.cwd().glob(f"{module}/**/*.ipynb"):
        exclude = [".ipynb_checkpoints", "jupyter_execute", ".jupyter_cache"]
        if any(x in path.parts for x in exclude):
            continue
        target = path.with_suffix(".html")
        manager(
            basename="compile_example",
            name=path.with_suffix("").name,
            file_dep=[path],
            targets=[target],
            actions=[f"jupyter nbconvert --execute --to=html {path}"],
        )


def add_profile_task(
    method: str,
    parameterization: str,
    log10_sigma: float,
    size: int,
    max_chains: int = None,
    timeout: float = None,
    iter_sampling: int = None,
    train_frac: float = 1,
    suffix: str = "",
):
    timeout = timeout or (10 if fast else 60)
    max_chains = max_chains or (2 if fast else 20)
    iter_sampling = iter_sampling or (10 if fast else 100)
    name = f"log10_noise_scale-{log10_sigma:.3f}_size-{size}{suffix}"
    target = workspace / f"profile/{method}/{parameterization}/{name}.pkl"
    args = [
        "python",
        "-m",
        "gptools.stan.profile",
        method,
        parameterization,
        10**log10_sigma,
        target,
        f"--iter_sampling={iter_sampling}",
        f"--n={size}",
        f"--max_chains={max_chains}",
        f"--timeout={timeout}",
        f"--train_frac={train_frac}",
    ]
    file_dep = [
        "profile/__main__.py",
        "gptools/fft1.stan",
        "gptools/graph.stan",
        "gptools/util.stan",
        "profile/data.stan",
        f"profile/{parameterization}.stan",
    ]
    prefix = Path("stan/gptools/stan")
    manager(
        basename=f"profile/{method}/{parameterization}",
        name=name,
        actions=[args],
        targets=[target],
        file_dep=[prefix / x for x in file_dep],
    )


# Run different profiling configurations. We expect the centered parameterization to be better for
# strong data and the non-centered parameterization to be better for weak data.
try:
    from gptools.stan.profile import (
        FOURIER_ONLY_SIZE_THRESHOLD,
        LOG10_NOISE_SCALES,
        PARAMETERIZATIONS,
        SIZES,
    )

    with di.group_tasks("profile") as profile_group:
        product = it.product(PARAMETERIZATIONS, LOG10_NOISE_SCALES, SIZES)
        for parameterization, log10_sigma, size in product:
            # Only run Fourier methods if the size threshold is exceeded.
            if size >= FOURIER_ONLY_SIZE_THRESHOLD and not parameterization.startswith(
                "fourier"
            ):
                continue
            add_profile_task("sample", parameterization, log10_sigma, size)

        # Add variational inference.
        for parameterization, log10_sigma in it.product(
            PARAMETERIZATIONS, LOG10_NOISE_SCALES
        ):
            add_profile_task(
                "variational", parameterization, log10_sigma, 1024, train_frac=0.8
            )
            # Here, we use a long timeout and many samples to ensure we get the distributions right.
            add_profile_task(
                "sample",
                parameterization,
                log10_sigma,
                1024,
                train_frac=0.8,
                suffix="-train-test",
                iter_sampling=500,
                timeout=300,
            )

        # Add a one-off task to calculate statistics for the abstract with 10k observations.
        add_profile_task("sample", "fourier_centered", 0, 10_000, timeout=300)
        add_profile_task("sample", "fourier_non_centered", 0, 10_000, timeout=300)
except ModuleNotFoundError:
    pass


# Tree data from https://datadryad.org/stash/dataset/doi:10.15146/5xcp-0d46.
with di.defaults(basename="trees"):
    # Download elevation data.
    target = "data/elevation.tsv"
    url = "https://datadryad.org/stash/downloads/file_stream/148941"
    manager(
        name="elevation",
        targets=[target],
        actions=[["curl", "-L", "-o", "$@", url]],
        uptodate=[True],
    )

    # Download the tree archive.
    archive = "data/bci.tree.zip"
    url = "https://datadryad.org/stash/downloads/file_stream/148942"
    manager(
        name="zip",
        targets=[archive],
        actions=[["curl", "-L", "-o", "$@", url]],
        uptodate=[True],
    )
    # Extract the rdata from the archive.
    rdata_target = "data/bci.tree8.rdata"
    manager(
        name="rdata",
        targets=[rdata_target],
        file_dep=[archive],
        actions=[
            [
                "unzip",
                "-jo",
                archive,
                "home/fullplotdata/tree/bci.tree8.rdata",
                "-d",
                "data",
            ]
        ],
    )
    # Convert to CSV.
    csv_target = "data/bci.tree8.csv"
    manager(
        name="csv",
        targets=[csv_target],
        file_dep=[rdata_target],
        actions=[
            [
                "Rscript",
                "--vanilla",
                "data/rda2csv.r",
                rdata_target,
                "bci.tree8",
                csv_target,
            ],
        ],
    )
    # Aggregate the trees to get a smaller dataset.
    for species in ["tachve"]:
        target = f"data/{species}.csv"
        script = "data/aggregate_trees.py"
        manager(
            name=species,
            targets=[target],
            file_dep=[csv_target, script],
            actions=[["$!", script, csv_target, species, target]],
        )


with di.defaults(basename="tube"):
    url = (
        "http://crowding.data.tfl.gov.uk/Annual%20Station%20Counts/2019/"
        "AnnualisedEntryExit_2019.xlsx"
    )
    entry_exit_target = "data/AnnualisedEntryExit_2019.xlsx"
    manager(
        name="entry-exit-data",
        targets=[entry_exit_target],
        actions=[["curl", "-L", "-o", "$@", url]],
        uptodate=[True],
    )
    graph_target = "data/tube.json"
    manager(
        name="graph",
        targets=[graph_target],
        file_dep=[entry_exit_target],
        actions=[
            ["$!", "data/construct_tube_network.py", entry_exit_target, graph_target]
        ],
    )
    # Tube data prepared data for Stan.
    target = "data/tube-stan.json"
    manager(
        name="prepared",
        targets=[target],
        file_dep=[graph_target],
        actions=[["$!", "data/prepare_tube_data.py", graph_target, target]],
    )


# Produce dedicated figures that aren't part of the documentation.
for notebook in Path.cwd().glob("figures/*.md"):
    # First use jupytext to convert to a classic notebook. Then execute.
    name = notebook.with_suffix("").name
    ipynb = notebook.with_suffix(".tmp.ipynb")
    pdf = workspace / f"{name}.pdf"
    html = workspace / f"{name}.tmp.html"
    actions = [
        SubprocessAction(
            f"jupytext --from md --to ipynb --output {ipynb} {notebook} "
            f"&& jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 --to=html "
            f"--output-dir={workspace} {ipynb}",
            env={"WORKSPACE": workspace},
            shell=True,
        ),
    ]
    # The profile target only exists for the Stan module.
    task_dep = []

    if name == "profile":
        try:
            import gptools.stan as _  # noqa: F401

            task_dep = ["profile"]
        except ModuleNotFoundError:
            pass
    manager(
        basename="figures",
        name=name,
        file_dep=[notebook],
        targets=[pdf, html],
        actions=actions,
        task_dep=task_dep,
    )

# Meta target to generate all results for Stan.
try:
    import gptools.stan as _  # noqa: F401, F811

    manager(basename="results", name="stan", task_dep=["docs:stan", "figures"])
except ModuleNotFoundError:
    pass
