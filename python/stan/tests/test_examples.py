from pathlib import Path


def test_minimal_trees(data_root: Path, docs_root: Path) -> None:
    """
    Minimal reproducible example to include in the publication.
    """
    from gptools.stan import compile_model
    import numpy as np

    # Load tree frequency matrix, define padding, and apply training mask.
    frequency = np.loadtxt(data_root / "tachve.csv", delimiter=",", dtype=int)
    num_rows, num_cols = frequency.shape
    padding = 10
    train_mask = np.random.binomial(1, 0.8, frequency.shape)
    data = {
        "num_rows": num_rows,
        "num_rows_padded": num_rows + padding,
        "num_cols": num_cols,
        "num_cols_padded": num_cols + padding,
        "frequency": np.where(train_mask, frequency, -1),
    }

    # Compile model and fit it.
    model = compile_model(stan_file=docs_root / "trees/trees.stan")
    fit = model.sample(data, iter_sampling=10, iter_warmup=10, chains=1, seed=0)
    print(fit.diagnose())


def test_minimal_tube(data_root: Path, docs_root: Path) -> None:
    """
    Minimal reproducible example to include in the publication.
    """
    from gptools.stan import compile_model
    import json
    import numpy as np

    # Load station locations, edges, passenger numbers, and apply training mask.
    with open(data_root / "tube-stan.json") as fp:
        data = json.load(fp)
    train_mask = np.random.binomial(1, 0.8, data["num_stations"])
    data["passengers"] = np.where(train_mask, data["passengers"], -1)

    data.update(
        {
            "include_degree_effect": 1,
            "include_zone_effect": 1,
        }
    )

    # Compile model and fit it.
    model = compile_model(stan_file=docs_root / "tube/tube.stan")
    fit = model.sample(data, iter_sampling=10, iter_warmup=10, chains=1, seed=0)
    print(fit.diagnose())
