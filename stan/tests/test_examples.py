from gptools.util.testing import parameterized_notebook_tests


test_example = parameterized_notebook_tests(__file__, "../docs")


def test_minimal_trees() -> None:
    """
    Minimal reproducible example to include in the publication.
    """
    from gptools.stan import compile_model
    import numpy as np

    # Load tree frequency matrix, define padding, and prepare data dictionary.
    frequency = np.loadtxt("data/tachve.csv", delimiter=",", dtype=int)
    num_rows, num_cols = frequency.shape
    padding = 10
    data = {
        "num_rows": num_rows,
        "num_rows_padded": num_rows + padding,
        "num_cols": num_cols,
        "num_cols_padded": num_cols + padding,
        "frequency": frequency,
    }

    # Compile model and fit it.
    model = compile_model(stan_file="stan/docs/trees/trees.stan")
    fit = model.sample(data, iter_sampling=10, iter_warmup=10, chains=1, seed=0)
    print(fit.diagnose())


def test_minimal_tube() -> None:
    """
    Minimal reproducible example to include in the publication.
    """
    from gptools.stan import compile_model
    import json

    # Load station locations, edges, passenger numbers, and prepare data dictionary.
    with open("data/tube-stan.json") as fp:
        data = json.load(fp)
    data.update({
        "include_degree_effect": 1,
        "include_zone_effect": 1,
    })

    # Compile model and fit it.
    model = compile_model(stan_file="stan/docs/tube/tube.stan")
    fit = model.sample(data, iter_sampling=10, iter_warmup=10, chains=1, seed=0)
    print(fit.diagnose())
