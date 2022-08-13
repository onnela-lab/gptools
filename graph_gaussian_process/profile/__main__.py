import argparse
import logging
import cmdstanpy
from graph_gaussian_process.kernels import ExpQuadKernel
from graph_gaussian_process.stan import get_include
from graph_gaussian_process.util import lattice_predecessors, predecessors_to_edge_index
import numpy as np
import pathlib
import pickle
import tabulate
import time
import typing


def __main__(args: typing.Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parametrization", help="parametrization of the model",
                        choices={"centered", "non_centered"})
    parser.add_argument("noise_scale", help="scale of observation noise", type=float)
    parser.add_argument("output", help="output path", nargs="?")
    parser.add_argument("--num_nodes", help="number of nodes", type=int, default=100)
    parser.add_argument("--num_parents", help="number of parents", type=int, default=5)
    parser.add_argument("--alpha", help="scale of Gaussian process covariance", type=float,
                        default=1.0)
    parser.add_argument("--rho", help="correlation length of Gaussian process covariance",
                        type=float, default=0.1)
    parser.add_argument("--epsilon", help="diagonal contribution to Gaussian process covariance",
                        type=float, default=1e-3)
    parser.add_argument("--seed", help="random number generator seed", type=int, default=42)
    parser.add_argument("--iter_sampling", help="number of posterior samples", type=int,
                        default=500)
    parser.add_argument("--show_progress", help="show progress bars", action="store_true")
    parser.add_argument("--show_diagnostics", help="show cmdstanpy diagnostics",
                        action="store_true")
    parser.add_argument("--compile", help="whether to compile the model", default="true",
                        choices={"false", "true", "force"})
    parser.add_argument("--iter_warmup", help="number of warmup samples", type=int)
    args = parser.parse_args(args)

    cmdstanpy_logger = cmdstanpy.utils.get_logger()
    for handler in cmdstanpy_logger.handlers:
        handler.setLevel(logging.WARNING)

    # Generate data from a Gaussian process with normal observation noise.
    np.random.seed(args.seed)
    X = np.linspace(0, 1, args.num_nodes)[:, None]
    kernel = ExpQuadKernel(args.alpha, args.rho, args.epsilon)
    cov = kernel(X)
    eta = np.random.multivariate_normal(np.zeros(args.num_nodes), cov)
    y = np.random.normal(eta, args.noise_scale)

    predecessors = lattice_predecessors((args.num_nodes,), args.num_parents)
    edge_index = predecessors_to_edge_index(predecessors)

    # Compile the model and fit it.
    compile = {"false": False, "true": True, "force": "force"}[args.compile]
    model = cmdstanpy.CmdStanModel(
        stan_file=pathlib.Path(__file__).parent / f"{args.parametrization}.stan",
        stanc_options={"include-paths": [get_include()]}, compile=compile,
    )
    data = {
        "num_nodes": args.num_nodes,
        "num_dims": 1,
        "X": X,
        "y": y,
        "alpha": kernel.alpha,
        "rho": kernel.rho,
        "epsilon": kernel.epsilon,
        "num_edges": edge_index.shape[1],
        "edge_index": edge_index,
        "noise_scale": args.noise_scale,
    }
    start = time.time()
    fit = model.sample(
        data, seed=args.seed, iter_sampling=args.iter_sampling, show_progress=args.show_progress,
        iter_warmup=args.iter_warmup or args.iter_sampling,
    )
    end = time.time()

    # Save the result.
    result = {
        "args": vars(args),
        "start": start,
        "end": end,
        "fit": fit,
    }
    if args.output:
        with open(args.output, "wb") as fp:
            pickle.dump(result, fp)

    # Report the results.
    values = vars(args) | {
        "duration": f"{end - start:.3f}",
        "divergences": f"{fit.divergences.sum()} / {fit.num_draws_sampling} "
            f"({100 * fit.divergences.sum() / fit.num_draws_sampling:.1f}%)",  # noqa: E131
        "max_treedepths": f"{fit.max_treedepths.sum()} / {fit.num_draws_sampling} "
            f"({100 * fit.max_treedepths.sum() / fit.num_draws_sampling:.1f}%)",  # noqa: E131
    }
    rows = [(key, str(value)) for key, value in values.items()]
    print(tabulate.tabulate(rows))

    if args.show_diagnostics:
        print(fit.diagnose())


if __name__ == "__main__":
    __main__()
