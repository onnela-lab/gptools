import argparse
import logging
import cmdstanpy
from gptools.stan import compile_model
from gptools.util import Timer
from gptools.util.kernels import ExpQuadKernel
from gptools.util.graph import lattice_predecessors, predecessors_to_edge_index
import numpy as np
import pathlib
import pickle
import tabulate
from tqdm import tqdm
from typing import Optional
from . import PARAMETERIZATIONS


def __main__(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parameterization", help="parameterization of the model",
                        choices=PARAMETERIZATIONS)
    parser.add_argument("noise_scale", help="scale of observation noise", type=float)
    parser.add_argument("output", help="output path", nargs="?")
    parser.add_argument("--n", help="number of observations", type=int, default=100)
    parser.add_argument("--num_parents", help="number of parents for GP on graphs", type=int,
                        default=5)
    parser.add_argument("--alpha", help="scale of Gaussian process covariance", type=float,
                        default=1.0)
    parser.add_argument("--rho", help="correlation length of Gaussian process covariance",
                        type=float, default=1.0)
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
    parser.add_argument("--timeout", help="timeout in seconds", type=float, default=60)
    parser.add_argument("--max_chains", type=int, default=1, help="maximum number of chains to "
                        "run; use -1 for an unlimited number of chains")
    args = parser.parse_args(args)

    # Make cmdstanpy less verbose.
    cmdstanpy_logger = cmdstanpy.utils.get_logger()
    for handler in cmdstanpy_logger.handlers:
        handler.setLevel(logging.WARNING)

    # Compile the model.
    compile = {"false": False, "true": True, "force": "force"}[args.compile]
    model = compile_model(
        stan_file=pathlib.Path(__file__).parent / f"{args.parameterization}.stan",
        compile=compile,
    )

    # Prepare the results container.
    result = {
        "args": vars(args),
    }

    np.random.seed(args.seed)
    i = 0
    with Timer() as total_timer, tqdm() as progress:
        while (args.max_chains == -1 or i < args.max_chains) \
                and (args.timeout is None or total_timer.duration < args.timeout):
            # Generate data from a Gaussian process with normal observation noise.
            X = np.arange(args.n)[:, None]
            kernel = ExpQuadKernel(args.alpha, args.rho, args.epsilon)
            cov = kernel(X)
            eta = np.random.multivariate_normal(np.zeros(args.n), cov)
            y = np.random.normal(eta, args.noise_scale)

            predecessors = lattice_predecessors((args.n,), args.num_parents)
            edge_index = predecessors_to_edge_index(predecessors)

            # Fit the model.
            data = {
                "n": args.n,
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

            with Timer() as timer:
                try:
                    fit = model.sample(
                        data, seed=args.seed, iter_sampling=args.iter_sampling, chains=1,
                        show_progress=args.show_progress, threads_per_chain=1,
                        iter_warmup=args.iter_warmup or args.iter_sampling, timeout=args.timeout,
                    )
                    timeout = False
                except TimeoutError:
                    fit = None
                    timeout = True

            result.setdefault("durations", []).append(timer.duration)
            result.setdefault("timeouts", []).append(timeout)
            result.setdefault("fits", []).append(fit)
            progress.update()
            i += 1

    for key in ["durations", "timeouts"]:
        result[key] = np.asarray(result[key])

    if args.output:
        with open(args.output, "wb") as fp:
            pickle.dump(result, fp)

    # Report the results.
    if all(result["timeouts"]):
        print(f"all chains timed out after {args.timeout:.3f} seconds")
        return

    # Show results on the first fit that didn't time out.
    for fit, timeout in zip(result["fits"], result["timeouts"]):
        if not timeout:
            break
    values = vars(args) | {
        "duration": f"{timer.duration:.3f}",
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
