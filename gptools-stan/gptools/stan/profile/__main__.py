import argparse
from gptools.stan import compile_model
from gptools.util import Timer
from gptools.util.kernels import ExpQuadKernel, DiagonalKernel
from gptools.util.graph import lattice_predecessors, predecessors_to_edge_index
from gptools.util.timeout import call_with_timeout
import numpy as np
import pathlib
import pickle
import tabulate
from scipy import stats
from tqdm import tqdm
from typing import Optional
from . import PARAMETERIZATIONS, sample_and_load_fit


def __main__(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="inference method to use", choices={"sample", "variational"})
    parser.add_argument("parameterization", help="parameterization of the model",
                        choices=PARAMETERIZATIONS)
    parser.add_argument("noise_scale", help="scale of observation noise", type=float)
    parser.add_argument("output", help="output path", nargs="?")
    parser.add_argument("--n", help="number of observations", type=int, default=100)
    parser.add_argument("--train_frac", help="fraction of points to use for training", type=float,
                        default=1.0)
    parser.add_argument("--num_parents", help="number of parents for GP on graphs", type=int,
                        default=5)
    parser.add_argument("--sigma", help="scale of Gaussian process covariance", type=float,
                        default=1.0)
    parser.add_argument("--length_scale", help="correlation length of Gaussian process covariance",
                        type=float, default=1.0)
    parser.add_argument("--epsilon", help="diagonal contribution to Gaussian process covariance",
                        type=float, default=1e-3)
    parser.add_argument("--seed", help="random number generator seed", type=int, default=42)
    parser.add_argument("--iter_sampling", help="number of posterior samples", type=int,
                        default=500)
    parser.add_argument("--show_progress", help="show progress bars", action="store_true")
    parser.add_argument("--show_diagnostics", help="show cmdstanpy diagnostics",
                        action="store_true")
    parser.add_argument("--iter_warmup", help="number of warmup samples", type=int)
    parser.add_argument("--timeout", help="timeout in seconds", type=float, default=60)
    parser.add_argument("--max_chains", type=int, default=1, help="maximum number of chains to "
                        "run; use -1 for an unlimited number of chains")
    parser.add_argument("--ignore_converged", action="store_true",
                        help="do not check if the variational algorithm has converged")
    args = parser.parse_args(args)

    # Compile the model.
    stan_file = pathlib.Path(__file__).parent / f"{args.parameterization}.stan"
    model = compile_model(stan_file=stan_file)

    # Prepare the results container.
    result = {"args": vars(args)}

    np.random.seed(args.seed)
    i = 0

    # Prepare the distribution outside the loop because matrix inversion can take a while.
    # Generate data from a Gaussian process with normal observation noise.
    X = np.arange(args.n)[:, None]
    kernel = ExpQuadKernel(args.sigma, args.length_scale) + DiagonalKernel(args.epsilon)
    cov = kernel.evaluate(X)
    dist = stats.multivariate_normal(np.zeros(args.n), cov)

    with Timer() as total_timer, tqdm() as progress:
        while (args.max_chains == -1 or i < args.max_chains) \
                and (args.timeout is None or total_timer.duration < args.timeout):

            # Sample the Gaussian process.
            eta = dist.rvs()
            y = np.random.normal(eta, args.noise_scale)

            # Construct the nearest-neighbor graph.
            predecessors = lattice_predecessors((args.n,), args.num_parents)
            edge_index = predecessors_to_edge_index(predecessors)

            # Sample observed points and fit the model.
            num_observed = np.random.binomial(args.n, args.train_frac)
            observed_idx = np.random.choice(args.n, size=num_observed, replace=False) + 1
            data = {
                "n": args.n,
                "num_dims": 1,
                "X": X,
                "y": y,
                "sigma": args.sigma,
                "length_scale": args.length_scale,
                "epsilon": args.epsilon,
                "num_edges": edge_index.shape[1],
                "edge_index": edge_index,
                "noise_scale": args.noise_scale,
                "num_observed": num_observed,
                "observed_idx": observed_idx,
            }

            with Timer() as timer:
                try:
                    kwargs = {
                        "data": data,
                        "seed": args.seed,
                    }
                    if args.method == "sample":
                        iter_warmup = args.iter_warmup or args.iter_sampling
                        fit = call_with_timeout(
                            args.timeout, sample_and_load_fit, model,
                            iter_sampling=args.iter_sampling, chains=1, threads_per_chain=1,
                            show_progress=args.show_progress, iter_warmup=iter_warmup, **kwargs,
                        )
                    elif args.method == "variational":
                        fit = call_with_timeout(
                            args.timeout, model.variational, output_samples=args.iter_sampling,
                            require_converged=not args.ignore_converged, **kwargs,
                        )
                    else:  # pragma: no cover
                        raise ValueError(args.method)
                    timeout = False
                except TimeoutError:
                    fit = None
                    timeout = True

            result.setdefault("durations", []).append(timer.duration)
            result.setdefault("timeouts", []).append(timeout)
            result.setdefault("fits", []).append(fit)
            result.setdefault("data", []).append(data)
            result.setdefault("etas", []).append(eta)
            progress.update()
            i += 1

    for key in ["durations", "timeouts", "etas"]:
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
    }
    if args.method == "sample":
        values.update({
            "divergences": f"{fit.divergences.sum()} / {fit.num_draws_sampling} "
                f"({100 * fit.divergences.sum() / fit.num_draws_sampling:.1f}%)",  # noqa: E131
            "max_treedepths": f"{fit.max_treedepths.sum()} / {fit.num_draws_sampling} "
                f"({100 * fit.max_treedepths.sum() / fit.num_draws_sampling:.1f}%)",  # noqa: E131
        })
    rows = [(key, str(value)) for key, value in values.items()]
    print(tabulate.tabulate(rows))

    if args.method == "sample" and args.show_diagnostics:
        print(fit.diagnose())


if __name__ == "__main__":
    __main__()
