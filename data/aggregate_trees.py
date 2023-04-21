import argparse
import numpy as np
import pandas as pd
from typing import List, Optional


def __main__(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="unaggregated trees extracted from rdata file")
    parser.add_argument("species", help="tree species to extract")
    parser.add_argument("output", help="output file")
    args = parser.parse_args(args)

    # Load all trees and filter to the desired species.
    data = pd.read_csv(args.input)
    data = data[data.sp == args.species]
    if len(data) == 0:
        raise ValueError(f"species {args.species} does not have any trees")
    print(f"loaded {len(data)} trees of species {args.species} from {args.input}")

    # Aggregate the trees and scatter them into a matrix.
    aggregated = data.groupby("quadrat").apply(len)
    i = (aggregated.index % 100).astype(int)
    j = (aggregated.index // 100).astype(int)
    frequency = np.zeros((25, 50), dtype=int)
    frequency[i, j] = aggregated.values

    # Save the results as comma-separated integers.
    header = f"{frequency.sum()} {args.species} trees from {args.input}."
    np.savetxt(args.output, frequency, delimiter=",", fmt="%d", header=header)
    print(f"dumped tree frequencies to {args.output}")
    return frequency


if __name__ == "__main__":
    __main__()
