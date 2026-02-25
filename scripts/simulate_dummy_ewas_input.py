import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd


def main(input_path, output_dir, n_simulations):
    """ Simulate dummy EWAS input files based on real data. To run it from the command line, use:
    python scripts/simulate_dummy_ewas_input.py data/prs.csv output/ewas_simulations 10
    Args:
        input_path (str): Path to the input data file (CSV).
        output_dir (str): Directory where the simulated files will be saved.
        n_simulations (int): Number of simulations to run.
    Returns:
        None
    """
    print(f"Reading input data from: {input_path}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    prs_col = "stdPRS_epilepsy"

    print(f"Simulating {n_simulations} EWAS input files.")
    for i in range(n_simulations):
        # Simulate group labels based on PRS values using a logistic function
        logits = df[prs_col]
        probs = 1 / (1 + np.exp(-logits))
        group = np.random.binomial(1, probs)

        # Create output DataFrame with IID and group labels
        out = df[["IID"]].copy()
        out["group"] = np.where(group == 1, "val", "ctrl")
        out_path = os.path.join(run_dir, f"{i}_ewas_input.csv")
        out.to_csv(out_path, index=False)

    print(f"Output will be saved to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate dummy EWAS input data from real data.")
    parser.add_argument("input_path", type=str, help="Path to the input data file.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("n_simulations", type=int, help="Number of simulations to run.")
    args = parser.parse_args()

    main(args.input_path, args.output_dir, args.n_simulations)
