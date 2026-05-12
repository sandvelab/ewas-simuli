import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_prob_histogram(probs, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs, bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("P(group = val)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of assignment probabilities")
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_simulation(logits, group, prs_col, sim_index, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(logits.min(), logits.max(), 40)
    ax.hist(logits[group == 0], bins=bins, alpha=0.6,
            label=f"ctrl (n={(group == 0).sum()})", color="lightcoral")
    ax.hist(logits[group == 1], bins=bins, alpha=0.6,
            label=f"val (n={(group == 1).sum()})", color="steelblue")
    ax.set_xlabel(prs_col)
    ax.set_ylabel("Count")
    ax.set_title(f"Simulation {sim_index}: PRS distribution by assigned group")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_meta(out_path, **params):
    with open(out_path, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


def to_quantiles(values):
    """Rank-transform to uniform [0,1], then center to [-0.5, 0.5]."""
    ranks = pd.Series(values).rank(method="average")
    q = ranks / (len(ranks) + 1)
    return q - 0.5   # center so intercept=0 gives ~50% prevalence


def main(input_path, output_dir, n_simulations, param, hard, seed, quantile):
    """ Simulate dummy EWAS input files based on real data. To run it from the command line, use:
    python scripts/simulate_dummy_ewas_input.py data/prs.csv output/ewas_simulations 10
    Args:
        input_path (str): Path to the input data file (CSV).
        output_dir (str): Directory where the simulated files will be saved.
        n_simulations (int): Number of simulations to run.
        param (float): Parameter to control the strength of association between PRS and group assignment.
        1 means a moderate association, higher values increase the strength of association.
        hard (bool): If True, assign groups based on a hard threshold of 0.5 on the probabilities.
        seed (int): Random seed for reproducibility.
        quantile (bool): If True, rank-transform PRS to uniform before applying sigmoid.
    Returns:
        None
    """
    print(f"Reading input data from: {input_path}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    if seed is not None:
        np.random.seed(seed)

    df = pd.read_csv(input_path)
    prs_col = "stdPRS_height"
    logits = df[prs_col]

    if quantile:
        logits = to_quantiles(logits)

    probs = 1 / (1 + np.exp(-param*logits))
    plot_prob_histogram(probs, os.path.join(run_dir, "prob_histogram.png"))

    write_meta(
        os.path.join(run_dir, "meta.txt"),
        timestamp=timestamp,
        input_path=input_path,
        n_simulations=n_simulations,
        n_samples=len(df),
        sigmoid_param=param,
        hard_thresholding=hard,
        random_seed=seed,
        quantile_transform=quantile
    )
    if hard:
        print("Using hard thresholding for group assignment (threshold = 0.5).")
        group = (probs >= 0.5).astype(int)

        out = df[["MID", "CID"]].copy()
        out["group"] = np.where(group == 1, "val", "ctrl")
        out.to_csv(os.path.join(run_dir, "0_ewas_input.csv"), index=False)

        plot_simulation(logits, group, prs_col, "hard",
                        os.path.join(run_dir, "0_ewas_input.png"))
    else:
        for i in range(n_simulations):
            print(f"Simulating {i} EWAS input file.")
            # Simulate group labels based on PRS values using a logistic function
            group = np.random.binomial(1, probs)
            # Create output DataFrame with IID and group labels
            out = df[["MID", "CID"]].copy()
            out["group"] = np.where(group == 1, "val", "ctrl")
            out_path = os.path.join(run_dir, f"{i}_ewas_input.csv")
            out.to_csv(out_path, index=False)
            plot_simulation(logits, group, prs_col, i, os.path.join(run_dir, f"{i}_ewas_input.png"))
    print(f"Output will be saved to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate dummy EWAS input data from real data.")
    parser.add_argument("input_path", type=str, help="Path to the input data file.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("n_simulations", type=int, help="Number of simulations to run.")
    parser.add_argument("param", type=float, help="Parameter to control the strength of association "
                                                  "between PRS and group assignment.")
    parser.add_argument("--hard", action="store_true", help="If True, assign groups based on a hard "
                                                            "threshold of 0.5 on the probabilities.")
    parser.add_argument("seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument("--quantile", action="store_true",
                        help="If set, rank-transform PRS to uniform before applying sigmoid.")
    args = parser.parse_args()

    main(args.input_path, args.output_dir, args.n_simulations, args.param, args.hard, args.seed, args.quantile)
