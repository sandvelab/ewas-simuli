import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHENOTYPE_COL = "matHeight"


def plot_simulation_continuous(heights, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(heights, bins=40, alpha=0.7, color="steelblue")
    ax.set_xlabel(PHENOTYPE_COL)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {PHENOTYPE_COL}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_simulation_group(heights, group, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(heights.min(), heights.max(), 40)
    ax.hist(heights[group == 0], bins=bins, alpha=0.6,
            label=f"ctrl (n={(group == 0).sum()})", color="lightcoral")
    ax.hist(heights[group == 1], bins=bins, alpha=0.6,
            label=f"val (n={(group == 1).sum()})", color="steelblue")
    ax.set_xlabel(PHENOTYPE_COL)
    ax.set_ylabel("Count")
    ax.set_title(f"Height distribution by assigned group (threshold = mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_meta(out_path, **params):
    with open(out_path, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


def main(input_path, output_dir, mode, standardize):
    print(f"Reading input data from: {input_path}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    before = len(df)
    df = df.dropna(subset=[PHENOTYPE_COL])
    df = df.drop_duplicates(subset=["MID"])
    after = len(df)
    print(f"Dropped {before - after} rows with missing '{PHENOTYPE_COL}'. {after} rows remaining.")

    heights = df[PHENOTYPE_COL].values
    mean_height = heights.mean()
    std_height = heights.std()
    print(f"Mean {PHENOTYPE_COL}: {mean_height:.4f}")
    print(f"Std  {PHENOTYPE_COL}: {std_height:.4f}")

    out = df[["MID", "CID"]].copy()

    if mode == "continuous":
        # CHANGED: --standardize flag instead of --include_height; raw height is default
        if standardize:
            out["phenotype"] = (heights - mean_height) / std_height
        else:
            out["phenotype"] = heights
        plot_simulation_continuous(heights, os.path.join(run_dir, "ewas_input.png"))
        write_meta(
            os.path.join(run_dir, "meta.txt"),
            timestamp=timestamp,
            input_path=input_path,
            phenotype_col=PHENOTYPE_COL,
            mode=mode,
            n_samples=after,
            n_dropped=before - after,
            mean=round(mean_height, 4),
            std=round(std_height, 4),
            # CHANGED: reflects standardize flag
            phenotype_values="standardized_height" if standardize else "raw_height",
        )

    elif mode == "group":
        group = (heights >= mean_height).astype(int)
        print(f"val (tall):   {(group == 1).sum()}")
        print(f"ctrl (short): {(group == 0).sum()}")
        out["group"] = np.where(group == 1, "val", "ctrl")
        plot_simulation_group(heights, group, os.path.join(run_dir, "ewas_input.png"))
        write_meta(
            os.path.join(run_dir, "meta.txt"),
            timestamp=timestamp,
            input_path=input_path,
            phenotype_col=PHENOTYPE_COL,
            mode=mode,
            n_samples=after,
            n_dropped=before - after,
            mean_threshold=round(mean_height, 4),
            n_val=int((group == 1).sum()),
            n_ctrl=int((group == 0).sum()),
        )

    out.to_csv(os.path.join(run_dir, "0_ewas_input.csv"), index=False)
    print(f"Output saved to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare phenotype for EWAS.")
    parser.add_argument("input_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("--mode", type=str, choices=["continuous", "group"], default="continuous",
                        help="Output mode: 'continuous' for phenotype column, 'group' for val/ctrl column.")
    parser.add_argument("--standardize", action="store_true",
                        help="(continuous mode only) Standardize height to mean=0, sd=1.")
    args = parser.parse_args()

    main(args.input_path, args.output_dir, args.mode, args.standardize)