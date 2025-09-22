import argparse
import os
import sys
import re

import pandas as pd


def main(input_path, output_dir):
    if not os.path.isfile(input_path):
        print(f"Error: The input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not os.path.isdir(output_dir):
        print(f"Error: The output path '{output_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing input file: {input_path}")

    input_df = pd.read_csv(input_path)
    cols = input_df.columns.tolist()

    methylation_cols = sorted(
        [c for c in input_df.columns if re.match(r"DNAm_\d+-\d+", c)],
        key=lambda x: int(re.search(r"DNAm_(\d+)-", x).group(1))
    )
    genotype_child_cols = sorted(
        [c for c in input_df.columns if re.match(r"geno_\d+-\d+", c)],
        key=lambda x: int(re.search(r"geno_(\d+)-", x).group(1))
    )
    genotype_m_cols = sorted(
        [c for c in input_df.columns if re.match(r"geno_M\d+", c)],
        key=lambda x: int(re.search(r"geno_M(\d+)", x).group(1))
    )
    exposure_m_cols = sorted(
        [c for c in input_df.columns if re.match(r"matEsc_\d+-\d+", c)],
        key=lambda x: int(re.search(r"matEsc_(\d+)-", x).group(1))
    )
    outcome_m_cols = sorted(
        [c for c in input_df.columns if re.match(r"matDep_\d+-\d+", c)],
        key=lambda x: int(re.search(r"matDep_(\d+)-", x).group(1))
    )

    for row in input_df.itertuples(index=False):
        cpg_id = row[cols.index("cpgID")]
        cpg_chrPos = row[cols.index("cpg_chrPos")]
        snp_id = row[cols.index("SNPID")]
        snp_chrPos = row[cols.index("SNP_chrPos")]

        for i in range(len(methylation_cols)):
            sample_id = i + 1
            methylation_value = row[cols.index(methylation_cols[i])]
            genotype_child_value = row[cols.index(genotype_child_cols[i])]
            genotype_m_value = row[cols.index(genotype_m_cols[i])]
            exposure_m_value = row[cols.index(exposure_m_cols[i])]
            outcome_m_value = row[cols.index(outcome_m_cols[i])]

            output_row = {
                "cpg_id": cpg_id,
                "cpg_chrPos": cpg_chrPos,
                "snp_id": snp_id,
                "snp_chr_pos": snp_chrPos,
                "sample_id": sample_id,
                "dna_methyl_child": methylation_value,
                "geno_child": genotype_child_value,
                "geno_mother": genotype_m_value,
                "exposure_mother": exposure_m_value,
                "outcome_mother": outcome_m_value
            }

            output_file = os.path.join(output_dir, f"{cpg_id}.csv")
            output_df = pd.DataFrame([output_row])
            if not os.path.isfile(output_file):
                output_df.to_csv(output_file, index=False)
            else:
                output_df.to_csv(output_file, mode='a', header=False, index=False)

    print(f"Output will be saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform input data and save results to an output directory.")
    parser.add_argument("input_path", nargs="?", type=str, help="Path to the input data file.",
                        default="data/synthetic_ewas_input_small.csv")
    parser.add_argument("output_dir", nargs="?", type=str, help="Path to the output directory.",
                        default="results/transformed_data")
    args = parser.parse_args()

    main(args.input_path, args.output_dir)
