import csv
import random

# Parameters
n_cpgs = 245862
n_samples = 773
file_name = "data/synthetic_ewas_input.csv"
possible_alleles = ['A', 'C', 'G', 'T']

# Helper functions
def random_dnam():
    """Return random DNA methylation value between 0 and 1 (3 decimals)."""
    return round(random.uniform(0, 1), 3)

def random_geno(alleles):
    """Return random genotype string (2 alleles A/C/G/T)."""
    return random.choice(alleles) + random.choice(alleles)

def random_binary():
    """Return 0 or 1 (maternal escape/depression indicators)."""
    return random.randint(0, 1)

# Build header
header = (
    ["cpgID", "cpg_chrPos", "SNPID", "SNP_chrPos"]
    + [f"DNAm_{i+1}-1" for i in range(n_samples)]
    + [f"geno_{i+1}-1" for i in range(n_samples)]
    + [f"geno_M{i+1}" for i in range(n_samples)]
    + [f"matEsc_{i+1}-1" for i in range(n_samples)]
    + [f"matDep_{i+1}-1" for i in range(n_samples)]
)

# Write CSV
with open(file_name, "w", newline="") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(header)

    for idx in range(1, n_cpgs + 1):
        cpg_id = f"cg{idx}"
        cpg_chrPos = f"chr{random.randint(1, 22)}:{random.randint(1, 1_000_000)}"
        snp_id = random.randint(100, 999_999)
        snp_chrPos = f"chr{random.randint(1, 22)}:{random.randint(1, 1_000_000)}"
        alleles = random.sample(possible_alleles, 2)

        dnam_vals = [random_dnam() for _ in range(n_samples)]
        geno_vals = [random_geno(alleles) for _ in range(n_samples)]
        genoM_vals = [random_geno(alleles) for _ in range(n_samples)]
        matEsc_vals = [random_binary() for _ in range(n_samples)]
        matDep_vals = [random_binary() for _ in range(n_samples)]

        row = [cpg_id, cpg_chrPos, snp_id, snp_chrPos] + \
              dnam_vals + geno_vals + genoM_vals + matEsc_vals + matDep_vals

        writer.writerow(row)

print(f"CSV file '{file_name}' created with {n_cpgs} CpGs.")
