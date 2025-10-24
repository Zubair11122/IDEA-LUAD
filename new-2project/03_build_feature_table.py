# 03_build_feature_table.py
import pandas as pd
from pathlib import Path

# --- Inputs (edit paths as needed)
# Your original merged OpenCRAVAT features (variant.csv) and base mutation table
mut_path = Path("mutations_with_signatures.tsv")  # or your cleaned_mutations.tsv (pre-OpenCRAVAT)
oc_path  = Path("variant.csv")                    # OpenCRAVAT output at variant level
spa_prob = Path("spa_out_restricted/Assignment_Solution/Decomposed_Mutation_Probabilities.txt")
spa_exp  = Path("spa_out_restricted/Assignment_Solution/Samples_Stats.txt")  # or per-sample exposures table

# --- Load
mut = pd.read_csv(mut_path, sep="\t", low_memory=False)
oc  = pd.read_csv(oc_path, low_memory=False)
prob = pd.read_csv(spa_prob, sep="\t", low_memory=False)
exp  = pd.read_csv(spa_exp, sep="\t", low_memory=False)

# Standardize keys for merging
# Ensure these columns exist; if not, adapt your column names accordingly
mut_cols = ['Tumor_Sample_Barcode','Chromosome','Start_Position','Reference_Allele','Tumor_Seq_Allele2']
for c in mut_cols:
    if c not in mut.columns:
        raise SystemExit(f"Missing {c} in {mut_path}")

# Normalize chromosome formats for consistent join
def norm_chr(x):
    x = str(x)
    return x if x.startswith('chr') else f'chr{x}'
mut['Chromosome'] = mut['Chromosome'].apply(norm_chr)

# OpenCRAVAT: lowercase and map to common keys if needed (based on your 5_opencravat script)
oc.columns = [c.strip().lower() for c in oc.columns]
# we expect at least 'hugo_symbol','chromosome','start_position' in oc after your remap
if 'chrom' in oc.columns: oc = oc.rename(columns={'chrom':'chromosome'})
if 'pos' in oc.columns:   oc = oc.rename(columns={'pos':'start_position'})

# --- Merge OpenCRAVAT subset you actually need (avoid CHASMplus if you use it as labels)
keep_cols = ['hugo_symbol','chromosome','start_position','cadd.phred','revel.score',
             'alphamissense.am_pathogenicity','sift.score','polyphen2.hdiv_rank',
             'gerp.gerp_rs']  # extend as you like, but avoid leakage from label source
oc_sub = oc[[c for c in keep_cols if c in oc.columns]].copy()

# --- Merge per-mutation probabilities from SPA
# prob table must have sample + variant coordinates + columns prob_SBS1, prob_SBS2, ...
prob_cols = ['Tumor_Sample_Barcode','Chromosome','Start_Position','Reference_Allele','Tumor_Seq_Allele2']
prob_cols += [c for c in prob.columns if c.startswith('prob_SBS')]
if not set(prob_cols).issubset(prob.columns):
    raise SystemExit("Per-mutation probability columns not found; check SPA output format.")
prob_sub = prob[prob_cols].copy()

# --- Merge per-sample exposures (exp_SBS*)
exp_cols = ['Sample'] + [c for c in exp.columns if c.startswith('exp_SBS')]
exp_sub = exp[exp_cols].rename(columns={'Sample':'Tumor_Sample_Barcode'})

# --- Build final
df = mut.merge(prob_sub, on=mut_cols, how='left') \
        .merge(exp_sub, on='Tumor_Sample_Barcode', how='left')

# Merge OpenCRAVAT by genomic coordinates (gene can vary by transcript; genomic match is safer)
df = df.merge(
    oc_sub,
    left_on=['Chromosome','Start_Position'],
    right_on=['chromosome','start_position'],
    how='left'
)

# Cleanup
df = df.drop(columns=[c for c in ['chromosome','start_position'] if c in df.columns])

# Remove any old columns that cause reviewer issues
for bad in ['Dominant_Signature','DP']:  # DP was fake, Dominant_Signature was idxmax
    if bad in df.columns: df = df.drop(columns=[bad])

df.to_csv("features_merged.tsv", sep="\t", index=False)
print("✅ wrote features_merged.tsv")
