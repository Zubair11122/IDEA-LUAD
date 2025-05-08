import pandas as pd

# ----------------------------------------------------------------------------
# 1. Load & Preprocess Data (Your existing code)
# ----------------------------------------------------------------------------
luad = pd.read_csv("TCGA-LUAD.somaticmutation_wxs.tsv", sep="\t", comment="#")
# Add cancer type labels
luad['Cancer_Type'] = 'LUAD'
# Combine and filter
combined = pd.concat([luad])
valid_effects = ['missense_variant', 'stop_gained', 'frameshift_variant']
combined = combined[combined['effect'].isin(valid_effects) & (combined['dna_vaf'] >= 0.05)]

# Standardize columns
combined = combined.rename(columns={
    'gene': 'Hugo_Symbol', 'chrom': 'Chromosome', 'start': 'Start_Position',
    'ref': 'Reference_Allele', 'alt': 'Tumor_Seq_Allele2',
    'effect': 'Variant_Classification', 'dna_vaf': 'AF'
})

# ----------------------------------------------------------------------------
# 2. Annotate with Cancer Gene Census (NEW)
# ----------------------------------------------------------------------------
cgc = pd.read_csv("Census_allTue Apr  1 13_43_24 2025.csv")
combined['Is_Driver'] = combined['Hugo_Symbol'].isin(cgc['Gene Symbol'])
print(f"Driver mutations: {combined['Is_Driver'].sum()}")

# ----------------------------------------------------------------------------
# 3. Prepare for COSMIC Signatures (NEW)
# ----------------------------------------------------------------------------
combined['Mutation'] = (
    combined['Chromosome'] + ' ' +
    combined['Start_Position'].astype(str) + ' ' +
    combined['Reference_Allele'] + ' ' +
    combined['Tumor_Seq_Allele2']
)
combined[['Mutation']].to_csv("sigprofiler_input.tsv", sep="\t", index=False)

# ----------------------------------------------------------------------------
# 4. Save and Report
# ----------------------------------------------------------------------------
combined.to_csv("cleaned_mutations.tsv", sep="\t", index=False)
print(f"Final mutations: {len(combined)}")
