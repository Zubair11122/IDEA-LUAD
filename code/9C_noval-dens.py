import pandas as pd

# Load your mutation data
df = pd.read_csv("C:/Users/Zubair/Desktop/Sup/mutations_variant_complete.tsv", sep="\t")

# Filter predicted driver mutations
df_predicted = df[df["is_driver"] == 1]

# Define LUAD COSMIC driver genes (18 genes from your paper)
cosmic_genes = {
    'TP53', 'EGFR', 'KRAS', 'STK11', 'KEAP1', 'NF1', 'RB1', 'PIK3CA',
    'SMARCA4', 'BRAF', 'CDKN2A', 'ARID1A', 'PTEN', 'NFE2L2', 'ATM', 'MET',
    'FGFR1', 'ERBB2'
}

# Remove known COSMIC LUAD drivers → keep only novel gene predictions
novel_genes = df_predicted[~df_predicted["hugo_symbol"].isin(cosmic_genes)]

# Count how often each novel gene appears
novel_gene_counts = (
    novel_genes["hugo_symbol"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "Gene", "hugo_symbol": "Mutation_Count"})
)

# Select the top 15 most frequently mutated novel genes
top_15_novel_genes = novel_gene_counts.head(15)

# Save to TSV for paper (supplementary table)
top_15_novel_genes.to_csv("C:/Users/Zubair/Desktop/Sup/Supplementary_Table X.csv", index=False)

# Optional: print preview
print("Top 15 Novel Candidate Drivers:")
print(top_15_novel_genes)
