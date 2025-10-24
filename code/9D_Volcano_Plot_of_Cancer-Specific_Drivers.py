import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

# Load your dataset
df = pd.read_csv("C:/Users/Zubair/Desktop/Sup/mutations_variant_complete.tsv", sep='\t')

# Count driver and non-driver mutations
gene_counts = df.groupby("hugo_symbol")["is_driver"].agg(['sum', 'count'])
gene_counts.columns = ['Driver_Mutations', 'Total_Mutations']
gene_counts['Non_Driver'] = gene_counts['Total_Mutations'] - gene_counts['Driver_Mutations']

# Total driver/non-driver across entire dataset
total_driver = df['is_driver'].sum()
total_non_driver = len(df) - total_driver

# Run Fisher's Exact Test
pvals = []
for _, row in gene_counts.iterrows():
    table = [
        [row['Driver_Mutations'], row['Non_Driver']],
        [total_driver - row['Driver_Mutations'], total_non_driver - row['Non_Driver']]
    ]
    _, p = fisher_exact(table, alternative='greater')
    pvals.append(p)

gene_counts['pval'] = pvals
gene_counts = gene_counts.replace([np.inf, -np.inf], np.nan).dropna()

# Compute log2 fold change and -log10 p-value
gene_counts['Log2FC'] = np.log2((gene_counts['Driver_Mutations'] + 1) / (gene_counts['Non_Driver'] + 1))
gene_counts['-log10(pval)'] = -np.log10(gene_counts['pval'])

# Identify top 15 significant genes
top_genes = gene_counts.nsmallest(10, 'pval')

# Volcano plot
plt.figure(figsize=(12, 7))
plt.scatter(gene_counts['Log2FC'], gene_counts['-log10(pval)'], alpha=0.4, color='gray', label='All Genes')

# Highlight significant points
significant = (gene_counts['-log10(pval)'] > 2) & (abs(gene_counts['Log2FC']) > 1)
plt.scatter(gene_counts[significant]['Log2FC'], gene_counts[significant]['-log10(pval)'], color='red', label='Significant')

# Add labels for top 15 genes
for gene, row in top_genes.iterrows():
    plt.text(row['Log2FC'], row['-log10(pval)'], gene, fontsize=9, ha='right' if row['Log2FC'] < 0 else 'left')

# Axes and styling
plt.axhline(y=2, color='red', linestyle='--')
plt.axvline(x=1, color='blue', linestyle='--')
plt.axvline(x=-1, color='blue', linestyle='--')
plt.title("Volcano Plot: Enrichment of Predicted Driver Genes")
plt.xlabel("log2(Fold Change: Driver / Non-Driver)")
plt.ylabel("-log10(p-value)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/Zubair/Desktop/Sup/volcano_plot_labeled_top15.png", dpi=1200)
plt.show()
