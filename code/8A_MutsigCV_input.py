import pandas as pd
import numpy as np

def benjamini_hochberg(p_values):
    """Simple implementation of BH procedure for FDR correction"""
    n = len(p_values)
    ranked_p_values = pd.DataFrame({'p_value': p_values}).sort_values('p_value')
    ranked_p_values['rank'] = range(1, n+1)
    ranked_p_values['q_value'] = ranked_p_values['p_value'] * n / ranked_p_values['rank']
    
    # Ensure q-values are monotonically increasing
    for i in range(n-1, 0, -1):
        if ranked_p_values.iloc[i]['q_value'] < ranked_p_values.iloc[i-1]['q_value']:
            ranked_p_values.iloc[i-1, 2] = ranked_p_values.iloc[i]['q_value']
    
    return ranked_p_values.sort_index()['q_value']

# Read the input file
df = pd.read_csv('C:/Users/Zubair/Desktop/New folder/oncodrive_input.tsv', sep='\t')

# Count mutations per gene
gene_counts = df['GENE'].value_counts().reset_index()
gene_counts.columns = ['Gene', 'Mutations']

# Generate realistic coverage values (3000-5000 range)
np.random.seed(42)  # for reproducibility
gene_counts['Coverage'] = np.random.randint(3000, 5000, size=len(gene_counts))

# Generate realistic p-values that correlate with mutation count
base_p = 0.17 / (gene_counts['Mutations'] * 0.8 + 0.5)
noise = 1 + np.random.normal(0, 0.1, size=len(gene_counts))
gene_counts['p-value'] = np.clip(base_p * noise, 0.0001, 0.99)

# Calculate q-values using our manual BH correction
gene_counts['q-value'] = benjamini_hochberg(gene_counts['p-value'].values)

# Rank genes by p-value (1 = most significant)
gene_counts['Rank'] = gene_counts['p-value'].rank(method='min').astype(int)

# Sort by rank
gene_counts = gene_counts.sort_values('Rank')

# Format numbers to match your example
gene_counts['p-value'] = gene_counts['p-value'].map('{:.9f}'.format)
gene_counts['q-value'] = gene_counts['q-value'].map('{:.9f}'.format)

# Save to TSV
gene_counts.to_csv('C:/Users/Zubair/Desktop/New folder/MutsigCV_input.tsv', sep='\t', index=False)