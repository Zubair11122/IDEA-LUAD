import pandas as pd
from pathlib import Path

# File paths
input_path = Path("C:/Users/Zubair/Desktop/New folder/mutations_variant_complete.tsv")
output_path = Path("C:/Users/Zubair/Desktop/New folder/oncodrive_input.tsv")

# Load the merged data
try:
    df = pd.read_csv(input_path, sep='\t', low_memory=False)
    
    # Standardize column names (case-insensitive)
    df.columns = df.columns.str.lower()
    
    # Required columns mapping
    column_mapping = {
        'chromosome': 'CHR',
        'start_position': 'POS',
        'reference_allele': 'REF',
        'tumor_seq_allele2': 'ALT',
        'hugo_symbol': 'GENE',
        'cadd.phred': 'SCORE'
    }
    
    # Verify all required columns exist
    missing_cols = [col for col in column_mapping.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create OncodriveFML input
    oncodrive_input = df.rename(columns=column_mapping)[list(column_mapping.values())].copy()
    
    # Remove 'chr' prefix if present
    oncodrive_input['CHR'] = oncodrive_input['CHR'].astype(str).str.replace('^chr', '', regex=True)
    
    # Ensure proper column order
    oncodrive_input = oncodrive_input[['CHR', 'POS', 'REF', 'ALT', 'GENE', 'SCORE']]
    
    # Save the output
    oncodrive_input.to_csv(output_path, sep='\t', index=False)
    print(f"✅ OncodriveFML input file saved to {output_path}")
    print(f"Total variants: {len(oncodrive_input)}")
    
except FileNotFoundError:
    print(f"❌ Error: Input file not found at {input_path}")
except Exception as e:
    print(f"❌ An error occurred: {str(e)}")