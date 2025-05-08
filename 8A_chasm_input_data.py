import pandas as pd
from pathlib import Path

# File paths
input_path = Path("C:/Users/Zubair/Desktop/New folder/mutations_variant_complete.tsv")
output_path = Path("C:/Users/Zubair/Desktop/New folder/chasm_input.tsv")

# Load the merged data
try:
    df = pd.read_csv(input_path, sep='\t', low_memory=False)
    
    # Standardize column names (case-insensitive)
    df.columns = df.columns.str.lower()  # Convert all column names to lowercase
    
    # Required columns mapping (case-insensitive)
    column_mapping = {
        'chromosome': 'Chromosome',
        'start_position': 'Start_Position',
        'reference_allele': 'Reference_Allele',
        'tumor_seq_allele2': 'Tumor_Seq_Allele2',
        'hugo_symbol': 'Hugo_Symbol',
        'variant_classification': 'Variant_Classification',
        'is_driver': 'Is_Driver',
        'chasmplus.score': 'chasmplus.score'
    }
    
    # Verify all required columns exist
    missing_cols = [col for col in column_mapping.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create CHASMplus input
    chasm_input = df.rename(columns=column_mapping)[list(column_mapping.values())]
    
    # Add 'chr' prefix if not present
    chasm_input['Chromosome'] = chasm_input['Chromosome'].astype(str)
    chasm_input['Chromosome'] = chasm_input['Chromosome'].apply(
        lambda x: x if x.startswith('chr') else f'chr{x}'
    )
    
    # Save the output
    chasm_input.to_csv(output_path, sep='\t', index=False)
    print(f"✅ CHASMplus input file saved to {output_path}")
    print(f"Total variants: {len(chasm_input)}")
    
except FileNotFoundError:
    print(f"❌ Error: Input file not found at {input_path}")
except Exception as e:
    print(f"❌ An error occurred: {str(e)}")