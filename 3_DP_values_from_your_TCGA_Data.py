import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_dp(maf_path):
    """Calculate DP while handling NA/inf values in dna_vaf."""
    df = pd.read_csv(maf_path, sep='\t', comment='#')

    # Clean dna_vaf: replace NA/0 with median VAF
    median_vaf = df['dna_vaf'].median()
    df['dna_vaf_clean'] = df['dna_vaf'].fillna(median_vaf).replace(0, median_vaf)

    # Calculate DP (minimum 1 read to avoid infinity) - FIXED LINE
    df['DP'] = np.maximum(1, (1 / df['dna_vaf_clean']).round().astype(int))

    return df[['gene', 'chrom', 'start', 'ref', 'alt', 'DP']].rename(
        columns={'gene': 'Hugo_Symbol', 'chrom': 'Chromosome',
                 'start': 'Start_Position', 'ref': 'Reference_Allele',
                 'alt': 'Tumor_Seq_Allele2'}
    )


# ----------------------------------------------------------------------------
# Process files with error handling
# ----------------------------------------------------------------------------
try:
    # Calculate DP for both cancer types
    luad_dp = calculate_dp("TCGA-LUAD.somaticmutation_wxs.tsv")
    combined_dp = pd.concat([luad_dp])

    # Merge with existing data
    df = pd.read_csv("mutations_with_signatures.tsv", sep='\t')
    df = df.merge(
        combined_dp,
        on=['Hugo_Symbol', 'Chromosome', 'Start_Position',
            'Reference_Allele', 'Tumor_Seq_Allele2'],
        how='left'
    )

    # Fill remaining NAs with sample-specific medians
    df['DP'] = df.groupby('Tumor_Sample_Barcode')['DP'].transform(
        lambda x: x.fillna(x.median())
    )

    # Final fallback: Cancer-type medians
    cancer_medians = {'LUAD': 60}
    df['DP'] = df['DP'].fillna(df['Cancer_Type'].map(cancer_medians))

    # Save and validate
    df.to_csv("mutations_with_signatures_and_DP.tsv", sep='\t', index=False)
    print("Success! DP values saved. Stats:\n", df['DP'].describe())

    # Plot distribution
    plt.hist(df['DP'], bins=50)
    plt.title("Final Read Depth Distribution")
    plt.xlabel("DP")
    plt.ylabel("Count")
    plt.show()

except Exception as e:
    print(f"Error: {str(e)}\n")
    print("Debugging Tips:")
    print("- Check for NA/0 values in 'dna_vaf' column")
    print("- Verify file encoding (try encoding='utf-8' in pd.read_csv)")
    print("- Inspect intermediate files with:\n  print(df.head())")