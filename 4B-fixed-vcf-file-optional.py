#!/usr/bin/env python3
"""
Convert a TSV of mutations into a minimal VCF.gz suitable for CADD upload.
"""
import pandas as pd
import gzip
import argparse

def tsv_to_vcf_gz(tsv_path: str, vcf_gz_path: str) -> None:
    # Load the TSV into a DataFrame
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)

    # Rename and select columns for VCF
    vcf_df = df.rename(columns={
        'Chromosome': '#CHROM',
        'Start_Position': 'POS',
        'Reference_Allele': 'REF',
        'Tumor_Seq_Allele2': 'ALT'
    })[['#CHROM', 'POS', 'REF', 'ALT']].copy()

    # Add required VCF fields
    vcf_df['ID']     = '.'
    vcf_df['QUAL']   = '.'
    vcf_df['FILTER'] = '.'
    vcf_df['INFO']   = '.'
    vcf_df['FORMAT'] = '.'

    # Reorder to VCF columns
    vcf_df = vcf_df[['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']]

    # Write to gzip-compressed VCF
    with gzip.open(vcf_gz_path, 'wt', newline='\n') as f:
        # Minimal VCF header
        f.write('##fileformat=VCFv4.2\n')
        f.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\n')
        # Write the variant lines
        vcf_df.to_csv(f, sep='\t', index=False, header=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert a mutations TSV to a gzipped minimal VCF.'
    )
    parser.add_argument(
        '-i', '--input',
        help='Path to the input TSV (mutations_with_signatures_and_DP.tsv)',
        default='C:/Users/Zubair/Desktop/New folder/mutations_with_signatures_and_DP.tsv'
    )
    parser.add_argument(
        '-o', '--output',
        help='Path for the output VCF.gz file',
        default='C:/Users/Zubair/Desktop/New folder/minimal.vcf.gz'
    )
    args = parser.parse_args()

    tsv_to_vcf_gz(args.input, args.output)
    print(f"✅ Wrote gzipped VCF to {args.output}")
