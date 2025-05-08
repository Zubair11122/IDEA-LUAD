import pandas as pd

# Load your TSV file
df = pd.read_csv('C:/Users/Zubair/Desktop/New folder/mutations_with_signatures_and_DP.tsv', sep='\t')

# Prepare VCF format (chromosome, position, reference, alternate)
vcf_data = df[['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2']]

# Create VCF header
vcf_header = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE"

# Prepare VCF body
vcf_body = "\n".join([
    f"{row['Chromosome']}\t{row['Start_Position']}\t.\t{row['Reference_Allele']}\t{row['Tumor_Seq_Allele2']}\t.\t.\t.\tGT\t1/1"
    for index, row in vcf_data.iterrows()
])

# Write to VCF file
with open("C:/Users/Zubair/Desktop/New folder/output.vcf", "w") as vcf_file:
    vcf_file.write(vcf_header + "\n" + vcf_body)
