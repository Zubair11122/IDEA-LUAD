import pandas as pd
from pyfaidx import Fasta
import os
import urllib.request
import gzip
import shutil

# --- Step 1: Download reference genome ---
if not os.path.exists("hg38.fa"):
    print("Downloading hg38 reference genome...")
    hg38_url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    urllib.request.urlretrieve(hg38_url, "hg38.fa.gz")
    with gzip.open("hg38.fa.gz", 'rb') as f_in:
        with open('hg38.fa', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove("hg38.fa.gz")

# --- Step 2: Load data ---
print("Loading data...")
combined = pd.read_csv("cleaned_mutations.tsv", sep="\t")
cosmic_sigs = pd.read_csv("COSMIC_v3.4_SBS_GRCh38.txt", sep="\t")

# Get only existing SBS columns
sbs_columns = [col for col in cosmic_sigs.columns if col.startswith('SBS')]
print(f"Using {len(sbs_columns)} signatures: {sbs_columns}")

# --- Step 3: Add genomic context ---
print("Adding genomic context...")
genome = Fasta("hg38.fa")

def get_context(row):
    try:
        chrom = row['Chromosome'] if str(row['Chromosome']).startswith('chr') else f"chr{row['Chromosome']}"
        return genome[chrom][row['Start_Position']-2 : row['Start_Position']+1].seq
    except:
        return "NNN"

combined['Context'] = combined.apply(get_context, axis=1)

# --- Step 4: Create mutation types ---
print("Creating mutation types...")
combined['Mutation_Type'] = (
    combined['Context'].str[0] + "[" +
    combined['Reference_Allele'] + ">" +
    combined['Tumor_Seq_Allele2'] + "]" +
    combined['Context'].str[2]
)

# --- Step 5: Annotate signatures ---
print("Annotating with COSMIC signatures...")
annotated = combined.merge(cosmic_sigs, left_on="Mutation_Type", right_on="Type", how="left")

# --- Step 6: Identify dominant signatures ---
print("Identifying dominant signatures...")
annotated['Dominant_Signature'] = annotated[sbs_columns].idxmax(axis=1)

# --- Step 7: Save results ---
print("Saving results...")
annotated.to_csv("mutations_with_signatures.tsv", sep="\t", index=False)

print("Analysis complete! Results saved to mutations_with_signatures.tsv")