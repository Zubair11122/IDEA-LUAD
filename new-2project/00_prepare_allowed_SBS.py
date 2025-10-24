# 00_prepare_allowed_SBS.py
import pandas as pd

# Load full COSMIC SBS reference (v3.4, SBS96 matrix)
# Columns: 'Type','SBS1','SBS2',...  (96 rows for contexts)
cosmic = pd.read_csv("C:/Users/Zubair/Desktop/A/COSMIC_v3.4_SBS_GRCh38.txt", sep="\t")

allowed = ['SBS1','SBS2','SBS3','SBS4','SBS5','SBS13','SBS18','SBS40']
keep = ['Type'] + [c for c in allowed if c in cosmic.columns]

allowed_df = cosmic[keep].copy()
allowed_df.to_csv("allowed_SBS.tsv", sep="\t", index=False)
print("✅ wrote allowed_SBS.tsv with", len(allowed), "signatures")
