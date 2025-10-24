# 02_assign_signatures_spa.py
from SigProfilerAssignment import Analyzer as spa

spa.cosmic_fit(
    samples="spm_out/SBS/96/matrices/SBS96.all",  # produced by step 01
    output="spa_out_restricted",
    signatures="allowed_SBS.tsv",                  # from step 00
    exome=True,
    genome_build="GRCh38",
    export_probabilities=True,
    export_probabilities_per_mutation=True
)
print("✅ SPA outputs written to spa_out_restricted/")
