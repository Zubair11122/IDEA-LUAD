# 01_make_matrices_exome.py
from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as matGen

# Input folder should contain per-sample MAF/VCF files.
# Use GRCh38 to match your hg38 usage.
out = matGen("GRCh38", "spm_out", "input_variants/", exome=True)
print("✅ matrices at:", out)
