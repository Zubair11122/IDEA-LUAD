from pathlib import Path
import pandas as pd
import numpy as np

# ── 1. File locations ────────────────────────────────────────────────
variant_path = Path("C:/Users/Zubair/Desktop/New folder/variant.csv")  # Replace with your path
mut_path     = Path("C:/Users/Zubair/Desktop/New folder/mutations_with_signatures_and_DP.tsv")  # Replace with your path
out_path     = Path("C:/Users/Zubair/Desktop/New folder/mutations_variant_complete.tsv")  # Output path
# ── 2. Columns to select from variant ───────────────────────────────
selected_columns = [
    'so', 'exonno', 'cchange', 'aloft.tolerant', 'aloft.recessive', 'aloft.dominant', 
    'aloft.pred', 'aloft.conf', 'alphamissense.am_pathogenicity', 'alphamissense.am_class',
    'alphamissense.pp3_pathogenic', 'bayesdel.bayesdel_addaf_score', 'bayesdel.bayesdel_addaf_rankscore',
    'bayesdel.bayesdel_addaf_pred', 'bayesdel.bayesdel_noaf_score', 'bayesdel.bayesdel_noaf_rankscore',
    'bayesdel.pp3_pathogenic', 'cadd.score', 'cadd.phred', 'cadd.pp3_pathogenic', 'chasmplus.pval',
    'chasmplus.score', 'chasmplus_acc.pval', 'chasmplus_acc.score', 'chasmplus_blca.pval',
    'chasmplus_blca.score', 'chasmplus_cesc.pval', 'chasmplus_cesc.score', 'chasmplus_chol.pval',
    'chasmplus_chol.score', 'chasmplus_coad.pval', 'chasmplus_coad.score', 'chasmplus_dlbc.pval',
    'chasmplus_dlbc.score', 'chasmplus_esca.pval', 'chasmplus_esca.score', 'chasmplus_gbm.pval',
    'chasmplus_gbm.score', 'chasmplus_hnsc.pval', 'chasmplus_hnsc.score', 'chasmplus_kich.pval',
    'chasmplus_kich.score', 'chasmplus_kirc.pval', 'chasmplus_kirc.score', 'chasmplus_kirp.pval',
    'chasmplus_kirp.score', 'chasmplus_laml.pval', 'chasmplus_laml.score', 'chasmplus_lgg.pval',
    'chasmplus_lgg.score', 'chasmplus_lihc.pval', 'chasmplus_lihc.score', 'chasmplus_luad.pval',
    'chasmplus_luad.score', 'chasmplus_lusc.pval', 'chasmplus_lusc.score', 'chasmplus_meso.pval',
    'chasmplus_meso.score', 'chasmplus_ov.pval', 'chasmplus_ov.score', 'chasmplus_paad.pval',
    'chasmplus_paad.score', 'chasmplus_pcpg.pval', 'chasmplus_pcpg.score', 'chasmplus_prad.pval',
    'chasmplus_prad.score', 'chasmplus_read.pval', 'chasmplus_read.score', 'chasmplus_sarc.pval',
    'chasmplus_sarc.score', 'chasmplus_skcm.pval', 'chasmplus_skcm.score', 'chasmplus_stad.pval',
    'chasmplus_stad.score', 'chasmplus_tgct.pval', 'chasmplus_tgct.score', 'chasmplus_thca.pval',
    'chasmplus_thca.score', 'chasmplus_thym.pval', 'chasmplus_thym.score', 'chasmplus_ucec.pval',
    'chasmplus_ucec.score', 'chasmplus_ucs.pval', 'chasmplus_ucs.score', 'chasmplus_uvm.pval',
    'chasmplus_uvm.score', 'cscape.score', 'cscape_coding.score', 'cscape_coding.rankscore',
    'clinpred.score', 'clinpred.rankscore', 'dann.score', 'dann_coding.dann_coding_score',
    'dann_coding.dann_rankscore', 'ditto.score', 'esm1b.score', 'esm1b.rankscore',
    'esm1b.prediction', 'esm1b.pp3_pathogenic', 'eve.score', 'eve.rank_score',
    'fathmm.fathmm_rscore', 'fathmm.score', 'fathmm_mkl.fathmm_mkl_coding_score',
    'fathmm_mkl.fathmm_mkl_coding_rankscore', 'fathmm_mkl.fathmm_mkl_coding_pred',
    'fathmm_xf_coding.fathmm_xf_coding_score', 'fathmm_xf_coding.fathmm_xf_coding_rankscore',
    'fathmm_xf_coding.fathmm_xf_coding_pred', 'fathmm_xf_coding.pp3_pathogenic',
    'funseq2.score', 'gerp.gerp_nr', 'gerp.gerp_rs', 'gerp.gerp_rs_rank',
    'lrt.lrt_score', 'lrt.lrt_converted_rankscore', 'lrt.lrt_pred', 'lrt.lrt_omega',
    'mistic.score', 'mistic.pred', 'metalr.score', 'metalr.rankscore', 'metalr.pred',
    'metalr.bp4_benign', 'metarnn.score', 'metarnn.pred', 'metarnn.rank_score',
    'metasvm.score', 'metasvm.rankscore', 'metasvm.pred', 'mutpred1.mutpred_general_score',
    'mutpred1.mutpred_rankscore', 'mutpred1.bp4_benign', 'mutation_assessor.score',
    'mutation_assessor.rankscore', 'mutation_assessor.impact', 'mutationtaster.score',
    'mutationtaster.rankscore', 'mutationtaster.prediction', 'mutationtaster.model',
    'provean.score', 'provean.rankscore', 'provean.prediction', 'phdsnpg.prediction',
    'phdsnpg.score', 'phdsnpg.fdr', 'phylop.phylop100_vert', 'phylop.phylop100_vert_r',
    'phylop.phylop470_mamm', 'phylop.phylop470_mamm_r', 'phylop.phylop17_primate',
    'phylop.phylop17_primate_r', 'polyphen2.hdiv_rank', 'polyphen2.hvar_rank',
    'primateai.primateai_score', 'primateai.primateai_rankscore', 'revel.score',
    'revel.rankscore', 'sift.prediction', 'sift.confidence', 'sift.score', 'sift.rankscore',
    'sift.med', 'sift.seqs', 'sift.multsite', 'varity_r.varity_r', 'varity_r.varity_er',
    'varity_r.varity_r_loo', 'varity_r.varity_er_loo', 'vest.score', 'vest.pval',
    'gmvp.score', 'gmvp.rank_score'
]

# ──────────────────────────────────────────────────────────────────────
# 3. Load Data with Proper Column Names
# ──────────────────────────────────────────────────────────────────────
# Load variant data
variant_df = pd.read_csv(variant_path, low_memory=False)
variant_df.columns = [col.strip().lower() for col in variant_df.columns]
variant_df = variant_df.rename(columns={
    'hugo': 'hugo_symbol',
    'chrom': 'chromosome',
    'pos': 'start_position'
})

# Load mutations data (preserve original order)
mut_df = pd.read_csv(mut_path, sep="\t", low_memory=False)
mut_df.columns = [col.strip().lower() for col in mut_df.columns]

# Add temporary index to maintain original order
mut_df['_original_index'] = range(len(mut_df))

# ──────────────────────────────────────────────────────────────────────
# 4. Perform Matching in Stages
# ──────────────────────────────────────────────────────────────────────
# Get available columns that exist in variant file
available_columns = [col for col in selected_columns if col in variant_df.columns]
variant_subset = variant_df[['hugo_symbol', 'chromosome', 'start_position'] + available_columns]

# Stage 1: Perfect 3/3 matches
perfect_matches = mut_df.merge(
    variant_subset,
    on=['hugo_symbol', 'chromosome', 'start_position'],
    how='inner',
    suffixes=('', '_variant')
)

# Get unmatched rows
matched_ids = perfect_matches['_original_index']
unmatched = mut_df[~mut_df['_original_index'].isin(matched_ids)]

# Stage 2: 2/3 matches (try all combinations)
match_combinations = [
    ['hugo_symbol', 'chromosome'],
    ['hugo_symbol', 'start_position'],
    ['chromosome', 'start_position']
]

partial_matches = []
for cols in match_combinations:
    # Find matches for this column combination
    temp_match = unmatched.merge(
        variant_subset.drop(columns=[c for c in ['hugo_symbol', 'chromosome', 'start_position'] if c not in cols]),
        on=cols,
        how='inner',
        suffixes=('', f'_{"_".join(cols)}')
    )
    
    # Avoid duplicate additions
    if not temp_match.empty:
        partial_matches.append(temp_match)
        unmatched = unmatched[~unmatched['_original_index'].isin(temp_match['_original_index'])]

# Combine all matches
all_matches = pd.concat([perfect_matches] + partial_matches, axis=0)

# ──────────────────────────────────────────────────────────────────────
# 5. Restore Original Order and Save
# ──────────────────────────────────────────────────────────────────────
# Sort by original index to maintain input order
final_df = all_matches.sort_values('_original_index')

# Drop temporary index column
final_df = final_df.drop(columns=['_original_index'])

# Save results
final_df.to_csv(out_path, sep="\t", index=False)
print(f"✅ Saved {len(final_df)} matched rows → {out_path}")
print("Match counts:")
print(f"- Perfect 3/3 matches: {len(perfect_matches)}")
print(f"- Partial 2/3 matches: {len(all_matches) - len(perfect_matches)}")