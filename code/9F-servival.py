import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.exceptions import ConvergenceError

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: Adjust these paths and column-names as needed
# ────────────────────────────────────────────────────────────────────────────────

root = Path(r"C:/Users/Zubair/Desktop/A")

clinical_file  = root / "luad_tcga_gdc_clinical_data.tsv"
mutations_file = root / "mutations_variant_complete.tsv"

clin_id_col = "Sample ID"                # column in clinical TSV for sample ID
mut_id_col  = "tumor_sample_barcode"     # column in mutation TSV for sample barcode

time_col  = "Overall Survival (Months)"  # survival time (in months)
event_col = "Overall Survival Status"    # survival status (e.g. "0:LIVING"/"1:DECEASED")

# ────────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD CLINICAL + MUTATION DATA
# ────────────────────────────────────────────────────────────────────────────────

# (a) Read clinical data as strings (so we can map statuses manually)
clin = pd.read_csv(clinical_file, sep="\t", dtype=str)

# (b) Read mutation calls
mut = pd.read_csv(mutations_file, sep="\t", low_memory=False, dtype=str)

# (c) Attempt to filter to only driver mutations if "is_driver" column exists
if "is_driver" in mut.columns:
    mut["is_driver"] = pd.to_numeric(mut["is_driver"], errors="coerce")
    mut_driver = mut.loc[mut["is_driver"] == 1]
    if mut_driver.shape[0] > 0:
        chosen_mut = mut_driver.copy()
        print(f"Using {chosen_mut.shape[0]} driver‐mutation rows (is_driver == 1).")
    else:
        chosen_mut = mut.copy()
        print("No rows with is_driver == 1; falling back to use all mutation rows.")
else:
    chosen_mut = mut.copy()
    print("'is_driver' column not found; using all mutation rows.")

# ────────────────────────────────────────────────────────────────────────────────
# STEP 2: CONVERT EVENT-STATUS TO 0/1
# ────────────────────────────────────────────────────────────────────────────────

print("\nUnique values in event column before conversion:")
print(clin[event_col].drop_duplicates().tolist())

# Map exactly "0:LIVING" → 0, "1:DECEASED" → 1
event_mapping = {
    "0:LIVING":   0,
    "1:DECEASED": 1,
}

# (a) Create a new column where we map strings → 0/1. Unmapped → NaN.
clin[event_col + "_num"] = clin[event_col].map(event_mapping)

# (b) If any entries are already "0" or "1" as strings, coerce them:
clin[event_col + "_num"] = pd.to_numeric(clin[event_col + "_num"], errors="coerce")

# (c) Drop the original string‐status column; rename our numeric column back to event_col
clin = clin.drop(columns=[event_col])
clin = clin.rename(columns={event_col + "_num": event_col})

print("\nAfter conversion, value counts for event column (0 vs 1 vs NaN):")
print(clin[event_col].value_counts(dropna=False))


# ────────────────────────────────────────────────────────────────────────────────
# STEP 3: BUILD BINARY MUTATION FLAGS FOR ALL GENES
# ────────────────────────────────────────────────────────────────────────────────

# Take all unique gene symbols (uppercase) from the chosen mutation DataFrame
all_genes = sorted(
    chosen_mut["hugo_symbol"]
    .dropna()
    .str.upper()
    .unique()
)

# If there are still no genes, abort
if len(all_genes) == 0:
    raise ValueError("No gene symbols found in mutation file; cannot proceed.")

# For each gene, add a column “<GENE>_mut” that is 1 if that sample has a mutation in GENE, else 0.
for g in all_genes:
    carriers = (
        chosen_mut.loc[
            chosen_mut["hugo_symbol"].str.upper() == g,
            mut_id_col
        ]
        .dropna()
        .unique()
    )
    clin[f"{g}_mut"] = clin[clin_id_col].isin(carriers).astype(int)

print(f"\nBuilt mutation flags for {len(all_genes)} genes.")


# ────────────────────────────────────────────────────────────────────────────────
# STEP 4: FIT COX PROPORTIONAL HAZARDS FOR EACH GENE
# ────────────────────────────────────────────────────────────────────────────────

results = []
skipped_genes = []

for g in all_genes:
    mut_col = f"{g}_mut"
    # (a) Subset to rows where time, event, and this gene’s mutation flag are non-null
    df_fit = clin[[time_col, event_col, mut_col]].dropna()
    df_fit = df_fit.rename(columns={time_col: "time", event_col: "event"})

    # (b) Ensure “time” is numeric
    df_fit["time"] = pd.to_numeric(df_fit["time"], errors="coerce")
    df_fit = df_fit.dropna(subset=["time", "event", mut_col])

    # If no data remains, skip
    if df_fit.shape[0] == 0:
        skipped_genes.append((g, "no non-null rows"))
        continue

    # (c) Check that the mutation flag has both 0 and 1 present
    unique_vals = df_fit[mut_col].unique()
    if len(unique_vals) < 2:
        skipped_genes.append((g, f"only one mutation status ({unique_vals.tolist()})"))
        continue

    # (d) Fit CoxPH inside try/except to catch convergence issues
    cph = CoxPHFitter()
    try:
        cph.fit(df_fit, duration_col="time", event_col="event")
    except ConvergenceError:
        skipped_genes.append((g, "convergence error"))
        continue

    # (e) Extract hazard ratio, Wald p‐value, and PH p‐value
    hr   = cph.hazard_ratios_[mut_col]
    pval = cph.summary.loc[mut_col, "p"]
    ph_test = proportional_hazard_test(cph, df_fit, time_transform="rank")
    ph_p    = ph_test.summary.loc[mut_col, "p"]

    results.append({
        "Gene":            g,
        "Hazard ratio":    round(hr, 2),
        "Wald p-value":    f"{pval:.3e}",
        "PH-assumption p": f"{ph_p:.3e}",
    })

# ────────────────────────────────────────────────────────────────────────────────
# STEP 5: SAVE RESULTS & REPORT SKIPPED GENES
# ────────────────────────────────────────────────────────────────────────────────

if results:
    out_df  = pd.DataFrame(results)
    out_file = root / "Supplementary_Table7_survival_all_genes.csv"
    out_df.to_csv(out_file, index=False)
    print(f"\n✅ Survival analysis complete. Results saved to:\n   {out_file}")
else:
    print("\n❌ No valid Cox results for any gene.")

# Report which genes were skipped (and why)
if skipped_genes:
    print(f"\n⚠️  Skipped {len(skipped_genes)} genes:")
    for g, reason in skipped_genes:
        print(f"   • {g}: {reason}")
