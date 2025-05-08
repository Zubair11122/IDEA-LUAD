import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

root            = Path(r"C:/Users/Zubair/Desktop/A")
clinical_file   = root / "luad_tcga_gdc_clinical_data.tsv"
mutations_file  = root / "mutations_variant_complete.tsv"

clin_id_col     = "Sample ID"
mut_id_col      = "tumor_sample_barcode"

time_col        = "Overall Survival (Months)"
event_col       = "Overall Survival Status"

genes = ["TP53", "CSMD3", "LRP1B"]

# ---- LOAD ----------------------------------------------------------------------
clin = pd.read_csv(clinical_file,  sep='\t')
mut  = pd.read_csv(mutations_file, sep='\t', low_memory=False)
mut  = mut.loc[mut["is_driver"] == 1]

# ---- EVENT COLUMN TO 0/1 --------------------------------------------------------
clin[event_col] = (
    clin[event_col].astype(str).str.strip().str[0].astype(int)
)

# ---- BUILD MUTATION FLAGS -------------------------------------------------------
for g in genes:
    carriers = mut.loc[
        mut["hugo_symbol"].str.upper() == g, mut_id_col
    ].unique()
    clin[f"{g}_mut"] = clin[clin_id_col].isin(carriers).astype(int)

# ---- COX PH ---------------------------------------------------------------------
rows = []
for g in genes:
    df_fit = clin[[time_col, event_col, f"{g}_mut"]].dropna()
    df_fit = df_fit.rename(columns={time_col: "time", event_col: "event"})
    cph = CoxPHFitter()
    cph.fit(df_fit, duration_col="time", event_col="event")

    hr   = cph.hazard_ratios_[f"{g}_mut"]
    pval = cph.summary.loc[f"{g}_mut", "p"]
    ph_p = proportional_hazard_test(
        cph, df_fit, time_transform="rank"
    ).summary["p"].values[0]

    rows.append({
        "Gene": g,
        "Hazard ratio": round(hr, 2),
        "Wald p-value": f"{pval:.3e}",
        "PH-assumption p": f"{ph_p:.3e}",
    })

pd.DataFrame(rows).to_csv(
    root / "Supplementary_Table7_survival.csv", index=False
)

print("✅ Supplementary Table 7 saved →",
      root / "Supplementary_Table7_survival.csv")
