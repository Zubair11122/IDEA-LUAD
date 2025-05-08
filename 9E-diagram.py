#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# ---------- 1. paths ----------
root = Path(r"C:/Users/Zubair/Desktop/A")          # change once if your folder moves
paths = {
    "mutations" : root / "mutations_variant_complete.tsv",
    "chasm"     : root / "chasm_input.tsv",
    "oncodrive" : root / "oncodrive_input.tsv",
    "mutsig"    : root / "MutsigCV_input.tsv",
}

# ---------- 2. COSMIC LUAD driver list ----------
cosmic_drivers = [
    'TP53','EGFR','KRAS','STK11','KEAP1','NF1','RB1','PIK3CA','SMARCA4',
    'BRAF','CDKN2A','ARID1A','PTEN','NFE2L2','ATM','MET','FGFR1','ERBB2'
]

# ---------- 3. IDEA‑LUAD (our model) ----------
df_pred = pd.read_csv(paths["mutations"], sep='\t', low_memory=False)
predicted = set(df_pred.loc[df_pred["is_driver"] == 1, "hugo_symbol"].str.upper())

# ---------- 4. CHASMplus top‑20 ----------
df_chasm = pd.read_csv(paths["chasm"], sep='\t')
chasm_top = (
    df_chasm.groupby("Hugo_Symbol")["chasmplus.score"].mean()
             .sort_values(ascending=False).head(20)
             .index.str.upper().tolist()
)

# ---------- 5. OncodriveFML top‑20 ----------
df_onco = pd.read_csv(paths["oncodrive"], sep='\t')
oncodrive_top = (
    df_onco.groupby("GENE")["SCORE"].mean()
           .sort_values(ascending=False).head(20)
           .index.str.upper().tolist()
)

# ---------- 6. MutSigCV top‑20 (lowest q‑values) ----------
df_mutsig = pd.read_csv(paths["mutsig"], sep='\t')
mutsig_top = (
    df_mutsig.groupby("Gene")["q-value"].mean()
             .sort_values(ascending=True).head(20)
             .index.str.upper().tolist()
)

# ---------- 7. assemble Supplementary Table 6 ----------
records = [
    {
        "Gene": g,
        "Detected_by_IDEA-LUAD":    "Yes" if g in predicted      else "No",
        "Detected_by_CHASMplus":    "Yes" if g in chasm_top      else "No",
        "Detected_by_OncodriveFML": "Yes" if g in oncodrive_top  else "No",
        "Detected_by_MutSigCV":     "Yes" if g in mutsig_top     else "No",
    }
    for g in cosmic_drivers
]

supp6 = pd.DataFrame(records)
out_path = root / "Supplementary_Table6_COSMIC_validation.csv"
supp6.to_csv(out_path, index=False)
print(f"✅ Supplementary Table 6 saved → {out_path}")
