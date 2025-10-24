import pandas as pd

# ── 1. Load the TSV ──────────────────────────────────────────────
tsv_path = r"C:/Users/Zubair/Desktop/New folder/mutations_variant_complete.tsv"          # ↖ change this if the file is elsewhere
df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

# ── 2. Display in 20-column chunks ───────────────────────────────
chunk_size = 5

                    # how many columns per page
n_rows    = 3                      # how many rows to preview

for start in range(0, len(df.columns), chunk_size):
    end   = start + chunk_size
    cols  = df.columns[start:end]

    # header
    print(f"\n=== Columns {start+1} – {min(end, len(df.columns))} of {len(df.columns)} ===")
    print(", ".join(cols))

    # data preview
    print(df.loc[0:n_rows-1, cols])   # rows 0,1,2  (first 3)

    # optional pause so the console doesn’t scroll past
    input("\nPress Enter for the next 20 columns…")
