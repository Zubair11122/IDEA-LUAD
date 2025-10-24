# 04_preprocess.py
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv("features_merged.tsv", sep="\t", low_memory=False)

# === TARGET ===
# Prefer OncoKB/Hotspots-derived mutation-level label column here:
# df['is_driver'] = ...
# If temporarily using CHASMplus as label, do NOT include CHASMplus features in df.

df['is_driver'] = df['is_driver'].astype(int)

# === FEATURES ===
cat_feats = ['Variant_Classification','Cancer_Type','so','sift.prediction']
cat_feats = [c for c in cat_feats if c in df.columns]

num_core = ['af','cadd.phred','revel.score','alphamissense.am_pathogenicity','gerp.gerp_rs']
num_core = [c for c in num_core if c in df.columns]

prob_cols = [c for c in df.columns if c.startswith('prob_SBS')]
exp_cols  = [c for c in df.columns if c.startswith('exp_SBS')]
num_feats = num_core + prob_cols + exp_cols

X = df[cat_feats + num_feats].copy()
y = df['is_driver']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Save
joblib.dump(preprocessor, "preprocessor.pkl")
pd.DataFrame(X_train_proc).to_csv("X_train_proc.tsv", sep="\t", index=False)
pd.DataFrame(X_test_proc).to_csv("X_test_proc.tsv", sep="\t", index=False)
y_train.to_csv("y_train.tsv", sep="\t", index=False)
y_test.to_csv("y_test.tsv", sep="\t", index=False)

print("✅ preprocessing done; saved preprocessor.pkl and splits")
