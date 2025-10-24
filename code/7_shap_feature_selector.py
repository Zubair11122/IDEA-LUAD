#!/usr/bin/env python3
import pandas as pd
import numpy as np
import shap
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ─── Configuration ──────────────────────────────────────────────────
os.makedirs("shap_results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ─── Data Loading ───────────────────────────────────────────────────
print("⏳ Loading data...")
df = pd.read_csv("mutations_variant_complete.tsv", sep='\t')
df.replace("-", np.nan, inplace=True)
y = df["is_driver"].astype(int)

# ─── Preprocessing ─────────────────────────────────────────────────
print("🔧 Transforming data...")
preprocessor = joblib.load("preprocessor.pkl")
X_raw = df[preprocessor.feature_names_in_]
X = preprocessor.transform(X_raw)
feature_names = preprocessor.get_feature_names_out()
X_df = pd.DataFrame(X, columns=feature_names)

# ─── Model Loading ─────────────────────────────────────────────────
print("🧠 Loading trained model...")
model = tf.keras.models.load_model("driver_prediction_model.keras")

# ─── Feature Groups (UPDATED FOR YOUR DATASET) ─────────────────────
feature_groups = {
    'SBS_signatures': [f for f in feature_names if f.startswith('sbs')],
    'AlphaMissense': [f for f in feature_names if 'alphamissense' in f.lower()],
    'CADD': [f for f in feature_names if 'cadd' in f.lower()],
    'SIFT': [f for f in feature_names if 'sift' in f.lower()],
    'PolyPhen': [f for f in feature_names if 'polyphen' in f.lower()],
    'REVEL': [f for f in feature_names if 'revel' in f.lower()],
    'GERP': [f for f in feature_names if 'gerp' in f.lower()],
    'Variant_Type': [f for f in feature_names if 'variant_classification' in f.lower()],
    'Clinical': [f for f in feature_names if f.lower() in ['dp', 'af']],
    'Other': [f for f in feature_names if not any(
        k in f.lower() for k in ['sbs', 'alphamissense', 'cadd', 'sift',
                                 'polyphen', 'revel', 'gerp', 'variant_classification']
    )]
}

# ─── Optimized SHAP Analysis ───────────────────────────────────────
print("🔍 Calculating SHAP values...")

# 1. Strategic sampling (smaller for speed)
sample_size = min(500, len(X_df))  # Reduced from 1000
stratified_sample, _ = train_test_split(
    X_df,
    test_size=len(X_df) - sample_size,
    stratify=y,
    random_state=42
)

# 2. Faster KernelExplainer
background = shap.sample(stratified_sample, 50)  # Smaller background
explainer = shap.KernelExplainer(
    model.predict,
    background,
    link="logit"  # Better for binary classification
)

# 3. Robust SHAP calculation
try:
    shap_values = explainer.shap_values(stratified_sample, nsamples=150)  # Reduced samples

    # Handle multi-output format
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values).sum(axis=0)  # Combine outputs for binary classification
    else:
        shap_values = np.array(shap_values)

    # Ensure proper dimensions
    if len(shap_values.shape) == 3:
        shap_means = np.abs(shap_values).sum(axis=2).mean(axis=0)  # Sum across outputs
    else:
        shap_means = np.abs(shap_values).mean(axis=0)

    shap_means = shap_means.flatten()  # Guarantee 1D array

except Exception as e:
    print(f"❌ SHAP calculation failed: {str(e)}")
    exit(1)

# ─── Feature Selection ─────────────────────────────────────────────
print("📊 Analyzing feature importance...")

# Create SHAP dataframe
shap_df = pd.DataFrame({
    'feature': feature_names,
    'shap': shap_means
}).sort_values("shap", ascending=False)

# Add groups (handling missing columns)
shap_df['group'] = 'Other'
for group, features in feature_groups.items():
    # Only add existing features
    existing_features = [f for f in features if f in feature_names]
    if existing_features:  # Only assign if features exist
        shap_df.loc[shap_df['feature'].isin(existing_features), 'group'] = group

# Dynamic selection
shap_df['cumulative'] = shap_df['shap'].cumsum() / shap_df['shap'].sum()
top_n = shap_df[shap_df['cumulative'] <= 0.95].shape[0]
top_n = max(50, min(top_n, 200))  # Keep 50-200 features

# Group quotas (only for existing groups)
min_features = {
    'SBS_signatures': 5, 'AlphaMissense': 2, 'CADD': 1,
    'SIFT': 1, 'PolyPhen': 1, 'REVEL': 1
}

top_features = []
for group, min_count in min_features.items():
    if group in shap_df['group'].unique():  # Only if group exists
        group_features = shap_df[shap_df['group'] == group].head(min_count)['feature'].tolist()
        top_features.extend(group_features)

# Fill remaining slots
remaining = top_n - len(top_features)
if remaining > 0:
    remaining_features = shap_df[~shap_df['feature'].isin(top_features)].head(remaining)['feature'].tolist()
    top_features.extend(remaining_features)

# ─── Visualization ─────────────────────────────────────────────────
print("📈 Generating plots...")

# 1. Top Features Plot
plt.figure(figsize=(12, 8))
top_30 = shap_df.head(30)
colors = plt.cm.tab20(np.linspace(0, 1, len(feature_groups)))

for i, (_, row) in enumerate(top_30.iterrows()):
    group_idx = list(feature_groups.keys()).index(row['group']) if row['group'] in feature_groups else -1
    color = colors[group_idx] if group_idx != -1 else 'gray'
    plt.barh(30 - i, row['shap'], color=color)

plt.yticks(range(30, 0, -1), top_30['feature'])
plt.title(f"Top 30 Features (SHAP Mean | {timestamp})")
plt.xlabel("Mean Absolute SHAP Value")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"shap_results/top_features_{timestamp}.png", dpi=300)
plt.close()

# 2. Group Importance (only for existing groups)
existing_groups = [g for g in feature_groups.keys() if g in shap_df['group'].unique()]
if existing_groups:
    group_importance = shap_df.groupby('group')['shap'].sum().loc[existing_groups]
    plt.figure(figsize=(10, 6))
    group_importance.plot(kind='bar', color=colors[:len(existing_groups)])
    plt.title("Feature Group Importance")
    plt.ylabel("Total SHAP Value")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"shap_results/group_importance_{timestamp}.png", dpi=300)
    plt.close()

# ─── Save Results ──────────────────────────────────────────────────
print("💾 Saving results...")

# Filtered dataset
X_selected = X_df[top_features]
X_selected["is_driver"] = y.values
X_selected.to_csv(f"shap_results/shap_filtered_{timestamp}.csv", index=False)

# Analysis report
report_path = f"shap_results/shap_report_{timestamp}.txt"
with open(report_path, "w") as f:
    f.write(f"SHAP Analysis Report ({timestamp})\n{'=' * 40}\n\n")
    f.write(f"Total features: {len(feature_names)}\nSelected features: {len(top_features)}\n\n")
    f.write("Top 20 Features:\n" + shap_df.head(20).to_string() + "\n\n")
    if existing_groups:
        f.write("Group Importance:\n" + group_importance.to_string() + "\n\n")
    f.write("Selected Features:\n" + "\n".join(top_features))

# ─── Final Output ──────────────────────────────────────────────────
print(f"""
✅ Successfully completed!
📁 Results saved in shap_results/:
   - Filtered data: shap_filtered_{timestamp}.csv
   - Top features: top_features_{timestamp}.png
   - Group importance: group_importance_{timestamp}.png
   - Full report: shap_report_{timestamp}.txt

🔍 Feature Selection Summary:
   - Selected {len(top_features)} features
   - Mean SHAP value: {shap_df['shap'].mean():.4f}
   - Top 3 features:
      1. {shap_df.iloc[0]['feature']} ({shap_df.iloc[0]['shap']:.4f})
      2. {shap_df.iloc[1]['feature']} ({shap_df.iloc[1]['shap']:.4f})
      3. {shap_df.iloc[2]['feature']} ({shap_df.iloc[2]['shap']:.4f})
""")