# 06_benchmark.py
import pandas as pd, numpy as np, joblib, tensorflow as tf, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
import os
sns.set()

model = tf.keras.models.load_model("driver_prediction_model.keras")
pre = joblib.load("preprocessor.pkl")

df = pd.read_csv("features_merged.tsv", sep='\t')
X_in = df[pre.feature_names_in_]  # scikit >=1.0
X = pre.transform(X_in)
y = df['is_driver'].astype(int).values

y_prob = model.predict(X).ravel()
df['Model_Score'] = y_prob
df['Model_Prediction'] = (y_prob > 0.5).astype(int)

# External tools prepared as before (chasm_input.tsv / oncodrive_input.tsv)
chasm = pd.read_csv("chasm_input.tsv", sep='\t')
onco  = pd.read_csv("oncodrive_input.tsv", sep='\t')

# Clean
chasm['chasmplus.score'] = pd.to_numeric(chasm['chasmplus.score'], errors='coerce')
onco['SCORE'] = pd.to_numeric(onco['SCORE'], errors='coerce')

# Gene-level aggregation for ROC/PR on genes if needed
gene_model = df.groupby('Hugo_Symbol').agg({'Model_Score':'mean','is_driver':'max'}).dropna()
gene_chasm = chasm.groupby('Hugo_Symbol')['chasmplus.score'].mean().rename('CHASM').to_frame()
gene_onco  = onco.groupby('GENE')['SCORE'].mean().rename('ONCODRIVE').to_frame()

comb = gene_model.join(gene_chasm, how='left').join(gene_onco, how='left').dropna()

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
for col,label in [('Model_Score','Our Model'),('CHASM','CHASMplus'),('ONCODRIVE','OncodriveFML')]:
    fpr,tpr,_ = roc_curve(comb['is_driver'], comb[col])
    auc = roc_auc_score(comb['is_driver'], comb[col])
    plt.plot(fpr,tpr,label=f'{label} (AUC={auc:.3f})')
plt.plot([0,1],[0,1],'k--'); plt.title('ROC'); plt.legend()

plt.subplot(1,2,2)
for col,label in [('Model_Score','Our Model'),('CHASM','CHASMplus'),('ONCODRIVE','OncodriveFML')]:
    prec,rec,_ = precision_recall_curve(comb['is_driver'], comb[col])
    ap = average_precision_score(comb['is_driver'], comb[col])
    plt.plot(rec,prec,label=f'{label} (AP={ap:.3f})')
plt.title('PR'); plt.legend(); plt.tight_layout()
plt.savefig("benchmark_results/performance_curves.png", dpi=300, bbox_inches='tight')
print("✅ wrote benchmark_results/performance_curves.png")
