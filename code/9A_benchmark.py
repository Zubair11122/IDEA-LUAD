import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from matplotlib_venn import venn3
import joblib
import tensorflow as tf
import os

# ─── Configuration ──────────────────────────────────────────────────────
plt.style.use('default')  # Ensures white background for all plots
output_dir = "C:/Users/Zubair/Desktop/A/benchmark_results"
os.makedirs(output_dir, exist_ok=True)

# ─── Load Model and Data ───────────────────────────────────────────────
print("Loading model and data...")
model = tf.keras.models.load_model("C:/Users/Zubair/Desktop/A/driver_prediction_model.keras")
preprocessor = joblib.load("C:/Users/Zubair/Desktop/A/preprocessor.pkl")

# Main dataset
df_real = pd.read_csv("C:/Users/Zubair/Desktop/A/mutations_variant_complete.tsv", sep='\t')
df_real.replace("-", np.nan, inplace=True)

# External tools
df_chasm = pd.read_csv("C:/Users/Zubair/Desktop/A/chasm_input.tsv", sep='\t')
df_oncodrive = pd.read_csv("C:/Users/Zubair/Desktop/A/oncodrive_input.tsv", sep='\t')
df_mutsig = pd.read_csv("C:/Users/Zubair/Desktop/A/MutsigCV_input.tsv", sep='\t')

# ─── Data Preparation ──────────────────────────────────────────────────
print("Preprocessing data...")

# Clean scores
df_chasm["chasmplus.score"] = pd.to_numeric(df_chasm["chasmplus.score"], errors="coerce")
df_oncodrive["SCORE"] = pd.to_numeric(df_oncodrive["SCORE"], errors="coerce")
df_mutsig["q-value"] = pd.to_numeric(df_mutsig["q-value"], errors="coerce")

# Generate model predictions
X_in = df_real[[col for col in preprocessor.feature_names_in_ if col in df_real.columns]]
X_trans = preprocessor.transform(X_in)
y_prob = model.predict(X_trans).ravel()
df_real["Model_Score"] = y_prob
df_real["Model_Prediction"] = (y_prob > 0.5).astype(int)

# ─── Top Gene Analysis ────────────────────────────────────────────────
def get_top_genes(df, score_col, gene_col, n=20, ascending=False):
    """Get top N genes by specified score column"""
    return (
        df.groupby(gene_col)[score_col]
        .mean()
        .sort_values(ascending=ascending)
        .head(n)
        .index
        .tolist()
    )

# Get top genes from each method
predicted_top = (
    df_real[df_real["Model_Prediction"] == 1]["hugo_symbol"]
    .value_counts()
    .head(20)
    .index
    .tolist()
)
oncodrive_top = get_top_genes(df_oncodrive, 'SCORE', 'GENE', 20, False)
chasm_top = get_top_genes(df_chasm, 'chasmplus.score', 'Hugo_Symbol', 20, False)
mutsig_top = get_top_genes(df_mutsig, 'q-value', 'Gene', 20, True)  # Lower q-value = more significant

# ─── Venn Diagram 1: Our Model vs OncodriveFML vs CHASMplus ───────────
def plot_venn_oncodrive():
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    venn3(
        [set(predicted_top), set(oncodrive_top), set(chasm_top)],
        set_labels=('IDEA-LUAD', 'OncodriveFML', 'CHASMplus')
    )
    plt.title("Top 20 Genes: IDEA-LUAD vs OncodriveFML vs CHASMplus", fontsize=14)
    plt.savefig(f"{output_dir}/venn_oncodrive_chasm.tiff", dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

# ─── Venn Diagram 2: Our Model vs MutSigCV vs CHASMplus ────────────────
def plot_venn_mutsig():
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    venn3(
        [set(predicted_top), set(mutsig_top), set(chasm_top)],
        set_labels=('IDEA-LUAD', 'MutSigCV', 'CHASMplus')
    )
    plt.title("Top 20 Genes: IDEA-LUAD vs MutSigCV vs CHASMplus", fontsize=14)
    plt.savefig(f"{output_dir}/venn_mutsig_chasm.tiff", dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

# ─── COSMIC Overlap ───────────────────────────────────────────────────
def plot_cosmic_overlap():
    cosmic_genes = {
        "TP53", "IDH1", "PIK3CA", "PTEN", "EGFR", "BRAF", "KRAS",
        "NRAS", "NF1", "ATRX", "CDKN2A", "CDK4", "MDM2", "RB1",
        "FGFR1", "FGFR3", "TERT", "PDGFRA", "CIC", "FUBP1"
    }
    overlaps = [
        len(set(predicted_top) & cosmic_genes),
        len(set(oncodrive_top) & cosmic_genes),
        len(set(chasm_top) & cosmic_genes),
        len(set(mutsig_top) & cosmic_genes)
    ]
    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    bars = plt.bar(
        ['IDEA-LUAD', 'OncodriveFML', 'CHASMplus', 'MutSigCV'],
        overlaps,
        color=['#D55E00','#009E73','#0072B2','#E69F00']
    )
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=13)
    plt.ylabel('Number of COSMIC Genes', fontsize=13)
    plt.title('Overlap with COSMIC Cancer Gene Census (Top 20)', fontsize=14)
    plt.ylim(0, 20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cosmic_overlap.tiff", dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

# ─── Performance Curves ────────────────────────────────────────────────
def plot_performance_curves():
    # Prepare combined scores at gene level
    gene_model = df_real.groupby('hugo_symbol').agg({
        'Model_Score': 'mean',
        'is_driver': 'max'  # Gene is driver if any variant is driver
    }).reset_index()
    gene_chasm = df_chasm.groupby('Hugo_Symbol')['chasmplus.score'].mean().reset_index()
    gene_onco = df_oncodrive.groupby('GENE')['SCORE'].mean().reset_index()
    gene_mutsig = df_mutsig.groupby('Gene')['q-value'].mean().reset_index()
    combined = gene_model.merge(
        gene_chasm,
        left_on='hugo_symbol',
        right_on='Hugo_Symbol',
        how='left'
    ).merge(
        gene_onco,
        left_on='hugo_symbol',
        right_on='GENE',
        how='left'
    ).merge(
        gene_mutsig,
        left_on='hugo_symbol',
        right_on='Gene',
        how='left'
    ).dropna()
    combined['MutSigCV'] = 1 - combined['q-value']

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    for ax in axs:
        ax.set_facecolor('white')

    # ROC Curve
    for method in ['Model_Score', 'SCORE', 'chasmplus.score', 'MutSigCV']:
        fpr, tpr, _ = roc_curve(combined['is_driver'], combined[method])
        auc = roc_auc_score(combined['is_driver'], combined[method])
        label = {
            'Model_Score': 'IDEA-LUAD',
            'SCORE': 'OncodriveFML',
            'chasmplus.score': 'CHASMplus',
            'MutSigCV': 'MutSigCV'
        }[method]
        axs[0].plot(fpr, tpr, lw=2, label=f'{label} (AUC={auc:.3f})')

    axs[0].plot([0, 1], [0, 1], 'k--')
    axs[0].set_xlabel('False Positive Rate', fontsize=13)
    axs[0].set_ylabel('True Positive Rate', fontsize=13)
    axs[0].set_title('ROC Curves', fontsize=14)
    axs[0].legend(fontsize=11)

    # PR Curve
    for method in ['Model_Score', 'SCORE', 'chasmplus.score', 'MutSigCV']:
        precision, recall, _ = precision_recall_curve(combined['is_driver'], combined[method])
        ap = average_precision_score(combined['is_driver'], combined[method])
        label = {
            'Model_Score': 'IDEA-LUAD',
            'SCORE': 'OncodriveFML',
            'chasmplus.score': 'CHASMplus',
            'MutSigCV': 'MutSigCV'
        }[method]
        axs[1].plot(recall, precision, lw=2, label=f'{label} (AP={ap:.3f})')

    axs[1].set_xlabel('Recall', fontsize=13)
    axs[1].set_ylabel('Precision', fontsize=13)
    axs[1].set_title('Precision-Recall Curves', fontsize=14)
    axs[1].legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_curves.tiff", dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

# ─── Generate Summary Statistics ──────────────────────────────────────
def generate_summary():
    cosmic_genes = {
        "TP53", "IDH1", "PIK3CA", "PTEN", "EGFR", "BRAF", "KRAS",
        "NRAS", "NF1", "ATRX", "CDKN2A", "CDK4", "MDM2", "RB1",
        "FGFR1", "FGFR3", "TERT", "PDGFRA", "CIC", "FUBP1"
    }
    data = {
        'Method': ['IDEA-LUAD', 'OncodriveFML', 'CHASMplus', 'MutSigCV'],
        'COSMIC_Overlap': [
            len(set(predicted_top) & cosmic_genes),
            len(set(oncodrive_top) & cosmic_genes),
            len(set(chasm_top) & cosmic_genes),
            len(set(mutsig_top) & cosmic_genes)
        ],
        'Unique_Genes': [
            len(set(predicted_top) - set(oncodrive_top) - set(chasm_top) - set(mutsig_top)),
            len(set(oncodrive_top) - set(predicted_top) - set(chasm_top) - set(mutsig_top)),
            len(set(chasm_top) - set(predicted_top) - set(oncodrive_top) - set(mutsig_top)),
            len(set(mutsig_top) - set(predicted_top) - set(oncodrive_top) - set(chasm_top))
        ]
    }
    summary = pd.DataFrame(data)
    summary.to_csv(f"{output_dir}/method_comparison_summary.csv", index=False)
    return summary

# ─── Main Execution ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating visualizations...")
    plot_venn_oncodrive()
    plot_venn_mutsig()
    plot_cosmic_overlap()
    plot_performance_curves()
    print("Generating summary statistics...")
    summary = generate_summary()
    print("\nBenchmarking Complete!")
    print(f"Results saved to: {output_dir}/")
    print("\nSummary Statistics:")
    print(summary.to_markdown(index=False))
