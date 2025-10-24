# ===============================================================
#  TabTransformer | CV evaluation + metric‑bar‑chart + diagram
# ===============================================================
import os, sys, joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, average_precision_score
)

# ---------------------------- CONFIG ---------------------------
RAW_FILE      = "C:/Users/Zubair/Desktop/Sup/mutations_variant_complete.tsv"
NPY_X         = "X_test.npy"          # delete or comment if not using .npy
NPY_Y         = "y_test.npy"
PREPROCESSOR  = "C:/Users/Zubair/Desktop/Sup/preprocessor.pkl"
MODEL_FILE    = "C:/Users/Zubair/Desktop/Sup/driver_prediction_model.keras"

FIG_OUT_BARS  = "C:/Users/Zubair/Desktop/Sup/model_metrics_Diagram1.png"
K_FOLDS       = 10                    # Stratified‑K
BATCH_SIZE    = 1024
LABEL_COLUMN  = "is_driver"
TEST_SPLIT_COL= None                  # e.g. "split" if you stored fold

ARCH_OUT_PNG  = "C:/Users/Zubair/Desktop/Sup/driver_prediction_architecture.png"
# ---------------------------------------------------------------


# ===============================================================
#  Helpers
# ===============================================================
def load_full_data():
    """Return X (array/sparse) and y (vector) ready for CV evaluation."""
    if os.path.exists(NPY_X) and os.path.exists(NPY_Y):
        print("[INFO] Loading X/Y from .npy files …")
        return (np.load(NPY_X, allow_pickle=True),
                np.load(NPY_Y, allow_pickle=True))

    print("[INFO] Loading raw TSV …")
    df = pd.read_csv(RAW_FILE, sep="\t")
    if TEST_SPLIT_COL:
        df = df[df[TEST_SPLIT_COL] == "test"]

    y = df[LABEL_COLUMN].values
    X = df.drop(columns=[LABEL_COLUMN])

    pre = joblib.load(PREPROCESSOR)
    print("[INFO] Transforming with fitted preprocessor …")
    X_ready = pre.transform(X)                  # ndarray or sparse matrix
    return X_ready, y


def evaluate_cv(model, X, y, k_folds=10):
    """Return dict of mean±std metrics over Stratified K folds (no retraining)."""
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    hist = {m: [] for m in ["Accuracy", "F1", "ROC-AUC", "PR-AUC"]}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"[CV] Fold {fold}/{k_folds}  |  test size = {len(test_idx)}")
        X_test, y_test = X[test_idx], y[test_idx]    # slices ok for sparse

        y_pred_proba = model.predict(X_test, batch_size=BATCH_SIZE).ravel()
        y_pred       = (y_pred_proba >= 0.5).astype(int)

        hist["Accuracy"].append(accuracy_score(y_test, y_pred))
        hist["F1"].append(f1_score(y_test, y_pred))
        hist["ROC-AUC"].append(roc_auc_score(y_test, y_pred_proba))
        hist["PR-AUC"].append(average_precision_score(y_test, y_pred_proba))

    return {k: (np.mean(v), np.std(v)) for k, v in hist.items()}


def plot_bar_means(cv_metrics, outfile):
    """Bar chart of mean scores; error bar = 1 σ."""
    names = list(cv_metrics.keys())
    means = [cv_metrics[k][0] for k in names]
    stds  = [cv_metrics[k][1] for k in names]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, means, yerr=stds, capsize=6, width=0.6)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    # ----------- fixed line -----------
    plt.title(f"Stratified {K_FOLDS}-Fold CV (mean ± σ)")
    # ----------------------------------

    for bar, mu in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2,
                 mu - 0.05,
                 f"{mu:.2f}",
                 ha="center", va="top", fontsize=11, color="white")

    plt.tight_layout()
    plt.savefig(outfile, dpi=1200)
    plt.show()
    print(f"[DONE] Metric figure →  {outfile}")


def make_architecture_diagram(model, out_png):
    """Draw the network with keras.utils.plot_model (Graphviz) or fallback."""
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file=out_png, show_shapes=True,
                   show_layer_names=True, dpi=300)
        print(f"[DONE] Architecture diagram →  {out_png}")
    except (ImportError, OSError) as e:
        print("[WARN] plot_model failed ⇒ trying VisualKeras …", e)
        try:
            import visualkeras
            img = visualkeras.layered_view(model, legend=True)
            img.save(out_png)
            print(f"[DONE] Architecture diagram (visualkeras) →  {out_png}")
        except Exception as err:
            print("[ERROR] Could not create architecture diagram:", err)
            print("Make sure Graphviz is installed OR `pip install visualkeras`.")


# ===============================================================
#  Main
# ===============================================================
if __name__ == "__main__":
    # 1) data -----------------------------------------------------
    X_all, y_all = load_full_data()
    print(f"[INFO] Feature matrix shape : {getattr(X_all, 'shape', 'sparse‑matrix')}")
    print(f"[INFO] Label vector length  : {len(y_all)}")

    # 2) model ----------------------------------------------------
    print("[INFO] Loading saved model …")
    model = tf.keras.models.load_model(MODEL_FILE)

    # 3) CV metrics ----------------------------------------------
    print(f"[INFO] Running Stratified {K_FOLDS}-fold evaluation …")
    cv_stats = evaluate_cv(model, X_all, y_all, k_folds=K_FOLDS)
    for k, (mu, sigma) in cv_stats.items():
        print(f"  • {k:<8}: {mu:.4f}  ± {sigma:.4f}")

    # 4) plot -----------------------------------------------------
    plot_bar_means(cv_stats, FIG_OUT_BARS)

    # 5) architecture diagram ------------------------------------
    make_architecture_diagram(model, ARCH_OUT_PNG)
