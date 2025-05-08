#!/usr/bin/env python3
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import ADASYN
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ─── Configuration ──────────────────────────────────────────────────
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ─── Data Loading ───────────────────────────────────────────────────
df = pd.read_csv("shap_filtered_data.csv")
y = df["is_driver"]
X = df.drop("is_driver", axis=1).astype(np.float32)

# ─── Cross-Validation ───────────────────────────────────────────────
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_aucs = []
os.makedirs("training_plots", exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"\n🔁 Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Apply ADASYN oversampling
    X_train_res, y_train_res = ADASYN(random_state=42).fit_resample(X_train, y_train)

    model = tf.keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            'accuracy'
        ]
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=8,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train_res, y_train_res,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    y_val_prob = model.predict(X_val, verbose=0).ravel()
    auc = roc_auc_score(y_val, y_val_prob)
    print(f"✅ Fold {fold + 1} AUC: {auc:.4f}")
    fold_aucs.append(auc)

    # Save fold training plot
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title(f'Fold {fold + 1} Training History')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f"training_plots/fold_{fold + 1}_history.png")
    plt.close()

# ─── Final Model Training ───────────────────────────────────────────
print("\n🚀 Training final model on full dataset...")
X_train_full, y_train_full = ADASYN(random_state=42).fit_resample(X, y)

final_model = tf.keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

history = final_model.fit(
    X_train_full, y_train_full,
    epochs=100,
    batch_size=64,
    verbose=1
)

# ─── Model Saving with Version Control ───────────────────────────────
mean_auc = np.mean(fold_aucs)
if mean_auc >= 0.88:
    model_path = f"{MODEL_DIR}/driver_prediction_SHAP_{TIMESTAMP}_auc{mean_auc:.4f}.keras"
    final_model.save(model_path)
    print(f"\n🧠 SHAP-filtered model saved as {model_path}")
    print("🔒 Original 94% AUC model remains unchanged in its location")
else:
    print("\n⛔ SHAP model underperformed (AUC < 0.88) - not saved")

# ─── Performance Analysis ───────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.boxplot(fold_aucs)
plt.title(f'CV AUC Distribution\nMean: {mean_auc:.4f} ± {np.std(fold_aucs):.4f}')
plt.ylabel('AUC')

plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='Train AUC')
plt.title('Final Model Training')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.tight_layout()
plt.savefig("training_plots/training_summary.png")
plt.close()

# ─── Final Report ───────────────────────────────────────────────────
print("\n📊 Final Report:")
print(f"📈 Cross-Validated AUCs: {[round(auc, 4) for auc in fold_aucs]}")
print(f"✅ Mean AUC: {mean_auc:.4f} ± {np.std(fold_aucs):.4f}")
print(f"🏆 Best Fold AUC: {max(fold_aucs):.4f}")

if mean_auc >= 0.88:
    print("🎯 SUCCESS: SHAP model meets target AUC!")
    print("💡 Compare with original model using:")
    print(f"   original = tf.keras.models.load_model('driver_prediction_model.keras')")
    print(f"   shap_model = tf.keras.models.load_model('{model_path}')")
else:
    print("⚠️ WARNING: SHAP model underperformed - keeping original 94% AUC model")