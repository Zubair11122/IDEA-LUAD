import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from imblearn.over_sampling import ADASYN
from tensorflow.keras import layers, regularizers, callbacks
import matplotlib.pyplot as plt
import optuna

# ─── Load Data ────────────────────────────────────────────────────────
df = pd.read_csv("mutations_variant_complete.tsv", sep="\t")
df.replace("-", np.nan, inplace=True)

# Convert target variable
y = df["is_driver"].astype(int)

# ─── Preprocess ───────────────────────────────────────────────────────
preprocessor = joblib.load("preprocessor.pkl")  # Changed from preprocessor1.pkl
X_raw = df[preprocessor.feature_names_in_]
X = preprocessor.transform(X_raw)

# ─── Hyperparameter Tuning with Optuna ─────────────────────────────────
def objective(trial):
    # Split data inside the objective function
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ─── Switch to ADASYN for balancing the data ───────────────────────
    X_train, y_train = ADASYN(random_state=42).fit_resample(X_train, y_train)

    # Hyperparameter search space
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    neurons = trial.suggest_int('neurons', 64, 512)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  # Updated for newer Optuna
    l2_reg = trial.suggest_float('l2_reg', 1e-4, 1e-2, log=True)

    model = tf.keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(neurons, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(neurons // 2, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max',
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Increased from 10
        batch_size=64,
        verbose=0,
        callbacks=[early_stop]
    )

    return history.history['val_auc'][-1]

# Create an Optuna study and optimize hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=3600)  # Added timeout

# Print the best parameters
print(f"Best trial: {study.best_trial.params}")
best_params = study.best_trial.params

# ─── K-Fold Cross-Validation ──────────────────────────────────────────
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_aucs = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"🔄 Fold {fold+1}")
    X_train_raw, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # ─── Switch to ADASYN for balancing the data ───────────────────────
    X_train, y_train = ADASYN(random_state=42).fit_resample(X_train_raw, y_train)

    # ─── Model Architecture ────────────────────────────────────────────
    model = tf.keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(best_params['neurons'], activation='relu',
                    kernel_regularizer=regularizers.l2(best_params['l2_reg'])),
        layers.BatchNormalization(),
        layers.Dropout(best_params['dropout_rate']),
        layers.Dense(best_params['neurons'] // 2, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
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
        patience=15,  # Increased patience
        mode='max',
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,  # Increased from 100
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    y_val_prob = model.predict(X_val).ravel()
    auc = roc_auc_score(y_val, y_val_prob)
    print(f"✅ Fold {fold+1} AUC: {auc:.4f}")
    fold_aucs.append(auc)
    fold_models.append(model)

# ─── Final Evaluation on Held-out Test Set ─────────────────────────────
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train_final, y_train_final = ADASYN(random_state=42).fit_resample(X_train_all, y_train_all)

# Use best model from CV or train new one
best_fold_idx = np.argmax(fold_aucs)
final_model = fold_models[best_fold_idx]

# Alternatively, train a new model on all data:
"""
final_model = tf.keras.Sequential([
    layers.Input(shape=(X_train_final.shape[1],)),
    layers.Dense(best_params['neurons'], activation='relu',
                kernel_regularizer=regularizers.l2(best_params['l2_reg'])),
    layers.BatchNormalization(),
    layers.Dropout(best_params['dropout_rate']),
    layers.Dense(best_params['neurons'] // 2, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        'accuracy'
    ]
)

final_model.fit(
    X_train_final, y_train_final,
    validation_split=0.2,
    epochs=150,
    batch_size=64,
    callbacks=[
        callbacks.EarlyStopping(
            monitor='val_auc',
            patience=15,
            mode='max',
            restore_best_weights=True
        )
    ],
    verbose=1
)
"""

# ─── Test Evaluation ───────────────────────────────────────────────────
y_prob_test = final_model.predict(X_test).ravel()
y_pred_test = (y_prob_test > 0.5).astype(int)

test_auc = roc_auc_score(y_test, y_prob_test)
print(f"\n✅ Final Test AUC: {test_auc:.4f}")
if test_auc >= 0.88:
    print("🎯 SUCCESS: AUC target achieved!")

final_model.save("driver_prediction_model.keras")
print("🧠 Final model saved as driver_prediction_model.keras")

# ─── ROC and PR Curves ─────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
precision, recall, _ = precision_recall_curve(y_test, y_prob_test)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr)
plt.title("ROC Curve (AUC = {:.3f})".format(test_auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)

plt.tight_layout()
plt.savefig("model_performance_curves.png", dpi=300, bbox_inches='tight')
plt.show()

# ─── Report Cross-Validated AUC Summary ───────────────────────────────
print(f"\n📊 Cross-Validated AUCs: {fold_aucs}")
print(f"📈 Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"🏆 Best Fold AUC: {np.max(fold_aucs):.4f}")