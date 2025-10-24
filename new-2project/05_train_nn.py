# 05_train_nn.py
import numpy as np, pandas as pd, joblib, tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from imblearn.over_sampling import ADASYN
from tensorflow.keras import layers, regularizers, callbacks
import matplotlib.pyplot as plt

X_train = pd.read_csv("X_train_proc.tsv", sep="\t").values
X_test  = pd.read_csv("X_test_proc.tsv",  sep="\t").values
y_train = pd.read_csv("y_train.tsv", sep="\t").values.ravel()
y_test  = pd.read_csv("y_test.tsv",  sep="\t").values.ravel()

# Oversample positives in training only
X_tr, y_tr = ADASYN(random_state=42).fit_resample(X_train, y_train)

model = tf.keras.Sequential([
    layers.Input(shape=(X_tr.shape[1],)),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(name='auc'),
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       'accuracy'])

es = callbacks.EarlyStopping(monitor='val_auc', patience=12, mode='max', restore_best_weights=True)
hist = model.fit(X_tr, y_tr, validation_data=(X_test, y_test),
                 epochs=120, batch_size=64, callbacks=[es], verbose=1)

y_prob = model.predict(X_test).ravel()
auc = roc_auc_score(y_test, y_prob)
ap  = average_precision_score(y_test, y_prob)
print(f"\n✅ Final Test AUROC: {auc:.4f} | AUPRC: {ap:.4f}")

model.save("driver_prediction_model.keras")
print("🧠 saved driver_prediction_model.keras")

# Curves
fpr,tpr,_ = roc_curve(y_test, y_prob)
prec,rec,_= precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.plot(fpr,tpr); plt.title(f"ROC (AUC={auc:.3f})")
plt.subplot(1,2,2); plt.plot(rec,prec); plt.title(f"PR (AP={ap:.3f})")
plt.tight_layout(); plt.savefig("model_performance_curves.png", dpi=300, bbox_inches='tight')
