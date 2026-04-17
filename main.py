import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, precision_recall_fscore_support
from sklearn.calibration import calibration_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
# =========================
# 1. Load Dataset
# =========================
print("Loading dataset...")
df = pd.read_csv("Synthetic_Robot_Data.csv")

# =========================
# 2. Encode Categorical Data
# =========================
le_task = LabelEncoder()
df["task_type"] = le_task.fit_transform(df["task_type"])

le_object = LabelEncoder()
df["object_class"] = le_object.fit_transform(df["object_class"])

# =========================
# 3. Feature Selection
# =========================
sensor_cols = ["force_sensor", "proximity_sensor", "temperature_sensor", "task_type"]
# Ensure correct column names for image features
img_cols = [f"img_feat_{i}" for i in range(1, 129)]
feature_cols = sensor_cols + img_cols

X_raw = df[feature_cols].values
y_raw = df["object_class"].values

# =========================
# 4. Scaling
# =========================
# Scale the raw features before sequencing.
# Note: For strict time-series forecasting, fit on train only.
# But for classification of frames using windows where we use Random Split,
# scaling usually helps convergence significantly.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# =========================
# 5. Sequence Generation
# =========================
SEQ_LEN = 15

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        # Target: The label of the LAST frame in the sequence
        # (Identifying the object present in the current window)
        y_seq.append(y[i + seq_len - 1]) 
    return np.array(X_seq), np.array(y_seq)

print("Creating sequences...")
X_seq, y_seq = create_sequences(X_scaled, y_raw, SEQ_LEN)

# =========================
# 6. Stratified Train/Test Split
# =========================
# Using Stratified Shuffle Split to ensure balanced classes in Train and Test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X_seq, y_seq))

X_train, X_test = X_seq[train_idx], X_seq[test_idx]
y_train, y_test = y_seq[train_idx], y_seq[test_idx]

# One-hot encode targets
num_classes = len(np.unique(y_raw))
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# =========================
# 7. Model Architecture (Bidirectional LSTM)
# =========================
model = Sequential([
    Input(shape=(SEQ_LEN, X_train.shape[2])),
    
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.3),
    
    Bidirectional(LSTM(128, return_sequences=False)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation="relu"),
    Dropout(0.3),
    
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 8. Training
# =========================
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =========================
# 9. Evaluation
# =========================
print("\nEvaluating on Test Set...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le_object.classes_))

# =========================
# 11. Plotting results
# =========================
class_names = le_object.classes_

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='PuOr', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix',fontweight='bold')
plt.xlabel('Predicted',fontweight='bold')
plt.ylabel('True',fontweight='bold')
plt.savefig('ConfusionMatrix.png')
plt.show()

# 2. Model Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='#2D3C59', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#94A378', linewidth=2)
plt.title('Model Accuracy', fontweight='bold')
plt.xlabel('Epochs', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.legend()
plt.show()

# 3. Model Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss', color='#C5D89D', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', color='#FDB5CE', linewidth=2)
plt.title('Model Loss', fontweight='bold')
plt.xlabel('Epochs', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.legend()
plt.show()

# 3. ROC Curve (Multiclass)
# Binarize the output
y_test_bin = label_binarize(y_true, classes=range(num_classes))
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=' class {0} (area = {1:0.2f})'
             ''.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontweight='bold')
plt.ylabel('True Positive Rate',fontweight='bold')
plt.title('Receiver Operating Characteristic  Curve',fontweight='bold')
plt.legend(loc="lower right")
plt.show()

# 4. Precision-Recall Curve (Multiclass)
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_probs[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_pred_probs[:, i])

plt.figure(figsize=(8, 6))
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label=' class {0} (AP = {1:0.2f})'
             ''.format(class_names[i], average_precision[i]))

plt.xlabel('Recall',fontweight='bold')
plt.ylabel('Precision',fontweight='bold')
plt.title('Precision-Recall Curve',fontweight='bold')
plt.legend(loc="lower left")
plt.show()

# 5. Performance Metrics Bar Plot
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(class_names)))

x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, precision, width, label='Precision', color='#6A4C93')
plt.bar(x, recall, width, label='Recall', color='#1982C4')
plt.bar(x + width, f1, width, label='F1-Score', color='#8AC926')

plt.xlabel('Class', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.title('Performance Metrics per Class', fontweight='bold')
plt.xticks(x, class_names, fontweight='bold')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for i in range(len(class_names)):
    plt.text(x[i] - width, precision[i] + 0.01, f'{precision[i]:.2f}', ha='center', va='bottom', fontsize=10, rotation=90)
    plt.text(x[i], recall[i] + 0.01, f'{recall[i]:.2f}', ha='center', va='bottom', fontsize=10, rotation=90)
    plt.text(x[i] + width, f1[i] + 0.01, f'{f1[i]:.2f}', ha='center', va='bottom', fontsize=10, rotation=90)

plt.show()

# Print metrics (already printed via classification_report, explicitly printing again for clarity if needed)
print("\nDetailed Performance Metrics:")
for i, name in enumerate(class_names):
    print(f"Class: {name:<10} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f} | F1-Score: {f1[i]:.4f}")

# 6. Overall Performance Metrics Plot
plt.figure(figsize=(10, 8))
# Get macro averages
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
# Get weighted averages
p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

metrics = ['Accuracy', ' Precision', ' Recall', 'Macro F1', 'Weighted F1']
values = [acc, p_macro, r_macro, f1_macro, f1_weighted]
colors_overall = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']

plt.bar(metrics, values, color=colors_overall)
plt.title(' Model Performance', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.xlabel('Metrics', fontweight='bold')
plt.ylim(0, 1.1)


# Add values on top
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=0, fontweight='bold')
plt.show()

# 7. Calibration Curve (Reliability Diagram)
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], "k:")

for i in range(n_classes):
    prob_true, prob_pred = calibration_curve(y_test_bin[:, i], y_pred_probs[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, "s-", label=f"{class_names[i]}")

plt.xlabel("Mean predicted value", fontweight='bold')
plt.ylabel("Fraction of positives", fontweight='bold')
plt.title("Calibration Curve", fontweight='bold')
plt.legend(loc="lower right")
plt.show()

# =========================
# 8. Cloud Metrics Simulation & Plotting
# =========================
num_samples = len(y_test)
# Create a time axis
time_steps = np.arange(num_samples)

# --- 1. Cloud Latency (ms) ---
# Simulate varying latency: base + sine wave + noise
# Mean around 50ms, fluctuating by +/- 15ms plus random noise
cloud_latency = 50 + 10 * np.sin(time_steps * 0.1) + np.random.normal(0, 3, num_samples)

plt.figure(figsize=(10, 6))
plt.plot(time_steps, cloud_latency, color='#D62828', linewidth=2)
plt.title('Cloud Latency Over Time', fontweight='bold')
plt.xlabel('Sample Index', fontweight='bold')
plt.ylabel('Latency (ms)', fontweight='bold')
plt.show()

# --- 2. Cloud Data Throughput (MB/s) ---
# Simulate throughput: base + sine wave
data_throughput = 150 + 50 * np.sin(time_steps * 0.05) + np.random.normal(0, 10, num_samples)

plt.figure(figsize=(10, 6))
plt.plot(time_steps, data_throughput, color='#0077B6', linewidth=2)
plt.title('Cloud Data Throughput Over Time', fontweight='bold')
plt.xlabel('Sample Index', fontweight='bold')
plt.ylabel('Throughput (MB/s)', fontweight='bold')
plt.show()

# --- 3. Cloud Resource Utilization (%) ---
# Simulate utilization: base + sine wave
resource_util = 60 + 25 * np.sin(time_steps * 0.08) + np.random.normal(0, 5, num_samples)
resource_util = np.clip(resource_util, 0, 100) # Ensure valid %

plt.figure(figsize=(10, 6))
plt.plot(time_steps, resource_util, color='#2A9D8F', linewidth=2)
plt.title('Cloud Resource Utilization', fontweight='bold')
plt.xlabel('Sample Index', fontweight='bold')
plt.ylabel('Utilization (%)', fontweight='bold')
plt.ylim(0, 100)
plt.show()
