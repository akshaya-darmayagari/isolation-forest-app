import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Resolve directory paths safely
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
file_path = None
possible_names = [
    os.path.join(DIR_PATH, "data", "Network Traffic.csv"),
    os.path.join(DIR_PATH, "data", "network_traffic.csv"),
    os.path.join(DIR_PATH, "data", "Network_Traffic.csv")
]
for name in possible_names:
    if os.path.exists(name):
        file_path = name
        break

if file_path is None:
    raise FileNotFoundError("Could not find Network Traffic CSV file.")

# Load dataset and artifacts
df = pd.read_csv(file_path)
model = joblib.load(os.path.join(DIR_PATH, "models", "if_model.pkl"))
scaler = joblib.load(os.path.join(DIR_PATH, "models", "scaler.pkl"))

X = df.drop("label", axis=1)
y_true = df["label"]

X_scaled = scaler.transform(X)

# Isolation forest outputs -1 for anomalies and 1 for normal data points
raw_pred = model.predict(X_scaled)

# Map output to match ground truth labels (0 for normal, 1 for anomaly)
y_pred = np.where(raw_pred == -1, 1, 0)

# Calculate Validation Metrics
print("=================== PERFORMANCE METRICS ===================")
print("Accuracy Score: ", accuracy_score(y_true, y_pred))
print("Precision Score:", precision_score(y_true, y_pred))
print("Recall Score:   ", recall_score(y_true, y_pred))
print("F1-Score:       ", f1_score(y_true, y_pred))
print("===========================================================")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))