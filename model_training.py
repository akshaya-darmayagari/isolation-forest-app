import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Create models directory
os.makedirs("models", exist_ok=True)

# Dynamic file resolver to handle spaces or capitalizations in "Network Traffic.csv"
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

if file_path is None and os.path.exists(os.path.join(DIR_PATH, "data")):
    for f in os.listdir(os.path.join(DIR_PATH, "data")):
        if "network" in f.lower() and f.endswith(".csv"):
            file_path = os.path.join(DIR_PATH, "data", f)
            break

if file_path is None:
    raise FileNotFoundError("Could not locate your Network Traffic CSV file inside the 'data/' directory.")

# Load data
df = pd.read_csv(file_path)

# Drop the ground-truth label to train as an unsupervised anomaly detection system
X = df.drop("label", axis=1)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Extract contamination rate from baseline label distribution (~10% anomalies in typical network datasets)
# If no label is present, defaults to 0.1 (10% contamination)
try:
    contamination_rate = df["label"].mean()
    if contamination_rate == 0.0 or np.isnan(contamination_rate):
        contamination_rate = 0.1
except KeyError:
    contamination_rate = 0.1

# Initialize and Train Isolation Forest
model = IsolationForest(contamination=contamination_rate, random_state=42, n_jobs=-1)
model.fit(X_scaled)

# Save artifacts
joblib.dump(model, "models/if_model.pkl", compress=3)
joblib.dump(scaler, "models/scaler.pkl", compress=3)

print(f"Isolation Forest trained successfully. Selected Contamination Rate: {contamination_rate:.4f}")