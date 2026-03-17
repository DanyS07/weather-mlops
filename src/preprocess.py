import pandas as pd
import numpy as np
import os
import yaml
from sklearn.preprocessing import MinMaxScaler
import pickle

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
MODEL_PATH = "models/scalers"

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

LOOKBACK = params["data"]["lookback"]
HORIZON = params["data"]["horizon"]

def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon, 0])  # predict temperature
    return np.array(X), np.array(y)

def process_file(file_name):
    df = pd.read_csv(f"{RAW_PATH}/{file_name}")

    # Drop null temperature
    df = df.dropna(subset=["temperature_2m"])

    # Feature engineering
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek

    df = df.drop(columns=["time"])

    # Scaling
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    # Save scaler (one per region)
    scaler_path = f"{MODEL_PATH}/{file_name}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Create sequences
    X, y = create_sequences(scaled, LOOKBACK, HORIZON)

    # Train-test split
    split = int(len(X) * (1 - params["model"]["test_size"]))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Save arrays
    name = file_name.replace(".csv", "")
    np.save(f"{PROCESSED_PATH}/X_train_{name}.npy", X_train)
    np.save(f"{PROCESSED_PATH}/y_train_{name}.npy", y_train)
    np.save(f"{PROCESSED_PATH}/X_test_{name}.npy", X_test)
    np.save(f"{PROCESSED_PATH}/y_test_{name}.npy", y_test)

def main():
    for file in os.listdir(RAW_PATH):
        if file.endswith(".csv"):
            process_file(file)

if __name__ == "__main__":
    main()