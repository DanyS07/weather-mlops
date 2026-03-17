import numpy as np
import os
import yaml
import json
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

PROCESSED_PATH = "data/processed"
MODEL_PATH = "models/trained"
os.makedirs(MODEL_PATH, exist_ok=True)

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

LOOKBACK = params["data"]["lookback"]
HORIZON = params["data"]["horizon"]

def build_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(HORIZON)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse",
                  metrics=["mae"])
    return model

def train_region(region):
    X_train = np.load(f"{PROCESSED_PATH}/X_train_{region}.npy")
    y_train = np.load(f"{PROCESSED_PATH}/y_train_{region}.npy")
    X_test = np.load(f"{PROCESSED_PATH}/X_test_{region}.npy")
    y_test = np.load(f"{PROCESSED_PATH}/y_test_{region}.npy")

    model = build_model((X_train.shape[1], X_train.shape[2]))

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    rmse = np.sqrt(loss)

    # Save model
    model.save(f"{MODEL_PATH}/{region}_model.keras")

    return mae, rmse

def main():
    metrics = {}
    regions = ["technopark", "thampanoor"]

    for region in regions:
        mae, rmse = train_region(region)
        metrics[f"mae_{region}"] = float(mae)
        metrics[f"rmse_{region}"] = float(rmse)

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save version info
    version_info = {
        "version": "1.0",
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **metrics
    }

    with open("version.json", "w") as f:
        json.dump(version_info, f, indent=4)

if __name__ == "__main__":
    main()