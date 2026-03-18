import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def train(region):
    # Load data
    df = pd.read_csv(f"data/raw/{region}.csv")
    df["time"] = pd.to_datetime(df["time"])

    # Feature engineering
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek

    df = df.dropna(subset=["temperature_2m"])
    df = df.drop(columns=["time"])

    # Split
    X = df.drop(columns=["temperature_2m"])
    y = df["temperature_2m"]

    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_scaled, y)

    # Save model
    with open(f"models/trained/{region}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save scaler
    with open(f"models/scalers/{region}.csv_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

# Train both regions
train("technopark")
train("thampanoor")