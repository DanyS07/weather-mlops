import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/processed/data.csv")

X = df[["hour"]]
y = df["temperature_2m"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

preds = model.predict(X)
rmse = float(np.sqrt(mean_squared_error(y, preds)))

joblib.dump(model, "model.pkl")

with open("metrics.json", "w") as f:
    json.dump({"rmse": rmse}, f, indent=4)

print("Training complete")