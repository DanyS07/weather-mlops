import streamlit as st
import pandas as pd
import joblib
import json

st.title("Weather Forecast (RandomForest)")

df = pd.read_csv("data/processed/data.csv")

model = joblib.load("model.pkl")

with open("metrics.json") as f:
    metrics = json.load(f)

st.write(f"RMSE: {metrics['rmse']:.2f}")

future = pd.DataFrame({"hour": list(range(24))})
preds = model.predict(future)

st.line_chart(preds)