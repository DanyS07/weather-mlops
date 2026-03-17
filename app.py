import streamlit as st
import numpy as np
import pandas as pd
import json
import tensorflow as tf
import plotly.graph_objects as go
import pickle

# -----------------------------
# Load version info
# -----------------------------
with open("version.json") as f:
    version = json.load(f)

st.title("Weather Forecast App")

st.caption(
    f"Model v{version['version']} | "
    f"Trained: {version['trained_on']} | "
    f"RMSE Technopark: {version['rmse_technopark']:.2f}°C | "
    f"RMSE Thampanoor: {version['rmse_thampanoor']:.2f}°C"
)

# -----------------------------
# Load model & scaler
# -----------------------------
def load_model(region):
    return tf.keras.models.load_model(f"models/trained/{region}_model.keras")

def load_scaler(region):
    with open(f"models/scalers/{region}.csv_scaler.pkl", "rb") as f:
        return pickle.load(f)

# -----------------------------
# Load data
# -----------------------------
def load_data(region):
    df = pd.read_csv(f"data/raw/{region}.csv")
    df["time"] = pd.to_datetime(df["time"])
    return df

# -----------------------------
# Prepare input
# -----------------------------
def create_input(df, region):
    df = df.dropna(subset=["temperature_2m"])

    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek

    df = df.drop(columns=["time"])

    scaler = load_scaler(region)
    scaled = scaler.transform(df)

    return scaled[-48:].reshape(1, 48, scaled.shape[1])

# -----------------------------
# Inverse scaling (temperature)
# -----------------------------
def inverse_temperature(scaler, forecast):
    temp_min = scaler.data_min_[0]
    temp_max = scaler.data_max_[0]
    return forecast * (temp_max - temp_min) + temp_min

# -----------------------------
# Plot
# -----------------------------
def plot_forecast(df, forecast):
    last_time = df["time"].iloc[-1]
    future_times = pd.date_range(last_time, periods=24, freq="H")

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=df["time"].tail(48),
        y=df["temperature_2m"].tail(48),
        mode='lines',
        name='Actual'
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=future_times,
        y=forecast.flatten(),
        mode='lines',
        name='Forecast'
    ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        legend_title="Legend"
    )

    return fig

# -----------------------------
# Region display
# -----------------------------
def show_region(region, tab):
    with tab:
        model = load_model(region)
        scaler = load_scaler(region)
        df = load_data(region)

        X = create_input(df, region)

        forecast = model.predict(X)
        forecast = inverse_temperature(scaler, forecast)

        fig = plot_forecast(df, forecast)
        st.plotly_chart(fig, use_container_width=True)

        # Sidebar stats
        st.sidebar.subheader(f"{region.capitalize()} Stats")
        st.sidebar.write(f"Min Temp: {forecast.min():.2f} °C")
        st.sidebar.write(f"Max Temp: {forecast.max():.2f} °C")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Technopark", "Thampanoor"])

show_region("technopark", tab1)
show_region("thampanoor", tab2)