import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

locations = {
    "technopark": (8.5241, 76.9366),
    "thampanoor": (8.4875, 76.9530)
}

def fetch_data(name, lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m"
        f"&past_days=7"
    )

    res = requests.get(url)
    data = res.json()

    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temperature_2m": data["hourly"]["temperature_2m"]
    })

    path = RAW_DIR / f"{name}.csv"
    df.to_csv(path, index=False)

    print(f"{name} saved → {path}")

for name, (lat, lon) in locations.items():
    fetch_data(name, lat, lon)

print("Collection complete")