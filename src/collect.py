import requests
import pandas as pd
from datetime import datetime, timedelta
import os

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

LOCATIONS = {
    "technopark": {"lat": 8.5574, "lon": 76.8800},
    "thampanoor": {"lat": 8.4875, "lon": 76.9525},
}

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m"
]


def fetch_data(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(FEATURES),
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    return df


def main():
    os.makedirs("data/raw", exist_ok=True)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    for name, coords in LOCATIONS.items():
        df = fetch_data(coords["lat"], coords["lon"], start_str, end_str)

        file_path = f"data/raw/{name}.csv"
        df.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")


if __name__ == "__main__":
    main()