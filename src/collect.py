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

    if "hourly" not in data:
        raise ValueError("API response missing 'hourly' data")

    df = pd.DataFrame(data["hourly"])
    return df


def update_or_create(region, coords):
    file_path = f"data/raw/{region}.csv"

    # -------------------------
    # CASE 1: File exists → incremental update
    # -------------------------
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)

        existing["time"] = pd.to_datetime(existing["time"])
        last_date = existing["time"].max().date()

        today = datetime.today().date()

        # If already up to date
        if last_date >= today:
            print(f"{region}: already up to date")
            return

        start_date = last_date + timedelta(days=1)
        end_date = today

        print(f"{region}: fetching {start_date} → {end_date}")

        new_data = fetch_data(
            coords["lat"],
            coords["lon"],
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

        new_data["time"] = pd.to_datetime(new_data["time"])

        # Append and remove duplicates
        df = pd.concat([existing, new_data])
        df = df.drop_duplicates(subset=["time"]).sort_values("time")

    # -------------------------
    # CASE 2: First run → full 6 months
    # -------------------------
    else:
        print(f"{region}: first run (6 months data)")

        end_date = datetime.today()
        start_date = end_date - timedelta(days=180)

        df = fetch_data(
            coords["lat"],
            coords["lon"],
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

    # Save
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(file_path, index=False)

    print(f"{region}: saved → {file_path}")


def main():
    for region, coords in LOCATIONS.items():
        try:
            update_or_create(region, coords)
        except Exception as e:
            print(f"Error in {region}: {e}")


if __name__ == "__main__":
    main()