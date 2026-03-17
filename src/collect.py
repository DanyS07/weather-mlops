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


# -----------------------------
# Fetch data from API
# -----------------------------
def fetch_data(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(FEATURES),
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"API failed: {response.status_code}")

    data = response.json()

    if "hourly" not in data:
        raise Exception("No 'hourly' data in response")

    df = pd.DataFrame(data["hourly"])
    return df


# -----------------------------
# Update or create CSV
# -----------------------------
def update_or_create(region, coords):
    file_path = f"data/raw/{region}.csv"

    print(f"\n--- {region.upper()} ---")

    try:
        # -------------------------
        # Case 1: File exists → incremental
        # -------------------------
        if os.path.exists(file_path):
            existing = pd.read_csv(file_path)
            existing["time"] = pd.to_datetime(existing["time"])

            last_date = existing["time"].max().date()
            today = datetime.today().date()

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

            print(f"{region}: new rows = {len(new_data)}")

            if new_data.empty:
                print(f"{region}: no new data")
                return

            new_data["time"] = pd.to_datetime(new_data["time"])

            df = pd.concat([existing, new_data])
            df = df.drop_duplicates(subset=["time"]).sort_values("time")

        # -------------------------
        # Case 2: First run → 6 months
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

            print(f"{region}: fetched rows = {len(df)}")

            if df.empty:
                print(f"{region}: ERROR → empty data")
                return

            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time")

        # -------------------------
        # Save file
        # -------------------------
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(file_path, index=False)

        print(f"{region}: saved → {file_path}")

    except Exception as e:
        print(f"{region}: FAILED → {e}")


# -----------------------------
# Main execution
# -----------------------------
def main():
    print("Starting data collection...")

    for region, coords in LOCATIONS.items():
        print(f"\nProcessing region: {region}")
        update_or_create(region, coords)

    print("\nData collection completed.")


if __name__ == "__main__":
    main()