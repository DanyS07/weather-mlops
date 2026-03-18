import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "data.csv"

files = list(RAW_PATH.glob("*.csv"))

if not files:
    raise ValueError("No raw files found")

dfs = [pd.read_csv(f) for f in files]
data = pd.concat(dfs, ignore_index=True)

data = data.dropna()

data["time"] = pd.to_datetime(data["time"])
data["hour"] = data["time"].dt.hour

data.to_csv(OUT_FILE, index=False)

print(f"Saved processed data → {OUT_FILE}")