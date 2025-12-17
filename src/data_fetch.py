# src/data_fetch.py
import os
import time
import json
import requests
from datetime import date, timedelta

import pandas as pd
from meteostat import Point, Daily

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
NASA_DIR = os.path.join(ROOT, "data", "nasa_power")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(NASA_DIR, exist_ok=True)

# Points: id, lat, lon
POINTS = [
    {"id": "field_1", "lat": 17.3850, "lon": 78.4867},
    {"id": "field_2", "lat": 13.0827, "lon": 80.2707},
]

VARS = "PRECTOT,T2M_MAX,T2M_MIN,T2M,RH2M,WS2M"


# ---------- NASA POWER (unchanged) ----------

def fetch_nasa_power_point(point, start: date, end: date, temporal="DAILY"):
    base = f"https://power.larc.nasa.gov/api/temporal/{temporal.lower()}/point"
    params = {
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "latitude": point["lat"],
        "longitude": point["lon"],
        "community": "ag",
        "parameters": VARS,
        "format": "CSV",
    }

    url = (
        f"{base}?latitude={params['latitude']}"
        f"&longitude={params['longitude']}"
        f"&start={params['start']}&end={params['end']}"
        f"&community=ag&parameters={params['parameters']}&format=CSV"
    )

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text

def fetch_meteostat(point, start: date, end: date):
    """
    Fetch daily historical weather from Meteostat ground stations.
    Accepts start/end as datetime.date or datetime-like and converts them
    to pandas.Timestamp for compatibility with Meteostat.
    """
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    location = Point(point["lat"], point["lon"])
    data = Daily(location, start_ts, end_ts)
    df = data.fetch()

    if df.empty:
        raise RuntimeError("Meteostat returned empty dataframe")

    df = df.rename(
        columns={
            "tmax": "T2M_MAX",
            "tmin": "T2M_MIN",
            "tavg": "T2M",
            "prcp": "PRECTOT",
            "wspd": "WS2M",
            "rhum": "RH2M",
        }
    )

    df.index.name = "date"
    return df

# ---------- IO helpers ----------

def save_text(text, path):
    with open(path, "w", encoding="utf8") as f:
        f.write(text)


def save_csv(df, path):
    df.to_csv(path)


# ---------- Main driver ----------

def fetch_all(days_back=3650):
    end = date.today()
    start = end - timedelta(days=days_back)

    for p in POINTS:
        print(f"[NASA] Fetching: {p['id']} ({p['lat']},{p['lon']}) {start} -> {end}")
        try:
            txt = fetch_nasa_power_point(p, start, end, temporal="DAILY")
            fname = f"{p['id']}_nasa_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
            out = os.path.join(NASA_DIR, fname)
            save_text(txt, out)
            print("Saved", out)
        except Exception as e:
            print("NASA fetch failed for", p["id"], e)

        print(f"[Meteostat] Fetching: {p['id']} {start} -> {end}")
        try:
            df = fetch_meteostat(p, start, end)
            fname = f"{p['id']}_meteostat_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
            out = os.path.join(RAW_DIR, fname)
            save_csv(df, out)
            print("Saved", out)
        except Exception as e:
            print("Meteostat fetch failed for", p["id"], e)

        time.sleep(1.0)


if __name__ == "__main__":
    fetch_all(days_back=3650)
