# src/data_fetch.py
import os
import time
import json
import requests
from datetime import date, timedelta

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

def fetch_nasa_power_point(point, start: date, end: date, temporal="DAILY"):
    base = f"https://power.larc.nasa.gov/api/temporal/{temporal.lower()}/point"
    params = {
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "latitude": point["lat"],
        "longitude": point["lon"],
        "community": "ag",
        "parameters": VARS,
        "format": "CSV"
    }
    url = f"{base}?latitude={params['latitude']}&longitude={params['longitude']}&start={params['start']}&end={params['end']}&community=ag&parameters={params['parameters']}&format=CSV"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text

def fetch_open_meteo(point, start: date, end: date):
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": point["lat"],
        "longitude": point["lon"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,temperature_2m_mean,relativehumidity_2m_mean,windspeed_10m_max",
        "timezone": "UTC"
    }
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def save_text(text, path):
    with open(path, "w", encoding="utf8") as f:
        f.write(text)

def save_json(obj, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2)

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

        print(f"[OpenMeteo] Fetching: {p['id']} {start} -> {end}")
        try:
            j = fetch_open_meteo(p, start, end)
            fname = f"{p['id']}_openmeteo_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json"
            out = os.path.join(RAW_DIR, fname)
            save_json(j, out)
            print("Saved", out)
        except Exception as e:
            print("OpenMeteo fetch failed for", p["id"], e)

        time.sleep(1.0)

if __name__ == "__main__":
    fetch_all(days_back=3650)
