python src/data_fetch.py
python src/preprocess.py
python src/train.py
python src/predict_and_postproc.pyimport os
import time
import requests
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "nasa_power")
os.makedirs(DATA_DIR, exist_ok=True)

# Replace with your own fields/lat-lon pairs
POINTS = [
    {"id":"field_1","lat":17.3850,"lon":78.4867},
    {"id":"field_2","lat":13.0827,"lon":80.2707}
]

VARS = "PRECTOT,T2M_MAX,T2M_MIN,T2M,RH2M,WS2M"

def fetch_nasa_power(point, start, end, temporal="DAILY"):
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

def save_csv(text, out_path):
    with open(out_path, "w", encoding="utf8") as f:
        f.write(text)

if __name__ == "__main__":
    end = date.today()
    start = end - timedelta(days=5*365)
    for p in POINTS:
        print("Fetching:", p["id"], p["lat"], p["lon"])
        try:
            csv_text = fetch_nasa_power(p, start, end, temporal="DAILY")
            fname = f"{p['id']}_power_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
            out_file = os.path.join(DATA_DIR, fname)
            save_csv(csv_text, out_file)
            print("Saved:", out_file)
            time.sleep(1.0)
        except Exception as e:
            print("Failed for", p["id"], "-", e)
