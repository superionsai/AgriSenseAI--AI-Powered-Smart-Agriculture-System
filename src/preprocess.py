# src/preprocess.py
import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from io import StringIO
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NASA_DIR = os.path.join(ROOT, "data", "nasa_power")
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

KEEP_VARS = ['PRECTOT', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M']

from utils_fusion import (
    detect_header_line if False else None  # placeholder for linter; we implement loaders below
)

# ---------- loaders ----------
def detect_header_line(lines):
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("YEAR") and "MO" in s:
            return i
        if s.startswith("DATE") and "," in s:
            return i
    for i, line in enumerate(lines):
        if "," in line and any(c.isalpha() for c in line):
            return i
    return None

def read_csv_from_header(path, header_line):
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        data = "".join(f.readlines()[header_line:])
    try:
        df = pd.read_csv(StringIO(data), sep=",", engine="python", skip_blank_lines=True)
        return df
    except Exception:
        reader = csv.reader(data.splitlines())
        rows = [r for r in reader if any(field.strip() for field in r)]
        head = [h.strip() for h in rows[0]]
        df = pd.DataFrame(rows[1:], columns=head)
        return df

def parse_dates_and_index(df):
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    if {'YEAR', 'MO'}.issubset(df.columns) and ('DY' in df.columns or 'DAY' in df.columns):
        day = 'DY' if 'DY' in df.columns else 'DAY'
        y = df['YEAR'].astype(str).str.zfill(4)
        m = df['MO'].astype(str).str.zfill(2)
        d = df[day].astype(str).str.zfill(2)
        df['date'] = pd.to_datetime(y + "-" + m + "-" + d, errors='coerce')
        df = df.set_index('date')
        df.index.name = 'date'
        return df
    if 'DATE' in df.columns:
        df['date'] = pd.to_datetime(df['DATE'], errors='coerce', infer_datetime_format=True)
        df = df.set_index('date')
        df.index.name = 'date'
        return df
    # fallback
    first = df.columns[0]
    df['date'] = pd.to_datetime(df[first].astype(str), errors='coerce', infer_datetime_format=True)
    df = df.set_index('date')
    df.index.name = 'date'
    return df

def load_power_csv(path):
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        lines = f.readlines()
    header_line = detect_header_line(lines)
    if header_line is None:
        raise RuntimeError("Header not found in " + path)
    df_raw = read_csv_from_header(path, header_line)
    dfi = parse_dates_and_index(df_raw)
    dfi = dfi[~dfi.index.isna()]
    for c in dfi.columns:
        dfi[c] = pd.to_numeric(dfi[c], errors='coerce')
    dfi = dfi.dropna(how='all')
    dfi = dfi.sort_index()
    return dfi

def parse_openmeteo_json(path):
    import json
    with open(path, 'r', encoding='utf8') as f:
        j = json.load(f)
    daily = j.get('daily', {})
    dates = daily.get('time', [])
    df = pd.DataFrame({'date': dates})
    mapping = {
        'precipitation_sum': 'PRECTOT',
        'temperature_2m_max': 'T2M_MAX',
        'temperature_2m_min': 'T2M_MIN',
        'temperature_2m_mean': 'T2M',
        'relativehumidity_2m_mean': 'RH2M',
        'windspeed_10m_max': 'WS2M'
    }
    for k, v in mapping.items():
        if k in daily:
            df[v] = daily[k]
    df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
    df = df.set_index('date')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# ---------- feature engineering ----------
def add_agro_features(df):
    for col in KEEP_VARS:
        if col not in df.columns:
            df[col] = np.nan
    df['precip_7d_sum'] = df['PRECTOT'].rolling(7, min_periods=1).sum()
    df['tmax_7d_mean'] = df['T2M_MAX'].rolling(7, min_periods=1).mean()
    df['tmin_7d_mean'] = df['T2M_MIN'].rolling(7, min_periods=1).mean()
    df['rh_7d_mean'] = df['RH2M'].rolling(7, min_periods=1).mean()
    df['t_range'] = df['T2M_MAX'] - df['T2M_MIN']
    Ra_approx = 20.0
    df['T_MEAN'] = df[['T2M', 'T2M_MAX', 'T2M_MIN']].mean(axis=1)
    df['hargreaves_et0'] = 0.0023 * (df['T_MEAN'] + 17.8) * np.sqrt(np.clip(df['t_range'].fillna(0), 0.0, None)) * Ra_approx
    df = df.resample('D').mean()
    df = df.interpolate(method='time').ffill().bfill()
    return df

def scale_and_save(df, out_csv_path, scaler_path):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    scaled_df.to_csv(out_csv_path)
    joblib.dump(scaler, scaler_path)
    return scaled_df, scaler

# ---------- main pipeline ----------
from utils_fusion import (
    align_sources,
    compute_api_errors,
    compute_weights_from_errors,
    smooth_weights,
    weighted_fusion,
    save_weights
)

WEIGHTS_FILE = os.path.join(OUT_DIR, "weights.json")
SMOOTH_ALPHA = 0.25
ERROR_WINDOW_DAYS = 90  # compute error over last N days

def load_raw_sources():
    # find most recent NASA CSV and OpenMeteo JSON per point
    nasa_files = sorted(glob.glob(os.path.join(NASA_DIR, "*.csv")))
    open_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    sources = {}
    # load all NASA files, name them 'nasa:<basename>'
    for p in nasa_files:
        name = Path(p).stem  # e.g. field_1_nasa_YYYY...
        try:
            df = load_power_csv(p)
            sources[name] = df
        except Exception as e:
            print("Failed loading NASA file", p, "->", e)
    # load openmeteo files
    for p in open_files:
        name = Path(p).stem
        try:
            df = parse_openmeteo_json(p)
            sources[name] = df
        except Exception as e:
            print("Failed loading OpenMeteo file", p, "->", e)
    return sources

def pick_named_groups(sources):
    """
    Group by logical API name: e.g. 'nasa' and 'openmeteo' per field.
    We expect file stems like field_1_nasa_YYYY... and field_1_openmeteo_...
    Returns dict of api_name -> df (concatenated across fields by union index)
    """
    apis = {}
    for key, df in sources.items():
        # key = field_1_nasa_...
        parts = key.split("_")
        # find 'nasa' or 'openmeteo' in the stem
        api = 'unknown'
        if 'nasa' in parts:
            api = 'nasa'
        elif 'openmeteo' in parts:
            api = 'openmeteo'
        else:
            # fallback: try contains string
            if 'nasa' in key:
                api = 'nasa'
            elif 'openmeteo' in key:
                api = 'openmeteo'
        # concat by union index (outer) per api
        if api not in apis:
            apis[api] = df.copy()
        else:
            # align and union (outer)
            apis[api] = pd.concat([apis[api], df]).sort_index().groupby(level=0).mean()
    return apis

def compute_weights_and_fuse(apis, variables=KEEP_VARS, reference='nasa'):
    # align
    aligned = align_sources(apis)
    # compute errors vs reference over recent window
    # restrict to recent window length
    last_date = None
    for df in aligned.values():
        if last_date is None or (len(df) > 0 and df.index.max() > last_date):
            last_date = df.index.max()
    if last_date is None:
        last_date = pd.Timestamp.today()
    window_start = last_date - pd.Timedelta(days=ERROR_WINDOW_DAYS)
    aligned_window = {k: v.loc[(v.index >= window_start) & (v.index <= last_date)] for k, v in aligned.items()}
    errors = compute_api_errors(aligned_window, reference=reference, variables=variables)
    new_weights = compute_weights_from_errors(errors)
    # smooth previous weights
    old_weights = {}
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r", encoding="utf8") as f:
                old_weights = json.load(f)
        except Exception:
            old_weights = {}
    smoothed = smooth_weights(old_weights, new_weights, alpha=SMOOTH_ALPHA)
    # save weights
    save_weights(smoothed, WEIGHTS_FILE)
    # fuse using smoothed weights
    fused = weighted_fusion(aligned, smoothed, variables=variables)
    return fused, smoothed, errors

if __name__ == "__main__":
    sources = load_raw_sources()
    if not sources:
        print("No raw sources found. Run data_fetch.py first.")
        raise SystemExit(1)
    apis = pick_named_groups(sources)
    print("APIs detected:", list(apis.keys()))
    fused, weights, errors = compute_weights_and_fuse(apis, variables=KEEP_VARS, reference='nasa')
    print("Computed weights:", weights)
    # apply feature engineering
    fused_features = add_agro_features(fused)
    out_csv = os.path.join(OUT_DIR, "fused_proc.csv")
    scaler_file = os.path.join(OUT_DIR, "fused_scaler.pkl")
    scaled_df, scaler = scale_and_save(fused_features, out_csv, scaler_file)
    print("Saved fused processed CSV:", out_csv)
    print("Saved scaler:", scaler_file)
    # also save weights & errors as json
    with open(os.path.join(OUT_DIR, "fused_errors.json"), "w", encoding="utf8") as f:
        json.dump({"errors": errors, "weights": weights}, f, indent=2)
    print("Saved fused_errors.json")
