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

# ---------- CSV loaders for NASA POWER ----------
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
        df['date'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.set_index('date')
        df.index.name = 'date'
        return df
    # fallback
    first = df.columns[0]
    df['date'] = pd.to_datetime(df[first].astype(str), errors='coerce')
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

# ---------- load raw CSV/JSON (meteostat CSVs or other CSVs) ----------
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
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.set_index('date')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# ---------- feature engineering ----------
def add_agro_features(df):
    if df is None or len(df) == 0:
        raise RuntimeError("No data passed to add_agro_features")

    # 🔧 FIX: enforce DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()]
    df = df.sort_index()

    for col in KEEP_VARS:
        if col not in df.columns:
            df[col] = np.nan

    df['precip_7d_sum'] = df['PRECTOT'].rolling(7, min_periods=1).sum()
    df['tmax_7d_mean'] = df['T2M_MAX'].rolling(7, min_periods=1).mean()
    df['tmin_7d_mean'] = df['T2M_MIN'].rolling(7, min_periods=1).mean()
    df['rh_7d_mean'] = df['RH2M'].rolling(7, min_periods=1).mean()
    df['t_range'] = df['T2M_MAX'] - df['T2M_MIN']
    df['T_MEAN'] = df[['T2M', 'T2M_MAX', 'T2M_MIN']].mean(axis=1)

    # Hargreaves ET₀ (approx)
    Ra = 20.0
    df['hargreaves_et0'] = (
        0.0023
        * (df['T_MEAN'] + 17.8)
        * np.sqrt(np.clip(df['t_range'], 0, None))
        * Ra
    )

    # 🔧 NOW resampling is safe
    df = df.resample('D').mean()
    df = df.interpolate(method='time').ffill().bfill()

    return df
def scale_and_save(df, out_csv_path, scaler_path):
    """
    Scaling with NO fabrication:
    - Columns with any remaining NaNs are dropped
    - Only fully-supported variables go to the model
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    import json
    import os

    df_clean = df.copy()

    # Drop columns that still contain NaNs
    nan_cols = df_clean.columns[df_clean.isna().any()].tolist()
    if nan_cols:
        print("Dropping columns with unresolved NaNs:", nan_cols)
        df_clean = df_clean.drop(columns=nan_cols)

    # Final safety
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df_clean.isna().any().any():
        raise RuntimeError("NaNs remain after fusion — aborting scaling.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_clean.values)

    scaled_df = pd.DataFrame(
        scaled, index=df_clean.index, columns=df_clean.columns
    )

    scaled_df.to_csv(out_csv_path)
    joblib.dump(scaler, scaler_path)

    # Save metadata
    meta = {
        "dropped_nan_columns": nan_cols,
        "final_columns": df_clean.columns.tolist(),
        "n_rows": int(len(df_clean)),
        "n_cols": int(len(df_clean.columns))
    }

    meta_path = os.path.join(os.path.dirname(out_csv_path), "fused_proc_meta.json")
    with open(meta_path, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)

    return scaled_df, scaler

# ---------- fusion helpers (imported) ----------
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

# ---------- load raw sources robustly ----------
def load_raw_sources():
    # NASA CSVs
    nasa_files = sorted(glob.glob(os.path.join(NASA_DIR, "*.csv")))
    # raw dir may contain meteostat CSVs or other csv/json
    raw_csvs = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    raw_jsons = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    sources = {}
    for p in nasa_files:
        name = Path(p).stem
        try:
            df = load_power_csv(p)
            sources[name] = df
        except Exception as e:
            print("Failed loading NASA file", p, "->", e)
    for p in raw_csvs + raw_jsons:
        name = Path(p).stem
        try:
            if p.lower().endswith(".json"):
                df = parse_openmeteo_json(p)
            else:
                # CSV from Meteostat or other tools - load with pandas and ensure datetime index
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                # coerce numeric
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            sources[name] = df
        except Exception as e:
            print("Failed loading raw file", p, "->", e)
    return sources

def pick_named_groups(sources):
    """
    Map file stems into API groups: 'nasa', 'meteostat', 'openmeteo', or 'other'.
    Concatenate multiple files per API by taking index-wise mean (outer union).
    """
    apis = {}
    for key, df in sources.items():
        parts = key.lower().split("_")
        if 'nasa' in parts or 'nasa' in key.lower():
            api = 'nasa'
        elif 'meteostat' in parts or 'meteostat' in key.lower():
            api = 'meteostat'
        elif 'openmeteo' in parts or 'openmeteo' in key.lower():
            api = 'openmeteo'
        else:
            api = 'other'
        if api not in apis:
            apis[api] = df.copy()
        else:
            # union by index, take mean where overlapping
            apis[api] = pd.concat([apis[api], df]).sort_index().groupby(level=0).mean()
    return apis

def compute_weights_and_fuse(apis, variables=KEEP_VARS, reference='nasa'):
    aligned = align_sources(apis)
    # if only one api present -> return it as fused with weight 1.0
    if len(aligned) == 0:
        raise RuntimeError("No API data available to fuse.")
    if len(aligned) == 1:
        only_name = list(aligned.keys())[0]
        fused = aligned[only_name].copy()
        weights = {only_name: 1.0}
        # ensure we only return requested variables (add missing as NaN)
        for v in variables:
            if v not in fused.columns:
                fused[v] = np.nan
        fused = fused[variables].sort_index()
        save_weights(weights, WEIGHTS_FILE)
        return fused, weights, {}
    # compute recent-window errors
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
    old_weights = {}
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r", encoding="utf8") as f:
                old_weights = json.load(f)
        except Exception:
            old_weights = {}
    smoothed = smooth_weights(old_weights, new_weights, alpha=SMOOTH_ALPHA)
    save_weights(smoothed, WEIGHTS_FILE)
    fused = weighted_fusion(aligned, smoothed, variables=variables)
    return fused, smoothed, errors

# ---------- main ----------
if __name__ == "__main__":
    sources = load_raw_sources()
    if not sources:
        print("No raw sources found. Run data_fetch.py first.")
        raise SystemExit(1)
    apis = pick_named_groups(sources)
    print("APIs detected:", list(apis.keys()))
    fused, weights, errors = compute_weights_and_fuse(apis, variables=KEEP_VARS, reference='nasa')
    print("Computed weights:", weights)
    fused_features = add_agro_features(fused)
    out_csv = os.path.join(OUT_DIR, "fused_proc.csv")
    scaler_file = os.path.join(OUT_DIR, "fused_scaler.pkl")
    scaled_df, scaler = scale_and_save(fused_features, out_csv, scaler_file)
    print("Saved fused processed CSV:", out_csv)
    print("Saved scaler:", scaler_file)
    with open(os.path.join(OUT_DIR, "fused_errors.json"), "w", encoding="utf8") as f:
        json.dump({"errors": errors, "weights": weights}, f, indent=2)
    print("Saved fused_errors.json")
