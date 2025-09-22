cat > src/preprocess.py <<'PY'
# src/preprocess.py
import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "nasa_power")
OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

KEEP_VARS = ['PRECTOT','T2M','T2M_MAX','T2M_MIN','RH2M','WS2M']

def load_power_csv(path):
    df = pd.read_csv(path, comment='#')
    if {'YEAR','MO','DY'}.issubset(df.columns):
        df['date'] = pd.to_datetime(df[['YEAR','MO','DY']])
        df.set_index('date', inplace=True)
    elif 'DATE' in df.columns:
        df['date'] = pd.to_datetime(df['DATE'])
        df.set_index('date', inplace=True)
    else:
        try:
            df.iloc[:,0] = pd.to_datetime(df.iloc[:,0])
            df.set_index(df.columns[0], inplace=True)
            df.index.name = 'date'
        except Exception:
            raise RuntimeError("Unrecognized CSV date columns in " + path)
    df = df.sort_index()
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numcols]

def impute_and_resample(df):
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

if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if len(files) == 0:
        print("No CSVs found in", DATA_DIR, " — run data_fetch.py first.")
        raise SystemExit(1)
    for fpath in files:
        print("Processing", fpath)
        df = load_power_csv(fpath)
        keep = [c for c in KEEP_VARS if c in df.columns]
        if len(keep) == 0:
            print("No expected columns found in", fpath, " — columns:", df.columns.tolist())
            continue
        df = df[keep]
        df = impute_and_resample(df)
        base_name = os.path.splitext(os.path.basename(fpath))[0]
        out_csv = os.path.join(OUT_DIR, base_name + "_proc.csv")
        scaler_file = os.path.join(OUT_DIR, base_name + "_scaler.pkl")
        scaled_df, scaler = scale_and_save(df, out_csv, scaler_file)
        print("Saved processed:", out_csv, "and scaler:", scaler_file)
PY
