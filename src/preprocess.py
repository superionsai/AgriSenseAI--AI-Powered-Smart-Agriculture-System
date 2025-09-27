# src/preprocess.py (robust NASA POWER CSV loader + preprocessing)
import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from io import StringIO
import csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "nasa_power")
OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

KEEP_VARS = ['PRECTOT', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M']


def detect_header_line(lines):
    """Return index of header line (0-based) or None."""
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # common header patterns
        if line_stripped.startswith("YEAR") and "MO" in line_stripped:
            return i
        if line_stripped.startswith("DATE") and "," in line_stripped:
            return i
    # fallback: first line that looks like CSV header (has letters and commas)
    for i, line in enumerate(lines):
        if "," in line and any(c.isalpha() for c in line):
            return i
    return None


def read_csv_from_header(path, header_line):
    """Read CSV starting from header_line using pandas with a safe engine."""
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        data_str = "".join(f.readlines()[header_line:])
    # Try pandas read_csv using python engine (more tolerant)
    try:
        df = pd.read_csv(StringIO(data_str), sep=",", engine="python", skip_blank_lines=True)
        return df
    except Exception as e:
        # Fallback: use csv.reader then construct DataFrame
        reader = csv.reader(data_str.splitlines())
        rows = [r for r in reader if any(field.strip() != "" for field in r)]
        if len(rows) < 1:
            raise RuntimeError(f"No data rows found after header in {path}")
        header = [h.strip() for h in rows[0]]
        data = rows[1:]
        df = pd.DataFrame(data, columns=header)
        return df


def parse_dates_and_index(df):
    """Handle DATE or YEAR/MO/DY style headers and return df indexed by datetime."""
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # YEAR / MO / DY (or DAY)
    if {'YEAR', 'MO'}.issubset(df.columns) and ('DY' in df.columns or 'DAY' in df.columns):
        day_col = 'DY' if 'DY' in df.columns else 'DAY'
        # coerce to numeric strings then to datetime
        try:
            y = df['YEAR'].astype(str).str.zfill(4)
            m = df['MO'].astype(str).str.zfill(2)
            d = df[day_col].astype(str).str.zfill(2)
            df['date'] = pd.to_datetime(y + "-" + m + "-" + d, errors='coerce')
        except Exception:
            df['date'] = pd.to_datetime(df[['YEAR', 'MO', day_col]].apply(lambda row: f"{int(float(row[0])):04d}-{int(float(row[1])):02d}-{int(float(row[2])):02d}", axis=1), errors='coerce')

        df = df.set_index('date')
        df.index.name = 'date'
        return df

    # Single DATE column
    if 'DATE' in df.columns:
        df['date'] = pd.to_datetime(df['DATE'], errors='coerce', infer_datetime_format=True)
        df = df.set_index('date')
        df.index.name = 'date'
        return df

    # Fallback: try parse first column as date
    first_col = df.columns[0]
    try:
        df[first_col] = df[first_col].astype(str)
        df['date'] = pd.to_datetime(df[first_col], errors='coerce', infer_datetime_format=True)
        df = df.set_index('date')
        df.index.name = 'date'
        return df
    except Exception as e:
        raise RuntimeError("Could not parse date columns")


def load_power_csv(path):
    """Load NASA POWER CSV robustly and return dataframe with numeric columns and datetime index."""
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        lines = f.readlines()

    header_line = detect_header_line(lines)
    if header_line is None:
        raise RuntimeError(f"Could not find header row in {path}")

    # Read df from header line
    df_raw = read_csv_from_header(path, header_line)

    # Debug info
    print(f"DEBUG: {os.path.basename(path)} header_line={header_line} raw_rows={len(df_raw)} raw_columns={list(df_raw.columns)[:10]}")

    # Parse dates and index
    df_indexed = parse_dates_and_index(df_raw)

    # drop rows where index is NaT
    df_indexed = df_indexed[~df_indexed.index.isna()]

    # convert numeric-like columns to numeric (coerce errors)
    for c in df_indexed.columns:
        df_indexed[c] = pd.to_numeric(df_indexed[c], errors='coerce')

    # drop rows where all values are NaN
    df_indexed = df_indexed.dropna(how='all')

    # sort & return only numeric columns
    df_indexed = df_indexed.sort_index()
    numcols = df_indexed.select_dtypes(include=[np.number]).columns.tolist()
    df_num = df_indexed[numcols].copy()
    print(f"DEBUG: after parsing {os.path.basename(path)} -> rows={len(df_num)} numeric_cols={numcols}")
    return df_num


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
        try:
            df = load_power_csv(fpath)
        except Exception as e:
            print(f"ERROR loading {fpath}: {e}")
            # print a few lines of the raw file to help debug
            with open(fpath, "r", encoding="utf8", errors="ignore") as fh:
                print("--- top 30 lines of file ---")
                for i, line in enumerate(fh):
                    if i >= 30:
                        break
                    print(i + 1, line.rstrip())
            continue

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
        print("✅ Saved processed:", out_csv, "and scaler:", scaler_file, " (rows:", len(scaled_df), ")")
