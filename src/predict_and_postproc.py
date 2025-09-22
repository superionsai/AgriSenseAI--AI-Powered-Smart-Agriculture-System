# src/predict_and_postproc.py
import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.lstm_weather import Seq2SeqLSTM

def load_model(n_features, out_len, device='cpu'):
    model = Seq2SeqLSTM(n_features=n_features, hidden_size=128, out_len=out_len)
    state_path = os.path.join(ROOT, "best_model.pth")
    if not os.path.exists(state_path):
        raise FileNotFoundError("best_model.pth not found — run train.py first.")
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_and_advisory(df_proc_path, scaler_path, model, in_len=30, out_len=15, device='cpu'):
    df = pd.read_csv(df_proc_path, index_col=0, parse_dates=True)
    scaler = joblib.load(scaler_path)
    arr = df.values
    if len(arr) < in_len:
        raise ValueError("Not enough history to predict.")
    X_in = arr[-in_len:].reshape(1, in_len, arr.shape[1]).astype(np.float32)
    with torch.no_grad():
        inp = torch.tensor(X_in).to(device)
        preds_scaled = model(inp).cpu().numpy()
    n_samples, ol, nf = preds_scaled.shape
    preds_2d = preds_scaled.reshape(-1, nf)
    preds_inv = scaler.inverse_transform(preds_2d).reshape(n_samples, ol, nf)
    preds = preds_inv[0]
    proc_df = pd.read_csv(df_proc_path, index_col=0)
    cols = proc_df.columns.tolist()
    def colidx(name):
        return cols.index(name) if name in cols else None

    precip_idx = colidx('PRECTOT')
    tmax_idx = colidx('T2M_MAX')
    tmin_idx = colidx('T2M_MIN')
    rh_idx = colidx('RH2M')

    advisory = {}
    cum7 = float(preds[:7, precip_idx].sum()) if precip_idx is not None else None
    advisory['cumulative_rain_next_7_days_mm'] = cum7

    if cum7 is not None:
        if cum7 < 10.0:
            advisory['irrigation_recommendation'] = {
                "recommend": True,
                "reason": f"Low predicted cumulative rain ({cum7:.1f} mm) over next 7 days",
                "timing": "within 48 hours"
            }
        else:
            advisory['irrigation_recommendation'] = {
                "recommend": False,
                "reason": f"Sufficient predicted rain ({cum7:.1f} mm) next 7 days"
            }

    risk = False
    if tmin_idx is not None and rh_idx is not None:
        tmin_avg = float(preds[:5, tmin_idx].mean())
        rh_avg = float(preds[:5, rh_idx].mean())
        advisory['tmin_next_5_days_avg'] = tmin_avg
        advisory['rh_next_5_days_avg'] = rh_avg
        if tmin_avg > 20.0 and rh_avg > 80.0:
            risk = True
    advisory['fungal_disease_risk'] = "high" if risk else "low"

    if tmax_idx is not None:
        advisory['max_tmax_next_15_days'] = float(preds[:, tmax_idx].max())

    result = {
        "forecast_horizon_days": out_len,
        "columns": cols,
        "advisory": advisory
    }
    return result

if __name__ == "__main__":
    PROC_DIR = os.path.join(ROOT, 'data', 'processed')
    proc_files = [f for f in os.listdir(PROC_DIR) if f.endswith('_proc.csv')]
    if len(proc_files) == 0:
        print("No processed CSVs found. Run preprocess.py first.")
        raise SystemExit(1)
    proc_csv = os.path.join(PROC_DIR, proc_files[0])
    scaler_file = proc_csv.replace('_proc.csv', '_scaler.pkl')
    df = pd.read_csv(proc_csv, index_col=0)
    n_features = df.shape[1]
    OUT_LEN = 15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(n_features=n_features, out_len=OUT_LEN, device=device)
    advisory = evaluate_and_advisory(proc_csv, scaler_file, model, in_len=30, out_len=OUT_LEN, device=device)
    import json
    print(json.dumps(advisory, indent=2))
