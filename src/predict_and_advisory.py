# src/predict_and_advisory.py
import os
import json
import numpy as np
import pandas as pd
import joblib
import torch
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR = os.path.join(ROOT, "data", "processed")
MODEL_PATH = os.path.join(ROOT, "best_model.pth")
from models.lstm_weather import Seq2SeqLSTM
from llm_interface import generate_advisory_text

FUSED_CSV = os.path.join(PROC_DIR, "fused_proc.csv")
FUSED_SCALER = os.path.join(PROC_DIR, "fused_scaler.pkl")
WEIGHTS_JSON = os.path.join(PROC_DIR, "weights.json")

def load_model(n_features, out_len, device='cpu'):
    model = Seq2SeqLSTM(n_features=n_features, hidden_size=128, out_len=out_len, num_layers=2, dropout=0.15, use_attention=False)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model checkpoint not found: " + MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def mc_dropout_predictions(model, x_in, n_samples=50, device='cpu'):
    model.train()  # enable dropout at inference
    X = torch.tensor(x_in.astype(np.float32)).to(device)
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(X).cpu().numpy()
            samples.append(out)
    model.eval()
    samples = np.concatenate(samples, axis=0)  # (n_samples, out_len, n_features)
    return samples

def predict_intervals(proc_csv_path, scaler_path, in_len=30, out_len=15, device='cpu'):
    df = pd.read_csv(proc_csv_path, index_col=0, parse_dates=True)
    scaler = joblib.load(scaler_path)
    arr = df.values
    if len(arr) < in_len:
        raise ValueError("Not enough history to predict.")
    X_in = arr[-in_len:].reshape(1, in_len, arr.shape[1]).astype(np.float32)
    model = load_model(n_features=arr.shape[1], out_len=out_len, device=device)
    samples = mc_dropout_predictions(model, X_in, n_samples=50, device=device)
    ns, ol, nf = samples.shape
    samples_2d = samples.reshape(ns*ol, nf)
    inv = scaler.inverse_transform(samples_2d).reshape(ns, ol, nf)
    p10 = np.percentile(inv, 10, axis=0)
    p50 = np.percentile(inv, 50, axis=0)
    p90 = np.percentile(inv, 90, axis=0)
    cols = df.columns.tolist()
    dates = [ (pd.to_datetime(df.index[-1]) + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(out_len) ]
    return {"dates": dates, "columns": cols, "p10": p10.tolist(), "p50": p50.tolist(), "p90": p90.tolist()}

def compute_advisory_from_intervals(intervals):
    cols = intervals['columns']
    try:
        idx_prec = cols.index('PRECTOT')
    except ValueError:
        idx_prec = None
    p50 = np.array(intervals['p50'])
    cum7 = float(p50[:7, idx_prec].sum()) if idx_prec is not None else None
    advisory = {"cumulative_rain_next_7_days_mm": cum7}
    if cum7 is not None:
        if cum7 < 10.0:
            advisory['irrigation'] = {"recommend": True, "reason": f"Low predicted median cumulative rain {cum7:.1f} mm over next 7 days", "when": "within 48 hours"}
        else:
            advisory['irrigation'] = {"recommend": False, "reason": f"Median predicted rain {cum7:.1f} mm over next 7 days"}
    advisory['fungal_risk'] = "low"
    return advisory

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(FUSED_CSV) or not os.path.exists(FUSED_SCALER):
        print("Fused processed CSV or scaler not found. Run preprocess.py first.")
        raise SystemExit(1)
    intervals = predict_intervals(FUSED_CSV, FUSED_SCALER, in_len=30, out_len=15, device=device)
    advisory = compute_advisory_from_intervals(intervals)
    final = {"forecast": intervals, "advisory": advisory}
    from rules_engine import generate_recommendations

    recs, acts = generate_recommendations(final["advisory"])

    final["recommendations"] = recs
    final["actions"] = acts

    llm_out = generate_advisory_text(final)
    final['llm'] = llm_out
    # attach weights info for transparency
    try:
        with open(WEIGHTS_JSON, "r", encoding="utf8") as f:
            final["fusion_weights"] = json.load(f)
    except Exception:
        final["fusion_weights"] = None
    print(json.dumps(final, indent=2))
    
    from llm_interface import generate_advisory_text, translate_text

    language = "te"  # user-selected
    clean_advisory = llm_out
    english = generate_advisory_text(clean_advisory)

    if language != "en":
        final_text = translate_text(english, target_lang=language)
    else:
        final_text = english

    print("\nFINAL ADVISORY:\n")
    print(final_text)

