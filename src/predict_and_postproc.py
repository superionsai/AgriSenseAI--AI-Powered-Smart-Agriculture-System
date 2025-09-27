"""
predict_and_postproc.py

Loads the trained Seq2Seq LSTM (best_model.pth) and matching scaler,
produces a multi-day forecast (all features), derives rule-based advisories,
and calls OpenAI to generate a structured JSON analysis (summary, risks,
recommendations, actions). Ensures outputs are JSON-serializable.

How to run (example):
    export OPENAI_API_KEY="sk-..."
    python3 src/predict_and_postproc.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
from datetime import timedelta
from openai import OpenAI

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.lstm_weather import Seq2SeqLSTM

# -------------------- Configuration --------------------
DEFAULT_IN_LEN = 30
DEFAULT_OUT_LEN = 15
LLM_MODEL_NAME = "gpt-4o-mini"  # change if you prefer another model


# -------------------- Utilities --------------------
def to_python_number(x):
    """Convert numpy numeric types to Python floats/ints for JSON safety."""
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    return x


def safe_json_dumps(obj):
    """Dump to JSON while converting numpy / pandas types to Python primitives."""
    def convert(o):
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        if isinstance(o, (np.generic,)):
            return to_python_number(o.item())
        if isinstance(o, (np.ndarray,)):
            return [convert(i) for i in o.tolist()]
        if isinstance(o, pd.Timestamp):
            return str(o)
        return o
    return json.dumps(convert(obj), indent=2, ensure_ascii=False)


# -------------------- Model Loader --------------------
def load_model(n_features, out_len, device="cpu"):
    model = Seq2SeqLSTM(n_features=n_features, hidden_size=128, out_len=out_len)
    state_path = os.path.join(ROOT, "best_model.pth")
    if not os.path.exists(state_path):
        raise FileNotFoundError("best_model.pth not found — run train.py first.")
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# -------------------- LLM Integration (OpenAI) --------------------
def generate_llm_analysis(result_json, model_name=LLM_MODEL_NAME):
    """
    Call OpenAI to generate a structured analysis.
    Returns a dict with keys: summary, risks, recommendations, actions.
    This function is robust: if the model doesn't return strict JSON, we will fall back
    to wrapping raw text into the schema.
    """
    client = OpenAI()

    # Clear instructions + schema
    prompt = (
        "You are an expert agricultural advisor. Given the structured forecast JSON "
        "and short system-level advisories, produce a JSON object that follows the "
        "exact schema shown below. Respond with JSON only (no extra commentary).\n\n"
        "Schema (exact):\n"
        "{\n"
        "  \"summary\": \"string - 1-3 sentence high-level overview\",\n"
        "  \"risks\": [\"string\", ...],\n"
        "  \"recommendations\": [\"string\", ...],\n"
        "  \"actions\": [\"string\", ...]\n"
        "}\n\n"
        "Guidance:\n"
        "- In `summary`, mention key trends (rain, temp, humidity) and time window.\n"
        "- In `risks`, list discrete risk items (e.g. 'fungal disease risk due to high RH').\n"
        "- In `recommendations`, give concise operational guidance (e.g. 'defer irrigation for 48h').\n"
        "- In `actions`, provide 2-5 concrete action items (imperative phrases) the farmer can follow.\n\n"
        "Now analyze this data and produce the JSON:\n\n"
        f"{json.dumps(result_json, indent=2)}\n\n"
        "Respond with JSON only and ensure it is valid JSON."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract content
        raw_out = ""
        # Safety: response format differences; try to access
        try:
            raw_out = response.choices[0].message.content
        except Exception:
            raw_out = response.choices[0].message["content"] if "message" in response.choices[0] else str(response)

        raw_out = raw_out.strip()
        if not raw_out:
            return {
                "summary": "No LLM output produced.",
                "risks": [],
                "recommendations": [],
                "actions": [],
                "raw_llm": ""
            }

        # Try extracting JSON substring
        start = raw_out.find("{")
        end = raw_out.rfind("}")
        if start != -1 and end != -1:
            candidate = raw_out[start:end + 1]
            try:
                parsed = json.loads(candidate)
                # ensure keys exist
                for k in ["summary", "risks", "recommendations", "actions"]:
                    if k not in parsed:
                        parsed[k] = [] if k != "summary" else ""
                parsed["raw_llm"] = raw_out
                return parsed
            except json.JSONDecodeError:
                # fallback wrap
                return {
                    "summary": raw_out,
                    "risks": [],
                    "recommendations": [],
                    "actions": [],
                    "raw_llm": raw_out,
                }
        else:
            # no JSON substring found — wrap whole text
            return {
                "summary": raw_out,
                "risks": [],
                "recommendations": [],
                "actions": [],
                "raw_llm": raw_out,
            }

    except Exception as e:
        return {
            "summary": f"LLM call failed: {e}",
            "risks": [],
            "recommendations": [],
            "actions": [],
            "raw_llm": ""
        }


# -------------------- Forecast + Advisories --------------------
def evaluate_and_advisory(df_proc_path, scaler_path, model, in_len=DEFAULT_IN_LEN, out_len=DEFAULT_OUT_LEN, device="cpu"):
    # Load processed scaled CSV and scaler
    df = pd.read_csv(df_proc_path, index_col=0, parse_dates=True)
    scaler = joblib.load(scaler_path)

    arr = df.values  # these are scaled values (MinMax)
    cols = df.columns.tolist()

    if len(arr) < in_len:
        raise ValueError(f"Not enough history to predict. Need {in_len} rows, got {len(arr)}")

    # Build input window (scaled)
    X_in = arr[-in_len:].reshape(1, in_len, arr.shape[1]).astype(np.float32)

    # Predict (scaled)
    with torch.no_grad():
        inp = torch.tensor(X_in).to(device)
        preds_scaled = model(inp).cpu().numpy()  # shape (1, out_len, n_features)

    # Inverse transform predictions to original units
    n_samples, ol, nf = preds_scaled.shape
    preds_2d = preds_scaled.reshape(-1, nf)
    preds_inv = scaler.inverse_transform(preds_2d).reshape(n_samples, ol, nf)
    preds = preds_inv[0]  # shape (out_len, n_features)

    # Build forecast dataframe
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(out_len)]
    forecast_df = pd.DataFrame(preds, index=future_dates, columns=cols)

    # Rule-based advisories (simple, extendable)
    advisory = {}

    # Irrigation advice: cumulative 7-day rainfall
    if "PRECTOT" in cols:
        cum7 = float(np.sum(forecast_df["PRECTOT"].iloc[:min(7, len(forecast_df))].values))
        advisory["cumulative_rain_next_7_days_mm"] = to_python_number(cum7)
        if cum7 < 10.0:
            advisory["irrigation_recommendation"] = {
                "recommend": True,
                "reason": f"Low predicted cumulative rain ({cum7:.1f} mm) over next 7 days",
                "timing": "within 48 hours"
            }
        else:
            advisory["irrigation_recommendation"] = {
                "recommend": False,
                "reason": f"Sufficient predicted rain ({cum7:.1f} mm) over next 7 days"
            }

    # Fungal disease risk based on avg Tmin and RH over next 5 days
    if "T2M_MIN" in cols and "RH2M" in cols:
        tmin_avg = float(np.mean(forecast_df["T2M_MIN"].iloc[:min(5, len(forecast_df))].values))
        rh_avg = float(np.mean(forecast_df["RH2M"].iloc[:min(5, len(forecast_df))].values))
        advisory["tmin_next_5_days_avg"] = to_python_number(tmin_avg)
        advisory["rh_next_5_days_avg"] = to_python_number(rh_avg)
        advisory["fungal_disease_risk"] = "high" if (tmin_avg > 20.0 and rh_avg > 80.0) else "low"

    # Heat stress indicator
    if "T2M_MAX" in cols:
        advisory["max_tmax_next_15_days"] = to_python_number(float(forecast_df["T2M_MAX"].max()))

    # Prepare forecast list with json-safe types
    forecast_records = []
    for row in forecast_df.reset_index().itertuples(index=False):
        # row[0] is date, others are feature values in order of cols
        rec = {"date": str(row[0])}
        for i, c in enumerate(cols):
            # pandas gives the date as first field, so shift index by 1
            val = getattr(row, c) if hasattr(row, c) else row[i+1]  # safe access
            # convert numpy types
            rec[c] = to_python_number(val)
        forecast_records.append(rec)

    result = {
        "forecast_horizon_days": out_len,
        "forecast": forecast_records,
        "advisory": advisory,
    }

    # Add LLM analysis (OpenAI)
    llm_analysis = generate_llm_analysis(result)
    result["llm_analysis"] = llm_analysis

    return result


# -------------------- Main --------------------
if __name__ == "__main__":
    PROC_DIR = os.path.join(ROOT, "data", "processed")
    proc_files = [f for f in os.listdir(PROC_DIR) if f.endswith("_proc.csv")]
    if len(proc_files) == 0:
        print("No processed CSVs found. Run preprocess.py first.")
        raise SystemExit(1)

    proc_csv = os.path.join(PROC_DIR, proc_files[0])
    scaler_file = proc_csv.replace("_proc.csv", "_scaler.pkl")

    df = pd.read_csv(proc_csv, index_col=0)
    n_features = df.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(n_features=n_features, out_len=DEFAULT_OUT_LEN, device=device)
    result = evaluate_and_advisory(proc_csv, scaler_file, model, in_len=DEFAULT_IN_LEN, out_len=DEFAULT_OUT_LEN, device=device)

    # Print JSON safely
    print(safe_json_dumps(result))
