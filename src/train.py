# src/train.py
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.lstm_weather import Seq2SeqLSTM


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_sequences(df, in_len=30, out_len=15):
    arr = df.values
    n = len(arr)
    if n < in_len + out_len:
        raise ValueError(
            f"Not enough data: have {n} rows, need at least {in_len + out_len}"
        )
    X, Y = [], []
    for i in range(in_len, n - out_len + 1):
        X.append(arr[i - in_len : i, :])
        Y.append(arr[i : i + out_len, :])
    return np.stack(X), np.stack(Y)


def train_model(train_loader, val_loader, model, epochs=40, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    best_val = 1e9

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        avg_train = np.mean(train_losses) if train_losses else float("nan")
        avg_val = np.mean(val_losses) if val_losses else float("nan")
        print(
            f"Epoch {epoch}/{epochs}  train_loss={avg_train:.6f}  val_loss={avg_val:.6f}"
        )

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(ROOT, "best_model.pth"))
            print("✅ Saved best_model.pth (val_loss improved)")
    return model


if __name__ == "__main__":
    PROC_DIR = os.path.join(ROOT, "data", "processed")
    proc_files = [f for f in os.listdir(PROC_DIR) if f.endswith("_proc.csv")]
    if len(proc_files) == 0:
        print("No processed CSVs found in", PROC_DIR, " — run preprocess.py first.")
        raise SystemExit(1)

    # Use the first processed CSV and its scaler
    csv_path = os.path.join(PROC_DIR, proc_files[0])
    scaler_path = csv_path.replace("_proc.csv", "_scaler.pkl")

    print(f"Loading scaled data from {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    print(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Default sequence lengths
    IN_LEN = 30
    OUT_LEN = 15

    if len(df) < IN_LEN + OUT_LEN:
        print(
            f"⚠️ Dataset too short ({len(df)} rows). Adjusting IN_LEN and OUT_LEN..."
        )
        IN_LEN = max(5, len(df) // 3)
        OUT_LEN = max(2, len(df) // 5)
        print(f"Using IN_LEN={IN_LEN}, OUT_LEN={OUT_LEN}")

    try:
        X, Y = make_sequences(df, in_len=IN_LEN, out_len=OUT_LEN)
    except ValueError as e:
        print("❌", e)
        raise SystemExit(1)

    # Train/val split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]

    train_ds = TimeSeriesDataset(X_train, Y_train)
    val_ds = TimeSeriesDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    n_features = X.shape[2]
    model = Seq2SeqLSTM(
        n_features=n_features, hidden_size=128, out_len=OUT_LEN, num_layers=2, dropout=0.1
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on device:", device)
    _ = train_model(train_loader, val_loader, model, epochs=10, lr=1e-3, device=device)
