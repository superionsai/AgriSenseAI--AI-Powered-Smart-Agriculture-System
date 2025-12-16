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

PROC_DIR = os.path.join(ROOT, "data", "processed")
FUSED_CSV = os.path.join(PROC_DIR, "fused_proc.csv")

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_sequences(arr, in_len=30, out_len=15):
    n = len(arr)
    if n < in_len + out_len:
        raise ValueError(f"Not enough data: {n} rows")
    X, Y = [], []
    for i in range(in_len, n - out_len + 1):
        X.append(arr[i - in_len:i, :])
        Y.append(arr[i:i + out_len, :])
    return np.stack(X), np.stack(Y)

def train_model(train_loader, val_loader, model, epochs=25, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        tr = np.mean(tr_losses)
        vl = np.mean(val_losses)
        print(f"Epoch {epoch}/{epochs} | train={tr:.6f} | val={vl:.6f}")

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(ROOT, "best_model.pth"))
            print("✅ Saved best_model.pth")

if __name__ == "__main__":
    if not os.path.exists(FUSED_CSV):
        print("❌ fused_proc.csv not found. Run preprocess.py first.")
        raise SystemExit(1)

    df = pd.read_csv(FUSED_CSV, index_col=0, parse_dates=True)
    arr = df.values

    IN_LEN = 30
    OUT_LEN = 15
    if len(arr) < IN_LEN + OUT_LEN:
        IN_LEN = max(5, len(arr)//3)
        OUT_LEN = max(2, len(arr)//5)
        print("Adjusted IN_LEN / OUT_LEN:", IN_LEN, OUT_LEN)

    X, Y = make_sequences(arr, IN_LEN, OUT_LEN)

    split = int(0.8 * len(X))
    train_ds = TimeSeriesDataset(X[:split], Y[:split])
    val_ds = TimeSeriesDataset(X[split:], Y[split:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = Seq2SeqLSTM(
        n_features=arr.shape[1],
        hidden_size=128,
        out_len=OUT_LEN,
        num_layers=2,
        dropout=0.15,
        use_attention=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)
    train_model(train_loader, val_loader, model, epochs=25, lr=1e-3, device=device)
