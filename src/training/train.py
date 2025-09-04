from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.features.preprocess import add_technical_features, build_labels, fit_transform_features, save_scaler
from src.features.dataset import build_sequences, SequenceDataset
from src.models.lstm import LSTMClassifier
from src.utils.paths import ensure_dirs, RUNS_DIR
from src.utils.seed import set_global_seed


def time_series_split(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def main():
    parser = argparse.ArgumentParser(description="Train LSTM for NVDA trend prediction")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    set_global_seed(args.seed)

    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    df = pd.read_csv(args.csv, parse_dates=["date"])  # expects columns from downloader
    df = add_technical_features(df)
    labels = build_labels(df).values
    df = df.dropna().reset_index(drop=True)
    labels = labels[-len(df):]

    feature_cols = tuple(c for c in df.columns if c not in ["date"])  # keep all numeric features
    X_scaled_df, scaler, _ = fit_transform_features(df, feature_cols)
    save_scaler(scaler, name=run_dir.name)

    X = X_scaled_df.values.astype(np.float32)
    y = labels.astype(np.int64)

    X_seq, y_seq = build_sequences(X, y, args.lookback)

    n = len(X_seq)
    train_idx, val_idx, test_idx = time_series_split(n)

    def subset(arr, idx):
        return arr[idx]

    ds_train = SequenceDataset(subset(X_seq, train_idx), subset(y_seq, train_idx))
    ds_val = SequenceDataset(subset(X_seq, val_idx), subset(y_seq, val_idx))
    ds_test = SequenceDataset(subset(X_seq, test_idx), subset(y_seq, test_idx))

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_size=X.shape[1], hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_path = run_dir / "model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(ds_train)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(ds_val)

        print({"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6)})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": vars(args),
                "input_size": X.shape[1],
                "lookback": args.lookback,
            }, best_path)

    # also save a symlink/copy as latest
    latest_dir = RUNS_DIR / "latest"
    if latest_dir.exists():
        # clean up previous
        for p in latest_dir.glob("*"):
            try:
                if p.is_dir():
                    for q in p.glob("**/*"):
                        q.unlink(missing_ok=True)
                    p.rmdir()
                else:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
        latest_dir.rmdir()
    latest_dir.mkdir(parents=True, exist_ok=True)
    # copy artifacts
    import shutil
    for p in run_dir.glob("**/*"):
        rel = p.relative_to(run_dir)
        dest = latest_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if p.is_file():
            shutil.copy2(p, dest)


if __name__ == "__main__":
    main()


