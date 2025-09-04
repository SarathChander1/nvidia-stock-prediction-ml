from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.features.preprocess import add_technical_features, build_labels, fit_transform_features, save_scaler
from src.features.dataset import build_sequences
from src.models.lstm import LSTMClassifier
from src.models.baselines import train_logistic_regression, train_random_forest, train_hist_gbdt
from src.utils.metrics import classification_metrics, confusion
from src.utils.plotting import plot_confusion
from src.utils.paths import RUNS_DIR, ensure_dirs
from src.utils.seed import set_global_seed


def time_series_split(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15):
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def train_lstm(X_seq_train, y_seq_train, X_seq_val, y_seq_val, input_size: int, lookback: int, hidden_size: int, num_layers: int, dropout: float, lr: float, epochs: int, device: torch.device):
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

    model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ds_train = TensorDataset(torch.from_numpy(X_seq_train).float(), torch.from_numpy(y_seq_train).long())
    dl_train = DataLoader(ds_train, batch_size=128, shuffle=True)
    ds_val = TensorDataset(torch.from_numpy(X_seq_val).float(), torch.from_numpy(y_seq_val).long())
    dl_val = DataLoader(ds_val, batch_size=256)

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(ds_val)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser(description="Compare multiple models on the same split")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    set_global_seed(args.seed)

    run_dir = RUNS_DIR / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Load and engineer features
    df = pd.read_csv(args.csv, parse_dates=["date"]) 
    df = add_technical_features(df)
    labels = build_labels(df).values
    df = df.dropna().reset_index(drop=True)
    labels = labels[-len(df):]

    feature_cols = tuple(c for c in df.columns if c not in ["date"]) 
    X_scaled_df, scaler, _ = fit_transform_features(df, feature_cols)
    save_scaler(scaler, name=run_dir.name)

    X_flat = X_scaled_df.values.astype(np.float32)
    y = labels.astype(np.int64)

    # Build sequences for LSTM
    X_seq, y_seq = build_sequences(X_flat, y, args.lookback)

    n_seq = len(X_seq)
    train_idx, val_idx, test_idx = time_series_split(n_seq)

    def subset(arr, idx):
        return arr[idx]

    X_seq_train, y_seq_train = subset(X_seq, train_idx), subset(y_seq, train_idx)
    X_seq_val, y_seq_val = subset(X_seq, val_idx), subset(y_seq, val_idx)
    X_seq_test, y_seq_test = subset(X_seq, test_idx), subset(y_seq, test_idx)

    # For tabular models, align to the same target indices using the last timestep of each sequence
    # Use the last timestep features as tabular input
    X_tab = X_seq[:, -1, :]
    X_tab_train, y_tab_train = subset(X_tab, train_idx), subset(y_seq, train_idx)
    X_tab_val, y_tab_val = subset(X_tab, val_idx), subset(y_seq, val_idx)
    X_tab_test, y_tab_test = subset(X_tab, test_idx), subset(y_seq, test_idx)

    results = {}

    # 1) Logistic Regression
    lr_model = train_logistic_regression(np.concatenate([X_tab_train, X_tab_val]), np.concatenate([y_tab_train, y_tab_val]))
    lr_probs = lr_model.predict_proba(X_tab_test)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)
    results["logreg"] = classification_metrics(y_tab_test, lr_preds, lr_probs)
    plot_confusion(confusion(y_tab_test, lr_preds), run_dir / "figures" / "cm_logreg.png")

    # 2) Random Forest
    rf_model = train_random_forest(np.concatenate([X_tab_train, X_tab_val]), np.concatenate([y_tab_train, y_tab_val]))
    rf_probs = rf_model.predict_proba(X_tab_test)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)
    results["random_forest"] = classification_metrics(y_tab_test, rf_preds, rf_probs)
    plot_confusion(confusion(y_tab_test, rf_preds), run_dir / "figures" / "cm_rf.png")

    # 3) HistGradientBoosting
    hgb_model = train_hist_gbdt(np.concatenate([X_tab_train, X_tab_val]), np.concatenate([y_tab_train, y_tab_val]))
    hgb_probs = hgb_model.predict_proba(X_tab_test)[:, 1]
    hgb_preds = (hgb_probs >= 0.5).astype(int)
    results["hist_gbdt"] = classification_metrics(y_tab_test, hgb_preds, hgb_probs)
    plot_confusion(confusion(y_tab_test, hgb_preds), run_dir / "figures" / "cm_hgb.png")

    # 4) LSTM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm = train_lstm(X_seq_train, y_seq_train, X_seq_val, y_seq_val, input_size=X_flat.shape[1], lookback=args.lookback,
                      hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, lr=args.lr, epochs=args.epochs, device=device)
    with torch.no_grad():
        logits = []
        for i in range(0, len(X_seq_test), 512):
            xb = torch.from_numpy(X_seq_test[i:i+512]).to(device)
            l = lstm(xb)
            logits.append(l.cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        probs = softmax_np(logits)[:, 1]
        preds = (probs >= 0.5).astype(int)
    results["lstm"] = classification_metrics(y_seq_test, preds, probs)
    plot_confusion(confusion(y_seq_test, preds), run_dir / "figures" / "cm_lstm.png")

    # Save results
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also save a simple markdown summary
    with open(run_dir / "RESULTS.md", "w") as f:
        f.write("# Model Comparison Results\n\n")
        for name, m in results.items():
            f.write(f"## {name}\n")
            for k, v in m.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")

    print("Results saved to:", run_dir)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()



