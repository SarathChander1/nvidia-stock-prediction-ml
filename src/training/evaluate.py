from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.features.preprocess import add_technical_features, build_labels, load_scaler, transform_features
from src.features.dataset import build_sequences, SequenceDataset
from src.models.lstm import LSTMClassifier
from src.utils.metrics import classification_metrics, confusion
from src.utils.plotting import plot_confusion, plot_prob_calibration, plot_equity_curve


def load_run(run_dir: Path):
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    
    # Try to find the scaler file - it might be named with timestamp or "latest"
    scaler_path = Path("outputs/scalers") / f"{run_dir.name}.joblib"
    if not scaler_path.exists():
        # If run_dir is "latest", try to find the actual timestamped scaler
        if run_dir.name == "latest":
            scaler_files = list(Path("outputs/scalers").glob("*.joblib"))
            if scaler_files:
                scaler_path = scaler_files[-1]  # Use the most recent one
            else:
                raise FileNotFoundError(f"No scaler files found in outputs/scalers/")
        else:
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    scaler = load_scaler(scaler_path)
    return ckpt, scaler


def simple_backtest(dates: np.ndarray, prices: np.ndarray, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # signals: 1 to be long next day, 0 otherwise
    returns = prices[1:] / prices[:-1] - 1.0
    strat_rets = returns * signals[:-1]
    eq = (1.0 + strat_rets).cumprod()
    eq = np.concatenate([[1.0], eq])
    # buy and hold
    bh = prices / prices[0]
    return eq, bh


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained run")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV to recompute test set")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt, scaler = load_run(run_dir)

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["date"])
    else:
        # try to infer from latest NVDA csv by glob
        import glob
        files = sorted(glob.glob("data/raw/NVDA_*_1d.csv"))
        if not files:
            raise FileNotFoundError("Provide --csv; no cached NVDA raw data found")
        df = pd.read_csv(files[-1], parse_dates=["date"])

    df = add_technical_features(df)
    labels = build_labels(df).values
    df = df.dropna().reset_index(drop=True)
    labels = labels[-len(df):]

    feature_cols = tuple(c for c in df.columns if c not in ["date"])  # same as training convention
    X_scaled_df = transform_features(df, feature_cols, scaler)

    X = X_scaled_df.values.astype(np.float32)
    y = labels.astype(np.int64)

    lookback = ckpt["lookback"]
    X_seq, y_seq = build_sequences(X, y, lookback)

    # Use the last 15% as test (same split as training script)
    n = len(X_seq)
    test_start = int(n * 0.85)
    X_test, y_test = X_seq[test_start:], y_seq[test_start:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_size=ckpt["input_size"], hidden_size=ckpt["config"]["hidden_size"], num_layers=ckpt["config"]["num_layers"], dropout=ckpt["config"]["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        logits = []
        for i in range(0, len(X_test), 512):
            xb = torch.from_numpy(X_test[i:i+512]).to(device)
            l = model(xb)
            logits.append(l.cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        probs = softmax_np(logits)[:, 1]
        preds = (probs >= 0.5).astype(int)

    m = classification_metrics(y_test, preds, probs)
    print(m)

    cm = confusion(y_test, preds)
    plot_confusion(cm, run_dir / "figures" / "confusion_matrix.png")
    plot_prob_calibration(y_test, probs, run_dir / "figures" / "calibration.png")

    # backtest on the same test window
    prices = df["adj_close"].values
    dates = df["date"].values
    # Align to test window indices after sequence building
    offset = len(df) - len(X_seq)
    test_dates = dates[offset + test_start:]
    test_prices = prices[offset + test_start:]
    equity, bh = simple_backtest(test_dates, test_prices, preds)
    plot_equity_curve(test_dates, equity, bh, run_dir / "figures" / "equity.png")


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()


