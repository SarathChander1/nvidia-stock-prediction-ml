from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils.paths import SCALERS_DIR


FEATURE_COLUMNS = [
    "open", "high", "low", "close", "adj_close", "volume",
    # engineered features appended later
]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    # Ensure numeric columns are properly typed
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    
    # Remove rows with NaN values in essential columns
    out = out.dropna(subset=["adj_close", "volume"])
    
    out["return_1d"] = out["adj_close"].pct_change()
    out["return_5d"] = out["adj_close"].pct_change(5)
    out["return_20d"] = out["adj_close"].pct_change(20)
    out["volatility_20d"] = out["return_1d"].rolling(20).std().fillna(0.0)
    out["rsi_14"] = compute_rsi(out["adj_close"], 14)
    out["sma_20"] = out["adj_close"].rolling(20).mean()
    out["sma_50"] = out["adj_close"].rolling(50).mean()
    out["sma_ratio_20_50"] = out["sma_20"] / out["sma_50"]
    out["volume_zscore_20"] = zscore_rolling(out["volume"], 20)
    out = out.drop(columns=["sma_20", "sma_50"])  # keep compact
    return out


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def zscore_rolling(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def build_labels(df: pd.DataFrame) -> pd.Series:
    # next-day up (1) if adj_close_{t+1} > adj_close_{t}, else 0
    future = df["adj_close"].shift(-1)
    label = (future > df["adj_close"]).astype(int)
    return label


@dataclass
class FittedScalers:
    feature_scaler_path: Path

    def save(self):
        # nothing else to save for now
        pass


def fit_transform_features(df: pd.DataFrame, feature_cols: Tuple[str, ...]) -> Tuple[pd.DataFrame, StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    features = df[list(feature_cols)].values
    features_scaled = scaler.fit_transform(features)
    return pd.DataFrame(features_scaled, columns=feature_cols, index=df.index), scaler, features


def transform_features(df: pd.DataFrame, feature_cols: Tuple[str, ...], scaler: StandardScaler) -> pd.DataFrame:
    features = df[list(feature_cols)].values
    features_scaled = scaler.transform(features)
    return pd.DataFrame(features_scaled, columns=feature_cols, index=df.index)


def save_scaler(scaler: StandardScaler, name: str) -> Path:
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    path = SCALERS_DIR / f"{name}.joblib"
    joblib.dump(scaler, path)
    return path


def load_scaler(path: Path) -> StandardScaler:
    return joblib.load(path)


