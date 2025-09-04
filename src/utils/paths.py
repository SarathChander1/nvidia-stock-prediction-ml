import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
SCALERS_DIR = OUTPUTS_DIR / "scalers"


def ensure_dirs() -> None:
    for d in [DATA_DIR, RAW_DATA_DIR, OUTPUTS_DIR, RUNS_DIR, FIGURES_DIR, MODELS_DIR, SCALERS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "OUTPUTS_DIR",
    "RUNS_DIR",
    "FIGURES_DIR",
    "MODELS_DIR",
    "SCALERS_DIR",
    "ensure_dirs",
]



