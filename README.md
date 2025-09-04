<<<<<<< HEAD
# nvidia-stock-prediction-ml
A machine learning project to predict the price trend of Nvidia (NVDA) stock. The project covers:  Data collection (stock historical prices)  Data normalization  Model training and evaluation  Observations on predicting a bullish stock vs cyclical or downtrending stocks
=======
# NVDA Stock Trend Prediction

Predict next-day price trend (up/down) for Nvidia (NVDA) stock using an LSTM on engineered OHLCV features.

## Project Overview

- Data collection: Download historical NVDA data from Yahoo Finance.
- Preprocessing: Feature engineering and normalization.
- Modeling: PyTorch LSTM sequence classifier to predict next-day trend.
- Evaluation: Classification metrics, plots, and a simple long-only backtest.

## Structure

```
.
├─ src/
│  ├─ data/
│  │  └─ download_data.py
│  ├─ features/
│  │  ├─ preprocess.py
│  │  └─ dataset.py
│  ├─ models/
│  │  └─ lstm.py
│  ├─ training/
│  │  ├─ train.py
│  │  └─ evaluate.py
│  └─ utils/
│     ├─ paths.py
│     ├─ seed.py
│     ├─ metrics.py
│     └─ plotting.py
├─ data/               # cached raw files
├─ outputs/            # models, scalers, figures, reports
├─ requirements.txt
└─ README.md
```

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download data (daily candles):

```bash
python -m src.data.download_data --ticker NVDA --start 2010-01-01 --end 2025-01-01 --interval 1d --cache
```

3. Train the model:

```bash
python -m src.training.train \
  --csv data/raw/NVDA_2010-01-01_2025-01-01_1d.csv \
  --lookback 60 --batch-size 128 --epochs 20 --hidden-size 128
```

4. Evaluate on the test split and generate plots:

```bash
python -m src.training.evaluate --run-dir outputs/runs/latest
```

The evaluation will output metrics, a confusion matrix, probability calibration plot, and a simple strategy backtest vs buy-and-hold.

## Compare Multiple Models

Train and evaluate Logistic Regression, Random Forest, HistGradientBoosting, and LSTM on the same time-based split, saving metrics and confusion matrices for each:

```bash
python -m src.training.compare \
  --csv data/raw/NVDA_2010-01-01_2025-01-01_1d.csv \
  --lookback 60 --epochs 10 --hidden-size 128
```

Outputs are saved under `outputs/runs/compare_<timestamp>/`:

- `results.json` and `RESULTS.md` with per-model metrics
- `figures/cm_*.png` confusion matrices

## Notes on Bullish vs Cyclical/Downtrending Stocks

- NVDA has long bullish regimes; naive class balance can be skewed toward "up". Use time-based splits and monitor precision/recall.
- Trend persistence helps sequence models; however, regime shifts (e.g., macro shocks) can hurt performance. Consider re-training and walk-forward validation.
- Simple long-only strategies may track buy-and-hold closely in strong bull markets; edge is often in drawdown management and avoiding whipsaws during consolidations.

## Reproducibility

- Experiments are logged in `outputs/runs/<timestamp>` including configs, scalers, model weights, and figures.
>>>>>>> 019e5ea (Initial commit: Add all project files)
