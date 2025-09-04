import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.utils.paths import RAW_DATA_DIR, ensure_dirs


def download_yahoo(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} {start} {end} {interval}")
    
    # Handle multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    df = df.reset_index().rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    
    # Ensure numeric columns are properly typed
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Download historical data from Yahoo Finance")
    parser.add_argument("--ticker", type=str, default="NVDA")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--interval", type=str, default="1d", choices=["1d", "1wk", "1mo"])
    parser.add_argument("--cache", action="store_true", help="Cache CSV to data/raw")
    args = parser.parse_args()

    ensure_dirs()

    df = download_yahoo(args.ticker, args.start, args.end, args.interval)
    print(df.head())

    if args.cache:
        filename = f"{args.ticker}_{args.start}_{args.end}_{args.interval}.csv"
        out_path = RAW_DATA_DIR / filename
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


