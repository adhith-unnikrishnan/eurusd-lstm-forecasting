"""Loading HistData EUR/USD 1-minute bars and building model features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import ta

PRICE_COLS = ["Open", "High", "Low", "Close"]


def load_histdata(path: str | Path) -> pd.DataFrame:
    """Load a HistData.com 1-minute bar file (XLSX or ASCII CSV).

    Returns a DataFrame indexed by timestamp with Open, High, Low, Close.
    HistData timestamps are EST without daylight saving; they are kept as-is.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None)
    else:  # ASCII format: 20230102 170000;1.070230;1.070230;1.070230;1.070230;0
        raw = pd.read_csv(path, sep=None, engine="python", header=None)

    raw = raw.iloc[:, :5]
    raw.columns = ["Date"] + PRICE_COLS
    # Drop a header row if the file has one
    raw = raw[pd.to_numeric(raw["Close"], errors="coerce").notna()]
    dates = raw["Date"]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = dates.astype(str).str.strip()
        parsed = pd.to_datetime(dates, format="%Y%m%d %H%M%S", errors="coerce")
        dates = parsed.fillna(pd.to_datetime(dates, errors="coerce"))
    df = pd.DataFrame({c: pd.to_numeric(raw[c]).to_numpy() for c in PRICE_COLS},
                      index=pd.DatetimeIndex(dates, name="Date"))
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def add_indicators(df: pd.DataFrame, close: str) -> pd.DataFrame:
    """Standard technical indicators computed on column `close`."""
    out = df.copy()
    c = out[close]
    out["SMA_50"] = ta.trend.sma_indicator(c, window=50)
    out["SMA_200"] = ta.trend.sma_indicator(c, window=200)
    out["RSI"] = ta.momentum.rsi(c, window=14)
    out["MACD"] = ta.trend.macd(c)
    out["MACD_Signal"] = ta.trend.macd_signal(c)
    out["MACD_Hist"] = ta.trend.macd_diff(c)
    bb = ta.volatility.BollingerBands(c, window=20)
    out["BB_Upper"] = bb.bollinger_hband()
    out["BB_Middle"] = bb.bollinger_mavg()
    out["BB_Lower"] = bb.bollinger_lband()
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stationary, scale-free features computed only from information up to each bar.

    Every feature at row t uses bars <= t. The target (built elsewhere) is the
    return from bar t to bar t+1, so no future information enters the inputs.
    """
    c = df["Close"]
    out = pd.DataFrame(index=df.index)
    out["ret_1"] = np.log(c / c.shift(1))
    out["ret_5"] = np.log(c / c.shift(5))
    out["ret_15"] = np.log(c / c.shift(15))
    out["range"] = (df["High"] - df["Low"]) / c
    out["body"] = (c - df["Open"]) / c
    ind = add_indicators(df, "Close")
    out["dist_sma50"] = c / ind["SMA_50"] - 1
    out["dist_sma200"] = c / ind["SMA_200"] - 1
    out["rsi"] = ind["RSI"] / 100 - 0.5
    out["macd"] = ind["MACD"] / c
    out["macd_signal"] = ind["MACD_Signal"] / c
    out["macd_hist"] = ind["MACD_Hist"] / c
    width = ind["BB_Upper"] - ind["BB_Lower"]
    out["bb_pos"] = (c - ind["BB_Lower"]) / width.replace(0, np.nan) - 0.5
    out["bb_width"] = width / c
    out["vol_30"] = out["ret_1"].rolling(30).std()
    minute = df.index.hour * 60 + df.index.minute
    out["tod_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * minute / 1440)
    return out
