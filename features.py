# features.py
import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Settle" not in out.columns:
        raise ValueError(f"Expected 'Settle' column. Got: {list(out.columns)}")

    # Ensure Settle is a 1D numeric Series
    if isinstance(out["Settle"], pd.DataFrame):
        out["Settle"] = out["Settle"].iloc[:, 0]

    out["Settle"] = pd.to_numeric(out["Settle"], errors="coerce")
    out = out.dropna(subset=["Settle"])

    # Returns
    out["ret_1d"] = np.log(out["Settle"]).diff()
    out["ret_5d"] = np.log(out["Settle"]).diff(5)
    out["ret_20d"] = np.log(out["Settle"]).diff(20)

    # Volatility (rolling std of daily log returns)
    out["vol_5d"] = out["ret_1d"].rolling(5).std()
    out["vol_20d"] = out["ret_1d"].rolling(20).std()
    out["vol_60d"] = out["ret_1d"].rolling(60).std()

    # Moving averages
    out["ma_5"] = out["Settle"].rolling(5).mean()
    out["ma_20"] = out["Settle"].rolling(20).mean()
    out["ma_50"] = out["Settle"].rolling(50).mean()

    # Ratios
    out["price_over_ma20"] = out["Settle"] / out["ma_20"].replace(0, np.nan)
    out["price_over_ma50"] = out["Settle"] / out["ma_50"].replace(0, np.nan)

    # Drawdown
    roll_max = out["Settle"].rolling(20).max()
    out["drawdown_20d"] = (out["Settle"] / roll_max) - 1.0

    return out


def make_targets(df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    """
    Creates:
      - y_week_return: forward log return over next `horizon_days` trading days
      - y_week_up: 1 if return > 0 else 0
    """
    out = df.copy()
    out["y_week_return"] = np.log(out["Settle"].shift(-horizon_days) / out["Settle"])
    out["y_week_up"] = (out["y_week_return"] > 0).astype(int)
    return out


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()