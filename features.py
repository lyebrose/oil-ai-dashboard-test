# features.py
import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = np.log(out["Settle"]).diff()
    out["ret_5d"] = np.log(out["Settle"]).diff(5)
    out["ret_20d"] = np.log(out["Settle"]).diff(20)

    out["vol_5d"] = out["ret_1d"].rolling(5).std()
    out["vol_20d"] = out["ret_1d"].rolling(20).std()
    out["vol_60d"] = out["ret_1d"].rolling(60).std()

    out["ma_5"] = out["Settle"].rolling(5).mean()
    out["ma_20"] = out["Settle"].rolling(20).mean()
    out["ma_50"] = out["Settle"].rolling(50).mean()

    out["price_over_ma20"] = out["Settle"] / out["ma_20"]
    out["price_over_ma50"] = out["Settle"] / out["ma_50"]

    # Drawdown (20d)
    roll_max = out["Settle"].rolling(20).max()
    out["drawdown_20d"] = (out["Settle"] / roll_max) - 1.0

    return out


def make_targets(df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    """
    Creates:
      - y_week_return: forward log return over next `horizon_days` trading days
      - y_week_up: classification label (1 if up, else 0)
    """
    out = df.copy()
    out["y_week_return"] = np.log(out["Settle"].shift(-horizon_days) / out["Settle"])
    out["y_week_up"] = (out["y_week_return"] > 0).astype(int)
    return out


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and keep model-ready columns.
    """
    out = df.copy()
    out = out.dropna()
    return out

