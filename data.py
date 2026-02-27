# data.py
import pandas as pd
import yfinance as yf
import requests
from urllib.parse import quote


def load_wti(start="2010-01-01") -> pd.DataFrame:
    """
    Reliable loader for Streamlit Cloud:
    1) Stooq CSV (usually stable, no rate limits)
    2) yfinance fallback (often rate-limited on Streamlit Cloud)
    Returns: DataFrame with DatetimeIndex and single column 'Settle' (float).
    """
    start_dt = pd.to_datetime(start)

    # 1) Stooq first (recommended on cloud)
    try:
        url = "https://stooq.com/q/d/l/?s=cl.f&i=d"
        raw = pd.read_csv(url)  # Date, Open, High, Low, Close, Volume
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw = raw.set_index("Date").sort_index()
        out = raw.rename(columns={"Close": "Settle"})[["Settle"]].copy()
        out["Settle"] = pd.to_numeric(out["Settle"], errors="coerce")
        out = out.dropna()
        out = out[out.index >= start_dt]
        if not out.empty:
            return out
    except Exception:
        pass

    # 2) yfinance fallback
    try:
        df = yf.download("CL=F", start=start, progress=False)
        if not df.empty and "Close" in df.columns:
            out = df.rename(columns={"Close": "Settle"})[["Settle"]].copy()
            out.index = pd.to_datetime(out.index)
            out = out.sort_index()
            out["Settle"] = pd.to_numeric(out["Settle"], errors="coerce")
            out = out.dropna()
            return out[out.index >= start_dt]
    except Exception:
        pass

    raise RuntimeError("Could not load WTI data from Stooq or yfinance.")