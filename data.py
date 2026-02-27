# data.py
import pandas as pd
import yfinance as yf


def load_wti(start: str = "2023-01-01") -> pd.DataFrame:
    """
    Load WTI daily settlement proxy.
    Streamlit Cloud often rate-limits Yahoo Finance, so we use Stooq first.

    Returns a DataFrame with:
      - DatetimeIndex
      - column: 'Settle' (float)
    """
    start_dt = pd.to_datetime(start)

    # 1) Stooq (recommended on Streamlit Cloud)
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

    # 2) yfinance fallback (may rate-limit on cloud)
    try:
        df = yf.download("CL=F", start=start, progress=False)
        if not df.empty and "Close" in df.columns:
            out = df.rename(columns={"Close": "Settle"})[["Settle"]].copy()
            out.index = pd.to_datetime(out.index)
            out = out.sort_index()
            out["Settle"] = pd.to_numeric(out["Settle"], errors="coerce")
            out = out.dropna()
            out = out[out.index >= start_dt]
            if not out.empty:
                return out
    except Exception:
        pass

    raise RuntimeError("Could not load WTI data from Stooq or yfinance.")