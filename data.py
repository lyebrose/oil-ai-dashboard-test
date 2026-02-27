# data.py
import pandas as pd
import yfinance as yf


def _load_fred_wti(start_dt: pd.Timestamp) -> pd.DataFrame:
    """
    FRED WTI spot (DCOILWTICO). Extremely reliable CSV endpoint.
    NOTE: Spot, not futures. Good fallback for deployment stability.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"
    raw = pd.read_csv(url)
    raw.columns = ["Date", "Settle"]
    raw["Date"] = pd.to_datetime(raw["Date"])
    # FRED uses '.' for missing values
    raw["Settle"] = pd.to_numeric(raw["Settle"].replace(".", pd.NA), errors="coerce")
    raw = raw.dropna(subset=["Settle"]).set_index("Date").sort_index()
    raw = raw[raw.index >= start_dt]
    if raw.empty:
        raise RuntimeError("FRED returned no usable WTI rows.")
    return raw


def _load_stooq(start_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Stooq crude oil continuous proxy.
    """
    url = "https://stooq.com/q/d/l/?s=cl.f&i=d"
    raw = pd.read_csv(url)  # Date, Open, High, Low, Close, Volume
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").sort_index()
    out = raw.rename(columns={"Close": "Settle"})[["Settle"]].copy()
    out["Settle"] = pd.to_numeric(out["Settle"], errors="coerce")
    out = out.dropna()
    out = out[out.index >= start_dt]
    if out.empty:
        raise RuntimeError("Stooq returned no usable rows.")
    return out


def _load_yfinance(start_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Yahoo Finance front-month futures proxy (CL=F). Often rate-limited on Streamlit Cloud.
    """
    df = yf.download("CL=F", start=start_dt.strftime("%Y-%m-%d"), progress=False)
    if df.empty or "Close" not in df.columns:
        raise RuntimeError("yfinance returned empty.")
    out = df.rename(columns={"Close": "Settle"})[["Settle"]].copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out["Settle"] = pd.to_numeric(out["Settle"], errors="coerce")
    out = out.dropna()
    out = out[out.index >= start_dt]
    if out.empty:
        raise RuntimeError("yfinance returned no usable rows after cleaning.")
    return out


def load_wti(start: str = "2010-01-01") -> pd.DataFrame:
    """
    Returns: DataFrame with DatetimeIndex and one column 'Settle' (float).
    Tries multiple sources to survive Streamlit Cloud network/rate limits.
    """
    start_dt = pd.to_datetime(start)

    # 1) FRED (most reliable)
    try:
        return _load_fred_wti(start_dt)
    except Exception:
        pass

    # 2) Stooq
    try:
        return _load_stooq(start_dt)
    except Exception:
        pass

    # 3) yfinance
    try:
        return _load_yfinance(start_dt)
    except Exception:
        pass

    raise RuntimeError("Could not load WTI data from FRED, Stooq, or yfinance.")