# data.py
import pandas as pd
import numpy as np
import requests
import yfinance as yf


# =========================
# Public loader
# =========================
def load_wti(start: str = "2023-01-01") -> pd.DataFrame:
    """
    Returns DataFrame with:
        - DatetimeIndex
        - 'Settle' column (float)
        - 'Volume' column (float, may be NaN if unavailable)

    Tries multiple sources and prefers one that contains usable Volume.
    """

    start_dt = pd.to_datetime(start)

    fred_df = None
    best_df = None

    # 1️⃣ FRED (very reliable, but no volume)
    try:
        fred_df = _load_fred_wti(start_dt)
    except Exception:
        fred_df = None

    # 2️⃣ Stooq (often includes volume)
    try:
        stooq_df = _load_stooq(start_dt)
        if (
            "Volume" in stooq_df.columns
            and stooq_df["Volume"].notna().any()
        ):
            return stooq_df
        if stooq_df is not None and not stooq_df.empty:
            best_df = stooq_df
    except Exception:
        pass

    # 3️⃣ yfinance (often includes volume)
    try:
        yf_df = _load_yfinance(start_dt)
        if (
            "Volume" in yf_df.columns
            and yf_df["Volume"].notna().any()
        ):
            return yf_df
        if best_df is None and yf_df is not None and not yf_df.empty:
            best_df = yf_df
    except Exception:
        pass

    # If we reach here:
    if best_df is not None:
        return best_df

    if fred_df is not None:
        return fred_df

    raise RuntimeError("Could not load WTI data from FRED, Stooq, or yfinance.")


# =========================
# FRED (price only)
# =========================
def _load_fred_wti(start_dt: pd.Timestamp) -> pd.DataFrame:
    """
    FRED WTI series:
    DCOILWTICO (Cushing, OK WTI Spot Price)
    """
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        "id=DCOILWTICO"
    )

    df = pd.read_csv(url)
    df.columns = ["Date", "Settle"]

    df["Date"] = pd.to_datetime(df["Date"])
    df["Settle"] = pd.to_numeric(df["Settle"], errors="coerce")

    df = df.set_index("Date").sort_index()
    df = df[df.index >= start_dt]

    # FRED has no volume
    df["Volume"] = np.nan

    return df.dropna(subset=["Settle"])


# =========================
# Stooq
# =========================
def _load_stooq(start_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Stooq continuous WTI futures
    """
    url = "https://stooq.com/q/d/l/?s=cl.f&i=d"

    df = pd.read_csv(url)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[df.index >= start_dt]

    out = pd.DataFrame(index=df.index)

    # Use Close as settlement proxy
    out["Settle"] = pd.to_numeric(df["Close"], errors="coerce")

    # Keep volume if available
    if "Volume" in df.columns:
        out["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    else:
        out["Volume"] = np.nan

    return out.dropna(subset=["Settle"])


# =========================
# yfinance
# =========================
def _load_yfinance(start_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Yahoo Finance CL=F (WTI continuous futures)
    """
    df = yf.download(
        "CL=F",
        start=start_dt.strftime("%Y-%m-%d"),
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data.")

    out = pd.DataFrame(index=pd.to_datetime(df.index))

    out["Settle"] = df["Close"].astype(float)

    if "Volume" in df.columns:
        out["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    else:
        out["Volume"] = np.nan

    return out.dropna(subset=["Settle"])