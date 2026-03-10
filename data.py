# data.py
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


def _load_stooq(start_dt: pd.Timestamp) -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=cl.f&i=d"
    raw = pd.read_csv(url)

    # Guard against Stooq returning an HTML error page
    if "Date" not in raw.columns:
        raise RuntimeError(f"Stooq returned unexpected columns: {raw.columns.tolist()}")

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date").sort_index()

    # Handle both "Close" and "Settle" column names
    if "Close" in raw.columns:
        raw = raw.rename(columns={"Close": "Settle"})
    if "Settle" not in raw.columns:
        raise RuntimeError(f"Stooq: no Close/Settle column. Got: {raw.columns.tolist()}")

    out = raw[["Settle"]].copy()
    out["Settle"] = pd.to_numeric(out["Settle"], errors="coerce")
    out = out.dropna()
    out = out[out.index >= start_dt]

    if out.empty:
        raise RuntimeError("Stooq returned no usable rows after filtering.")
    return out


def _load_yfinance(start_dt: pd.Timestamp) -> pd.DataFrame:
    raw = yf.download("CL=F", start=start_dt.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)

    if raw.empty:
        raise RuntimeError("yfinance returned empty DataFrame.")

    # Flatten MultiIndex columns (newer yfinance returns ("Close", "CL=F") etc.)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    if "Close" not in raw.columns:
        raise RuntimeError(f"yfinance: no Close column. Got: {raw.columns.tolist()}")

    out = raw[["Close"]].rename(columns={"Close": "Settle"}).copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out["Settle"] = pd.to_numeric(out["Settle"], errors="coerce")
    out = out.dropna()
    out = out[out.index >= start_dt]

    if out.empty:
        raise RuntimeError("yfinance: no usable rows after cleaning.")
    return out


def _load_fred_wti(start_dt: pd.Timestamp) -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"
    raw = pd.read_csv(url)
    raw.columns = ["Date", "Settle"]
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["Settle"] = pd.to_numeric(raw["Settle"].replace(".", pd.NA), errors="coerce")
    raw = raw.dropna(subset=["Settle"]).set_index("Date").sort_index()
    raw = raw[raw.index >= start_dt]
    if raw.empty:
        raise RuntimeError("FRED returned no usable WTI rows.")
    return raw


def load_wti(start: str = "2023-01-01") -> pd.DataFrame:
    start_dt = pd.to_datetime(start)

    for name, loader in [
        ("Stooq",    lambda: _load_stooq(start_dt)),
        ("yfinance", lambda: _load_yfinance(start_dt)),
        ("FRED",     lambda: _load_fred_wti(start_dt)),
    ]:
        try:
            df = loader()
            logger.info(f"WTI data loaded from {name}: {len(df)} rows, latest={df.index[-1].date()}")
            return df
        except Exception as e:
            logger.warning(f"{name} failed: {e}")

    raise RuntimeError("Could not load WTI data from Stooq, yfinance, or FRED.")
