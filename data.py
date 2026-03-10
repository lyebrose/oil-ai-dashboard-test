# data.py
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import logging

logger = logging.getLogger(__name__)

# ── Put your EIA API key here (free at https://www.eia.gov/opendata/) ──────────
EIA_API_KEY = "ByOQgCLHkMjNN2smurIgrhoRSnEXChaYfk1A0uNC"


# ══════════════════════════════════════════════════════════════════════════════
# PRICE SOURCES
# ══════════════════════════════════════════════════════════════════════════════
def _load_yfinance(start_dt: pd.Timestamp) -> pd.DataFrame:

    raw = yf.download(
        "CL=F",
        start=start_dt.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned empty DataFrame.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].rename(columns={"Close": "Settle"})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.dropna()

    return df


def _load_eia_price(start_dt: pd.Timestamp) -> pd.DataFrame:
    """
    EIA WTI spot price (daily).
    Series: PET.RWTC.D
    Requires EIA_API_KEY.
    """
    if EIA_API_KEY == "YOUR_EIA_API_KEY_HERE":
        raise RuntimeError("EIA API key not set.")

    url = (
        "https://api.eia.gov/v2/petroleum/pri/spt/data/"
        f"?api_key={EIA_API_KEY}"
        "&frequency=daily"
        "&data[0]=value"
        "&facets[series][]=RWTC"
        "&sort[0][column]=period"
        "&sort[0][direction]=asc"
        "&offset=0&length=5000"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    j = r.json()
    rows = j.get("response", {}).get("data", [])
    if not rows:
        raise RuntimeError("EIA price: empty response.")

    df = pd.DataFrame(rows)
    df["Date"]   = pd.to_datetime(df["period"])
    df["Settle"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["Date", "Settle"]].dropna().set_index("Date").sort_index()
    df = df[df.index >= start_dt]

    if df.empty:
        raise RuntimeError("EIA price: no rows after filtering.")
    logger.info(f"EIA price: {len(df)} rows, latest={df.index[-1].date()}")
    return df


def _load_stooq(start_dt: pd.Timestamp) -> pd.DataFrame:
    """Stooq CL.F continuous — fallback."""
    url = "https://stooq.com/q/d/l/?s=cl.f&i=d"
    raw = pd.read_csv(url)
    if "Date" not in raw.columns:
        raise RuntimeError(f"Stooq unexpected columns: {raw.columns.tolist()}")
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date").sort_index()
    if "Close" in raw.columns:
        raw = raw.rename(columns={"Close": "Settle"})
    if "Settle" not in raw.columns:
        raise RuntimeError(f"Stooq: no Close/Settle column.")
    out = raw[["Settle"]].copy()
    out["Settle"] = pd.to_numeric(out["Settle"], errors="coerce")
    out = out.dropna()
    out = out[out.index >= start_dt]
    if out.empty:
        raise RuntimeError("Stooq: no rows after filtering.")
    logger.info(f"Stooq: {len(out)} rows, latest={out.index[-1].date()}")
    return out


def _load_fred_price(start_dt: pd.Timestamp) -> pd.DataFrame:
    """FRED DCOILWTICO — ~2 week lag, last resort."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"
    raw = pd.read_csv(url)
    raw.columns = ["Date", "Settle"]
    raw["Date"]   = pd.to_datetime(raw["Date"])
    raw["Settle"] = pd.to_numeric(raw["Settle"].replace(".", pd.NA), errors="coerce")
    raw = raw.dropna(subset=["Settle"]).set_index("Date").sort_index()
    raw = raw[raw.index >= start_dt]
    if raw.empty:
        raise RuntimeError("FRED price: no rows.")
    logger.info(f"FRED price: {len(raw)} rows, latest={raw.index[-1].date()}")
    return raw


def load_wti(start: str = "2023-01-01") -> pd.DataFrame:
    """
    Returns DataFrame with DatetimeIndex and column 'Settle' (float).
    Priority: yfinance (live) → EIA spot (daily fundamental) → Stooq → FRED
    """
    start_dt = pd.to_datetime(start)
    for name, loader in [
        ("yfinance", lambda: _load_yfinance(start_dt)),
        ("EIA spot", lambda: _load_eia_price(start_dt)),
        ("Stooq",    lambda: _load_stooq(start_dt)),
        ("FRED",     lambda: _load_fred_price(start_dt)),
    ]:
        try:
            return loader()
        except Exception as e:
            logger.warning(f"{name} failed: {e}")

    raise RuntimeError("All WTI price sources failed.")


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def load_eia_inventory(start_dt: pd.Timestamp) -> pd.Series:
    """
    EIA US crude oil inventory (weekly, thousand barrels).
    Series: PET.WCRSTUS1.W
    Returns a weekly Series reindexed to daily (forward-filled).
    """
    if EIA_API_KEY == "YOUR_EIA_API_KEY_HERE":
        logger.warning("EIA API key not set — skipping inventory.")
        return pd.Series(dtype=float, name="inventory_kbd")

    url = (
        "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
        f"?api_key={EIA_API_KEY}"
        "&frequency=weekly"
        "&data[0]=value"
        "&facets[series][]=WCRSTUS1"
        "&sort[0][column]=period"
        "&sort[0][direction]=asc"
        "&offset=0&length=2000"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
        if not rows:
            raise RuntimeError("EIA inventory: empty response.")
        df = pd.DataFrame(rows)
        df["Date"]  = pd.to_datetime(df["period"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("Date")["value"].dropna().sort_index()
        s = s[s.index >= start_dt]
        # Week-over-week change (draw = negative = bullish)
        s_chg = s.diff().rename("inventory_chg_kbd")
        logger.info(f"EIA inventory: {len(s)} weeks")
        return s_chg
    except Exception as e:
        logger.warning(f"EIA inventory failed: {e}")
        return pd.Series(dtype=float, name="inventory_chg_kbd")


def load_dollar_index(start_dt: pd.Timestamp) -> pd.Series:
    """
    US Dollar Index (DXY) via FRED series DTWEXBGS.
    Broad dollar index — negative correlation with oil typically.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"
    try:
        raw = pd.read_csv(url)
        raw.columns = ["Date", "dxy"]
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw["dxy"]  = pd.to_numeric(raw["dxy"].replace(".", pd.NA), errors="coerce")
        s = raw.dropna().set_index("Date")["dxy"].sort_index()
        s = s[s.index >= start_dt]
        # Use % change so it's stationary
        s_chg = s.pct_change().rename("dxy_ret")
        logger.info(f"DXY: {len(s)} rows")
        return s_chg
    except Exception as e:
        logger.warning(f"DXY (FRED) failed: {e}")
        return pd.Series(dtype=float, name="dxy_ret")


def load_opec_production(start_dt: pd.Timestamp) -> pd.Series:
    """
    OPEC crude production proxy via EIA monthly supply data.
    Series: PET.PATC_OPEC_1.M  (OPEC total liquids, thousand b/d)
    Falls back to a world production proxy if OPEC-specific fails.
    """
    if EIA_API_KEY == "YOUR_EIA_API_KEY_HERE":
        logger.warning("EIA API key not set — skipping OPEC production.")
        return pd.Series(dtype=float, name="opec_prod_kbd")

    # Try OPEC total crude first, then world as fallback
    series_candidates = ["PATC_OPEC_1", "PASC_OPEC_1"]
    for series_id in series_candidates:
        url = (
            "https://api.eia.gov/v2/petroleum/supply/monthly/data/"
            f"?api_key={EIA_API_KEY}"
            "&frequency=monthly"
            "&data[0]=value"
            f"&facets[series][]={series_id}"
            "&sort[0][column]=period"
            "&sort[0][direction]=asc"
            "&offset=0&length=500"
        )
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            rows = r.json().get("response", {}).get("data", [])
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["Date"]  = pd.to_datetime(df["period"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            s = df.set_index("Date")["value"].dropna().sort_index()
            s = s[s.index >= start_dt]
            s_chg = s.diff().rename("opec_prod_chg_kbd")
            logger.info(f"OPEC production ({series_id}): {len(s)} months")
            return s_chg
        except Exception as e:
            logger.warning(f"OPEC production {series_id} failed: {e}")

    return pd.Series(dtype=float, name="opec_prod_chg_kbd")


# ══════════════════════════════════════════════════════════════════════════════
# MERGE FUNDAMENTALS INTO PRICE DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════

def load_fundamentals(df_price: pd.DataFrame) -> pd.DataFrame:
    """
    Merges fundamental features into the daily price DataFrame.
    All series are forward-filled to align with daily trading calendar.
    Missing data for any series just results in NaN (model handles it).
    """
    start_dt = df_price.index.min()
    df = df_price.copy()

    # ── Dollar index (daily FRED) ──────────────────────────────────────────
    dxy = load_dollar_index(start_dt)
    if not dxy.empty:
        df = df.join(dxy.reindex(df.index, method="ffill"), how="left")

    # ── EIA inventory change (weekly → daily ffill) ────────────────────────
    inv = load_eia_inventory(start_dt)
    if not inv.empty:
        df = df.join(inv.reindex(df.index, method="ffill"), how="left")

    # ── OPEC production change (monthly → daily ffill) ─────────────────────
    opec = load_opec_production(start_dt)
    if not opec.empty:
        df = df.join(opec.reindex(df.index, method="ffill"), how="left")

    logger.info(f"Fundamentals merged. Columns: {df.columns.tolist()}")
    return df
