# data.py
import pandas as pd
import yfinance as yf
import requests
from urllib.parse import quote


def load_wti(start="2010-01-01") -> pd.DataFrame:
    """
    WTI front-month continuous proxy via Yahoo Finance ticker CL=F.
    Returns a daily dataframe with settlement proxy as 'Settle'.
    """
    df = yf.download("CL=F", start=start, progress=False)
    if df.empty:
        raise RuntimeError("No data returned from yfinance for CL=F. Check internet/ticker.")
    df = df.rename(columns={"Close": "Settle"})[["Settle"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def gdelt_daily_signals(date: pd.Timestamp, query: str) -> dict:
    """
    Pull a *daily* signal from GDELT 2.1 DOC API for the given date.
    We compute:
      - article_count: number of matching docs that day (via timelinevol)
      - tone_avg: average tone that day (via timelinevol)
    If anything fails, returns zeros.

    Note: This is a lightweight “event pressure” signal.
    """
    # GDELT wants YYYYMMDDHHMMSS; we use day bounds
    day = pd.Timestamp(date).tz_localize(None)
    start = day.strftime("%Y%m%d000000")
    end = day.strftime("%Y%m%d235959")

    # Timeline endpoint gives daily buckets with vol + tone
    # Docs: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
    base = "https://api.gdeltproject.org/api/v2/doc/doc"

    params = (
        f"?query={quote(query)}"
        f"&mode=timelinevol"
        f"&format=json"
        f"&startdatetime={start}"
        f"&enddatetime={end}"
        f"&timelinesmooth=0"
        f"&format=json"
    )

    url = base + params
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        j = r.json()
        timeline = j.get("timeline", [])
        if not timeline:
            return {"article_count": 0.0, "tone_avg": 0.0}

        # For a single day range, timeline usually has 1 item
        item = timeline[0]
        count = float(item.get("value", 0.0))

        # tone can be nested
        tone = item.get("tone", None)
        tone_avg = float(tone) if tone is not None else 0.0

        return {"article_count": count, "tone_avg": tone_avg}
    except Exception:
        return {"article_count": 0.0, "tone_avg": 0.0}


def add_gdelt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds GDELT-derived features:
      - energy_news_count
      - energy_news_tone
      - spike_ratio (count / rolling 30d mean)
    """
    # You can adjust query terms here:
    # Keep it broad, but energy/oil/geopolitics focused.
    gdelt_query = '(oil OR "crude oil" OR WTI OR OPEC OR sanctions OR refinery OR pipeline) AND (geopolitics OR war OR "Middle East" OR Russia OR Iran OR shipping)'

    counts = []
    tones = []
    for dt in df.index:
        sig = gdelt_daily_signals(dt, gdelt_query)
        counts.append(sig["article_count"])
        tones.append(sig["tone_avg"])

    out = df.copy()
    out["energy_news_count"] = counts
    out["energy_news_tone"] = tones

    # Spike vs baseline
    out["news_count_ma30"] = out["energy_news_count"].rolling(30, min_periods=5).mean()
    out["news_spike_ratio"] = out["energy_news_count"] / out["news_count_ma30"].replace(0, pd.NA)
    out["news_spike_ratio"] = out["news_spike_ratio"].fillna(0.0)

    return out