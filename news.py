import requests
import pandas as pd
from datetime import datetime, timedelta


def fetch_oil_geopolitics(max_records=20):

    query = """
    (oil OR crude OR opec OR tanker OR "strait of hormuz" OR refinery OR sanctions)
    """

    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": max_records,
        "format": "json",
        "sort": "DateDesc",
        "timespan": "1day"
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()

    data = r.json()

    articles = []

    for a in data.get("articles", []):
        articles.append({
            "title": a["title"],
            "url": a["url"],
            "source": a.get("sourceCommonName", ""),
            "date": a["seendate"]
        })

    return pd.DataFrame(articles)
