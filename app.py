# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime, timedelta
import math

from data import load_wti
from features import add_technical_features, make_targets, build_model_frame
from model import walk_forward_backtest, train_latest_model, FEATURE_COLS

def add_participation_ratio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Ensure ret_1d exists
    if "ret_1d" not in out.columns:
        out["ret_1d"] = np.log(out["Settle"]).diff()

    # If volume missing, create it (prevents crashes)
    if "Volume" not in out.columns:
        out["Volume"] = np.nan

    # Rolling baseline + relative volume
    out["vol_avg_20"] = out["Volume"].rolling(20).mean()
    out["vol_rel_20"] = out["Volume"] / out["vol_avg_20"]

    denom = out["vol_rel_20"].replace([0, np.inf, -np.inf], np.nan)

    # High = big move on low participation ("thin/flukey")
    out["move_per_participation"] = out["ret_1d"].abs() / denom

    return out
# ---------------- Page config ----------------
st.set_page_config(
    page_title="WTI AI Dashboard",
    page_icon="🛢️",
    layout="wide",
)

# ---------------- CSS (sleek blue + cream) ----------------
st.markdown(
    """
<style>
.stApp { background: #FBF6EC; }
.block-container { padding-top: 1.1rem; padding-bottom: 1.2rem; }

/* Headings */
h1, h2, h3, h4 { color: #0B1F44; letter-spacing: -0.2px; }

/* Cards */
.card {
    background: #FFFDF6;
    border: 1px solid rgba(16,42,67,0.10);
    border-radius: 18px;
    padding: 16px 16px;
    box-shadow: 0 6px 18px rgba(16,42,67,0.07);
}
.kicker { color: rgba(16,42,67,0.70); font-size: 0.85rem; margin-bottom: 4px; }
.big { font-size: 1.65rem; font-weight: 800; color: #0B1F44; margin: 0; }
.delta { font-size: 0.95rem; margin-top: 6px; font-weight: 650; color: rgba(11,31,68,0.90); }

/* Pills */
.pill {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 0.83rem;
    letter-spacing: 0.3px;
}
.pill-up {
    background: rgba(30, 77, 183, 0.10);
    color: #1E4DB7;
    border: 1px solid rgba(30, 77, 183, 0.25);
}
.pill-down {
    background: rgba(180, 83, 9, 0.10);
    color: #9A3412;
    border: 1px solid rgba(180, 83, 9, 0.22);
}
.pill-neutral {
    background: rgba(16,42,67,0.08);
    color: rgba(16,42,67,0.85);
    border: 1px solid rgba(16,42,67,0.15);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #FFFDF6;
    border-right: 1px solid rgba(16,42,67,0.08);
}

/* Buttons */
.stButton button {
    background: #1E4DB7 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 0 !important;
    padding: 10px 14px !important;
    font-weight: 800 !important;
}
.stButton button:hover { background: #173E93 !important; }

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(16,42,67,0.10);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Header ----------------
st.markdown("## 🛢️ WTI Market Dashboard")
st.markdown(
    "<div class='kicker'>1-week horizon • Technical model • Educational decision support (not financial advice)</div>",
    unsafe_allow_html=True,
)

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.markdown("### Controls")
    start_date = st.text_input("Start date (YYYY-MM-DD)", "2023-01-01")
    horizon_days = st.selectbox("Forecast horizon (trading days)", [5, 10], index=0)
    backtest_start_year = st.slider("Backtest start year", 2012, 2025, 2018)

    st.markdown("---")
    st.markdown("### Display")
    show_feature_table = st.checkbox("Show latest feature table", value=True)
    show_importance = st.checkbox("Show feature importance", value=True)
    geo_days = st.selectbox("Geopolitics lookback (days)", [3, 7, 14], index=1)
    geo_max_items = st.selectbox("Max headlines", [6, 8, 10, 12], index=2)

    st.markdown("---")
    st.markdown("### What-if / Position sizing")
    use_custom_return = st.checkbox("Use custom predicted return (%)", value=False)
    st.markdown("---")
st.markdown("### Trade simulator")

sim_mode = st.selectbox(
    "Mode",
    ["Percent exposure (simple)", "WTI futures (CL)"],
    index=0,
)

amount = st.number_input("Capital ($)", min_value=0.0, value=10000.0, step=500.0)

use_custom_return = st.checkbox("Use custom predicted return (%)", value=False)

custom_return_pct = None
if use_custom_return:
    custom_return_pct = st.number_input("Custom predicted return (%)", value=0.0, step=0.25)

contracts = 1
margin_per_contract = 8000.0
if sim_mode == "WTI futures (CL)":
    contracts = st.number_input("Contracts (CL)", min_value=0, value=1, step=1)
    margin_per_contract = st.number_input("Margin per contract ($) (assumption)", min_value=0.0, value=8000.0, step=250.0)
    

# ---------------- Data build ----------------
@st.cache_data(show_spinner=False)
def build_dataset(start_date: str, horizon_days: int) -> pd.DataFrame:
    df = load_wti(start=start_date)
    df = add_technical_features(df)
    df = make_targets(df, horizon_days=horizon_days)
    df = build_model_frame(df)
    df = add_participation_ratio(df)
    return df

df = build_dataset(start_date, horizon_days)

# ---------------- Model training (latest) ----------------
model, latest_pred = train_latest_model(df)
pred_pct = (np.exp(latest_pred) - 1) * 100
pred_dir = "UP" if latest_pred > 0 else "DOWN"

# ---------------- Top KPIs ----------------
latest_price = float(df["Settle"].iloc[-1])

week_change = np.nan
if len(df) > 6:
    week_change = float((df["Settle"].iloc[-1] / df["Settle"].iloc[-6] - 1) * 100)

risk = float(df["vol_20d"].iloc[-1]) if "vol_20d" in df.columns else np.nan
vol_label = "High" if risk >= 0.03 else "Normal"

# Signal strength proxy (based on forecast magnitude, capped)
signal_strength = min(abs(pred_pct) / 3.0, 1.0)

pill_class = "pill-up" if pred_dir == "UP" else "pill-down"
pill_text = f"<span class='pill {pill_class}'>SIGNAL: {pred_dir}</span>"

c1, c2, c3, c4 = st.columns([1.25, 1.25, 1.0, 1.1])

with c1:
    st.markdown(
        f"""
        <div class="card">
            <div class="kicker">WTI (daily proxy)</div>
            <p class="big">${latest_price:,.2f}</p>
            <div class="delta">1-week change: {week_change:+.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="card">
            <div class="kicker">Forecast (next {horizon_days} trading days)</div>
            <p class="big">{pred_pct:+.2f}%</p>
            <div class="delta">{pill_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="card">
            <div class="kicker">Risk (20d vol)</div>
            <p class="big">{risk:.4f}</p>
            <div class="delta">Regime: {vol_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="card">
            <div class="kicker">Signal strength (proxy)</div>
            <p class="big">{signal_strength*100:.0f}%</p>
            <div class="delta">Based on |forecast| (capped)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(signal_strength)

st.markdown("---")
# ---------------- Trade simulator (exposure + futures + risk bands) ----------------
st.markdown("### 🧮 Trade simulator (return + futures P/L + risk bands)")

# Use model forecast unless custom return is enabled
used_return_pct = float(custom_return_pct) if (use_custom_return and custom_return_pct is not None) else float(pred_pct)
used_logret = np.log1p(used_return_pct / 100.0)  # convert % to log return

price_now = float(df["Settle"].iloc[-1])

# Volatility scaling to horizon: vol_20d is daily log-return std
daily_vol = float(df["vol_20d"].iloc[-1]) if "vol_20d" in df.columns and pd.notna(df["vol_20d"].iloc[-1]) else 0.0
h = int(horizon_days)
sigma_h = daily_vol * np.sqrt(max(h, 1))  # horizon std in log-return space

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# Expected / band returns (log-return model: r ~ N(mu, sigma_h))
mu = used_logret
r_1s_low, r_1s_high = mu - sigma_h, mu + sigma_h
r_2s_low, r_2s_high = mu - 2 * sigma_h, mu + 2 * sigma_h

# Convert to prices
price_exp = price_now * np.exp(mu)
price_1s_low = price_now * np.exp(r_1s_low)
price_1s_high = price_now * np.exp(r_1s_high)
price_2s_low = price_now * np.exp(r_2s_low)
price_2s_high = price_now * np.exp(r_2s_high)

# Probability of profit (P[r > 0]) under normal approximation
p_profit = 0.0
if sigma_h > 0:
    z = (0.0 - mu) / sigma_h
    p_profit = 1.0 - norm_cdf(z)
else:
    p_profit = 1.0 if mu > 0 else (0.0 if mu < 0 else 0.5)

# Compute P/L depending on mode
# - Percent exposure: P/L = capital * return
# - Futures: CL contract = 1,000 barrels, P/L = (price_change) * 1000 * contracts
CONTRACT_SIZE_BBL = 1000.0

def pl_percent(capital: float, r_log: float) -> float:
    return capital * (np.exp(r_log) - 1.0)

def pl_futures(contracts_n: int, price_start: float, r_log: float) -> float:
    price_end = price_start * np.exp(r_log)
    return (price_end - price_start) * CONTRACT_SIZE_BBL * float(contracts_n)

if sim_mode == "Percent exposure (simple)":
    pl_exp = pl_percent(amount, mu)
    pl_1s_low = pl_percent(amount, r_1s_low)
    pl_1s_high = pl_percent(amount, r_1s_high)
    pl_2s_low = pl_percent(amount, r_2s_low)
    pl_2s_high = pl_percent(amount, r_2s_high)

    notional = amount
    leverage_label = "1.0× (capital exposure)"
    margin_used = 0.0

else:
    c = int(contracts)
    pl_exp = pl_futures(c, price_now, mu)
    pl_1s_low = pl_futures(c, price_now, r_1s_low)
    pl_1s_high = pl_futures(c, price_now, r_1s_high)
    pl_2s_low = pl_futures(c, price_now, r_2s_low)
    pl_2s_high = pl_futures(c, price_now, r_2s_high)

    notional = price_now * CONTRACT_SIZE_BBL * float(c)
    margin_used = float(c) * float(margin_per_contract)
    leverage_label = f"{(notional / amount) if amount > 0 else 0.0:.2f}× notional/capital"

# UI layout
sim_left, sim_right = st.columns([1.25, 1])

with sim_left:
    st.markdown(
        f"""
        <div class="card">
            <div class="kicker">Forecast summary (next {horizon_days} trading days)</div>
            <p class="big">{used_return_pct:+.2f}%</p>
            <div class="delta">Implied price: ${price_exp:,.2f} (from ${price_now:,.2f})</div>
            <div class="kicker" style="margin-top:10px;">
                Daily vol (20d): <b>{daily_vol:.4f}</b> · Horizon σ: <b>{sigma_h:.4f}</b> · P(profit): <b>{p_profit*100:.1f}%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Bands in a clean table
    bands = pd.DataFrame(
        {
            "Scenario": ["Expected", "1σ low", "1σ high", "2σ low", "2σ high"],
            "Price": [price_exp, price_1s_low, price_1s_high, price_2s_low, price_2s_high],
            "P/L ($)": [pl_exp, pl_1s_low, pl_1s_high, pl_2s_low, pl_2s_high],
        }
    )
    st.markdown("#### Risk bands (volatility-based)")
    st.dataframe(
        bands.style.format({"Price": "${:,.2f}", "P/L ($)": "{:+,.2f}"}),
        use_container_width=True,
    )

with sim_right:
    # Position + margin card
    if sim_mode == "WTI futures (CL)":
        margin_status = "OK" if (amount >= margin_used) else "⚠️ Under-margined (capital < margin assumption)"
        st.markdown(
            f"""
            <div class="card">
                <div class="kicker">Futures position</div>
                <p class="big">{int(contracts)} contract(s)</p>
                <div class="delta">Notional: ${notional:,.0f}</div>
                <div class="kicker" style="margin-top:10px;">
                    Margin used (assumed): <b>${margin_used:,.0f}</b> · Status: <b>{margin_status}</b><br/>
                    Leverage proxy: <b>{leverage_label}</b><br/>
                    Contract size: <b>1,000 barrels</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Margin varies by broker/exchange and market conditions. This is an assumption for simulation.")
    else:
        st.markdown(
            f"""
            <div class="card">
                <div class="kicker">Capital exposure</div>
                <p class="big">${amount:,.0f}</p>
                <div class="delta">Assumes return applies to capital directly</div>
                <div class="kicker" style="margin-top:10px;">
                    Exposure: <b>${notional:,.0f}</b> · Leverage: <b>{leverage_label}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Quick interpretation
    if pl_exp > 0:
        st.success("📈 Expected P/L is positive under the selected forecast.")
    elif pl_exp < 0:
        st.error("📉 Expected P/L is negative under the selected forecast.")
    else:
        st.info("⚪ Expected P/L is near zero under the selected forecast.")

    st.caption("Bands use a normal approximation with σ estimated from 20-day realized volatility (log returns).")
# ---------------- Position / What-if calculator ----------------
st.markdown("### 💵 What-if return calculator")

calc_left, calc_right = st.columns([1.2, 1])

with calc_left:
    amount = st.number_input(
        "Amount to invest ($)",
        min_value=0.0,
        value=10000.0,
        step=500.0,
    )

    if use_custom_return:
        custom_return = st.number_input(
            "Predicted return (%)",
            value=float(pred_pct),
            step=0.25,
        )
        used_return_pct = float(custom_return)
        source_label = "Custom input"
    else:
        used_return_pct = float(pred_pct)
        source_label = "Model forecast"

    st.caption(f"Return source: **{source_label}**")

with calc_right:
    projected_profit = amount * (used_return_pct / 100.0)
    ending_value = amount + projected_profit

    st.markdown(
        f"""
        <div class="card">
            <div class="kicker">Projected outcome (next {horizon_days} trading days)</div>
            <p class="big">${ending_value:,.2f}</p>
            <div class="delta">P/L: {projected_profit:+,.2f} &nbsp; • &nbsp; Return: {used_return_pct:+.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if used_return_pct > 0:
        st.success("📈 Positive expected return (based on selected return input).")
    elif used_return_pct < 0:
        st.error("📉 Negative expected return (based on selected return input).")
    else:
        st.info("⚪ Flat expected return.")

# ---------------- Geopolitics: lightweight headlines + coherent scoring ----------------
UP_KEYS = [
    "attack", "strike", "drone", "missile", "explosion",
    "pipeline", "refinery", "terminal", "shutdown", "outage",
    "red sea", "houthi", "strait of hormuz", "tanker", "shipping",
    "sanction", "embargo", "escalat", "war", "conflict",
    "opec cuts", "cut output", "production cut", "supply disruption",
]
DOWN_KEYS = [
    "ceasefire", "truce", "deal", "agreement", "talks", "diplom",
    "raise output", "increase output", "production hike", "opec+ hikes",
    "release reserves", "spr release",
    "demand slump", "recession", "weak demand", "slowdown",
]
DRIVER_BUCKETS = {
    "Supply disruptions (infrastructure/shipping)": [
        "pipeline", "refinery", "terminal", "shutdown", "outage", "tanker", "shipping", "red sea", "strait of hormuz"
    ],
    "Conflict escalation / security risk": ["war", "conflict", "attack", "strike", "drone", "missile", "explosion", "escalat"],
    "Sanctions & policy": ["sanction", "embargo"],
    "OPEC policy": ["opec", "opec+", "production cut", "cut output", "increase output", "raise output", "production hike"],
    "Demand macro": ["recession", "weak demand", "slowdown", "demand slump"],
}

def classify_impact(title: str):
    t = (title or "").lower()
    up_hits = sum(1 for k in UP_KEYS if k in t)
    down_hits = sum(1 for k in DOWN_KEYS if k in t)

    if up_hits == 0 and down_hits == 0:
        return ("Unclear", 0.45, "Headline is relevant, but impact depends on details.")
    if up_hits > down_hits:
        conf = min(0.55 + 0.08 * (up_hits - down_hits), 0.85)
        return ("Upward pressure", conf, "More cues for tighter supply / risk premium.")
    if down_hits > up_hits:
        conf = min(0.55 + 0.08 * (down_hits - up_hits), 0.85)
        return ("Downward pressure", conf, "More cues for easing risk / higher supply / weaker demand.")
    return ("Mixed", 0.55, "Contains both tightening and easing cues.")

def extract_drivers(headlines):
    counts = {k: 0 for k in DRIVER_BUCKETS.keys()}
    for h in headlines:
        t = (h.get("title") or "").lower()
        for bucket, keys in DRIVER_BUCKETS.items():
            if any(k in t for k in keys):
                counts[bucket] += 1
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [(k, v) for k, v in top if v > 0][:4]

@st.cache_data(show_spinner=False, ttl=60*30)  # cache for 30 minutes
def fetch_geo_headlines(days: int = 7, max_items: int = 10):
    """
    Fetch recent geopolitics/oil-related headlines from GDELT (lightweight).
    Returns list of dicts: title, url, date, source.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    start_s = start.strftime("%Y%m%d%H%M%S")
    end_s = end.strftime("%Y%m%d%H%M%S")

    query = (
        '(oil OR "crude oil" OR WTI OR Brent OR OPEC OR refinery OR pipeline OR tanker) '
        'AND (sanctions OR war OR conflict OR "Middle East" OR Russia OR Iran OR "Red Sea" '
        'OR "Strait of Hormuz" OR Venezuela OR Libya OR Nigeria)'
    )

    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={requests.utils.quote(query)}"
        f"&mode=artlist&format=json"
        f"&startdatetime={start_s}&enddatetime={end_s}"
        f"&maxrecords={max_items}&sort=hybridrel"
    )

    r = requests.get(url, timeout=12)
    r.raise_for_status()
    j = r.json()
    articles = j.get("articles", []) or []

    out = []
    for a in articles:
        out.append(
            {
                "title": (a.get("title") or "").strip(),
                "url": a.get("url") or "",
                "date": a.get("seendate") or "",
                "source": a.get("sourceCountry") or a.get("sourceCollection") or "",
            }
        )
    return out

def geo_summary(headlines):
    """
    Build:
      - net risk score [-100..100] (positive = upward pressure)
      - counts
      - coherent takeaway
    """
    if not headlines:
        return 0.0, {"up": 0, "down": 0, "mixed": 0, "unclear": 0}, "No headlines loaded."

    score = 0.0
    counts = {"up": 0, "down": 0, "mixed": 0, "unclear": 0}

    for h in headlines:
        direction, conf, _ = classify_impact(h.get("title", ""))
        if direction == "Upward pressure":
            counts["up"] += 1
            score += conf
        elif direction == "Downward pressure":
            counts["down"] += 1
            score -= conf
        elif direction == "Mixed":
            counts["mixed"] += 1
            score += 0.0
        else:
            counts["unclear"] += 1
            score += 0.0

    # Normalize roughly to [-100..100]
    denom = max(len(headlines) * 0.85, 1)
    net = float(np.clip((score / denom) * 100, -100, 100))

    # Coherent takeaway text
    if net >= 25:
        takeaway = "Geopolitical tape is skewed toward **supply risk / tighter supply** → upward pressure bias."
    elif net <= -25:
        takeaway = "Tape is skewed toward **risk easing / higher supply / weaker demand** → downward pressure bias."
    else:
        takeaway = "Geopolitical tape is **mixed** → likely noise-driven; price may be more technical/macro-led."

    return net, counts, takeaway

def pill_for_geo(net_score: float):
    if net_score >= 25:
        return ("pill-up", "RISK BIAS: UP")
    if net_score <= -25:
        return ("pill-down", "RISK BIAS: DOWN")
    return ("pill-neutral", "RISK BIAS: MIXED")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🧪 Backtest", "🧠 Model Inputs"])

# ---------- Tab 1: Overview ----------
with tab1:
    # Top charts row: remove empty space by making the right column fully stacked
    left, right = st.columns([1.55, 1])

    with left:
        st.markdown("### Price")
        fig_price = px.line(df.reset_index(), x="Date", y="Settle", title="WTI Price (daily)")
        fig_price.update_layout(
            template="simple_white",
            paper_bgcolor="#FFFDF6",
            plot_bgcolor="#FFFDF6",
            title_font_size=18,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            margin=dict(l=10, r=10, t=55, b=10),
            height=520,
        )
        fig_price.update_traces(line=dict(width=2))
        st.plotly_chart(fig_price, use_container_width=True)

    with right:
        st.markdown("### Volatility & Returns")
        g = df.reset_index()

        fig_vol = px.line(g, x="Date", y="vol_20d", title="20-day volatility (std)")
        fig_vol.update_layout(
            template="simple_white",
            paper_bgcolor="#FFFDF6",
            plot_bgcolor="#FFFDF6",
            title_font_size=14,
            xaxis_title="Date",
            yaxis_title="Vol",
            margin=dict(l=10, r=10, t=45, b=10),
            height=245,
        )
        fig_vol.update_traces(line=dict(width=2))
        st.plotly_chart(fig_vol, use_container_width=True)

        fig_ret = px.bar(g.tail(120), x="Date", y="ret_1d", title="Daily log returns (last ~6 months)")
        fig_ret.update_layout(
            template="simple_white",
            paper_bgcolor="#FFFDF6",
            plot_bgcolor="#FFFDF6",
            title_font_size=14,
            xaxis_title="Date",
            yaxis_title="Log return",
            margin=dict(l=10, r=10, t=45, b=10),
            height=245,
        )
        st.plotly_chart(fig_ret, use_container_width=True)

        st.markdown("### Participation ratio (thin-move detector)")

if "Volume" not in df.columns or df["Volume"].dropna().empty:
    st.info("Volume isn’t available from the current data source, so the ratio panel is hidden.")
else:
    g = df.reset_index()
    fig_ratio = px.line(
        g,
        x="Date",
        y="move_per_participation",
        title="|Return| per relative volume (higher = thinner move)",
    )
    fig_ratio.update_layout(
        template="simple_white",
        paper_bgcolor="#FFFDF6",
        plot_bgcolor="#FFFDF6",
        title_font_size=14,
        xaxis_title="Date",
        yaxis_title="Ratio",
        margin=dict(l=10, r=10, t=45, b=10),
        height=245,
    )
    fig_ratio.update_traces(line=dict(width=2))
    st.plotly_chart(fig_ratio, use_container_width=True)

    st.markdown("---")

    # Bottom row: geopolitics (left) + summary/drivers (right)
    geo_col, context_col = st.columns([1.35, 1])

    with geo_col:
        st.markdown("### 🌍 Geopolitical events")
        st.caption("Recent headlines + an explainable heuristic about oil price pressure.")

        try:
            headlines = fetch_geo_headlines(days=geo_days, max_items=geo_max_items)
        except Exception:
            headlines = []

        if not headlines:
            st.info("Couldn’t load headlines right now. (API/network). Try again later.")
        else:
            # Render each headline compactly
            for h in headlines:
                direction, conf, reason = classify_impact(h.get("title", ""))
                if direction == "Upward pressure":
                    badge = "<span class='pill pill-up'>UPWARD</span>"
                elif direction == "Downward pressure":
                    badge = "<span class='pill pill-down'>DOWNWARD</span>"
                elif direction == "Mixed":
                    badge = "<span class='pill pill-neutral'>MIXED</span>"
                else:
                    badge = "<span class='pill pill-neutral'>UNCLEAR</span>"

                title = h.get("title", "").strip()
                url = h.get("url", "")
                source = h.get("source", "")
                conf_txt = f"{conf*100:.0f}%"

                st.markdown(
                    f"""
                    <div class="card" style="padding: 12px 12px; margin-bottom: 10px;">
                        <div class="kicker">{badge} &nbsp; Confidence: <b>{conf_txt}</b> &nbsp; · &nbsp; {source}</div>
                        <div style="font-weight: 800; color:#0B1F44; line-height: 1.25;">
                            {f'<a href="{url}" target="_blank" style="text-decoration:none; color:#0B1F44;">{title}</a>' if url else title}
                        </div>
                        <div class="kicker" style="margin-top:6px;">{reason}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with context_col:
        st.markdown("### 🧭 Today’s context")
        st.caption("A coherent summary built from the headlines (not a guarantee).")

        net, counts, takeaway = geo_summary(headlines if "headlines" in locals() else [])
        pill_cls, pill_lbl = pill_for_geo(net)

        st.markdown(
            f"""
            <div class="card">
                <div class="kicker">Geopolitical risk score</div>
                <p class="big">{net:+.0f}</p>
                <div class="delta"><span class="pill {pill_cls}">{pill_lbl}</span></div>
                <div class="kicker" style="margin-top:10px;">
                    Up: <b>{counts['up']}</b> · Down: <b>{counts['down']}</b> · Mixed: <b>{counts['mixed']}</b> · Unclear: <b>{counts['unclear']}</b>
                </div>
                <div style="margin-top:10px; color: rgba(11,31,68,0.92); font-weight:650;">
                    {takeaway}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Driver breakdown
        drivers = extract_drivers(headlines if "headlines" in locals() else [])
        st.markdown("#### Top drivers detected")
        if not drivers:
            st.write("No dominant driver category detected from the current headlines.")
        else:
            for name, ct in drivers:
                st.markdown(f"- **{name}** — {ct} headline(s)")

        # Actionable interpretation
        st.markdown("#### How this could affect oil")
        st.write(
            "- **Supply risk (shipping chokepoints, outages, attacks)** tends to add a *risk premium* → upward pressure.\n"
            "- **Sanctions / escalation** can tighten expected supply or increase uncertainty.\n"
            "- **De-escalation / higher output / demand weakness** can reduce risk premium → downward pressure.\n"
            "- If geopolitics is mixed, price action is often more **technical/macro-led** short-term."
        )

# ---------- Tab 2: Backtest ----------
with tab2:
    st.markdown("### Walk-forward backtest (by year)")
    with st.spinner("Running walk-forward backtest..."):
        bt = walk_forward_backtest(df, start_year=backtest_start_year)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (log return)", f"{bt.mae:.5f}")
    m2.metric("RMSE (log return)", f"{bt.rmse:.5f}")
    m3.metric("Direction accuracy", f"{bt.dir_acc*100:.2f}%")

    bt_df = (
        pd.DataFrame({"truth": bt.truth, "pred": bt.preds})
        .dropna()
        .reset_index()
        .rename(columns={"index": "Date"})
    )

    fig_bt = px.line(bt_df, x="Date", y=["truth", "pred"], title="Truth vs Predicted (1-week log return)")
    fig_bt.update_layout(
        template="simple_white",
        paper_bgcolor="#FFFDF6",
        plot_bgcolor="#FFFDF6",
        title_font_size=18,
        xaxis_title="Date",
        yaxis_title="Log return",
        margin=dict(l=10, r=10, t=55, b=10),
        legend_title_text="",
        height=520,
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    if show_importance:
        st.markdown("### Feature importance")
        st.caption("Permutation importance on the most recent backtest fold (approx).")
        st.dataframe(bt.importance, use_container_width=True)

# ---------- Tab 3: Model inputs ----------
with tab3:
    st.markdown("### Latest model inputs")
    st.caption("These are the feature values used for the most recent forecast.")
    if show_feature_table:
        latest = df[FEATURE_COLS].tail(1).T.rename(columns={df.index[-1]: "latest"})
        st.dataframe(latest, use_container_width=True)

    st.markdown("### Model notes")
    st.write(
        "- Forecast horizon is **next trading week** (5 or 10 trading days).\n"
        "- Signal strength shown is a **proxy** based on forecast magnitude.\n"
        "- Geopolitics panel uses **recent headlines** + a transparent heuristic to summarize possible pressure."
    )