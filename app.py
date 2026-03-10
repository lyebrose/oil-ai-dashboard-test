# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import math
import feedparser

from data import load_wti
from features import add_technical_features, make_targets, build_model_frame
from model import walk_forward_backtest, train_latest_model, FEATURE_COLS

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.ft.com/rss/home",
]

OIL_KEYWORDS = [
    "oil", "crude", "wti", "brent", "opec", "energy", "petroleum",
    "refinery", "pipeline", "tanker", "iran", "russia", "saudi",
    "middle east", "red sea", "houthi", "sanctions", "ukraine",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BarrelX Dashboard",
    page_icon="🛢️",
    layout="wide",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
# BG:        #EEF4FB  (pale sky)
# Surface:   #FFFFFF  (pure white cards)
# Border:    #C8DCF0  (steel-blue border)
# Primary:   #2563EB  (vivid blue)
# Accent:    #38BDF8  (sky-blue highlight)
# Text:      #0F2A4A  (near-black navy)
# Muted:     #6B8DAD  (slate)
# Up:        #0EA5E9  (cyan-blue)
# Down:      #F97316  (amber-orange — warm contrast, not harsh red)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #EEF4FB !important;
    font-family: 'DM Sans', sans-serif;
    color: #0F2A4A;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family: 'DM Sans', sans-serif;
    color: #0F2A4A;
    font-weight: 800;
    letter-spacing: -0.4px;
}

/* ── Cards ── */
.card {
    background: #FFFFFF;
    border: 1.5px solid #C8DCF0;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 2px 12px rgba(37,99,235,0.07), 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease;
}
.card:hover {
    box-shadow: 0 6px 24px rgba(37,99,235,0.12), 0 2px 6px rgba(0,0,0,0.05);
}

.kicker {
    color: #6B8DAD;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}

.big {
    font-size: 1.75rem;
    font-weight: 800;
    color: #0F2A4A;
    margin: 0;
    font-variant-numeric: tabular-nums;
    font-family: 'DM Mono', monospace;
}

.delta {
    font-size: 0.88rem;
    margin-top: 7px;
    font-weight: 600;
    color: #6B8DAD;
}

/* ── Accent bar on cards ── */
.card-accent {
    border-left: 4px solid #38BDF8;
    padding-left: 16px;
}
.card-accent-up   { border-left-color: #0EA5E9; }
.card-accent-down { border-left-color: #F97316; }

/* ── Pills ── */
.pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.pill-up {
    background: rgba(14,165,233,0.12);
    color: #0369A1;
    border: 1.5px solid rgba(14,165,233,0.30);
}
.pill-down {
    background: rgba(249,115,22,0.10);
    color: #C2410C;
    border: 1.5px solid rgba(249,115,22,0.28);
}
.pill-neutral {
    background: rgba(107,141,173,0.12);
    color: #4A6B8A;
    border: 1.5px solid rgba(107,141,173,0.25);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #DAEAF8 !important;
    border-right: 1.5px solid #C8DCF0 !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #0F2A4A !important;
}
section[data-testid="stSidebar"] h3 {
    color: #2563EB !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #38BDF8 100%) !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    border: 0 !important;
    padding: 9px 18px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 16px rgba(37,99,235,0.40) !important;
    transform: translateY(-1px) !important;
}

/* ── DataFrame ── */
div[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1.5px solid #C8DCF0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    border-bottom: 2px solid #C8DCF0;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    color: #6B8DAD !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    background: #FFFFFF !important;
    border-bottom: 2px solid #2563EB !important;
}

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1.5px solid #C8DCF0;
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.06);
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #2563EB, #38BDF8) !important;
    border-radius: 999px !important;
}
.stProgress > div {
    background: #C8DCF0 !important;
    border-radius: 999px !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1.5px solid #C8DCF0 !important;
    margin: 1.5rem 0 !important;
}

/* ── Caption ── */
.stCaption, caption {
    color: #6B8DAD !important;
    font-size: 0.78rem !important;
}

/* ── Alert / info boxes ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
}

/* ── Header bar ── */
header[data-testid="stHeader"] {
    background: #EEF4FB !important;
    border-bottom: 1px solid #C8DCF0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme helper ────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F5F9FE",
    font=dict(family="DM Sans, sans-serif", color="#0F2A4A"),
    xaxis=dict(
        color="#6B8DAD",
        gridcolor="#E2EDF8",
        linecolor="#C8DCF0",
        tickfont=dict(color="#6B8DAD", size=11),
        zerolinecolor="#C8DCF0",
    ),
    yaxis=dict(
        color="#6B8DAD",
        gridcolor="#E2EDF8",
        linecolor="#C8DCF0",
        tickfont=dict(color="#6B8DAD", size=11),
        zerolinecolor="#C8DCF0",
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#C8DCF0",
        borderwidth=1,
        font=dict(color="#0F2A4A", size=11),
    ),
    hoverlabel=dict(
        bgcolor="#0F2A4A",
        bordercolor="#38BDF8",
        font=dict(color="#EEF4FB", family="DM Mono, monospace", size=12),
    ),
    margin=dict(l=12, r=12, t=50, b=12),
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
    <div style="
        background: linear-gradient(135deg, #2563EB 0%, #38BDF8 100%);
        border-radius: 12px;
        width: 42px; height: 42px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem; box-shadow: 0 4px 12px rgba(37,99,235,0.30);
    ">🛢️</div>
    <div>
        <div style="font-size:1.5rem; font-weight:800; color:#0F2A4A; line-height:1.1;">BarrelX</div>
        <div style="font-size:0.78rem; color:#6B8DAD; font-weight:600; letter-spacing:0.4px;">
            AI-POWERED WTI MARKET DASHBOARD
        </div>
    </div>
</div>
<div style="height:3px; background:linear-gradient(90deg,#2563EB,#38BDF8,transparent); border-radius:999px; margin-bottom:1rem;"></div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
last_updated = "loading..."

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    start_date = st.text_input("Start date (YYYY-MM-DD)", "2023-01-01")
    horizon_days = st.selectbox("Forecast horizon (trading days)", [5, 10], index=0)
    backtest_start_year = st.slider("Backtest start year", 2012, 2025, 2018)

    st.markdown("---")
    st.markdown("### 🖥️ Display")
    show_feature_table = st.checkbox("Show latest feature table", value=True)
    show_importance = st.checkbox("Show feature importance", value=True)
    geo_days = st.selectbox("Geopolitics lookback (days)", [3, 7, 14], index=1)
    geo_max_items = st.selectbox("Max headlines", [6, 8, 10, 12], index=2)

    st.markdown("---")
    st.markdown("### 📊 Position sizing")
    use_custom_return = st.checkbox("Use custom predicted return (%)", value=False)

    st.markdown("---")
    st.caption(f"Data last updated: **{last_updated}**")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

# ── Outside sidebar: trade simulator inputs ───────────────────────────────────
sim_mode = st.selectbox(
    "Simulator mode",
    ["Percent exposure (simple)", "WTI futures (CL)"],
    index=0,
)
amount = st.number_input("Capital ($)", min_value=0.0, value=10000.0, step=500.0)

use_custom_return = st.checkbox("Override with custom predicted return (%)", value=False)
custom_return_pct = None
if use_custom_return:
    custom_return_pct = st.number_input("Custom predicted return (%)", value=0.0, step=0.25)

contracts = 1
margin_per_contract = 8000.0
if sim_mode == "WTI futures (CL)":
    contracts = st.number_input("Contracts (CL)", min_value=0, value=1, step=1)
    margin_per_contract = st.number_input("Margin per contract ($)", min_value=0.0, value=8000.0, step=250.0)

# ── Data build ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def build_dataset(start_date: str, horizon_days: int):
    df = load_wti(start=start_date)
    df = add_technical_features(df)
    df = make_targets(df, horizon_days=horizon_days)
    df = build_model_frame(df)
    return df, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

df, last_updated = build_dataset(start_date, horizon_days)

# ── Model ─────────────────────────────────────────────────────────────────────
model, latest_pred = train_latest_model(df)
pred_pct = (np.exp(latest_pred) - 1) * 100
pred_dir = "UP" if latest_pred > 0 else "DOWN"

# ── KPI values ────────────────────────────────────────────────────────────────
latest_price = float(df["Settle"].iloc[-1])
week_change = float((df["Settle"].iloc[-1] / df["Settle"].iloc[-6] - 1) * 100) if len(df) > 6 else float("nan")
risk = float(df["vol_20d"].iloc[-1]) if "vol_20d" in df.columns else float("nan")
vol_label = "⚠️ High" if risk >= 0.03 else "✅ Normal"
signal_strength = min(abs(pred_pct) / 3.0, 1.0)

pill_class = "pill-up" if pred_dir == "UP" else "pill-down"
pill_text = f"<span class='pill {pill_class}'>{'▲' if pred_dir=='UP' else '▼'} {pred_dir}</span>"

wk_color = "#0EA5E9" if week_change >= 0 else "#F97316"
wk_arrow = "▲" if week_change >= 0 else "▼"

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1.25, 1.25, 1.0, 1.1])

with c1:
    st.markdown(f"""
    <div class="card card-accent" style="border-left-color:#2563EB;">
        <div class="kicker">WTI Settlement Price</div>
        <p class="big">${latest_price:,.2f}</p>
        <div class="delta" style="color:{wk_color};">{wk_arrow} {abs(week_change):.2f}% past week</div>
    </div>""", unsafe_allow_html=True)

with c2:
    acc = "card-accent-up" if pred_dir == "UP" else "card-accent-down"
    st.markdown(f"""
    <div class="card card-accent {acc}">
        <div class="kicker">Forecast · Next {horizon_days} trading days</div>
        <p class="big">{pred_pct:+.2f}%</p>
        <div class="delta">{pill_text}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    risk_color = "#F97316" if risk >= 0.03 else "#0EA5E9"
    st.markdown(f"""
    <div class="card card-accent" style="border-left-color:{risk_color};">
        <div class="kicker">20-day Volatility</div>
        <p class="big" style="font-size:1.4rem;">{risk:.4f}</p>
        <div class="delta">{vol_label}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card card-accent" style="border-left-color:#38BDF8;">
        <div class="kicker">Signal Strength</div>
        <p class="big" style="font-size:1.4rem;">{signal_strength*100:.0f}%</p>
        <div class="delta">Based on |forecast| magnitude</div>
    </div>""", unsafe_allow_html=True)
    st.progress(signal_strength)

st.markdown("---")

# ── Trade simulator ───────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.5rem;">
    <div style="width:4px;height:22px;background:linear-gradient(180deg,#2563EB,#38BDF8);border-radius:2px;"></div>
    <h3 style="margin:0;">Trade Simulator</h3>
</div>""", unsafe_allow_html=True)

used_return_pct = float(custom_return_pct) if (use_custom_return and custom_return_pct is not None) else float(pred_pct)
used_logret = np.log1p(used_return_pct / 100.0)
price_now = float(df["Settle"].iloc[-1])
daily_vol = float(df["vol_20d"].iloc[-1]) if "vol_20d" in df.columns and pd.notna(df["vol_20d"].iloc[-1]) else 0.0
sigma_h = daily_vol * np.sqrt(max(int(horizon_days), 1))

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

mu = used_logret
r_1s_low,  r_1s_high  = mu - sigma_h,     mu + sigma_h
r_2s_low,  r_2s_high  = mu - 2*sigma_h,   mu + 2*sigma_h

price_exp     = price_now * np.exp(mu)
price_1s_low  = price_now * np.exp(r_1s_low)
price_1s_high = price_now * np.exp(r_1s_high)
price_2s_low  = price_now * np.exp(r_2s_low)
price_2s_high = price_now * np.exp(r_2s_high)

p_profit = (1.0 - norm_cdf(-mu / sigma_h)) if sigma_h > 0 else (1.0 if mu > 0 else 0.0)

CONTRACT_SIZE_BBL = 1000.0

def pl_percent(capital, r_log): return capital * (np.exp(r_log) - 1.0)
def pl_futures(n, p0, r_log):   return (p0 * np.exp(r_log) - p0) * CONTRACT_SIZE_BBL * float(n)

if sim_mode == "Percent exposure (simple)":
    pl_exp     = pl_percent(amount, mu)
    pl_1s_low  = pl_percent(amount, r_1s_low)
    pl_1s_high = pl_percent(amount, r_1s_high)
    pl_2s_low  = pl_percent(amount, r_2s_low)
    pl_2s_high = pl_percent(amount, r_2s_high)
    notional, leverage_label, margin_used = amount, "1.0× (capital exposure)", 0.0
else:
    c = int(contracts)
    pl_exp     = pl_futures(c, price_now, mu)
    pl_1s_low  = pl_futures(c, price_now, r_1s_low)
    pl_1s_high = pl_futures(c, price_now, r_1s_high)
    pl_2s_low  = pl_futures(c, price_now, r_2s_low)
    pl_2s_high = pl_futures(c, price_now, r_2s_high)
    notional     = price_now * CONTRACT_SIZE_BBL * float(c)
    margin_used  = float(c) * float(margin_per_contract)
    leverage_label = f"{(notional/amount if amount>0 else 0):.2f}× notional/capital"

sim_left, sim_right = st.columns([1.25, 1])

with sim_left:
    st.markdown(f"""
    <div class="card">
        <div class="kicker">Forecast summary · next {horizon_days} trading days</div>
        <p class="big">{used_return_pct:+.2f}%</p>
        <div class="delta">Implied price: <b>${price_exp:,.2f}</b> from <b>${price_now:,.2f}</b></div>
        <div style="margin-top:10px;display:flex;gap:20px;flex-wrap:wrap;">
            <span class="kicker">Daily σ <b style="color:#0F2A4A;">{daily_vol:.4f}</b></span>
            <span class="kicker">Horizon σ <b style="color:#0F2A4A;">{sigma_h:.4f}</b></span>
            <span class="kicker">P(profit) <b style="color:#0EA5E9;">{p_profit*100:.1f}%</b></span>
        </div>
    </div>""", unsafe_allow_html=True)

    bands = pd.DataFrame({
        "Scenario": ["Expected", "1σ low", "1σ high", "2σ low", "2σ high"],
        "Price ($)": [price_exp, price_1s_low, price_1s_high, price_2s_low, price_2s_high],
        "P/L ($)":   [pl_exp,    pl_1s_low,    pl_1s_high,    pl_2s_low,    pl_2s_high],
    })
    st.markdown("##### Risk bands (volatility-based)")
    st.dataframe(bands.style.format({"Price ($)": "${:,.2f}", "P/L ($)": "{:+,.2f}"}), use_container_width=True)

with sim_right:
    if sim_mode == "WTI futures (CL)":
        margin_ok = amount >= margin_used
        status_color = "#0EA5E9" if margin_ok else "#F97316"
        status_txt   = "✅ OK" if margin_ok else "⚠️ Under-margined"
        st.markdown(f"""
        <div class="card card-accent" style="border-left-color:{status_color};">
            <div class="kicker">Futures position</div>
            <p class="big">{int(contracts)} contract(s)</p>
            <div class="delta">Notional: ${notional:,.0f}</div>
            <div style="margin-top:10px;font-size:0.82rem;color:#6B8DAD;line-height:1.7;">
                Margin (assumed): <b style="color:#0F2A4A;">${margin_used:,.0f}</b> · {status_txt}<br>
                Leverage: <b style="color:#0F2A4A;">{leverage_label}</b><br>
                Contract size: <b style="color:#0F2A4A;">1,000 bbl</b>
            </div>
        </div>""", unsafe_allow_html=True)
        st.caption("Margin assumptions only — verify with your broker.")
    else:
        st.markdown(f"""
        <div class="card card-accent" style="border-left-color:#2563EB;">
            <div class="kicker">Capital exposure</div>
            <p class="big">${amount:,.0f}</p>
            <div class="delta">Return applied directly to capital</div>
            <div style="margin-top:10px;font-size:0.82rem;color:#6B8DAD;">
                Leverage: <b style="color:#0F2A4A;">{leverage_label}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    if pl_exp > 0:
        st.success(f"📈 Expected P/L: **+${pl_exp:,.2f}**")
    elif pl_exp < 0:
        st.error(f"📉 Expected P/L: **${pl_exp:,.2f}**")
    else:
        st.info("⚪ Expected P/L near zero.")
    st.caption("Normal approximation · σ from 20-day realized vol (log returns)")

st.markdown("---")

# ── What-if calculator ────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.5rem;">
    <div style="width:4px;height:22px;background:linear-gradient(180deg,#38BDF8,#2563EB);border-radius:2px;"></div>
    <h3 style="margin:0;">What-if Return Calculator</h3>
</div>""", unsafe_allow_html=True)

calc_left, calc_right = st.columns([1.2, 1])

with calc_left:
    calc_amount = st.number_input("Amount to invest ($)", min_value=0.0, value=10000.0, step=500.0)
    if use_custom_return:
        custom_return = st.number_input("Predicted return (%)", value=float(pred_pct), step=0.25)
        calc_return_pct = float(custom_return)
        source_label = "Custom input"
    else:
        calc_return_pct = float(pred_pct)
        source_label = "Model forecast"
    st.caption(f"Return source: **{source_label}**")

with calc_right:
    projected_profit = calc_amount * (calc_return_pct / 100.0)
    ending_value = calc_amount + projected_profit
    pnl_color = "#0EA5E9" if projected_profit >= 0 else "#F97316"
    st.markdown(f"""
    <div class="card card-accent" style="border-left-color:{pnl_color};">
        <div class="kicker">Projected outcome · next {horizon_days} trading days</div>
        <p class="big">${ending_value:,.2f}</p>
        <div class="delta" style="color:{pnl_color};">
            P/L: {projected_profit:+,.2f} &nbsp;·&nbsp; Return: {calc_return_pct:+.2f}%
        </div>
    </div>""", unsafe_allow_html=True)

# ── Geo helpers ───────────────────────────────────────────────────────────────
UP_KEYS = [
    "attack","strike","drone","missile","explosion","pipeline","refinery",
    "terminal","shutdown","outage","red sea","houthi","strait of hormuz",
    "tanker","shipping","sanction","embargo","escalat","war","conflict",
    "opec cuts","cut output","production cut","supply disruption",
]
DOWN_KEYS = [
    "ceasefire","truce","deal","agreement","talks","diplom","raise output",
    "increase output","production hike","opec+ hikes","release reserves",
    "spr release","demand slump","recession","weak demand","slowdown",
]
DRIVER_BUCKETS = {
    "Supply disruptions": ["pipeline","refinery","terminal","shutdown","outage","tanker","shipping","red sea","strait of hormuz"],
    "Conflict / security risk": ["war","conflict","attack","strike","drone","missile","explosion","escalat"],
    "Sanctions & policy": ["sanction","embargo"],
    "OPEC policy": ["opec","opec+","production cut","cut output","increase output","raise output","production hike"],
    "Demand macro": ["recession","weak demand","slowdown","demand slump"],
}

def classify_impact(title):
    t = (title or "").lower()
    up   = sum(1 for k in UP_KEYS   if k in t)
    down = sum(1 for k in DOWN_KEYS if k in t)
    if up == 0 and down == 0:
        return "Unclear", 0.45, "Impact unclear from headline alone."
    if up > down:
        return "Upward pressure",   min(0.55+0.08*(up-down), 0.85),   "Cues for tighter supply / risk premium."
    if down > up:
        return "Downward pressure", min(0.55+0.08*(down-up), 0.85),   "Cues for easing risk / higher supply / weaker demand."
    return "Mixed", 0.55, "Both tightening and easing cues present."

def extract_drivers(headlines):
    counts = {k: 0 for k in DRIVER_BUCKETS}
    for h in headlines:
        t = (h.get("title") or "").lower()
        for bucket, keys in DRIVER_BUCKETS.items():
            if any(k in t for k in keys):
                counts[bucket] += 1
    return [(k, v) for k, v in sorted(counts.items(), key=lambda x: -x[1]) if v > 0][:4]

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_geo_headlines(days=7, max_items=10):
    cutoff = datetime.utcnow() - timedelta(days=days)
    out = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title = (entry.get("title") or "").strip()
                if not title or not any(kw in title.lower() for kw in OIL_KEYWORDS):
                    continue
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    pub_dt = datetime(*published[:6])
                    if pub_dt < cutoff:
                        continue
                    date_str = pub_dt.strftime("%Y-%m-%d %H:%M")
                else:
                    date_str = ""
                out.append({"title": title, "url": entry.get("link",""), "date": date_str,
                             "source": feed.feed.get("title") or feed_url})
        except Exception as e:
            st.warning(f"RSS fetch failed for {feed_url}: {e}")
    seen, deduped = set(), []
    for h in sorted(out, key=lambda x: x["date"], reverse=True):
        if h["title"] not in seen:
            seen.add(h["title"]); deduped.append(h)
        if len(deduped) >= max_items:
            break
    return deduped

def geo_summary(headlines):
    if not headlines:
        return 0.0, {"up":0,"down":0,"mixed":0,"unclear":0}, "No headlines loaded."
    score, counts = 0.0, {"up":0,"down":0,"mixed":0,"unclear":0}
    for h in headlines:
        d, conf, _ = classify_impact(h.get("title",""))
        if d == "Upward pressure":   counts["up"]    += 1; score += conf
        elif d == "Downward pressure": counts["down"] += 1; score -= conf
        elif d == "Mixed":             counts["mixed"] += 1
        else:                          counts["unclear"] += 1
    net = float(np.clip((score / max(len(headlines)*0.85,1))*100, -100, 100))
    if net >= 25:  takeaway = "Tape skewed toward **supply risk / tighter supply** → upward pressure bias."
    elif net <= -25: takeaway = "Tape skewed toward **risk easing / higher supply** → downward pressure bias."
    else:            takeaway = "Tape is **mixed** → price likely more technical/macro-led short-term."
    return net, counts, takeaway

def pill_for_geo(s):
    if s >= 25:  return "pill-up",   "RISK BIAS: UP"
    if s <= -25: return "pill-down", "RISK BIAS: DOWN"
    return "pill-neutral", "RISK BIAS: MIXED"

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🧪 Backtest", "🧠 Model Inputs"])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    left, right = st.columns([1.55, 1])

    with left:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <div style="width:4px;height:18px;background:linear-gradient(180deg,#2563EB,#38BDF8);border-radius:2px;"></div>
            <span style="font-weight:700;color:#0F2A4A;font-size:1rem;">WTI Price History</span>
        </div>""", unsafe_allow_html=True)

        df_plot = df.reset_index().copy()
        df_plot["ma_30"] = df_plot["Settle"].rolling(30).mean()

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["Settle"],
            name="WTI Price", line=dict(color="#2563EB", width=2),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
        ))
        fig_price.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["ma_30"],
            name="30-day MA", line=dict(color="#38BDF8", width=1.5, dash="dot"),
        ))
        fig_price.add_hline(
            y=latest_price,
            line_dash="dot", line_color="rgba(249,115,22,0.6)",
            annotation_text=f"  ${latest_price:,.2f}",
            annotation_position="top left",
            annotation_font_color="#F97316",
        )
        fig_price.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="WTI Crude Oil — Daily Settlement", font=dict(size=14, color="#0F2A4A"), x=0),
            height=500,
            hovermode="x unified",
            legend=dict(**PLOT_LAYOUT["legend"], orientation="h", y=1.08, x=0),
            margin=dict(l=12, r=12, t=55, b=12),
        )
        st.plotly_chart(fig_price, use_container_width=True)
        st.caption(f"Last updated: **{last_updated}** · Auto-refreshes every 24 h · Dotted orange = latest close")

    with right:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <div style="width:4px;height:18px;background:linear-gradient(180deg,#38BDF8,#2563EB);border-radius:2px;"></div>
            <span style="font-weight:700;color:#0F2A4A;font-size:1rem;">Price Forecast — Next {h} Days</span>
        </div>""".replace("{h}", str(horizon_days)), unsafe_allow_html=True)

        last_date = df.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)
        daily_log_step = latest_pred / horizon_days
        forecast_prices = [price_now * np.exp(daily_log_step * i) for i in range(1, horizon_days + 1)]

        sigma_steps = [daily_vol * np.sqrt(i) for i in range(1, horizon_days + 1)]
        upper_1 = [p * np.exp(s)    for p, s in zip(forecast_prices, sigma_steps)]
        lower_1 = [p * np.exp(-s)   for p, s in zip(forecast_prices, sigma_steps)]
        upper_2 = [p * np.exp(2*s)  for p, s in zip(forecast_prices, sigma_steps)]
        lower_2 = [p * np.exp(-2*s) for p, s in zip(forecast_prices, sigma_steps)]

        history_tail = df["Settle"].tail(20).reset_index()
        history_tail.columns = ["Date", "Price"]

        anchor_dates  = [last_date] + list(forecast_dates)
        anchor_prices = [price_now] + forecast_prices

        fig_fwd = go.Figure()

        # 2σ band
        fig_fwd.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=upper_2 + lower_2[::-1],
            fill="toself", fillcolor="rgba(37,99,235,0.06)",
            line=dict(color="rgba(0,0,0,0)"), name="2σ range", hoverinfo="skip",
        ))
        # 1σ band
        fig_fwd.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=upper_1 + lower_1[::-1],
            fill="toself", fillcolor="rgba(37,99,235,0.13)",
            line=dict(color="rgba(0,0,0,0)"), name="1σ range", hoverinfo="skip",
        ))
        # Historical
        fig_fwd.add_trace(go.Scatter(
            x=history_tail["Date"], y=history_tail["Price"],
            name="Historical", line=dict(color="#2563EB", width=2),
        ))
        # Expected path
        fig_fwd.add_trace(go.Scatter(
            x=anchor_dates, y=anchor_prices,
            name="Expected path",
            mode="lines+markers",
            line=dict(color="#F97316", width=2, dash="dash"),
            marker=dict(size=5, color="#F97316",
                        line=dict(color="#FFFFFF", width=1.5)),
        ))

        # Today line via shape + annotation
        fig_fwd.add_shape(
            type="line",
            x0=str(last_date), x1=str(last_date),
            y0=0, y1=1, yref="paper",
            line=dict(dash="dot", color="rgba(107,141,173,0.5)", width=1.5),
        )
        fig_fwd.add_annotation(
            x=str(last_date), y=0.97, yref="paper",
            text="Today", showarrow=False,
            font=dict(color="#6B8DAD", size=11, family="DM Sans"),
            xanchor="left", bgcolor="rgba(238,244,251,0.8)",
            bordercolor="#C8DCF0", borderwidth=1, borderpad=3,
        )

        fig_fwd.update_layout(
            **PLOT_LAYOUT,
            title=dict(text=f"Forecast cone · {horizon_days}-day horizon", font=dict(size=14, color="#0F2A4A"), x=0),
            height=500,
            hovermode="x unified",
            legend=dict(**PLOT_LAYOUT["legend"], orientation="h", y=1.08, x=0),
            margin=dict(l=12, r=12, t=55, b=12),
        )
        st.plotly_chart(fig_fwd, use_container_width=True)
        st.caption(f"Orange dashed = expected path · Bands = 1σ / 2σ volatility cone · Daily σ: {daily_vol:.4f}")

    st.markdown("---")

    # ── Geopolitics ───────────────────────────────────────────────────────────
    geo_col, context_col = st.columns([1.35, 1])

    with geo_col:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <div style="width:4px;height:18px;background:linear-gradient(180deg,#2563EB,#38BDF8);border-radius:2px;"></div>
            <span style="font-weight:700;color:#0F2A4A;font-size:1rem;">🌍 Geopolitical Headlines</span>
        </div>""", unsafe_allow_html=True)
        st.caption("Recent oil-relevant headlines · heuristic impact scoring")

        try:
            headlines = fetch_geo_headlines(days=geo_days, max_items=geo_max_items)
        except Exception:
            headlines = []

        if not headlines:
            st.info("Couldn't load headlines right now. Try again later.")
        else:
            for h in headlines:
                direction, conf, reason = classify_impact(h.get("title",""))
                if direction == "Upward pressure":
                    badge = "<span class='pill pill-up'>▲ UPWARD</span>"
                    border = "#C8DCF0"
                elif direction == "Downward pressure":
                    badge = "<span class='pill pill-down'>▼ DOWNWARD</span>"
                    border = "#FED7AA"
                elif direction == "Mixed":
                    badge = "<span class='pill pill-neutral'>↔ MIXED</span>"
                    border = "#C8DCF0"
                else:
                    badge = "<span class='pill pill-neutral'>? UNCLEAR</span>"
                    border = "#C8DCF0"

                title = h.get("title","").strip()
                url   = h.get("url","")
                source = h.get("source","")
                link_html = f'<a href="{url}" target="_blank" style="text-decoration:none;color:#0F2A4A;">{title}</a>' if url else title

                st.markdown(f"""
                <div class="card" style="padding:12px 14px;margin-bottom:10px;border-left:3px solid {border};">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        {badge}
                        <span class="kicker" style="margin:0;">{source} · {conf*100:.0f}% conf</span>
                    </div>
                    <div style="font-weight:700;color:#0F2A4A;font-size:0.88rem;line-height:1.35;">{link_html}</div>
                    <div class="kicker" style="margin-top:5px;margin-bottom:0;">{reason}</div>
                </div>""", unsafe_allow_html=True)

    with context_col:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <div style="width:4px;height:18px;background:linear-gradient(180deg,#38BDF8,#2563EB);border-radius:2px;"></div>
            <span style="font-weight:700;color:#0F2A4A;font-size:1rem;">🧭 Today's Context</span>
        </div>""", unsafe_allow_html=True)
        st.caption("Coherent summary from headlines — not a guarantee")

        net, counts, takeaway = geo_summary(headlines if "headlines" in locals() else [])
        pill_cls, pill_lbl = pill_for_geo(net)
        score_color = "#0EA5E9" if net >= 25 else ("#F97316" if net <= -25 else "#6B8DAD")

        st.markdown(f"""
        <div class="card card-accent" style="border-left-color:{score_color};">
            <div class="kicker">Geopolitical Risk Score</div>
            <p class="big" style="color:{score_color};">{net:+.0f}</p>
            <div class="delta"><span class="pill {pill_cls}">{pill_lbl}</span></div>
            <div style="margin-top:12px;display:flex;gap:16px;flex-wrap:wrap;">
                <span class="kicker">Up <b style="color:#0EA5E9;">{counts['up']}</b></span>
                <span class="kicker">Down <b style="color:#F97316;">{counts['down']}</b></span>
                <span class="kicker">Mixed <b style="color:#0F2A4A;">{counts['mixed']}</b></span>
                <span class="kicker">Unclear <b style="color:#0F2A4A;">{counts['unclear']}</b></span>
            </div>
            <div style="margin-top:10px;font-size:0.85rem;color:#0F2A4A;font-weight:500;line-height:1.5;">{takeaway}</div>
        </div>""", unsafe_allow_html=True)

        drivers = extract_drivers(headlines if "headlines" in locals() else [])
        st.markdown("##### Top drivers detected")
        if not drivers:
            st.write("No dominant driver category in current headlines.")
        else:
            for name, ct in drivers:
                bar_w = int(ct / max(d[1] for d in drivers) * 100)
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                        <span style="font-size:0.82rem;font-weight:600;color:#0F2A4A;">{name}</span>
                        <span style="font-size:0.78rem;color:#6B8DAD;">{ct} headline(s)</span>
                    </div>
                    <div style="background:#E2EDF8;border-radius:999px;height:5px;">
                        <div style="background:linear-gradient(90deg,#2563EB,#38BDF8);width:{bar_w}%;height:5px;border-radius:999px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("##### Price impact guide")
        for icon, text in [
            ("🔴", "**Supply risk** (chokepoints, outages, attacks) → risk premium → upward"),
            ("🟠", "**Sanctions / escalation** → tighter expected supply → upward"),
            ("🟢", "**De-escalation / higher output / demand weakness** → downward"),
            ("⚪", "**Mixed tape** → price often more technical/macro-led"),
        ]:
            st.markdown(f"{icon} {text}")

# ── Tab 2: Backtest ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.5rem;">
        <div style="width:4px;height:22px;background:linear-gradient(180deg,#2563EB,#38BDF8);border-radius:2px;"></div>
        <h3 style="margin:0;">Walk-forward Backtest</h3>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Running walk-forward backtest..."):
        bt = walk_forward_backtest(df, start_year=backtest_start_year)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (log return)",   f"{bt.mae:.5f}")
    m2.metric("RMSE (log return)",  f"{bt.rmse:.5f}")
    m3.metric("Direction accuracy", f"{bt.dir_acc*100:.2f}%")

    bt_df = (pd.DataFrame({"truth": bt.truth, "pred": bt.preds})
             .dropna().reset_index().rename(columns={"index": "Date"}))

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["truth"],
                                name="Truth", line=dict(color="#2563EB", width=1.5)))
    fig_bt.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["pred"],
                                name="Predicted", line=dict(color="#F97316", width=1.5, dash="dot")))
    fig_bt.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Truth vs Predicted — 1-week log return", font=dict(size=14, color="#0F2A4A"), x=0),
        height=480, hovermode="x unified",
        legend=dict(**PLOT_LAYOUT["legend"], orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    if show_importance:
        st.markdown("##### Feature importance (permutation, last fold)")
        st.dataframe(bt.importance, use_container_width=True)

# ── Tab 3: Model inputs ───────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.5rem;">
        <div style="width:4px;height:22px;background:linear-gradient(180deg,#2563EB,#38BDF8);border-radius:2px;"></div>
        <h3 style="margin:0;">Latest Model Inputs</h3>
    </div>""", unsafe_allow_html=True)
    st.caption("Feature values used for the most recent forecast.")

    if show_feature_table:
        latest = df[FEATURE_COLS].tail(1).T.rename(columns={df.index[-1]: "latest"})
        st.dataframe(latest, use_container_width=True)

    st.markdown("##### Model notes")
    st.markdown("""
    <div class="card" style="font-size:0.88rem;line-height:1.7;color:#0F2A4A;">
        <b>Forecast horizon:</b> next 5 or 10 trading days (configurable in sidebar)<br>
        <b>Signal strength:</b> proxy based on forecast magnitude (capped at 3%)<br>
        <b>Volatility cone:</b> 20-day realized vol scaled by √t to each horizon step<br>
        <b>Geopolitics:</b> RSS headline heuristic — transparent keyword scoring, not ML<br>
        <b>Disclaimer:</b> Educational tool only. Not financial advice.
    </div>""", unsafe_allow_html=True)
