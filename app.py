# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime, timedelta

from data import load_wti
from features import add_technical_features, make_targets, build_model_frame
from model import walk_forward_backtest, train_latest_model, FEATURE_COLS

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
.block-container { padding-top: 1.1rem; padding-bottom: 1.2rem; max-width: 1400px; }

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
    "<div class='kicker'>1-week horizon • Technical model • Geopolitical context • Educational decision support (not financial advice)</div>",
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

    st.markdown("---")
    st.markdown("### Geopolitics panel")
    geo_days = st.selectbox("Lookback (days)", [3, 7, 14], index=1)
    geo_max_items = st.selectbox("Max headlines", [6, 8, 10, 12], index=2)

    st.markdown("---")
    st.markdown("### Chart smoothing")
    show_ma_overlay = st.checkbox("Show 20d moving average overlay", value=True)
    smooth_lines = st.checkbox("Use smooth curve rendering", value=True)
    show_fill = st.checkbox("Subtle fill under price", value=True)


# ---------------- Data build ----------------
@st.cache_data(show_spinner=False)
def build_dataset(start_date: str, horizon_days: int) -> pd.DataFrame:
    df = load_wti(start=start_date)
    df = add_technical_features(df)
    df = make_targets(df, horizon_days=horizon_days)
    df = build_model_frame(df)
    return df


df = build_dataset(start_date, horizon_days)

# Extra plotting columns (design / readability)
df["ma_20_plot"] = df["Settle"].rolling(20).mean()
df["ma_50_plot"] = df["Settle"].rolling(50).mean()

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

# Volatility regime (more nuanced than a single cutoff)
if risk < 0.02:
    vol_label = "Low"
elif risk < 0.035:
    vol_label = "Normal"
elif risk < 0.05:
    vol_label = "Elevated"
else:
    vol_label = "Extreme"

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

# ---------------- Geopolitics: headlines + scoring ----------------
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


@st.cache_data(show_spinner=False, ttl=60 * 30)  # cache for 30 minutes
def fetch_geo_headlines(days: int = 7, max_items: int = 10):
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
        else:
            counts["unclear"] += 1

    denom = max(len(headlines) * 0.85, 1)
    net = float(np.clip((score / denom) * 100, -100, 100))

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
    left, right = st.columns([1.55, 1])

    # ---- Price chart (smoother design) ----
    with left:
        st.markdown("### Price")

        plot_df = df.reset_index().copy()
        plot_series = ["Settle"]
        if show_ma_overlay:
            plot_series.append("ma_20_plot")

        fig_price = px.line(
            plot_df,
            x="Date",
            y=plot_series,
            title="WTI Price (Daily) + 20D Moving Average",
        )

        # Style traces: make raw price nicer + optional smoothing + subtle fill
        # Plotly assigns names equal to column names
        # Price trace
        price_shape = "spline" if smooth_lines else "linear"
        fig_price.update_traces(
            selector=dict(name="Settle"),
            line=dict(width=3, shape=price_shape, smoothing=0.45 if smooth_lines else 0),
        )

        if show_fill:
            fig_price.update_traces(
                selector=dict(name="Settle"),
                fill="tozeroy",
                fillcolor="rgba(30,77,183,0.08)",
            )

        # MA trace
        if show_ma_overlay:
            fig_price.update_traces(
                selector=dict(name="ma_20_plot"),
                line=dict(width=3, dash="dash", shape="spline", smoothing=0.6),
                opacity=0.9,
            )

        # Calm layout (less erratic feel)
        fig_price.update_layout(
            template="simple_white",
            paper_bgcolor="#FFFDF6",
            plot_bgcolor="#FFFDF6",
            title_font_size=18,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            margin=dict(l=10, r=10, t=55, b=10),
            height=520,
            legend_title_text="",
        )
        fig_price.update_xaxes(showgrid=False, zeroline=False)
        fig_price.update_yaxes(showgrid=True, gridcolor="rgba(16,42,67,0.08)", zeroline=False)

        st.plotly_chart(fig_price, use_container_width=True)

    # ---- Volatility + Returns (no empty gap; stacked charts fill column) ----
    with right:
        st.markdown("### Volatility & Returns")
        g = df.reset_index()

        # Vol chart smoother line + calmer grid
        fig_vol = px.line(g, x="Date", y="vol_20d", title="20-day volatility (std of daily log returns)")
        fig_vol.update_traces(line=dict(width=2.6, shape="spline", smoothing=0.55))
        fig_vol.update_layout(
            template="simple_white",
            paper_bgcolor="#FFFDF6",
            plot_bgcolor="#FFFDF6",
            title_font_size=14,
            xaxis_title="Date",
            yaxis_title="Vol",
            margin=dict(l=10, r=10, t=45, b=10),
            height=245,
            legend_title_text="",
        )
        fig_vol.update_xaxes(showgrid=False, zeroline=False)
        fig_vol.update_yaxes(showgrid=True, gridcolor="rgba(16,42,67,0.08)", zeroline=False)
        st.plotly_chart(fig_vol, use_container_width=True)

        # Returns bars - clean formatting, a bit more history
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
        fig_ret.update_xaxes(showgrid=False, zeroline=False)
        fig_ret.update_yaxes(showgrid=True, gridcolor="rgba(16,42,67,0.08)", zeroline=False)
        st.plotly_chart(fig_ret, use_container_width=True)

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

        drivers = extract_drivers(headlines if "headlines" in locals() else [])
        st.markdown("#### Top drivers detected")
        if not drivers:
            st.write("No dominant driver category detected from the current headlines.")
        else:
            for name, ct in drivers:
                st.markdown(f"- **{name}** — {ct} headline(s)")

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
    fig_bt.update_traces(line=dict(width=2.5, shape="spline", smoothing=0.5))
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
    fig_bt.update_xaxes(showgrid=False, zeroline=False)
    fig_bt.update_yaxes(showgrid=True, gridcolor="rgba(16,42,67,0.08)", zeroline=False)
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
        "- The price line uses optional **spline rendering** + an optional **moving average overlay** (visual smoothing only).\n"
        "- Geopolitics panel uses **recent headlines** + a transparent heuristic to summarize possible pressure."
    )