# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from data import load_wti
from features import add_technical_features, make_targets, build_model_frame
from model import walk_forward_backtest, train_latest_model, FEATURE_COLS

# ---------------- Page config ----------------
st.set_page_config(
    page_title="WTI AI Dashboard",
    page_icon="🛢️",
    layout="wide",
)

# ---------------- CSS for sleek blue + cream ----------------
st.markdown(
    """
<style>
/* Cream background consistency */
.stApp {
    background: #FBF6EC;
}

/* Reduce top padding */
.block-container {
    padding-top: 1.2rem;
}

/* Headings */
h1, h2, h3, h4 {
    color: #0B1F44;
    letter-spacing: -0.2px;
}

/* A “card” look */
.card {
    background: #FFFDF6;
    border: 1px solid rgba(16,42,67,0.10);
    border-radius: 18px;
    padding: 16px 16px;
    box-shadow: 0 6px 18px rgba(16,42,67,0.07);
}

/* Small label */
.kicker {
    color: rgba(16,42,67,0.65);
    font-size: 0.85rem;
    margin-bottom: 4px;
}

/* Big number */
.big {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0B1F44;
    margin: 0;
}

/* Delta */
.delta {
    font-size: 0.95rem;
    margin-top: 4px;
    font-weight: 600;
}

/* Up / Down pill */
.pill {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.2px;
}
.pill-up {
    background: rgba(22, 163, 74, 0.12);
    color: #166534;
    border: 1px solid rgba(22, 163, 74, 0.25);
}
.pill-down {
    background: rgba(220, 38, 38, 0.10);
    color: #7F1D1D;
    border: 1px solid rgba(220, 38, 38, 0.22);
}

/* Blue accent divider */
hr {
    border: none;
    border-top: 1px solid rgba(30,77,183,0.18);
    margin: 1.0rem 0;
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
    font-weight: 700 !important;
}
.stButton button:hover {
    background: #173E93 !important;
}

/* Dataframe styling */
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

# ---------------- Data build ----------------
@st.cache_data(show_spinner=False)
def build_dataset(start_date: str, horizon_days: int) -> pd.DataFrame:
    df = load_wti(start=start_date)
    df = add_technical_features(df)
    df = make_targets(df, horizon_days=horizon_days)
    df = build_model_frame(df)
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

pill_class = "pill-up" if pred_dir == "UP" else "pill-down"
pill_text = f"<span class='pill {pill_class}'>SIGNAL: {pred_dir}</span>"

c1, c2, c3, c4 = st.columns([1.25, 1.25, 1.0, 1.1])

with c1:
    st.markdown(
        f"""
        <div class="card">
            <div class="kicker">WTI settlement (proxy)</div>
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
            <div class="kicker">1-week forecast (return)</div>
            <p class="big">{pred_pct:+.2f}%</p>
            <div class="delta">{pill_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    vol_label = "High" if risk >= 0.03 else "Normal"
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
    # quick confidence proxy: scale predicted magnitude (cap at 3%)
    conf = min(abs(pred_pct) / 3.0, 1.0)
    st.markdown(
        f"""
        <div class="card">
            <div class="kicker">Signal strength (proxy)</div>
            <p class="big">{conf*100:.0f}%</p>
            <div class="delta">Based on |forecast| (capped)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(conf)

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🧪 Backtest", "🧠 Model Inputs"])

# ---------- Tab 1: Overview ----------
with tab1:
    left, right = st.columns([1.6, 1])

    with left:
        st.markdown("### Price")
        fig_price = px.line(
            df.reset_index(),
            x="Date",
            y="Settle",
            title="WTI Price (daily)",
        )
        fig_price.update_layout(
            template="simple_white",
            paper_bgcolor="#FFFDF6",
            plot_bgcolor="#FFFDF6",
            title_font_size=18,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            margin=dict(l=10, r=10, t=55, b=10),
        )
        fig_price.update_traces(line=dict(width=2))
        st.plotly_chart(fig_price, use_container_width=True)

    with right:
        st.markdown("### Volatility & Returns")
        g = df.reset_index()

        fig_vol = px.line(g, x="Date", y="vol_20d", title="20-day volatility (log-return std)")
        fig_vol.update_layout(
            template="simple_white",
            paper_bgcolor="#FFFDF6",
            plot_bgcolor="#FFFDF6",
            title_font_size=16,
            xaxis_title="Date",
            yaxis_title="Vol (std)",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        fig_ret = px.bar(g.tail(90), x="Date", y="ret_1d", title="Daily log returns (last 90 days)")
        fig_ret.update_layout(
            template="simple_white",
            paper_bgcolor="#FFFDF6",
            plot_bgcolor="#FFFDF6",
            title_font_size=16,
            xaxis_title="Date",
            yaxis_title="Log return",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_ret, use_container_width=True)

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

    st.markdown("### Notes")
    st.write(
        "- This UI is styled with custom CSS + a Streamlit theme (blue + cream).\n"
        "- Next upgrade: add geopolitics safely via weekly aggregated signals (so it stays fast)."
    )