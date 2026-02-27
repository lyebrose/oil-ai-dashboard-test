# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from data import load_wti
from features import add_technical_features, make_targets, build_model_frame
from model import walk_forward_backtest, train_latest_model, FEATURE_COLS

st.set_page_config(page_title="WTI AI Trader Dashboard", layout="wide")

st.title("WTI Futures AI Dashboard (1-week horizon)")
st.caption("Educational decision-support tool. Not financial advice.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    start_date = st.text_input("Start date (YYYY-MM-DD)", "2023-01-01")
    horizon_days = st.selectbox("Forecast horizon (trading days)", [5, 10], index=0)
    backtest_start_year = st.slider("Backtest start year", 2012, 2025, 2018)

# ---------------- Data build ----------------
@st.cache_data(show_spinner=False)
def build_dataset(start_date: str, horizon_days: int) -> pd.DataFrame:
    df = load_wti(start=start_date)
    df = add_technical_features(df)
    df = make_targets(df, horizon_days=horizon_days)
    df = build_model_frame(df)
    return df

df = build_dataset(start_date, horizon_days)

# ---------------- Top row metrics ----------------
col1, col2, col3 = st.columns([1.3, 1, 1])

latest_price = float(df["Settle"].iloc[-1])

week_change = np.nan
if len(df) > 6:
    week_change = float((df["Settle"].iloc[-1] / df["Settle"].iloc[-6] - 1) * 100)

with col1:
    st.metric("Latest WTI settlement (proxy)", f"${latest_price:,.2f}", f"{week_change:+.2f}% (1w)")

with st.spinner("Training latest model..."):
    model, latest_pred = train_latest_model(df)

pred_dir = "UP" if latest_pred > 0 else "DOWN"
pred_pct = (np.exp(latest_pred) - 1) * 100

with col2:
    st.metric("Next-week forecast (return)", f"{pred_pct:+.2f}%", f"Direction: {pred_dir}")

risk = float(df["vol_20d"].iloc[-1]) if "vol_20d" in df.columns else np.nan
with col3:
    st.metric("Risk (20d vol)", f"{risk:.4f}")

st.divider()

# ---------------- Charts ----------------
left, right = st.columns([1.4, 1])

with left:
    st.subheader("WTI Price Over Time")
    fig = px.line(df.reset_index(), x="Date", y="Settle", title="WTI (continuous proxy) Settlement")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Volatility + Returns")
    g = df.reset_index()

    figv = px.line(g, x="Date", y="vol_20d", title="20-day volatility (log return std)")
    st.plotly_chart(figv, use_container_width=True)

    figr = px.line(g, x="Date", y="ret_1d", title="1-day log return")
    st.plotly_chart(figr, use_container_width=True)

st.divider()

# ---------------- Backtest ----------------
st.subheader("Walk-forward backtest (by year)")
with st.spinner("Running walk-forward backtest..."):
    bt = walk_forward_backtest(df, start_year=backtest_start_year)

m1, m2, m3 = st.columns(3)
m1.metric("MAE (log return)", f"{bt.mae:.5f}")
m2.metric("RMSE (log return)", f"{bt.rmse:.5f}")
m3.metric("Direction accuracy", f"{bt.dir_acc*100:.2f}%")

bt_df = pd.DataFrame({"truth": bt.truth, "pred": bt.preds}).dropna().reset_index().rename(columns={"index": "Date"})
fig_bt = px.line(bt_df, x="Date", y=["truth", "pred"], title="Truth vs Predicted (1-week log return)")
st.plotly_chart(fig_bt, use_container_width=True)

st.subheader("What’s driving the model?")
st.caption("Permutation importance on the most recent backtest fold (approx).")
st.dataframe(bt.importance, use_container_width=True)

st.divider()

st.subheader("Latest feature values (model inputs)")
st.dataframe(df[FEATURE_COLS].tail(1).T.rename(columns={df.index[-1]: "latest"}), use_container_width=True)