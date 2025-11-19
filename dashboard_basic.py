# COMPLETE MUTUAL FUND STRATEGY BACKTESTING CODE
# Includes:
# - Sharpe & Sortino calculation
# - 6-month rolling selection
# - Strategy CAGR
# - NIFTY CAGR (same window)
# - Clean + corrected methodology

import streamlit as st
import pandas as pd
import numpy as np
import datetime

# =======================================
# 1) SHARPE RATIO
# =======================================
def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    excess = returns - risk_free_rate/252
    return (excess.mean() / returns.std()) * np.sqrt(252)

# =======================================
# 2) SORTINO RATIO
# =======================================
def calculate_sortino_ratio(returns, risk_free_rate=0.05):
    if len(returns) == 0:
        return np.nan
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    excess = returns - risk_free_rate/252
    return (excess.mean() / downside.std()) * np.sqrt(252)

# =======================================
# 3) CAGR
# =======================================
def compute_strategy_cagr(selection_results, method="sharpe", top_n=5):
    capital = 1.0
    dates = []

    for result in selection_results:
        if method == "sharpe":
            selected = result["sharpe_selected"][:top_n]
        else:
            selected = result["sortino_selected"][:top_n]

        fwd = result["forward_returns"]
        period_returns = [fwd[f] for f in selected if f in fwd.index]

        if len(period_returns) == 0:
            continue

        capital *= (1 + np.mean(period_returns))
        dates.append(result["selection_date"])

    if len(dates) < 2:
        return np.nan

    years = (dates[-1] - dates[0]).days / 365.25
    return (capital ** (1/years)) - 1

# =======================================
# 4) NIFTY CAGR (MATCH WINDOW)
# =======================================
def compute_nifty_cagr_same_window(nifty_series, start_date, end_date):
    sliced = nifty_series.loc[(nifty_series.index >= start_date) & (nifty_series.index <= end_date)]
    if len(sliced) < 2:
        return np.nan

    start_val = sliced.iloc[0]
    end_val = sliced.iloc[-1]
    years = (sliced.index[-1] - sliced.index[0]).days / 365.25

    return (end_val / start_val) ** (1/years) - 1

# =======================================
# 5) SELECTION ENGINE
# =======================================
def process_all_selections(nav_wide, top_n):
    results = []

    dates = nav_wide.index
    six_month = 126  # 6 months trading

    for i in range(len(dates) - six_month):
        start_date = dates[i]
        end_date = dates[i + six_month]

        window_data = nav_wide.loc[:start_date].tail(252)  # 1yr
        fwd_slice = nav_wide.loc[start_date:end_date]

        if len(window_data) < 30 or len(fwd_slice) < 10:
            continue

        ret = window_data.pct_change().dropna()
        sr = ret.apply(calculate_sharpe_ratio)
        sor = ret.apply(calculate_sortino_ratio)

        sharpe_ranked = sr.sort_values(ascending=False).index.tolist()
        sortino_ranked = sor.sort_values(ascending=False).index.tolist()

        fwd_returns = fwd_slice.pct_change().dropna().sum()

        results.append({
            "selection_date": start_date,
            "sharpe_selected": sharpe_ranked,
            "sortino_selected": sortino_ranked,
            "forward_returns": fwd_returns
        })

    return results

# =======================================
# 6) MAIN APP
# =======================================
def main():
    st.title("Mutual Fund Selection Strategy Backtest")

    uploaded = st.file_uploader("Upload Mutual Fund NAV CSV", type=["csv"])
    nifty_file = st.file_uploader("Upload NIFTY NAV CSV", type=["csv"])

    if uploaded is None or nifty_file is None:
        st.info("Upload both files to continue.")
        return

    df = pd.read_csv(uploaded)
    nifty = pd.read_csv(nifty_file)

    df["Date"] = pd.to_datetime(df["Date"])
    nifty["Date"] = pd.to_datetime(nifty["Date"])

    df = df.set_index("Date").sort_index()
    nifty = nifty.set_index("Date").sort_index()

    nav_wide = df.pivot(columns="Scheme Name", values="Adj Nav").dropna()
    nifty_series = nifty["Adj Close"].dropna()

    top_n = st.slider("Top N funds per selection", 3, 10, 5)

    st.subheader("Running Selection Model…")
    selection_results = process_all_selections(nav_wide, top_n)

    if len(selection_results) == 0:
        st.error("No selections generated. Check data.")
        return

    # STRATEGY CAGRs
    sharpe_cagr = compute_strategy_cagr(selection_results, "sharpe", top_n)
    sortino_cagr = compute_strategy_cagr(selection_results, "sortino", top_n)

    st.subheader("📈 Strategy CAGR Results")
    st.write(f"Sharpe Strategy CAGR: **{sharpe_cagr*100:.2f}%**")
    st.write(f"Sortino Strategy CAGR: **{sortino_cagr*100:.2f}%**")

    # NIFTY CAGR (matched period)
    start_date = selection_results[0]["selection_date"]
    end_date = selection_results[-1]["selection_date"]

    nifty_cagr = compute_nifty_cagr_same_window(nifty_series, start_date, end_date)

    st.subheader("📊 NIFTY Benchmark CAGR (Same Window)")
    st.write(f"NIFTY CAGR (Matched Period): **{nifty_cagr*100:.2f}%**")


if __name__ == "__main__":
    main()
