"""
Basic Dashboard - Detailed Fund Selection Only
Shows selected funds by Sharpe and Sortino ratios for a selected date.
This version reads data from local CSV files in the `data/` folder (no upload required).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

LOOKBACK = 252          # 1 year in trading days
HOLDING_PERIOD = 126    # 6 months in trading days (approx)
RISK_FREE_RATE = 0.05   # 5% annual
DEFAULT_TOP_N = 5

CATEGORY_MAP = {
    "Large Cap": "largecap",
    "Small Cap": "smallcap",
    "Mid Cap": "midcap",
    "Large & Mid Cap": "large_and_midcap",
    "Multi Cap": "multicap",
    "International": "international",
}

DATA_FOLDER = "data"
FUNDS_CSV_MAP = {
    "largecap": "largecap_funds.csv",
    "smallcap": "smallcap_funds.csv",
    "midcap": "midcap_funds.csv",
    "large_and_midcap": "large_and_midcap_funds.csv",
    "multicap": "multicap_funds.csv",
    "international": "international_funds.csv",
}
NIFTY_CSV = "nifty100_fileter_data.csv"

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Basic Fund Selection Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """Calculate annualized Sharpe ratio."""
    returns = returns.dropna()
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    excess_returns = returns - risk_free_rate / 252
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
    return sharpe


def calculate_sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """Calculate annualized Sortino ratio."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.nan
    sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    return sortino

# ============================================================================
# DATA LOADING FUNCTIONS (LOCAL FILEs)
# ============================================================================

@st.cache_data
def load_fund_data(category_key: str):
    """Load and prepare fund NAV data from local CSV file under data/"""
    fname = FUNDS_CSV_MAP.get(category_key)
    if fname is None:
        return None, None
    path = os.path.join(DATA_FOLDER, fname)
    if not os.path.exists(path):
        return None, None

    df = pd.read_csv(path)
    # Expecting columns: date, scheme_code, scheme_name, nav
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Remove duplicates and pivot
    df = df.drop_duplicates(subset=['scheme_code', 'date'], keep='last')
    nav_wide = df.pivot(index='date', columns='scheme_code', values='nav')
    nav_wide = nav_wide.sort_index()

    # Forward fill small gaps
    nav_wide = nav_wide.ffill(limit=5)

    scheme_names = df[['scheme_code', 'scheme_name']].drop_duplicates().set_index('scheme_code')['scheme_name'].to_dict()

    return nav_wide, scheme_names

@st.cache_data
def load_nifty_data():
    path = os.path.join(DATA_FOLDER, NIFTY_CSV)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Expecting columns: Date, Close (or Close adjusted)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date').set_index('Date')
    # Try common column names for price
    price_col = None
    for c in ['Close', 'Adj Close', 'ClosePrice', 'close']:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        # If there's no obvious column, try the first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
        else:
            return None

    series = pd.to_numeric(df[price_col].astype(str).str.replace(' ', ''), errors='coerce').dropna()
    series.index = df.loc[series.index].index
    return series

# ============================================================================
# CAGR & STRATEGY FUNCTIONS
# ============================================================================

def compute_strategy_cagr(selection_results, method="sharpe", top_n=DEFAULT_TOP_N):
    """
    Compounds the 6-month forward returns of selected funds to compute strategy CAGR.
    selection_results is the list produced by process_all_selections.
    """
    capital = 1.0
    dates = []

    for result in selection_results:
        if method == "sharpe":
            selected = result['sharpe_selected'][:top_n]
        else:
            selected = result['sortino_selected'][:top_n]

        forward_returns = result['forward_returns']
        period_returns = [forward_returns.get(f) for f in selected if f in forward_returns.index]

        if len(period_returns) == 0:
            continue

        period_return = np.nanmean(period_returns)
        capital *= (1 + period_return)
        dates.append(result['selection_date'])

    if len(dates) < 2:
        return np.nan

    start_date = dates[0]
    end_date = dates[-1]
    total_years = (end_date - start_date).days / 365.25
    if total_years <= 0:
        return np.nan

    return (capital ** (1 / total_years)) - 1


def compute_nifty_cagr_same_window(nifty_series, start_date, end_date):
    if nifty_series is None or len(nifty_series) == 0:
        return np.nan

    sliced = nifty_series.loc[(nifty_series.index >= start_date) & (nifty_series.index <= end_date)]
    if len(sliced) < 2:
        return np.nan

    start_val = float(sliced.iloc[0])
    end_val = float(sliced.iloc[-1])
    years = (sliced.index[-1] - sliced.index[0]).days / 365.25
    if years <= 0:
        return np.nan

    return (end_val / start_val) ** (1 / years) - 1

# ============================================================================
# SELECTION ENGINE
# ============================================================================

def calculate_metrics_for_date(nav_wide, date_idx, lookback=LOOKBACK):
    if date_idx < lookback:
        return None

    lookback_data = nav_wide.iloc[date_idx - lookback:date_idx]
    metrics = {}

    for fund in nav_wide.columns:
        fund_prices = lookback_data[fund].astype(float).dropna()
        if len(fund_prices) < 2:
            metrics[fund] = {'sharpe': np.nan, 'sortino': np.nan}
            continue

        fund_returns = fund_prices.pct_change().dropna()
        sharpe = calculate_sharpe_ratio(fund_returns)
        sortino = calculate_sortino_ratio(fund_returns)
        metrics[fund] = {'sharpe': sharpe, 'sortino': sortino}

    return metrics


def calculate_forward_returns(nav_wide, date_idx, holding_period=HOLDING_PERIOD):
    if date_idx + holding_period >= len(nav_wide):
        return None, None

    current_prices = nav_wide.iloc[date_idx]
    future_prices = nav_wide.iloc[date_idx + holding_period]

    forward_returns = (future_prices / current_prices - 1).dropna()
    forward_ranks = forward_returns.rank(ascending=False, method='first')

    return forward_returns, forward_ranks


def process_all_selections(nav_wide, top_n=DEFAULT_TOP_N):
    start_idx = LOOKBACK
    end_idx = len(nav_wide) - HOLDING_PERIOD
    if start_idx >= end_idx:
        return []

    selection_indices = range(start_idx, end_idx, HOLDING_PERIOD)
    results = []

    for date_idx in selection_indices:
        selection_date = nav_wide.index[date_idx]

        metrics = calculate_metrics_for_date(nav_wide, date_idx, LOOKBACK)
        if metrics is None:
            continue

        forward_returns, forward_ranks = calculate_forward_returns(nav_wide, date_idx, HOLDING_PERIOD)
        if forward_returns is None:
            continue

        metrics_df = pd.DataFrame(metrics).T
        metrics_df['forward_return'] = metrics_df.index.map(forward_returns.to_dict())
        metrics_df['forward_rank'] = metrics_df.index.map(forward_ranks.to_dict())

        sharpe_selected = metrics_df.nlargest(top_n, 'sharpe').index.tolist()
        sortino_selected = metrics_df.nlargest(top_n, 'sortino').index.tolist()

        results.append({
            'selection_date': selection_date,
            'date_idx': date_idx,
            'sharpe_selected': sharpe_selected,
            'sortino_selected': sortino_selected,
            'metrics_df': metrics_df,
            'forward_returns': forward_returns,
            'forward_ranks': forward_ranks
        })

    return results

# ============================================================================
# DASHBOARD UI
# ============================================================================

def main():
    st.title("📊 Basic Fund Selection Dashboard")
    st.markdown("Detailed Fund Selection")

    st.sidebar.header("⚙️ Configuration")
    cat_name = st.sidebar.selectbox("Category", list(CATEGORY_MAP.keys()), index=0)
    cat_key = CATEGORY_MAP[cat_name]
    top_n = st.sidebar.number_input("Top N Funds", min_value=1, max_value=20, value=DEFAULT_TOP_N, step=1)

    # Load data from local files
    nav_wide, scheme_names = load_fund_data(cat_key)
    nifty_series = load_nifty_data()

    if nav_wide is None:
        st.error("❌ Fund data file not found. Please check the data directory and filenames.")
        return

    if len(nav_wide) < LOOKBACK + HOLDING_PERIOD:
        st.error("❌ Insufficient data. Need at least 1.5 years of data.")
        return

    # Calculate and display Nifty 100 CAGR over full fund window (initial quick display)
    if nifty_series is not None and len(nav_wide) > 0:
        fund_start_date = nav_wide.index[0]
        fund_end_date = nav_wide.index[-1]
        nifty_cagr_full = compute_nifty_cagr_same_window(nifty_series, fund_start_date, fund_end_date)
        if not np.isnan(nifty_cagr_full):
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Nifty 100 CAGR (fund data window)", f"{nifty_cagr_full*100:.2f}%")
            with col2: st.caption(f"From: {fund_start_date.strftime('%Y-%m-%d')}")
            with col3: st.caption(f"To: {fund_end_date.strftime('%Y-%m-%d')}")

    st.markdown("---")

    with st.spinner("Calculating metrics and selecting funds..."):
        selection_results = process_all_selections(nav_wide, top_n)

    if len(selection_results) == 0:
        st.error("❌ No selection results generated. Check data availability.")
        return

    # Compute strategy CAGRs
    sharpe_cagr = compute_strategy_cagr(selection_results, method='sharpe', top_n=top_n)
    sortino_cagr = compute_strategy_cagr(selection_results, method='sortino', top_n=top_n)

    # Compute Nifty CAGR for the exact same strategy window
    strategy_start = selection_results[0]['selection_date']
    strategy_end = selection_results[-1]['selection_date']
    nifty_cagr_matched = compute_nifty_cagr_same_window(nifty_series, strategy_start, strategy_end) if nifty_series is not None else np.nan

    # Display
    st.subheader("📈 Overall Strategy Performance (CAGR)")
    c1, c2, c3 = st.columns(3)
    with c1:
        if not np.isnan(sharpe_cagr):
            st.metric("Sharpe Strategy CAGR", f"{sharpe_cagr*100:.2f}%")
        else:
            st.warning("Sharpe Strategy CAGR not available")
    with c2:
        if not np.isnan(sortino_cagr):
            st.metric("Sortino Strategy CAGR", f"{sortino_cagr*100:.2f}%")
        else:
            st.warning("Sortino Strategy CAGR not available")
    with c3:
        if not np.isnan(nifty_cagr_matched):
            st.metric("Nifty 100 CAGR (matched window)", f"{nifty_cagr_matched*100:.2f}%")
        else:
            st.warning("Nifty CAGR (matched) not available")

    st.markdown("---")

    # Detailed view for selected date
    st.subheader("🔍 Detailed Fund Selection")
    dates = [r['selection_date'] for r in selection_results]
    selected_date_idx = st.selectbox("Select Date", range(len(dates)), format_func=lambda x: dates[x].strftime('%Y-%m-%d'))

    if selected_date_idx < len(selection_results):
        result = selection_results[selected_date_idx]
        metrics_df = result['metrics_df']

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Selected by Sharpe Ratio")
            sharpe_funds = result['sharpe_selected']
            sharpe_display = []
            for fund in sharpe_funds:
                if fund in metrics_df.index:
                    row = metrics_df.loc[fund]
                    sharpe_display.append({
                        'Fund Name': scheme_names.get(fund, fund),
                        'Sharpe Ratio': f"{row['sharpe']:.3f}" if not np.isnan(row['sharpe']) else 'N/A',
                        'Forward Return (%)': f"{row['forward_return']*100:.2f}" if not np.isnan(row.get('forward_return', np.nan)) else 'N/A',
                        'Forward Rank': f"{row['forward_rank']:.0f}" if not np.isnan(row.get('forward_rank', np.nan)) else 'N/A'
                    })
            if sharpe_display:
                st.dataframe(pd.DataFrame(sharpe_display), use_container_width=True)

        with col2:
            st.markdown("#### Selected by Sortino Ratio")
            sortino_funds = result['sortino_selected']
            sortino_display = []
            for fund in sortino_funds:
                if fund in metrics_df.index:
                    row = metrics_df.loc[fund]
                    sortino_display.append({
                        'Fund Name': scheme_names.get(fund, fund),
                        'Sortino Ratio': f"{row['sortino']:.3f}" if not np.isnan(row['sortino']) else 'N/A',
                        'Forward Return (%)': f"{row['forward_return']*100:.2f}" if not np.isnan(row.get('forward_return', np.nan)) else 'N/A',
                        'Forward Rank': f"{row['forward_rank']:.0f}" if not np.isnan(row.get('forward_rank', np.nan)) else 'N/A'
                    })
            if sortino_display:
                st.dataframe(pd.DataFrame(sortino_display), use_container_width=True)

if __name__ == '__main__':
    main()
