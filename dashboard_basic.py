"""
Basic Dashboard - Detailed Fund Selection Only
Shows selected funds by Sharpe and Sortino ratios for a selected date.
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
HOLDING_PERIOD = 126    # 6 months in trading days
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

def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    excess_returns = returns - risk_free_rate / 252
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
    return sharpe

def calculate_sortino_ratio(returns, risk_free_rate=0.05):
    """Calculate annualized Sortino ratio."""
    if len(returns) == 0:
        return np.nan
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.nan
    sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    return sortino

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_fund_data(category_key: str):
    category_to_csv = {
        "largecap": "data/largecap_funds.csv",
        "smallcap": "data/smallcap_funds.csv",
        "midcap": "data/midcap_funds.csv",
        "large_and_midcap": "data/large_and_midcap_funds.csv",
        "multicap": "data/multicap_funds.csv",
        "international": "data/international_funds.csv",
    }
    
    funds_csv = category_to_csv.get(category_key, "data/largecap_funds.csv")
    
    if not os.path.exists(funds_csv):
        return None, None
    
    df = pd.read_csv(funds_csv)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    
    df = df.drop_duplicates(subset=['scheme_code', 'date'], keep='last')
    
    nav_wide = df.pivot(index='date', columns='scheme_code', values='nav')
    nav_wide = nav_wide.sort_index()
    
    nav_wide = nav_wide.ffill(limit=5)
    
    scheme_names = df[['scheme_code', 'scheme_name']].drop_duplicates().set_index('scheme_code')['scheme_name'].to_dict()
    
    return nav_wide, scheme_names

@st.cache_data
def load_nifty_data():
    try:
        nifty_df = pd.read_csv('data/nifty100_fileter_data.csv')
        nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
        nifty_df = nifty_df.sort_values('Date').set_index('Date')
        nifty_df['Close'] = nifty_df['Close'].astype(str).str.replace(' ', '').astype(float)
        return nifty_df['Close']
    except Exception:
        return None

def calculate_simple_nifty_cagr(nifty_data, start_date, end_date):
    if nifty_data is None or len(nifty_data) == 0:
        return np.nan, None, None
    
    try:
        start_idx = nifty_data.index.get_indexer([start_date], method='nearest')[0]
        end_idx = nifty_data.index.get_indexer([end_date], method='nearest')[0]
        
        initial_nifty = float(nifty_data.iloc[start_idx])
        final_nifty = float(nifty_data.iloc[end_idx])
        
        if initial_nifty <= 0 or final_nifty <= 0:
            return np.nan, None, None
        
        actual_start = nifty_data.index[start_idx]
        actual_end = nifty_data.index[end_idx]
        
        total_days = (actual_end - actual_start).days
        total_years = total_days / 365.25
        
        if total_years <= 0:
            return np.nan, actual_start, actual_end
        
        cagr = ((final_nifty / initial_nifty) ** (1 / total_years)) - 1
        
        return cagr, actual_start, actual_end
    except Exception:
        return np.nan, None, None

# ============================================================================
# SELECTION & METRIC CALCULATION
# ============================================================================

def calculate_metrics_for_date(nav_wide, date_idx, lookback=252):
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
        
        sharpe = calculate_sharpe_ratio(fund_returns, RISK_FREE_RATE)
        sortino = calculate_sortino_ratio(fund_returns, RISK_FREE_RATE)
        
        metrics[fund] = {
            'sharpe': sharpe,
            'sortino': sortino
        }
    
    return metrics

def calculate_forward_returns(nav_wide, date_idx, holding_period=126):
    if date_idx + holding_period >= len(nav_wide):
        return None, None
    
    current_prices = nav_wide.iloc[date_idx]
    future_prices = nav_wide.iloc[date_idx + holding_period]
    
    forward_returns = (future_prices / current_prices - 1).dropna()
    forward_ranks = forward_returns.rank(ascending=False, method='first')
    
    return forward_returns, forward_ranks

def process_all_selections(nav_wide, top_n=5):
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
# STRATEGY CAGR FUNCTION (NEW)
# ============================================================================

def compute_strategy_cagr(selection_results, method="sharpe", top_n=5):
    """
    Compounds the 6-month forward returns of selected funds to compute strategy CAGR.
    """
    capital = 1.0
    dates = []

    for result in selection_results:
        if method == "sharpe":
            selected_funds = result["sharpe_selected"][:top_n]
        else:
            selected_funds = result["sortino_selected"][:top_n]

        forward_returns = result["forward_returns"]

        period_returns = []
        for fund in selected_funds:
            if fund in forward_returns.index:
                period_returns.append(forward_returns[fund])

        if len(period_returns) == 0:
            continue

        period_return = np.mean(period_returns)
        capital *= (1 + period_return)
        dates.append(result["selection_date"])

    if len(dates) < 2:
        return np.nan

    start_date = dates[0]
    end_date = dates[-1]
    years = (end_date - start_date).days / 365.25

    if years <= 0:
        return np.nan

    cagr = (capital ** (1 / years)) - 1
    return cagr

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
    
    nav_wide, scheme_names = load_fund_data(cat_key)
    
    if nav_wide is None:
        st.error("❌ Fund data file not found.")
        return
    
    if len(nav_wide) < LOOKBACK + HOLDING_PERIOD:
        st.error("❌ Insufficient data. Need at least 1.5 years of data.")
        return
    
    nifty_data = load_nifty_data()
    if nifty_data is not None and len(nav_wide) > 0:
        fund_start = nav_wide.index[0]
        fund_end = nav_wide.index[-1]
        
        nifty_cagr, nifty_start, nifty_end = calculate_simple_nifty_cagr(nifty_data, fund_start, fund_end)
        
        if not np.isnan(nifty_cagr):
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Nifty 100 CAGR", f"{nifty_cagr*100:.2f}%")
            with col2: st.caption(f"From: {nifty_start.strftime('%Y-%m-%d')}")
            with col3: st.caption(f"To: {nifty_end.strftime('%Y-%m-%d')}")

    st.markdown("---")
    
    with st.spinner("Calculating fund selections..."):
        selection_results = process_all_selections(nav_wide, top_n)
    
    if len(selection_results) == 0:
        st.error("❌ No selection results generated.")
        return

    # ============================================================================
    # DISPLAY STRATEGY CAGR (NEW)
    # ============================================================================

    sharpe_cagr = compute_strategy_cagr(selection_results, method="sharpe", top_n=top_n)
    sortino_cagr = compute_strategy_cagr(selection_results, method="sortino", top_n=top_n)

    st.subheader("📈 Overall Strategy CAGR")

    col1, col2 = st.columns(2)

    with col1:
        if not np.isnan(sharpe_cagr):
            st.metric("Sharpe Strategy CAGR", f"{sharpe_cagr*100:.2f}%")
        else:
            st.warning("Sharpe Strategy CAGR unavailable")

    with col2:
        if not np.isnan(sortino_cagr):
            st.metric("Sortino Strategy CAGR", f"{sortino_cagr*100:.2f}%")
        else:
            st.warning("Sortino Strategy CAGR unavailable")

    st.markdown("---")

    # ============================================================================
    # DETAILED SELECTION VIEW
    # ============================================================================

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
                row = metrics_df.loc[fund]
                sharpe_display.append({
                    'Fund Name': scheme_names.get(fund, fund),
                    'Sharpe Ratio': f"{row['sharpe']:.3f}",
                    'Forward Return (%)': f"{row['forward_return']*100:.2f}",
                    'Forward Rank': f"{row['forward_rank']:.0f}"
                })
            
            st.dataframe(pd.DataFrame(sharpe_display), use_container_width=True)
        
        with col2:
            st.markdown("#### Selected by Sortino Ratio")
            sortino_funds = result['sortino_selected']
            sortino_display = []
            
            for fund in sortino_funds:
                row = metrics_df.loc[fund]
                sortino_display.append({
                    'Fund Name': scheme_names.get(fund, fund),
                    'Sortino Ratio': f"{row['sortino']:.3f}",
                    'Forward Return (%)': f"{row['forward_return']*100:.2f}",
                    'Forward Rank': f"{row['forward_rank']:.0f}"
                })
            
            st.dataframe(pd.DataFrame(sortino_display), use_container_width=True)

if __name__ == "__main__":
    main()
