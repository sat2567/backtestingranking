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
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_fund_data(category_key: str):
    """Load and prepare fund NAV data."""
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
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['scheme_code', 'date'], keep='last')
    
    # Pivot to wide format
    nav_wide = df.pivot(index='date', columns='scheme_code', values='nav')
    nav_wide = nav_wide.sort_index()
    
    # Forward fill missing NAVs (up to 5 days)
    nav_wide = nav_wide.ffill(limit=5)
    
    # Store scheme names
    scheme_names = df[['scheme_code', 'scheme_name']].drop_duplicates().set_index('scheme_code')['scheme_name'].to_dict()
    
    return nav_wide, scheme_names

@st.cache_data
def load_nifty_data():
    """Load Nifty 100 benchmark data."""
    try:
        nifty_df = pd.read_csv('data/nifty100_fileter_data.csv')
        nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
        nifty_df = nifty_df.sort_values('Date').set_index('Date')
        nifty_df['Close'] = nifty_df['Close'].astype(str).str.replace(' ', '').astype(float)
        return nifty_df['Close']
    except Exception:
        return None

def calculate_simple_nifty_cagr(nifty_data, start_date, end_date):
    """Calculate simple CAGR from first to last Nifty price."""
    if nifty_data is None or len(nifty_data) == 0:
        return np.nan, None, None
    
    try:
        # Find closest dates
        start_idx = nifty_data.index.get_indexer([start_date], method='nearest')[0]
        end_idx = nifty_data.index.get_indexer([end_date], method='nearest')[0]
        
        # Use actual available dates
        if start_idx < 0 or start_idx >= len(nifty_data):
            start_idx = 0
        if end_idx < 0 or end_idx >= len(nifty_data):
            end_idx = len(nifty_data) - 1
        
        initial_nifty = float(nifty_data.iloc[start_idx])
        final_nifty = float(nifty_data.iloc[end_idx])
        
        if initial_nifty <= 0 or final_nifty <= 0 or np.isnan(initial_nifty) or np.isnan(final_nifty):
            return np.nan, None, None
        
        actual_start = nifty_data.index[start_idx]
        actual_end = nifty_data.index[end_idx]
        
        # Calculate CAGR
        total_days = (actual_end - actual_start).days
        total_years = total_days / 365.25
        
        if total_years <= 0:
            return np.nan, actual_start, actual_end
        
        cagr = ((final_nifty / initial_nifty) ** (1 / total_years)) - 1
        
        return cagr, actual_start, actual_end
    except Exception:
        return np.nan, None, None

# ============================================================================
# SELECTION AND METRIC CALCULATION
# ============================================================================

def calculate_metrics_for_date(nav_wide, date_idx, lookback=252):
    """Calculate Sharpe and Sortino ratios for all funds at a given date."""
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
    """Calculate forward returns and ranks for all funds."""
    if date_idx + holding_period >= len(nav_wide):
        return None, None
    
    current_prices = nav_wide.iloc[date_idx]
    future_prices = nav_wide.iloc[date_idx + holding_period]
    
    forward_returns = (future_prices / current_prices - 1).dropna()
    forward_ranks = forward_returns.rank(ascending=False, method='first')
    
    return forward_returns, forward_ranks

def process_all_selections(nav_wide, top_n=5):
    """Process all selection dates and calculate metrics."""
    start_idx = LOOKBACK
    end_idx = len(nav_wide) - HOLDING_PERIOD
    
    if start_idx >= end_idx:
        return []
    
    selection_indices = range(start_idx, end_idx, HOLDING_PERIOD)
    results = []
    
    for date_idx in selection_indices:
        selection_date = nav_wide.index[date_idx]
        
        # Calculate metrics
        metrics = calculate_metrics_for_date(nav_wide, date_idx, LOOKBACK)
        if metrics is None:
            continue
        
        # Calculate forward returns
        forward_returns, forward_ranks = calculate_forward_returns(nav_wide, date_idx, HOLDING_PERIOD)
        if forward_returns is None:
            continue
        
        # Create DataFrame with metrics
        metrics_df = pd.DataFrame(metrics).T
        metrics_df['forward_return'] = metrics_df.index.map(forward_returns.to_dict())
        metrics_df['forward_rank'] = metrics_df.index.map(forward_ranks.to_dict())
        
        # Select top N by Sharpe
        sharpe_selected = metrics_df.nlargest(top_n, 'sharpe').index.tolist()
        
        # Select top N by Sortino
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

def create_six_month_windows(start_date, end_date):
    """Create non-overlapping 6-month windows for selection and evaluation."""
    windows = []
    current = start_date
    window_id = 1
    while current < end_date:
        selection_start = current
        selection_end = selection_start + pd.DateOffset(months=6)
        evaluation_start = selection_end
        evaluation_end = evaluation_start + pd.DateOffset(months=6)
        
        if evaluation_end > end_date:
            evaluation_end = end_date
        
        windows.append({
            'Window ID': window_id,
            'Selection Start': selection_start.strftime('%Y-%m-%d'),
            'Selection End': selection_end.strftime('%Y-%m-%d'),
            'Evaluation Start': evaluation_start.strftime('%Y-%m-%d'),
            'Evaluation End': evaluation_end.strftime('%Y-%m-%d')
        })
        
        current = evaluation_end
        window_id += 1
    
    return pd.DataFrame(windows)

def simulate_portfolio(selection_results, strategy='sharpe'):
    """Simulate equally weighted portfolio for selected funds."""
    portfolio_value = 1.0
    portfolio_values = []
    start_date = None
    end_date = None
    
    for result in selection_results:
        selected_funds = result[f'{strategy}_selected']
        forward_returns = result['forward_returns']
        
        if forward_returns is None or len(selected_funds) == 0:
            continue
        
        # Get returns for selected funds
        selected_returns = [forward_returns.get(fund, np.nan) for fund in selected_funds]
        selected_returns = [r for r in selected_returns if not np.isnan(r)]
        
        if len(selected_returns) == 0:
            continue
        
        # Average return
        avg_return = np.mean(selected_returns)
        
        # Apply to portfolio
        portfolio_value *= (1 + avg_return)
        portfolio_values.append({
            'date': result['selection_date'],
            'value': portfolio_value,
            'avg_return': avg_return
        })
        
        if start_date is None:
            start_date = result['selection_date']
        end_date = result['selection_date'] + pd.DateOffset(months=6)
    
    return portfolio_value, portfolio_values, start_date, end_date

def calculate_portfolio_cagr(portfolio_value, start_date, end_date):
    """Calculate CAGR for portfolio."""
    if portfolio_value <= 0 or start_date is None or end_date is None:
        return np.nan
    
    total_days = (end_date - start_date).days
    total_years = total_days / 365.25
    
    if total_years <= 0:
        return np.nan
    
    cagr = (portfolio_value ** (1 / total_years)) - 1
    return cagr

# ============================================================================
# DASHBOARD UI
# ============================================================================

def main():
    st.title("📊 Basic Fund Selection Dashboard")
    st.markdown("Detailed Fund Selection")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    cat_name = st.sidebar.selectbox("Category", list(CATEGORY_MAP.keys()), index=0)
    cat_key = CATEGORY_MAP[cat_name]
    
    top_n = st.sidebar.number_input("Top N Funds", min_value=1, max_value=20, value=DEFAULT_TOP_N, step=1)
    
    # Load data
    nav_wide, scheme_names = load_fund_data(cat_key)
    
    if nav_wide is None:
        st.error("❌ Fund data file not found. Please check the data directory.")
        return
    
    if len(nav_wide) < LOOKBACK + HOLDING_PERIOD:
        st.error("❌ Insufficient data. Need at least 1.5 years of data.")
        return
    
    # Calculate and display Nifty 100 CAGR
    nifty_data = load_nifty_data()
    if nifty_data is not None and len(nav_wide) > 0:
        # Determine date range from fund data
        fund_start_date = nav_wide.index[0]  # First date in fund data
        fund_end_date = nav_wide.index[-1]    # Last date in fund data
        
        nifty_cagr, nifty_start, nifty_end = calculate_simple_nifty_cagr(nifty_data, fund_start_date, fund_end_date)
        
        if not np.isnan(nifty_cagr) and nifty_start and nifty_end:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nifty 100 CAGR", f"{nifty_cagr*100:.2f}%")
            with col2:
                st.caption(f"From: {nifty_start.strftime('%Y-%m-%d')}")
            with col3:
                st.caption(f"To: {nifty_end.strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    
    # Process selections
    with st.spinner("Calculating metrics and selecting funds..."):
        selection_results = process_all_selections(nav_wide, top_n)
    
    if len(selection_results) == 0:
        st.error("❌ No selection results generated. Check data availability.")
        return
    
    # Calculate portfolio CAGRs
    sharpe_final_value, _, sharpe_start, sharpe_end = simulate_portfolio(selection_results, 'sharpe')
    sharpe_cagr = calculate_portfolio_cagr(sharpe_final_value, sharpe_start, sharpe_end)
    
    sortino_final_value, _, sortino_start, sortino_end = simulate_portfolio(selection_results, 'sortino')
    sortino_cagr = calculate_portfolio_cagr(sortino_final_value, sortino_start, sortino_end)
    
    st.markdown("### Portfolio Performance (Top 5 Funds, 6-Month Rebalance)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sharpe Strategy CAGR", f"{sharpe_cagr*100:.2f}%" if not np.isnan(sharpe_cagr) else "N/A")
    with col2:
        st.metric("Sortino Strategy CAGR", f"{sortino_cagr*100:.2f}%" if not np.isnan(sortino_cagr) else "N/A")
    with col3:
        st.metric("Nifty 100 CAGR", f"{nifty_cagr*100:.2f}%" if not np.isnan(nifty_cagr) else "N/A")
    
    st.markdown("#### 6-Month Windows")
    windows_df = create_six_month_windows(fund_start_date, fund_end_date)
    st.dataframe(windows_df, use_container_width=True)
    
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
                        'Sharpe Ratio': f"{row['sharpe']:.3f}" if not np.isnan(row['sharpe']) else "N/A",
                        'Forward Return (%)': f"{row['forward_return']*100:.2f}" if not np.isnan(row['forward_return']) else "N/A",
                        'Forward Rank': f"{row['forward_rank']:.0f}" if not np.isnan(row['forward_rank']) else "N/A"
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
                        'Sortino Ratio': f"{row['sortino']:.3f}" if not np.isnan(row['sortino']) else "N/A",
                        'Forward Return (%)': f"{row['forward_return']*100:.2f}" if not np.isnan(row['forward_return']) else "N/A",
                        'Forward Rank': f"{row['forward_rank']:.0f}" if not np.isnan(row['forward_rank']) else "N/A"
                    })
            
            if sortino_display:
                st.dataframe(pd.DataFrame(sortino_display), use_container_width=True)

if __name__ == "__main__":
    main()
