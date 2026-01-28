"""
Advanced Fund Analysis Dashboard - Enhanced Version
====================================================
Features:
- Comprehensive Category Explorer with 1Y and 3Y rolling analysis
- Beautiful modern UI with better styling
- More metrics and columns
- Improved visualizations
- Better data presentation

Run: streamlit run enhanced_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import linregress

warnings.filterwarnings('ignore')

# ============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="Fund Analysis Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Headers */
    h1 {
        color: #1E3A5F;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #4CAF50;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #2E4A6F;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    h3 {
        color: #3E5A7F;
        font-weight: 500;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.85rem;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 700;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* DataFrames */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Selectbox */
    div[data-baseweb="select"] {
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Dividers */
    hr {
        margin: 1.5rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, #ddd, transparent);
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #4CAF50;
        margin-bottom: 1rem;
    }
    
    .metric-card-warning {
        border-left-color: #ff9800;
    }
    
    .metric-card-danger {
        border-left-color: #f44336;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. CONSTANTS
# ============================================================================

DEFAULT_HOLDING = 126
DEFAULT_TOP_N = 2
DEFAULT_TARGET_N = 4
RISK_FREE_RATE = 0.06
TRADING_DAYS_YEAR = 252
DAILY_RISK_FREE_RATE = (1 + RISK_FREE_RATE) ** (1/TRADING_DAYS_YEAR) - 1
DATA_DIR = "data"
MAX_DATA_DATE = pd.Timestamp('2025-12-05')

FILE_MAPPING = {
    "Large Cap": "largecap_merged.xlsx",
    "Mid Cap": "midcap.xlsx",
    "Small Cap": "smallcap.xlsx",
    "Large & Mid Cap": "large_and_midcap_fund.xlsx",
    "Multi Cap": "MULTICAP.xlsx",
    "International": "international_merged.xlsx"
}

# Color schemes
COLORS = {
    'primary': '#4CAF50',
    'secondary': '#2196F3',
    'success': '#00C853',
    'warning': '#FF9800',
    'danger': '#F44336',
    'info': '#00BCD4',
    'purple': '#9C27B0',
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2'
}

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def clean_weekday_data(df):
    if df is None or df.empty: return df
    df = df[df.index <= MAX_DATA_DATE]
    df = df[df.index.dayofweek < 5]
    if len(df) > 0:
        start_date = df.index.min()
        end_date = min(df.index.max(), MAX_DATA_DATE)
        all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
        df = df.reindex(all_weekdays)
        df = df.ffill(limit=5)
    return df

# ============================================================================
# 4. METRIC CALCULATIONS
# ============================================================================

def calculate_sharpe_ratio(returns):
    if len(returns) < 10 or returns.std() == 0: return np.nan
    excess_returns = returns - DAILY_RISK_FREE_RATE
    return (excess_returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS_YEAR)

def calculate_sortino_ratio(returns):
    if len(returns) < 10: return np.nan
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0: return np.nan
    mean_return = (returns - DAILY_RISK_FREE_RATE).mean()
    return (mean_return / downside.std()) * np.sqrt(TRADING_DAYS_YEAR)

def calculate_omega_ratio(returns, target=DAILY_RISK_FREE_RATE):
    if len(returns) < 10: return np.nan
    diff = returns - target
    pos = diff[diff > 0].sum()
    neg = -diff[diff < 0].sum()
    if neg == 0: return np.nan
    return pos / neg

def calculate_martin_ratio(series):
    if len(series) < 30: return np.nan
    cum_max = series.expanding(min_periods=1).max()
    drawdowns = (series / cum_max) - 1
    ulcer_index = np.sqrt(np.mean(drawdowns.fillna(0) ** 2))
    if ulcer_index == 0: return np.nan
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0 or series.iloc[0] <= 0: return np.nan
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
    return (cagr - RISK_FREE_RATE) / ulcer_index

def calculate_calmar_ratio(series):
    if len(series) < 252: return np.nan
    max_dd = calculate_max_dd(series)
    if pd.isna(max_dd) or max_dd >= 0: return np.nan
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0 or series.iloc[0] <= 0: return np.nan
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
    return cagr / abs(max_dd)

def calculate_capture_score(fund_rets, bench_rets):
    common_idx = fund_rets.index.intersection(bench_rets.index)
    if len(common_idx) < 30: return np.nan, np.nan, np.nan
    f = fund_rets.loc[common_idx]
    b = bench_rets.loc[common_idx]
    up_market = b[b > 0]
    down_market = b[b < 0]
    
    up_cap = f.loc[up_market.index].mean() / up_market.mean() if not up_market.empty and up_market.mean() != 0 else np.nan
    down_cap = f.loc[down_market.index].mean() / down_market.mean() if not down_market.empty and down_market.mean() != 0 else np.nan
    
    if pd.notna(up_cap) and pd.notna(down_cap) and down_cap > 0:
        ratio = up_cap / down_cap
    else:
        ratio = np.nan
    
    return up_cap, down_cap, ratio

def calculate_volatility(returns):
    if len(returns) < 10: return np.nan
    return returns.std() * np.sqrt(TRADING_DAYS_YEAR)

def calculate_max_dd(series):
    if len(series) < 10 or series.isna().all(): return np.nan
    comp_ret = (1 + series.pct_change().fillna(0)).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret / peak) - 1
    return dd.min()

def calculate_cagr(series):
    if len(series) < 30 or series.iloc[0] <= 0: return np.nan
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0: return np.nan
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1

def calculate_information_ratio(fund_returns, bench_returns):
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 30: return np.nan
    f_ret = fund_returns.loc[common_idx]
    b_ret = bench_returns.loc[common_idx]
    active_return = f_ret - b_ret
    tracking_error = active_return.std(ddof=1) * np.sqrt(TRADING_DAYS_YEAR)
    if tracking_error == 0: return np.nan
    return (active_return.mean() * TRADING_DAYS_YEAR) / tracking_error

def calculate_beta_alpha(fund_returns, bench_returns):
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 60: return np.nan, np.nan
    f_ret = fund_returns.loc[common_idx]
    b_ret = bench_returns.loc[common_idx]
    
    cov = np.cov(f_ret, b_ret)
    if cov[1, 1] == 0: return np.nan, np.nan
    
    beta = cov[0, 1] / cov[1, 1]
    alpha = (f_ret.mean() - beta * b_ret.mean()) * TRADING_DAYS_YEAR
    
    return beta, alpha

def calculate_rolling_metrics(series, benchmark_series, window_days):
    """Calculate rolling returns and outperformance metrics."""
    if len(series) < window_days + 30: 
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Rolling returns
    fund_rolling = series.pct_change(window_days).dropna()
    bench_rolling = benchmark_series.pct_change(window_days).dropna()
    
    common_idx = fund_rolling.index.intersection(bench_rolling.index)
    if len(common_idx) < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    f_roll = fund_rolling.loc[common_idx]
    b_roll = bench_rolling.loc[common_idx]
    
    # Average rolling return
    avg_rolling_return = f_roll.mean()
    
    # % times beat benchmark
    diff = f_roll - b_roll
    pct_beat = (diff > 0).mean()
    
    # Average outperformance when beating
    beats = diff[diff > 0]
    avg_outperformance = beats.mean() if len(beats) > 0 else 0
    
    # Average underperformance when losing
    losses = diff[diff < 0]
    avg_underperformance = losses.mean() if len(losses) > 0 else 0
    
    # Consistency score
    consistency = pct_beat * (1 + avg_outperformance) / (1 + abs(avg_underperformance)) if avg_underperformance != 0 else pct_beat
    
    return avg_rolling_return, pct_beat, avg_outperformance, avg_underperformance, consistency

def calculate_risk_adjusted_metrics(series, benchmark_series):
    """Calculate comprehensive risk-adjusted metrics."""
    returns = series.pct_change().dropna()
    
    metrics = {}
    
    # Basic metrics
    metrics['volatility'] = calculate_volatility(returns)
    metrics['sharpe'] = calculate_sharpe_ratio(returns)
    metrics['sortino'] = calculate_sortino_ratio(returns)
    metrics['max_dd'] = calculate_max_dd(series)
    metrics['calmar'] = calculate_calmar_ratio(series)
    metrics['cagr'] = calculate_cagr(series)
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    metrics['downside_vol'] = downside_returns.std() * np.sqrt(TRADING_DAYS_YEAR) if len(downside_returns) > 10 else np.nan
    
    # VaR and CVaR
    metrics['var_95'] = returns.quantile(0.05) if len(returns) > 30 else np.nan
    metrics['cvar_95'] = returns[returns <= returns.quantile(0.05)].mean() if len(returns) > 30 else np.nan
    
    # Benchmark relative metrics
    if benchmark_series is not None:
        bench_returns = benchmark_series.pct_change().dropna()
        metrics['beta'], metrics['alpha'] = calculate_beta_alpha(returns, bench_returns)
        metrics['info_ratio'] = calculate_information_ratio(returns, bench_returns)
        up_cap, down_cap, cap_ratio = calculate_capture_score(returns, bench_returns)
        metrics['up_capture'] = up_cap
        metrics['down_capture'] = down_cap
        metrics['capture_ratio'] = cap_ratio
    
    return metrics

# ============================================================================
# 5. DATA LOADING
# ============================================================================

@st.cache_data
def load_fund_data_raw(category_key: str):
    filename = FILE_MAPPING.get(category_key)
    if not filename: return None, None
    local_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(local_path): return None, None
    try:
        df = pd.read_excel(local_path, header=None)
        fund_names = df.iloc[2, 1:].tolist()
        data_df = df.iloc[4:, :].copy()
        if isinstance(data_df.iloc[-1, 0], str) and 'Accord' in str(data_df.iloc[-1, 0]):
            data_df = data_df.iloc[:-1, :]
        dates = pd.to_datetime(data_df.iloc[:, 0], errors='coerce')
        nav_wide = pd.DataFrame(index=dates)
        scheme_map = {}
        for i, fund_name in enumerate(fund_names):
            if pd.notna(fund_name) and str(fund_name).strip() != '':
                scheme_code = str(abs(hash(fund_name)) % (10 ** 8))
                scheme_map[scheme_code] = fund_name
                nav_values = pd.to_numeric(data_df.iloc[:, i+1], errors='coerce')
                nav_wide[scheme_code] = nav_values.values
        nav_wide = nav_wide.sort_index()
        nav_wide = nav_wide[~nav_wide.index.duplicated(keep='last')]
        nav_wide = clean_weekday_data(nav_wide)
        return nav_wide, scheme_map
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None, None

@st.cache_data
def load_nifty_data():
    local_path = os.path.join(DATA_DIR, "nifty100_data.csv")
    if not os.path.exists(local_path): return None
    try:
        df = pd.read_csv(local_path)
        df.columns = [c.lower().strip() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df = df.dropna(subset=['date', 'nav']).set_index('date').sort_index()
        return clean_weekday_data(df).squeeze()
    except: return None

# ============================================================================
# 6. COMPREHENSIVE ANALYTICS
# ============================================================================

def calculate_comprehensive_metrics(nav_df, scheme_map, benchmark_series):
    """Calculate comprehensive metrics for all funds including 1Y and 3Y rolling analysis."""
    
    if nav_df is None or nav_df.empty or benchmark_series is None:
        return pd.DataFrame()
    
    metrics_list = []
    
    for col in nav_df.columns:
        series = nav_df[col].dropna()
        if len(series) < 260: continue
        
        fund_name = scheme_map.get(col, col)
        
        # Basic info
        row = {
            'Fund Name': fund_name,
            'fund_id': col
        }
        
        # Data range
        row['Data Start'] = series.index[0].strftime('%Y-%m-%d')
        row['Data End'] = series.index[-1].strftime('%Y-%m-%d')
        row['Days of Data'] = len(series)
        
        # Returns
        returns = series.pct_change().dropna()
        
        # Point-in-time returns
        if len(series) >= 63:
            row['Return 3M'] = (series.iloc[-1] / series.iloc[-63] - 1) * 100
        if len(series) >= 126:
            row['Return 6M'] = (series.iloc[-1] / series.iloc[-126] - 1) * 100
        if len(series) >= 252:
            row['Return 1Y'] = (series.iloc[-1] / series.iloc[-252] - 1) * 100
        if len(series) >= 756:
            row['Return 3Y'] = ((series.iloc[-1] / series.iloc[-756]) ** (1/3) - 1) * 100
        if len(series) >= 1260:
            row['Return 5Y'] = ((series.iloc[-1] / series.iloc[-1260]) ** (1/5) - 1) * 100
        
        # CAGR (full period)
        row['CAGR %'] = calculate_cagr(series) * 100 if calculate_cagr(series) else np.nan
        
        # 1Y Rolling Analysis
        roll_1y_ret, roll_1y_beat, roll_1y_outperf, roll_1y_underperf, roll_1y_consistency = calculate_rolling_metrics(
            series, benchmark_series, 252
        )
        row['1Y Rolling Avg Return %'] = roll_1y_ret * 100 if pd.notna(roll_1y_ret) else np.nan
        row['1Y % Times Beat Benchmark'] = roll_1y_beat * 100 if pd.notna(roll_1y_beat) else np.nan
        row['1Y Avg Outperformance %'] = roll_1y_outperf * 100 if pd.notna(roll_1y_outperf) else np.nan
        row['1Y Avg Underperformance %'] = roll_1y_underperf * 100 if pd.notna(roll_1y_underperf) else np.nan
        row['1Y Consistency Score'] = roll_1y_consistency if pd.notna(roll_1y_consistency) else np.nan
        
        # 3Y Rolling Analysis
        roll_3y_ret, roll_3y_beat, roll_3y_outperf, roll_3y_underperf, roll_3y_consistency = calculate_rolling_metrics(
            series, benchmark_series, 756
        )
        row['3Y Rolling Avg Return %'] = roll_3y_ret * 100 if pd.notna(roll_3y_ret) else np.nan
        row['3Y % Times Beat Benchmark'] = roll_3y_beat * 100 if pd.notna(roll_3y_beat) else np.nan
        row['3Y Avg Outperformance %'] = roll_3y_outperf * 100 if pd.notna(roll_3y_outperf) else np.nan
        row['3Y Consistency Score'] = roll_3y_consistency if pd.notna(roll_3y_consistency) else np.nan
        
        # Risk Metrics
        row['Volatility %'] = calculate_volatility(returns) * 100 if calculate_volatility(returns) else np.nan
        row['Max Drawdown %'] = calculate_max_dd(series) * 100 if calculate_max_dd(series) else np.nan
        
        downside_returns = returns[returns < 0]
        row['Downside Vol %'] = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 10 else np.nan
        
        # Risk-Adjusted Metrics
        row['Sharpe Ratio'] = calculate_sharpe_ratio(returns)
        row['Sortino Ratio'] = calculate_sortino_ratio(returns)
        row['Calmar Ratio'] = calculate_calmar_ratio(series)
        row['Martin Ratio'] = calculate_martin_ratio(series)
        
        # Benchmark Relative Metrics
        if benchmark_series is not None:
            bench_returns = benchmark_series.pct_change().dropna()
            
            beta, alpha = calculate_beta_alpha(returns, bench_returns)
            row['Beta'] = beta
            row['Alpha %'] = alpha * 100 if pd.notna(alpha) else np.nan
            
            row['Information Ratio'] = calculate_information_ratio(returns, bench_returns)
            
            up_cap, down_cap, cap_ratio = calculate_capture_score(returns, bench_returns)
            row['Up Capture %'] = up_cap * 100 if pd.notna(up_cap) else np.nan
            row['Down Capture %'] = down_cap * 100 if pd.notna(down_cap) else np.nan
            row['Capture Ratio'] = cap_ratio
            
            # Batting Average (daily)
            common_idx = returns.index.intersection(bench_returns.index)
            if len(common_idx) > 30:
                f_ret = returns.loc[common_idx]
                b_ret = bench_returns.loc[common_idx]
                row['Batting Avg %'] = (f_ret > b_ret).mean() * 100
        
        # VaR metrics
        if len(returns) > 30:
            row['VaR 95 %'] = returns.quantile(0.05) * 100
            row['CVaR 95 %'] = returns[returns <= returns.quantile(0.05)].mean() * 100
        
        # Positive months %
        monthly_returns = series.resample('ME').last().pct_change().dropna()
        if len(monthly_returns) > 6:
            row['% Positive Months'] = (monthly_returns > 0).mean() * 100
        
        # Best/Worst periods
        if len(monthly_returns) > 12:
            row['Best Month %'] = monthly_returns.max() * 100
            row['Worst Month %'] = monthly_returns.min() * 100
        
        metrics_list.append(row)
    
    df = pd.DataFrame(metrics_list)
    
    if not df.empty:
        # Add ranking columns
        if '1Y Rolling Avg Return %' in df.columns:
            df['1Y Return Rank'] = df['1Y Rolling Avg Return %'].rank(ascending=False, method='min')
        if '3Y Rolling Avg Return %' in df.columns:
            df['3Y Return Rank'] = df['3Y Rolling Avg Return %'].rank(ascending=False, method='min')
        if 'Sharpe Ratio' in df.columns:
            df['Sharpe Rank'] = df['Sharpe Ratio'].rank(ascending=False, method='min')
        if '1Y Consistency Score' in df.columns:
            df['Consistency Rank'] = df['1Y Consistency Score'].rank(ascending=False, method='min')
        
        # Composite score
        rank_cols = [c for c in ['1Y Return Rank', '3Y Return Rank', 'Sharpe Rank', 'Consistency Rank'] if c in df.columns]
        if rank_cols:
            df['Composite Rank'] = df[rank_cols].mean(axis=1)
            df = df.sort_values('Composite Rank')
    
    return df

def calculate_quarterly_ranks(nav_df, scheme_map):
    """Calculate quarterly rank history."""
    if nav_df is None or nav_df.empty: return pd.DataFrame()
    
    quarter_ends = pd.date_range(start=nav_df.index.min(), end=nav_df.index.max(), freq='Q')
    history_data = {}
    
    for q_date in quarter_ends:
        start_lookback = q_date - pd.Timedelta(days=365)
        if start_lookback < nav_df.index.min(): continue
        try:
            idx_now = nav_df.index.asof(q_date)
            idx_prev = nav_df.index.asof(start_lookback)
            if pd.isna(idx_now) or pd.isna(idx_prev): continue
            rets = (nav_df.loc[idx_now] / nav_df.loc[idx_prev]) - 1
            ranks = rets.rank(ascending=False, method='min')
            history_data[q_date.strftime('%Y-Q%q')] = ranks
        except: continue
    
    rank_df = pd.DataFrame(history_data)
    rank_df.index = rank_df.index.map(lambda x: scheme_map.get(x, x))
    rank_df = rank_df.dropna(how='all')
    
    if not rank_df.empty:
        # Calculate summary statistics
        def calc_stats(row):
            valid = row.dropna()
            if len(valid) == 0: return pd.Series({'% Top 3': 0, '% Top 5': 0, '% Top 10': 0, 'Avg Rank': np.nan, 'Best Rank': np.nan, 'Worst Rank': np.nan})
            return pd.Series({
                '% Top 3': (valid <= 3).sum() / len(valid) * 100,
                '% Top 5': (valid <= 5).sum() / len(valid) * 100,
                '% Top 10': (valid <= 10).sum() / len(valid) * 100,
                'Avg Rank': valid.mean(),
                'Best Rank': valid.min(),
                'Worst Rank': valid.max()
            })
        
        stats = rank_df.apply(calc_stats, axis=1)
        rank_df = pd.concat([stats, rank_df], axis=1)
        rank_df = rank_df.sort_values('% Top 5', ascending=False)
    
    return rank_df

# ============================================================================
# 7. VISUALIZATION HELPERS
# ============================================================================

def create_performance_chart(nav_df, selected_funds, scheme_map, benchmark_series=None, normalize=True):
    """Create interactive performance chart."""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for idx, fund_id in enumerate(selected_funds[:10]):  # Limit to 10 funds
        series = nav_df[fund_id].dropna()
        if normalize and len(series) > 0:
            series = series / series.iloc[0] * 100
        
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name=scheme_map.get(fund_id, fund_id)[:30],
            line=dict(color=colors[idx % len(colors)], width=2),
            hovertemplate='%{y:.2f}<extra>%{fullData.name}</extra>'
        ))
    
    if benchmark_series is not None:
        bench = benchmark_series.dropna()
        if normalize and len(bench) > 0:
            # Align benchmark to fund data range
            common_start = max(nav_df[selected_funds[0]].dropna().index[0], bench.index[0])
            bench = bench[bench.index >= common_start]
            if len(bench) > 0:
                bench = bench / bench.iloc[0] * 100
        
        fig.add_trace(go.Scatter(
            x=bench.index,
            y=bench.values,
            name='Nifty 100',
            line=dict(color='rgba(128, 128, 128, 0.7)', width=2, dash='dot'),
            hovertemplate='%{y:.2f}<extra>Nifty 100</extra>'
        ))
    
    fig.update_layout(
        title=dict(text='Performance Comparison', font=dict(size=16)),
        xaxis_title='Date',
        yaxis_title='Value (Normalized to 100)' if normalize else 'NAV',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        height=450,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_rolling_return_chart(nav_df, fund_id, scheme_map, benchmark_series, window=252):
    """Create rolling return comparison chart."""
    series = nav_df[fund_id].dropna()
    fund_rolling = series.pct_change(window).dropna() * 100
    bench_rolling = benchmark_series.pct_change(window).dropna() * 100
    
    common_idx = fund_rolling.index.intersection(bench_rolling.index)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=common_idx,
        y=fund_rolling.loc[common_idx],
        name=scheme_map.get(fund_id, fund_id)[:25],
        fill='tozeroy',
        line=dict(color=COLORS['primary'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=common_idx,
        y=bench_rolling.loc[common_idx],
        name='Nifty 100',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{window//252}Y Rolling Returns',
        yaxis_title='Rolling Return %',
        hovermode='x unified',
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_drawdown_chart(series, fund_name):
    """Create drawdown chart."""
    returns = series.pct_change().fillna(0)
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.expanding().max()
    drawdown = (cum_ret / peak - 1) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        line=dict(color=COLORS['danger'], width=1),
        fillcolor='rgba(244, 67, 54, 0.3)',
        name='Drawdown'
    ))
    
    fig.update_layout(
        title=f'Drawdown History - {fund_name[:30]}',
        yaxis_title='Drawdown %',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# ============================================================================
# 8. UI COMPONENTS
# ============================================================================

def render_metric_card(title, value, subtitle=None, delta=None, color='primary'):
    """Render a styled metric card."""
    delta_html = f'<div style="color: {"green" if delta and delta > 0 else "red"}; font-size: 0.9rem;">{delta:+.2f}%</div>' if delta else ''
    subtitle_html = f'<div style="color: #666; font-size: 0.8rem;">{subtitle}</div>' if subtitle else ''
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid {COLORS.get(color, COLORS['primary'])};
        margin-bottom: 10px;
    ">
        <div style="color: #666; font-size: 0.85rem; margin-bottom: 5px;">{title}</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #1a1a2e;">{value}</div>
        {delta_html}
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)

def render_explorer_tab():
    """Render the enhanced Category Explorer tab."""
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0;">📊 Category Explorer</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Comprehensive fund analysis with rolling metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        category = st.selectbox(
            "📁 Select Category",
            list(FILE_MAPPING.keys()),
            help="Choose a fund category to analyze"
        )
    
    with col2:
        view_mode = st.selectbox(
            "👁️ View Mode",
            ["📈 Comprehensive Metrics", "📊 Quarterly Rank History", "🔍 Fund Deep Dive"],
            help="Select analysis view"
        )
    
    with col3:
        st.write("")
        st.write("")
        refresh = st.button("🔄 Refresh", use_container_width=True)
    
    st.divider()
    
    # Load data
    with st.spinner(f"Loading {category} data..."):
        nav_df, scheme_map = load_fund_data_raw(category)
        benchmark = load_nifty_data()
    
    if nav_df is None:
        st.error("❌ Could not load data. Please check if data files exist.")
        return
    
    # Data summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Funds", len(nav_df.columns))
    col2.metric("📅 Data Range", f"{nav_df.index.min().strftime('%Y-%m')} to {nav_df.index.max().strftime('%Y-%m')}")
    col3.metric("📆 Trading Days", len(nav_df))
    col4.metric("🏛️ Benchmark", "Nifty 100" if benchmark is not None else "N/A")
    
    st.divider()
    
    # View Mode: Comprehensive Metrics
    if "Comprehensive Metrics" in view_mode:
        with st.spinner("Calculating comprehensive metrics..."):
            metrics_df = calculate_comprehensive_metrics(nav_df, scheme_map, benchmark)
        
        if metrics_df.empty:
            st.warning("No funds with sufficient data history.")
            return
        
        # Column groups for display
        st.markdown("### 📋 Fund Performance Summary")
        
        # Tabs for different metric groups
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Overall Ranking",
            "📈 Returns Analysis",
            "🔄 Rolling Analysis (1Y & 3Y)",
            "⚠️ Risk Metrics",
            "📊 Benchmark Comparison"
        ])
        
        with tab1:
            st.markdown("#### Top Funds by Composite Score")
            display_cols = ['Fund Name', 'Composite Rank', '1Y Return Rank', '3Y Return Rank', 
                           'Sharpe Rank', 'Consistency Rank', 'CAGR %', 'Max Drawdown %']
            display_cols = [c for c in display_cols if c in metrics_df.columns]
            
            st.dataframe(
                metrics_df[display_cols].head(20).style.format({
                    'Composite Rank': '{:.1f}',
                    '1Y Return Rank': '{:.0f}',
                    '3Y Return Rank': '{:.0f}',
                    'Sharpe Rank': '{:.0f}',
                    'Consistency Rank': '{:.0f}',
                    'CAGR %': '{:.2f}%',
                    'Max Drawdown %': '{:.2f}%'
                }).background_gradient(subset=['Composite Rank'], cmap='Greens_r'),
                use_container_width=True,
                height=500
            )
        
        with tab2:
            st.markdown("#### Return Analysis")
            return_cols = ['Fund Name', 'Return 3M', 'Return 6M', 'Return 1Y', 'Return 3Y', 
                          'Return 5Y', 'CAGR %', '% Positive Months', 'Best Month %', 'Worst Month %']
            return_cols = [c for c in return_cols if c in metrics_df.columns]
            
            st.dataframe(
                metrics_df[return_cols].style.format({
                    'Return 3M': '{:.2f}%',
                    'Return 6M': '{:.2f}%',
                    'Return 1Y': '{:.2f}%',
                    'Return 3Y': '{:.2f}%',
                    'Return 5Y': '{:.2f}%',
                    'CAGR %': '{:.2f}%',
                    '% Positive Months': '{:.1f}%',
                    'Best Month %': '{:.2f}%',
                    'Worst Month %': '{:.2f}%'
                }).background_gradient(subset=['Return 1Y'], cmap='RdYlGn'),
                use_container_width=True,
                height=500
            )
        
        with tab3:
            st.markdown("#### Rolling Analysis - 1 Year & 3 Year")
            st.info("💡 Rolling analysis shows how consistently a fund performs over time, not just point-in-time returns.")
            
            rolling_cols = ['Fund Name', 
                           '1Y Rolling Avg Return %', '1Y % Times Beat Benchmark', '1Y Avg Outperformance %', '1Y Consistency Score',
                           '3Y Rolling Avg Return %', '3Y % Times Beat Benchmark', '3Y Avg Outperformance %', '3Y Consistency Score']
            rolling_cols = [c for c in rolling_cols if c in metrics_df.columns]
            
            st.dataframe(
                metrics_df[rolling_cols].style.format({
                    '1Y Rolling Avg Return %': '{:.2f}%',
                    '1Y % Times Beat Benchmark': '{:.1f}%',
                    '1Y Avg Outperformance %': '{:.2f}%',
                    '1Y Consistency Score': '{:.3f}',
                    '3Y Rolling Avg Return %': '{:.2f}%',
                    '3Y % Times Beat Benchmark': '{:.1f}%',
                    '3Y Avg Outperformance %': '{:.2f}%',
                    '3Y Consistency Score': '{:.3f}'
                }).background_gradient(subset=['1Y Consistency Score'], cmap='Greens'),
                use_container_width=True,
                height=500
            )
            
            # Explanation
            st.markdown("""
            **Column Explanations:**
            - **Rolling Avg Return %**: Average of all rolling 1Y/3Y returns over the fund's history
            - **% Times Beat Benchmark**: How often the fund beat the benchmark on a rolling basis
            - **Avg Outperformance %**: When the fund beat the benchmark, by how much on average
            - **Consistency Score**: Combined measure of beat rate and outperformance magnitude
            """)
        
        with tab4:
            st.markdown("#### Risk Metrics")
            risk_cols = ['Fund Name', 'Volatility %', 'Downside Vol %', 'Max Drawdown %', 
                        'VaR 95 %', 'CVaR 95 %', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Martin Ratio']
            risk_cols = [c for c in risk_cols if c in metrics_df.columns]
            
            st.dataframe(
                metrics_df[risk_cols].style.format({
                    'Volatility %': '{:.2f}%',
                    'Downside Vol %': '{:.2f}%',
                    'Max Drawdown %': '{:.2f}%',
                    'VaR 95 %': '{:.2f}%',
                    'CVaR 95 %': '{:.2f}%',
                    'Sharpe Ratio': '{:.2f}',
                    'Sortino Ratio': '{:.2f}',
                    'Calmar Ratio': '{:.2f}',
                    'Martin Ratio': '{:.2f}'
                }).background_gradient(subset=['Sharpe Ratio'], cmap='Greens'),
                use_container_width=True,
                height=500
            )
        
        with tab5:
            st.markdown("#### Benchmark Comparison")
            bench_cols = ['Fund Name', 'Beta', 'Alpha %', 'Information Ratio', 
                         'Up Capture %', 'Down Capture %', 'Capture Ratio', 'Batting Avg %']
            bench_cols = [c for c in bench_cols if c in metrics_df.columns]
            
            st.dataframe(
                metrics_df[bench_cols].style.format({
                    'Beta': '{:.2f}',
                    'Alpha %': '{:.2f}%',
                    'Information Ratio': '{:.2f}',
                    'Up Capture %': '{:.1f}%',
                    'Down Capture %': '{:.1f}%',
                    'Capture Ratio': '{:.2f}',
                    'Batting Avg %': '{:.1f}%'
                }).background_gradient(subset=['Alpha %'], cmap='RdYlGn'),
                use_container_width=True,
                height=500
            )
        
        # Download button
        st.divider()
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            "📥 Download Complete Analysis (CSV)",
            csv,
            f"{category}_comprehensive_analysis.csv",
            "text/csv",
            use_container_width=True
        )
    
    # View Mode: Quarterly Rank History
    elif "Quarterly Rank History" in view_mode:
        with st.spinner("Calculating quarterly rankings..."):
            rank_df = calculate_quarterly_ranks(nav_df, scheme_map)
        
        if rank_df.empty:
            st.warning("Insufficient data for quarterly analysis.")
            return
        
        st.markdown("### 📊 Quarterly Performance Rank History")
        st.info("💡 Lower rank = better performance. See how funds rank each quarter based on 1-year returns.")
        
        # Summary columns
        summary_cols = ['% Top 3', '% Top 5', '% Top 10', 'Avg Rank', 'Best Rank', 'Worst Rank']
        quarter_cols = [c for c in rank_df.columns if c not in summary_cols]
        
        # Display with formatting
        st.dataframe(
            rank_df.style.format({
                '% Top 3': '{:.1f}%',
                '% Top 5': '{:.1f}%',
                '% Top 10': '{:.1f}%',
                'Avg Rank': '{:.1f}',
                'Best Rank': '{:.0f}',
                'Worst Rank': '{:.0f}'
            }).background_gradient(subset=['% Top 5'], cmap='Greens'),
            use_container_width=True,
            height=600
        )
    
    # View Mode: Fund Deep Dive
    elif "Fund Deep Dive" in view_mode:
        st.markdown("### 🔍 Individual Fund Analysis")
        
        # Fund selector
        fund_options = {scheme_map.get(col, col): col for col in nav_df.columns}
        selected_fund_name = st.selectbox("Select Fund", sorted(fund_options.keys()))
        selected_fund_id = fund_options[selected_fund_name]
        
        series = nav_df[selected_fund_id].dropna()
        
        if len(series) < 252:
            st.warning("Insufficient data history for this fund.")
            return
        
        # Fund metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cagr = calculate_cagr(series)
            st.metric("📈 CAGR", f"{cagr*100:.2f}%" if cagr else "N/A")
        
        with col2:
            sharpe = calculate_sharpe_ratio(series.pct_change().dropna())
            st.metric("⚖️ Sharpe Ratio", f"{sharpe:.2f}" if sharpe else "N/A")
        
        with col3:
            max_dd = calculate_max_dd(series)
            st.metric("📉 Max Drawdown", f"{max_dd*100:.2f}%" if max_dd else "N/A")
        
        with col4:
            vol = calculate_volatility(series.pct_change().dropna())
            st.metric("📊 Volatility", f"{vol*100:.2f}%" if vol else "N/A")
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_performance_chart(nav_df, [selected_fund_id], scheme_map, benchmark)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_drawdown_chart(series, selected_fund_name)
            st.plotly_chart(fig, use_container_width=True)
        
        # Rolling return charts
        st.markdown("#### Rolling Return Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_rolling_return_chart(nav_df, selected_fund_id, scheme_map, benchmark, 252)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(series) >= 756:
                fig = create_rolling_return_chart(nav_df, selected_fund_id, scheme_map, benchmark, 756)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for 3Y rolling analysis")

# ============================================================================
# 9. BACKTESTER (Keeping original logic but with better UI)
# ============================================================================

def get_lookback_data(nav, analysis_date):
    max_days = 400
    start_date = analysis_date - pd.Timedelta(days=max_days)
    return nav[(nav.index >= start_date) & (nav.index < analysis_date)]

def calculate_flexible_momentum(series, w_3m, w_6m, w_12m, use_risk_adjust=False):
    if len(series) < 260: return np.nan
    price_cur = series.iloc[-1]
    current_date = series.index[-1]
    
    def get_past_price(days_ago):
        target_date = current_date - pd.Timedelta(days=days_ago)
        sub_series = series[series.index <= target_date]
        if sub_series.empty: return np.nan
        return sub_series.iloc[-1]
    
    p91 = get_past_price(91)
    p182 = get_past_price(182)
    p365 = get_past_price(365)
    
    r3 = (price_cur / p91) - 1 if not pd.isna(p91) else 0
    r6 = (price_cur / p182) - 1 if not pd.isna(p182) else 0
    r12 = (price_cur / p365) - 1 if not pd.isna(p365) else 0
    
    raw_score = (r3 * w_3m) + (r6 * w_6m) + (r12 * w_12m)
    
    if use_risk_adjust:
        date_1y_ago = current_date - pd.Timedelta(days=365)
        hist_vol_data = series[series.index >= date_1y_ago]
        if len(hist_vol_data) < 20: return np.nan
        vol = hist_vol_data.pct_change().dropna().std() * np.sqrt(TRADING_DAYS_YEAR)
        if vol == 0: return np.nan
        return raw_score / vol
    return raw_score

def get_market_regime(benchmark_series, current_date, window=200):
    subset = benchmark_series[benchmark_series.index <= current_date]
    if len(subset) < window: return 'neutral'
    current_price = subset.iloc[-1]
    dma = subset.iloc[-window:].mean()
    return 'bull' if current_price > dma else 'bear'

def run_backtest(nav, strategy_type, top_n, target_n, holding_days, custom_weights, momentum_config, benchmark_series, ensemble_weights=None):
    start_date = nav.index.min() + pd.Timedelta(days=370)
    if start_date >= nav.index.max():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    start_idx = nav.index.searchsorted(start_date)
    rebal_idx = list(range(start_idx, len(nav) - 1, holding_days))
    
    if not rebal_idx: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    history, eq_curve, bench_curve = [], [{'date': nav.index[rebal_idx[0]], 'value': 100.0}], [{'date': nav.index[rebal_idx[0]], 'value': 100.0}]
    cap, b_cap = 100.0, 100.0
    
    for i in rebal_idx:
        date = nav.index[i]
        hist = get_lookback_data(nav, date)
        
        bench_rets = None
        if benchmark_series is not None:
            try:
                b_slice = get_lookback_data(benchmark_series.to_frame(), date)
                bench_rets = b_slice['nav'].pct_change().dropna()
            except: pass
        
        scores = {}
        selected = []
        regime_status = "neutral"
        
        # Strategy implementations (keeping original logic)
        if strategy_type in ['momentum', 'sharpe', 'sortino']:
            for col in nav.columns:
                s = hist[col].dropna()
                if len(s) < 126: continue
                rets = s.pct_change().dropna()
                
                if strategy_type == 'momentum':
                    val = calculate_flexible_momentum(s, momentum_config.get('w_3m', 0.33), 
                                                     momentum_config.get('w_6m', 0.33), 
                                                     momentum_config.get('w_12m', 0.33),
                                                     momentum_config.get('risk_adjust', False))
                elif strategy_type == 'sharpe':
                    val = calculate_sharpe_ratio(rets)
                elif strategy_type == 'sortino':
                    val = calculate_sortino_ratio(rets)
                
                if not pd.isna(val): scores[col] = val
            selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
        
        elif strategy_type == 'regime_switch':
            if benchmark_series is not None:
                regime = get_market_regime(benchmark_series, date)
                regime_status = regime
                
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) < 126: continue
                    
                    if regime == 'bull':
                        val = calculate_flexible_momentum(s, 0.3, 0.3, 0.4, False)
                    else:
                        val = calculate_sharpe_ratio(s.pct_change().dropna())
                    
                    if not pd.isna(val): scores[col] = val
                selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
        
        elif strategy_type == 'stable_momentum':
            mom_scores = {}
            for col in nav.columns:
                s = hist[col].dropna()
                if len(s) >= 260:
                    val = calculate_flexible_momentum(s, 0.33, 0.33, 0.33, False)
                    if not pd.isna(val): mom_scores[col] = val
            
            pool = sorted(mom_scores, key=mom_scores.get, reverse=True)[:top_n*2]
            
            dd_scores = {}
            for col in pool:
                dd = calculate_max_dd(hist[col].dropna())
                if not pd.isna(dd): dd_scores[col] = dd
            selected = sorted(dd_scores, key=dd_scores.get, reverse=True)[:top_n]
        
        # Execution
        entry = i + 1
        exit_i = min(i + 1 + holding_days, len(nav) - 1)
        
        b_ret = 0.0
        if benchmark_series is not None:
            try: b_ret = (benchmark_series.asof(nav.index[exit_i]) / benchmark_series.asof(nav.index[entry])) - 1
            except: pass
        
        port_ret = 0.0
        hit_rate = 0.0
        
        if selected:
            period_ret_all = (nav.iloc[exit_i] / nav.iloc[entry]) - 1
            port_ret = period_ret_all[selected].mean()
            
            actual_top = period_ret_all.dropna().nlargest(target_n).index.tolist()
            matches = len(set(selected).intersection(set(actual_top)))
            hit_rate = matches / top_n if top_n > 0 else 0
        
        cap *= (1 + (port_ret if not pd.isna(port_ret) else 0))
        b_cap *= (1 + b_ret)
        
        history.append({
            'date': date,
            'selected': selected,
            'return': port_ret,
            'hit_rate': hit_rate,
            'regime': regime_status
        })
        eq_curve.append({'date': nav.index[exit_i], 'value': cap})
        bench_curve.append({'date': nav.index[exit_i], 'value': b_cap})
    
    return pd.DataFrame(history), pd.DataFrame(eq_curve), pd.DataFrame(bench_curve)

def run_backtest_detailed(nav, strategy_type, top_n, target_n, holding_days, custom_weights, momentum_config, benchmark_series, scheme_map, ensemble_weights=None):
    """Enhanced backtest that returns detailed fund selections and actual top performers."""
    start_date = nav.index.min() + pd.Timedelta(days=370)
    if start_date >= nav.index.max():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    start_idx = nav.index.searchsorted(start_date)
    rebal_idx = list(range(start_idx, len(nav) - 1, holding_days))
    
    if not rebal_idx: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    history, eq_curve, bench_curve = [], [{'date': nav.index[rebal_idx[0]], 'value': 100.0}], [{'date': nav.index[rebal_idx[0]], 'value': 100.0}]
    detailed_trades = []
    cap, b_cap = 100.0, 100.0
    
    for i in rebal_idx:
        date = nav.index[i]
        hist = get_lookback_data(nav, date)
        
        bench_rets = None
        if benchmark_series is not None:
            try:
                b_slice = get_lookback_data(benchmark_series.to_frame(), date)
                bench_rets = b_slice['nav'].pct_change().dropna()
            except: pass
        
        scores = {}
        selected = []
        regime_status = "neutral"
        
        # Strategy implementations
        if strategy_type in ['momentum', 'sharpe', 'sortino']:
            for col in nav.columns:
                s = hist[col].dropna()
                if len(s) < 126: continue
                rets = s.pct_change().dropna()
                
                if strategy_type == 'momentum':
                    val = calculate_flexible_momentum(s, momentum_config.get('w_3m', 0.33), 
                                                     momentum_config.get('w_6m', 0.33), 
                                                     momentum_config.get('w_12m', 0.33),
                                                     momentum_config.get('risk_adjust', False))
                elif strategy_type == 'sharpe':
                    val = calculate_sharpe_ratio(rets)
                elif strategy_type == 'sortino':
                    val = calculate_sortino_ratio(rets)
                
                if not pd.isna(val): scores[col] = val
            selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
        
        elif strategy_type == 'regime_switch':
            if benchmark_series is not None:
                regime = get_market_regime(benchmark_series, date)
                regime_status = regime
                
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) < 126: continue
                    
                    if regime == 'bull':
                        val = calculate_flexible_momentum(s, 0.3, 0.3, 0.4, False)
                    else:
                        val = calculate_sharpe_ratio(s.pct_change().dropna())
                    
                    if not pd.isna(val): scores[col] = val
                selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
        
        elif strategy_type == 'stable_momentum':
            mom_scores = {}
            for col in nav.columns:
                s = hist[col].dropna()
                if len(s) >= 260:
                    val = calculate_flexible_momentum(s, 0.33, 0.33, 0.33, False)
                    if not pd.isna(val): mom_scores[col] = val
            
            pool = sorted(mom_scores, key=mom_scores.get, reverse=True)[:top_n*2]
            
            dd_scores = {}
            for col in pool:
                dd = calculate_max_dd(hist[col].dropna())
                if not pd.isna(dd): dd_scores[col] = dd
            selected = sorted(dd_scores, key=dd_scores.get, reverse=True)[:top_n]
        
        elif strategy_type == 'elimination':
            # Elimination strategy
            fund_data = {}
            for col in nav.columns:
                s = hist[col].dropna()
                if len(s) < 200: continue
                rets = s.pct_change().dropna()
                fund_data[col] = {
                    'max_dd': calculate_max_dd(s),
                    'volatility': calculate_volatility(rets),
                    'momentum_6m': (s.iloc[-1] / s.iloc[-126] - 1) if len(s) >= 126 else np.nan,
                    'sharpe': calculate_sharpe_ratio(rets)
                }
            
            if fund_data:
                df = pd.DataFrame(fund_data).T.dropna()
                if len(df) > top_n * 2:
                    # Eliminate worst drawdowns
                    dd_threshold = df['max_dd'].quantile(0.25)
                    df = df[df['max_dd'] >= dd_threshold]
                    
                    # Eliminate highest volatility
                    if len(df) > top_n * 2:
                        vol_threshold = df['volatility'].quantile(0.75)
                        df = df[df['volatility'] <= vol_threshold]
                    
                    # Pick top by Sharpe
                    selected = df.nlargest(top_n, 'sharpe').index.tolist()
                else:
                    selected = df.nlargest(top_n, 'sharpe').index.tolist() if len(df) > 0 else []
        
        elif strategy_type == 'consistency':
            # Consistency-first strategy
            consistent_funds = []
            for col in nav.columns:
                s = hist[col].dropna()
                if len(s) < 300: continue
                
                quarters_good = 0
                for q in range(4):
                    q_end = date - pd.Timedelta(days=q * 91)
                    q_start = q_end - pd.Timedelta(days=91)
                    try:
                        idx_s = hist.index.asof(q_start)
                        idx_e = hist.index.asof(q_end)
                        if pd.isna(idx_s) or pd.isna(idx_e): continue
                        all_rets = (hist.loc[idx_e] / hist.loc[idx_s] - 1).dropna()
                        if col in all_rets.index and all_rets[col] >= all_rets.median():
                            quarters_good += 1
                    except: continue
                
                if quarters_good >= 3:
                    consistent_funds.append(col)
            
            if consistent_funds:
                mom_scores = {}
                for col in consistent_funds:
                    s = hist[col].dropna()
                    if len(s) >= 63:
                        mom_scores[col] = (s.iloc[-1] / s.iloc[-63] - 1)
                selected = sorted(mom_scores, key=mom_scores.get, reverse=True)[:top_n] if mom_scores else []
            else:
                # Fallback to momentum
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) >= 126:
                        scores[col] = calculate_flexible_momentum(s, 0.33, 0.33, 0.33, False)
                selected = sorted([k for k,v in scores.items() if not pd.isna(v)], 
                                 key=lambda x: scores[x], reverse=True)[:top_n]
        
        # Execution
        entry = i + 1
        exit_i = min(i + 1 + holding_days, len(nav) - 1)
        
        b_ret = 0.0
        if benchmark_series is not None:
            try: b_ret = (benchmark_series.asof(nav.index[exit_i]) / benchmark_series.asof(nav.index[entry])) - 1
            except: pass
        
        port_ret = 0.0
        hit_rate = 0.0
        hits = 0
        
        # Calculate returns and get actual top performers
        period_ret_all = (nav.iloc[exit_i] / nav.iloc[entry]) - 1
        period_ret_all = period_ret_all.dropna()
        
        actual_top_funds = period_ret_all.nlargest(target_n).index.tolist()
        actual_top_names = [scheme_map.get(f, f) for f in actual_top_funds]
        actual_top_returns = [period_ret_all[f] for f in actual_top_funds]
        
        if selected:
            port_ret = period_ret_all[selected].mean()
            hits = len(set(selected).intersection(set(actual_top_funds)))
            hit_rate = hits / top_n if top_n > 0 else 0
        
        cap *= (1 + (port_ret if not pd.isna(port_ret) else 0))
        b_cap *= (1 + b_ret)
        
        # Store basic history
        history.append({
            'date': date,
            'selected': selected,
            'return': port_ret,
            'hit_rate': hit_rate,
            'regime': regime_status
        })
        
        # Store detailed trade info
        selected_names = [scheme_map.get(f, f) for f in selected]
        selected_returns = [period_ret_all.get(f, np.nan) for f in selected]
        
        # Check which selected funds were hits
        selected_hits = [f in actual_top_funds for f in selected]
        
        detailed_trades.append({
            'Period Start': date.strftime('%Y-%m-%d'),
            'Period End': nav.index[exit_i].strftime('%Y-%m-%d'),
            'Regime': regime_status,
            'Selected Fund 1': selected_names[0] if len(selected_names) > 0 else '',
            'Return 1': selected_returns[0] if len(selected_returns) > 0 else np.nan,
            'Hit 1': '✅' if len(selected_hits) > 0 and selected_hits[0] else '❌',
            'Selected Fund 2': selected_names[1] if len(selected_names) > 1 else '',
            'Return 2': selected_returns[1] if len(selected_returns) > 1 else np.nan,
            'Hit 2': '✅' if len(selected_hits) > 1 and selected_hits[1] else '❌',
            'Selected Fund 3': selected_names[2] if len(selected_names) > 2 else '',
            'Return 3': selected_returns[2] if len(selected_returns) > 2 else np.nan,
            'Hit 3': '✅' if len(selected_hits) > 2 and selected_hits[2] else '❌',
            'Portfolio Return': port_ret,
            'Benchmark Return': b_ret,
            'Hits': hits,
            'Hit Rate': hit_rate,
            'Actual Top 1': actual_top_names[0] if len(actual_top_names) > 0 else '',
            'Actual Return 1': actual_top_returns[0] if len(actual_top_returns) > 0 else np.nan,
            'Actual Top 2': actual_top_names[1] if len(actual_top_names) > 1 else '',
            'Actual Return 2': actual_top_returns[1] if len(actual_top_returns) > 1 else np.nan,
            'Actual Top 3': actual_top_names[2] if len(actual_top_names) > 2 else '',
            'Actual Return 3': actual_top_returns[2] if len(actual_top_returns) > 2 else np.nan,
        })
        
        eq_curve.append({'date': nav.index[exit_i], 'value': cap})
        bench_curve.append({'date': nav.index[exit_i], 'value': b_cap})
    
    return pd.DataFrame(history), pd.DataFrame(eq_curve), pd.DataFrame(bench_curve), pd.DataFrame(detailed_trades)


def display_strategy_results(nav_df, scheme_map, benchmark, strat_key, strat_name, mom_config, top_n, target_n, holding):
    """Display comprehensive results for a strategy including detailed trade history."""
    
    with st.spinner(f"Running {strat_name} backtest..."):
        history, eq_curve, bench_curve, detailed_trades = run_backtest_detailed(
            nav_df, strat_key, top_n, target_n, holding, {}, mom_config, benchmark, scheme_map
        )
    
    if eq_curve.empty:
        st.warning("No trades generated. Insufficient data.")
        return
    
    # Summary Metrics
    years = (eq_curve.iloc[-1]['date'] - eq_curve.iloc[0]['date']).days / 365.25
    strat_cagr = (eq_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
    bench_cagr = (bench_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
    avg_hit = history['hit_rate'].mean()
    total_trades = len(history)
    total_hits = int(detailed_trades['Hits'].sum()) if 'Hits' in detailed_trades.columns else 0
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📈 Strategy CAGR", f"{strat_cagr*100:.2f}%")
    col2.metric("📊 Benchmark CAGR", f"{bench_cagr*100:.2f}%")
    col3.metric("🎯 Outperformance", f"{(strat_cagr-bench_cagr)*100:+.2f}%")
    col4.metric("🏆 Avg Hit Rate", f"{avg_hit*100:.1f}%")
    col5.metric("📋 Total Trades", f"{total_trades}")
    
    # Sub-tabs for different views
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "📈 Equity Curve", 
        "📋 All Trades Detail", 
        "🎯 Hit Analysis",
        "📊 Period Summary"
    ])
    
    with sub_tab1:
        # Equity curve chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_curve['date'], y=eq_curve['value'],
            name='Strategy', line=dict(color='green', width=2),
            fill='tozeroy', fillcolor='rgba(0, 200, 0, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=bench_curve['date'], y=bench_curve['value'],
            name='Benchmark (Nifty 100)', line=dict(color='gray', width=2, dash='dot')
        ))
        fig.update_layout(
            height=400, 
            title=f'{strat_name} - Equity Curve',
            yaxis_title='Value (100 = Start)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Hit rate over time chart
        if not detailed_trades.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=detailed_trades['Period Start'],
                y=detailed_trades['Hit Rate'] * 100,
                name='Hit Rate',
                marker_color=detailed_trades['Hit Rate'].apply(
                    lambda x: 'green' if x >= 0.5 else 'orange' if x > 0 else 'red'
                )
            ))
            fig2.add_hline(y=avg_hit*100, line_dash="dash", line_color="blue", 
                          annotation_text=f"Avg: {avg_hit*100:.1f}%")
            fig2.update_layout(
                height=300,
                title='Hit Rate by Period',
                yaxis_title='Hit Rate %',
                xaxis_title='Period'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with sub_tab2:
        st.markdown("### 📋 Complete Trade History")
        st.markdown(f"*Showing all {len(detailed_trades)} trading periods with selected funds and actual top performers*")
        
        if not detailed_trades.empty:
            # Format the display
            display_df = detailed_trades.copy()
            
            # Create a cleaner view
            st.dataframe(
                display_df.style.format({
                    'Return 1': '{:.2%}',
                    'Return 2': '{:.2%}',
                    'Return 3': '{:.2%}',
                    'Portfolio Return': '{:.2%}',
                    'Benchmark Return': '{:.2%}',
                    'Hit Rate': '{:.0%}',
                    'Actual Return 1': '{:.2%}',
                    'Actual Return 2': '{:.2%}',
                    'Actual Return 3': '{:.2%}',
                }).applymap(
                    lambda x: 'background-color: #90EE90' if x == '✅' else 
                              'background-color: #FFB6C1' if x == '❌' else '',
                    subset=['Hit 1', 'Hit 2', 'Hit 3']
                ),
                use_container_width=True,
                height=500
            )
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Trade History (CSV)",
                csv,
                f"{strat_key}_trade_history.csv",
                "text/csv"
            )
    
    with sub_tab3:
        st.markdown("### 🎯 Hit Rate Analysis")
        
        if not detailed_trades.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Hit rate distribution
                hit_counts = detailed_trades['Hits'].value_counts().sort_index()
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"{int(k)} hits" for k in hit_counts.index],
                        y=hit_counts.values,
                        marker_color=['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(hit_counts)]
                    )
                ])
                fig.update_layout(
                    title=f'Distribution of Hits per Period (out of {top_n} picks)',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats
                st.markdown("**Hit Rate Statistics:**")
                st.write(f"- Periods with 0 hits: {(detailed_trades['Hits'] == 0).sum()} ({(detailed_trades['Hits'] == 0).mean()*100:.1f}%)")
                st.write(f"- Periods with 1+ hits: {(detailed_trades['Hits'] >= 1).sum()} ({(detailed_trades['Hits'] >= 1).mean()*100:.1f}%)")
                st.write(f"- Periods with 2+ hits: {(detailed_trades['Hits'] >= 2).sum()} ({(detailed_trades['Hits'] >= 2).mean()*100:.1f}%)")
                if top_n >= 3:
                    st.write(f"- Periods with 3+ hits: {(detailed_trades['Hits'] >= 3).sum()} ({(detailed_trades['Hits'] >= 3).mean()*100:.1f}%)")
            
            with col2:
                # Returns when hitting vs missing
                hit_periods = detailed_trades[detailed_trades['Hits'] > 0]['Portfolio Return']
                miss_periods = detailed_trades[detailed_trades['Hits'] == 0]['Portfolio Return']
                
                fig = go.Figure()
                if len(hit_periods) > 0:
                    fig.add_trace(go.Box(y=hit_periods * 100, name='Periods with Hits', marker_color='green'))
                if len(miss_periods) > 0:
                    fig.add_trace(go.Box(y=miss_periods * 100, name='Periods with No Hits', marker_color='red'))
                fig.update_layout(title='Returns Distribution: Hits vs No Hits', yaxis_title='Return %', height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Return Statistics:**")
                if len(hit_periods) > 0:
                    st.write(f"- Avg return when hitting: {hit_periods.mean()*100:.2f}%")
                if len(miss_periods) > 0:
                    st.write(f"- Avg return when missing: {miss_periods.mean()*100:.2f}%")
                st.write(f"- Overall avg return: {detailed_trades['Portfolio Return'].mean()*100:.2f}%")
    
    with sub_tab4:
        st.markdown("### 📊 Period-by-Period Summary")
        
        if not detailed_trades.empty:
            # Simplified view showing just selections and outcomes
            summary_data = []
            for _, row in detailed_trades.iterrows():
                selected = []
                for i in range(1, 4):
                    if row.get(f'Selected Fund {i}', ''):
                        ret = row.get(f'Return {i}', np.nan)
                        hit = row.get(f'Hit {i}', '')
                        selected.append(f"{row[f'Selected Fund {i}'][:25]} ({ret*100:.1f}%) {hit}")
                
                actual = []
                for i in range(1, 4):
                    if row.get(f'Actual Top {i}', ''):
                        ret = row.get(f'Actual Return {i}', np.nan)
                        actual.append(f"{row[f'Actual Top {i}'][:25]} ({ret*100:.1f}%)")
                
                summary_data.append({
                    'Period': f"{row['Period Start']} → {row['Period End']}",
                    'Selected Funds': ' | '.join(selected),
                    'Hits': f"{int(row['Hits'])}/{top_n}",
                    'Port Return': f"{row['Portfolio Return']*100:.1f}%",
                    'Actual Top Funds': ' | '.join(actual)
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, height=500)


def render_backtest_tab():
    """Render the Backtester tab with improved UI and detailed trade history."""
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0;">🚀 Strategy Backtester</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Test fund selection strategies with detailed trade-by-trade analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        category = st.selectbox("📁 Category", list(FILE_MAPPING.keys()), key="bt_cat")
    with col2:
        top_n = st.number_input("🎯 Funds to Select", 1, 10, 3, key="bt_topn")
    with col3:
        target_n = st.number_input("🏆 Target Top N", 1, 15, 5, key="bt_target")
    with col4:
        holding = st.selectbox("📅 Holding Period", [63, 126, 252, 378, 504], index=1,
                               format_func=lambda x: f"{x} days (~{x//21}M)")
    
    st.divider()
    
    # Load data
    with st.spinner("Loading data..."):
        nav_df, scheme_map = load_fund_data_raw(category)
        benchmark = load_nifty_data()
    
    if nav_df is None:
        st.error("Could not load data.")
        return
    
    st.success(f"✅ Loaded {len(nav_df.columns)} funds | Data: {nav_df.index.min().strftime('%Y-%m')} to {nav_df.index.max().strftime('%Y-%m')}")
    
    # Strategy definitions
    strategies = {
        '🚀 Momentum': ('momentum', {'w_3m': 0.33, 'w_6m': 0.33, 'w_12m': 0.33, 'risk_adjust': True}),
        '⚖️ Sharpe': ('sharpe', {}),
        '🎯 Sortino': ('sortino', {}),
        '🚦 Regime Switch': ('regime_switch', {}),
        '⚓ Stable Momentum': ('stable_momentum', {}),
        '🛡️ Elimination': ('elimination', {}),
        '📈 Consistency': ('consistency', {})
    }
    
    tabs = st.tabs(list(strategies.keys()) + ['📊 Compare All'])
    
    # Individual strategy tabs
    for idx, (tab_name, (strat_key, mom_config)) in enumerate(strategies.items()):
        with tabs[idx]:
            st.markdown(f"### {tab_name} Strategy")
            
            # Strategy description
            descriptions = {
                'momentum': "Ranks funds by weighted average of 3M, 6M, 12M returns (risk-adjusted).",
                'sharpe': "Ranks funds by Sharpe Ratio (excess return / volatility).",
                'sortino': "Ranks funds by Sortino Ratio (excess return / downside volatility).",
                'regime_switch': "Uses Momentum in bull markets (price > 200 DMA), Sharpe in bear markets.",
                'stable_momentum': "Selects top momentum funds, then filters for lowest drawdown.",
                'elimination': "Eliminates worst 25% by drawdown, worst 25% by volatility, picks top by Sharpe.",
                'consistency': "Requires fund to be top 50% for 3+ of last 4 quarters, then picks by momentum."
            }
            st.info(f"📌 **Logic:** {descriptions.get(strat_key, 'Custom strategy')}")
            
            display_strategy_results(nav_df, scheme_map, benchmark, strat_key, tab_name, 
                                    mom_config, top_n, target_n, holding)
    
    # Compare All tab
    with tabs[-1]:
        st.markdown("### 📊 Strategy Comparison")
        st.markdown("*Compare all strategies side by side*")
        
        results = []
        all_equity_curves = {}
        
        progress = st.progress(0)
        for idx, (name, (strat_key, mom_config)) in enumerate(strategies.items()):
            history, eq_curve, bench_curve, detailed = run_backtest_detailed(
                nav_df, strat_key, top_n, target_n, holding, {}, mom_config, benchmark, scheme_map
            )
            
            if not eq_curve.empty:
                years = (eq_curve.iloc[-1]['date'] - eq_curve.iloc[0]['date']).days / 365.25
                cagr = (eq_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                bench_cagr_val = (bench_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                hit_rate = history['hit_rate'].mean()
                max_dd = calculate_max_dd(pd.Series(eq_curve['value'].values, index=eq_curve['date']))
                
                # Additional metrics
                win_rate = (history['return'] > 0).mean()
                avg_return = history['return'].mean()
                
                results.append({
                    'Strategy': name,
                    'CAGR %': cagr * 100,
                    'Benchmark CAGR %': bench_cagr_val * 100,
                    'Alpha %': (cagr - bench_cagr_val) * 100,
                    'Max DD %': max_dd * 100 if max_dd else 0,
                    'Hit Rate %': hit_rate * 100,
                    'Win Rate %': win_rate * 100,
                    'Avg Period Return %': avg_return * 100,
                    'Total Trades': len(history)
                })
                
                all_equity_curves[name] = eq_curve
            
            progress.progress((idx + 1) / len(strategies))
        
        progress.empty()
        
        # Results table
        if results:
            results_df = pd.DataFrame(results).sort_values('CAGR %', ascending=False)
            
            st.dataframe(
                results_df.style.format({
                    'CAGR %': '{:.2f}',
                    'Benchmark CAGR %': '{:.2f}',
                    'Alpha %': '{:+.2f}',
                    'Max DD %': '{:.2f}',
                    'Hit Rate %': '{:.1f}',
                    'Win Rate %': '{:.1f}',
                    'Avg Period Return %': '{:.2f}',
                    'Total Trades': '{:.0f}'
                }).background_gradient(subset=['CAGR %', 'Alpha %'], cmap='RdYlGn')
                .background_gradient(subset=['Hit Rate %'], cmap='Greens'),
                use_container_width=True
            )
        
        # Comparison chart
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        
        for idx, (name, eq_curve) in enumerate(all_equity_curves.items()):
            fig.add_trace(go.Scatter(
                x=eq_curve['date'], 
                y=eq_curve['value'],
                name=name,
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
        
        if 'bench_curve' in dir() and not bench_curve.empty:
            fig.add_trace(go.Scatter(
                x=bench_curve['date'], 
                y=bench_curve['value'],
                name='Benchmark (Nifty 100)', 
                line=dict(color='black', width=2, dash='dot')
            ))
        
        fig.update_layout(
            height=500, 
            title='Strategy Equity Curves Comparison',
            yaxis_title='Value (100 = Start)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Hit rate comparison chart
        if results:
            fig2 = go.Figure(data=[
                go.Bar(
                    x=[r['Strategy'] for r in results],
                    y=[r['Hit Rate %'] for r in results],
                    marker_color=[colors[i % len(colors)] for i in range(len(results))],
                    text=[f"{r['Hit Rate %']:.1f}%" for r in results],
                    textposition='outside'
                )
            ])
            fig2.update_layout(
                height=350,
                title='Hit Rate Comparison by Strategy',
                yaxis_title='Hit Rate %'
            )
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# 10. MAIN APP
# ============================================================================

def main():
    # Title
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h1 style="margin: 0; border: none;">📈 Fund Analysis Pro</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Comprehensive mutual fund analysis and backtesting platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2 = st.tabs(["📊 Category Explorer", "🚀 Strategy Backtester"])
    
    with tab1:
        render_explorer_tab()
    
    with tab2:
        render_backtest_tab()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #999; font-size: 0.8rem; margin-top: 40px; border-top: 1px solid #eee;">
        Fund Analysis Pro • Data updated through Dec 2025 • Risk-free rate: 6%
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
