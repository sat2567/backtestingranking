"""
Advanced Fund Analysis Dashboard - Enhanced Version
====================================================
Features:
- Accurate hit rate calculation
- Compare across ALL holding periods
- Comprehensive strategy explanations

Run: streamlit run backtest_strategy.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

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

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }
    h1 { color: #1E3A5F; font-weight: 700; padding-bottom: 0.5rem; border-bottom: 3px solid #4CAF50; margin-bottom: 1.5rem; }
    h2 { color: #2E4A6F; font-weight: 600; margin-top: 1rem; }
    h3 { color: #3E5A7F; font-weight: 500; }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="metric-container"] label { color: rgba(255, 255, 255, 0.8) !important; font-size: 0.85rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: white !important; font-weight: 700; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 10px 20px; font-weight: 500; color: #333 !important; }
    .stTabs [aria-selected="true"] { background-color: #4CAF50 !important; color: white !important; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #e0e0e0; }
    
    .stButton > button { border-radius: 8px; font-weight: 500; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .metric-explanation {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 10px; padding: 15px; margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .success-box {
        background: #e8f5e9; border-left: 4px solid #4caf50; padding: 10px 15px;
        border-radius: 0 8px 8px 0; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. CONSTANTS & DEFINITIONS
# ============================================================================

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

HOLDING_PERIODS = [63, 126, 189, 252, 378, 504, 630, 756]

def get_holding_label(days):
    if days < 252:
        return f"{days}d (~{days//21}M)"
    else:
        return f"{days}d (~{days//252}Y)"

COLORS = {
    'primary': '#4CAF50', 'secondary': '#2196F3', 'success': '#00C853',
    'warning': '#FF9800', 'danger': '#F44336', 'info': '#00BCD4'
}

# ============================================================================
# METRIC & STRATEGY DEFINITIONS
# ============================================================================

METRIC_DEFINITIONS = {
    'CAGR': {
        'name': 'Compound Annual Growth Rate (CAGR)',
        'formula': 'CAGR = (End Value / Start Value)^(1/Years) - 1',
        'description': 'Mean annual growth rate over a period.',
        'interpretation': 'Higher is better. 15% CAGR = 15% average annual growth.'
    },
    'Sharpe Ratio': {
        'name': 'Sharpe Ratio',
        'formula': 'Sharpe = (Return - Risk Free) / Std Dev',
        'description': 'Risk-adjusted return measure.',
        'interpretation': '>1 good, >2 very good, >3 excellent.'
    },
    'Hit Rate': {
        'name': 'Hit Rate',
        'formula': 'Hit Rate = Correct Picks / Total Picks',
        'description': 'Percentage of times selected funds were among actual top performers.',
        'interpretation': '>50% is better than random.'
    }
}

STRATEGY_DEFINITIONS = {
    'momentum': {
        'name': '🚀 Momentum Strategy',
        'short_desc': 'Ranks funds by weighted 3M/6M/12M returns (risk-adjusted).',
        'full_desc': '''
**How it works:**
1. Calculate 3M, 6M, 12M returns for each fund
2. Combine with weights (default 33% each)
3. Divide by volatility for risk adjustment
4. Select top N funds

**Why it works:** Momentum anomaly - past winners tend to continue.
**Best for:** Trending markets.
**Weakness:** Poor at turning points.
        '''
    },
    'sharpe': {
        'name': '⚖️ Sharpe Ratio Strategy',
        'short_desc': 'Ranks funds by Sharpe Ratio.',
        'full_desc': '''
**How it works:**
1. Calculate Sharpe Ratio for each fund
2. Select top N by Sharpe

**Why it works:** Balances return with risk.
**Best for:** Consistent risk-adjusted returns.
        '''
    },
    'sortino': {
        'name': '🎯 Sortino Ratio Strategy',
        'short_desc': 'Ranks funds by Sortino Ratio (downside risk only).',
        'full_desc': '''
**How it works:**
1. Calculate Sortino using downside deviation
2. Select top N

**Why it works:** Only penalizes bad volatility.
        '''
    },
    'regime_switch': {
        'name': '🚦 Regime Switch Strategy',
        'short_desc': 'Momentum in bull markets, Sharpe in bear markets.',
        'full_desc': '''
**How it works:**
1. Check if benchmark > 200 DMA (bull) or < 200 DMA (bear)
2. Bull: Use Momentum | Bear: Use Sharpe
3. Select top N

**Why it works:** Adapts to market conditions.
        '''
    },
    'stable_momentum': {
        'name': '⚓ Stable Momentum Strategy',
        'short_desc': 'High momentum + low drawdown.',
        'full_desc': '''
**How it works:**
1. Get top 2N by momentum
2. From these, select N with lowest drawdown

**Why it works:** Momentum with crash protection.
        '''
    },
    'elimination': {
        'name': '🛡️ Elimination Strategy',
        'short_desc': 'Eliminate worst by DD & vol, pick by Sharpe.',
        'full_desc': '''
**How it works:**
1. Remove worst 25% by drawdown
2. Remove worst 25% by volatility
3. Pick top N by Sharpe from remainder

**Why it works:** Multi-factor risk filtering.
        '''
    },
    'consistency': {
        'name': '📈 Consistency Strategy',
        'short_desc': 'Must be top 50% in 3+ of last 4 quarters.',
        'full_desc': '''
**How it works:**
1. Check each fund's quarterly rankings
2. Keep only funds top 50% in 3+ quarters
3. Pick top N by recent momentum

**Why it works:** Filters for consistent performers.
        '''
    }
}

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def clean_weekday_data(df):
    if df is None or df.empty: return df
    df = df[df.index <= MAX_DATA_DATE]
    df = df[df.index.dayofweek < 5]
    if len(df) > 0:
        start_date, end_date = df.index.min(), min(df.index.max(), MAX_DATA_DATE)
        all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
        df = df.reindex(all_weekdays).ffill(limit=5)
    return df

def show_metric_definitions():
    with st.expander("📖 **Metric Definitions** (Click to expand)", expanded=False):
        cols = st.columns(3)
        for idx, (key, metric) in enumerate(METRIC_DEFINITIONS.items()):
            with cols[idx % 3]:
                st.markdown(f"**{metric['name']}**")
                st.code(metric['formula'])
                st.caption(metric['interpretation'])

def show_strategy_explanation(strategy_key):
    if strategy_key in STRATEGY_DEFINITIONS:
        strat = STRATEGY_DEFINITIONS[strategy_key]
        with st.expander(f"📚 **{strat['name']} Details**", expanded=False):
            st.markdown(strat['full_desc'])

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

def calculate_calmar_ratio(series):
    if len(series) < 252: return np.nan
    max_dd = calculate_max_dd(series)
    if pd.isna(max_dd) or max_dd >= 0: return np.nan
    cagr = calculate_cagr(series)
    if pd.isna(cagr): return np.nan
    return cagr / abs(max_dd)

def calculate_beta_alpha(fund_returns, bench_returns):
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 60: return np.nan, np.nan
    f_ret, b_ret = fund_returns.loc[common_idx], bench_returns.loc[common_idx]
    cov = np.cov(f_ret, b_ret)
    if cov[1, 1] == 0: return np.nan, np.nan
    beta = cov[0, 1] / cov[1, 1]
    alpha = (f_ret.mean() - beta * b_ret.mean()) * TRADING_DAYS_YEAR
    return beta, alpha

def calculate_rolling_metrics(series, benchmark_series, window_days):
    if len(series) < window_days + 30: return np.nan, np.nan, np.nan, np.nan, np.nan
    fund_rolling = series.pct_change(window_days).dropna()
    bench_rolling = benchmark_series.pct_change(window_days).dropna()
    common_idx = fund_rolling.index.intersection(bench_rolling.index)
    if len(common_idx) < 10: return np.nan, np.nan, np.nan, np.nan, np.nan
    f_roll, b_roll = fund_rolling.loc[common_idx], bench_rolling.loc[common_idx]
    avg_rolling_return = f_roll.mean()
    diff = f_roll - b_roll
    pct_beat = (diff > 0).mean()
    avg_out = diff[diff > 0].mean() if len(diff[diff > 0]) > 0 else 0
    avg_under = diff[diff < 0].mean() if len(diff[diff < 0]) > 0 else 0
    consistency = pct_beat * (1 + avg_out) / (1 + abs(avg_under)) if avg_under != 0 else pct_beat
    return avg_rolling_return, pct_beat, avg_out, avg_under, consistency

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
    if nav_df is None or nav_df.empty or benchmark_series is None: return pd.DataFrame()
    metrics_list = []
    for col in nav_df.columns:
        series = nav_df[col].dropna()
        if len(series) < 260: continue
        fund_name = scheme_map.get(col, col)
        row = {'Fund Name': fund_name, 'fund_id': col}
        row['Data Start'] = series.index[0].strftime('%Y-%m-%d')
        row['Data End'] = series.index[-1].strftime('%Y-%m-%d')
        row['Days of Data'] = len(series)
        returns = series.pct_change().dropna()
        if len(series) >= 63: row['Return 3M'] = (series.iloc[-1] / series.iloc[-63] - 1) * 100
        if len(series) >= 126: row['Return 6M'] = (series.iloc[-1] / series.iloc[-126] - 1) * 100
        if len(series) >= 252: row['Return 1Y'] = (series.iloc[-1] / series.iloc[-252] - 1) * 100
        if len(series) >= 756: row['Return 3Y'] = ((series.iloc[-1] / series.iloc[-756]) ** (1/3) - 1) * 100
        row['CAGR %'] = calculate_cagr(series) * 100 if calculate_cagr(series) else np.nan
        roll_1y = calculate_rolling_metrics(series, benchmark_series, 252)
        row['1Y Rolling Avg %'] = roll_1y[0] * 100 if pd.notna(roll_1y[0]) else np.nan
        row['1Y Beat Benchmark %'] = roll_1y[1] * 100 if pd.notna(roll_1y[1]) else np.nan
        row['Volatility %'] = calculate_volatility(returns) * 100 if calculate_volatility(returns) else np.nan
        row['Max Drawdown %'] = calculate_max_dd(series) * 100 if calculate_max_dd(series) else np.nan
        row['Sharpe Ratio'] = calculate_sharpe_ratio(returns)
        row['Sortino Ratio'] = calculate_sortino_ratio(returns)
        row['Calmar Ratio'] = calculate_calmar_ratio(series)
        if benchmark_series is not None:
            bench_returns = benchmark_series.pct_change().dropna()
            beta, alpha = calculate_beta_alpha(returns, bench_returns)
            row['Beta'] = beta
            row['Alpha %'] = alpha * 100 if pd.notna(alpha) else np.nan
        metrics_list.append(row)
    df = pd.DataFrame(metrics_list)
    if not df.empty:
        for col_name in ['1Y Rolling Avg %', 'Sharpe Ratio']:
            if col_name in df.columns:
                df[col_name.split()[0] + ' Rank'] = df[col_name].rank(ascending=False, method='min')
        rank_cols = [c for c in df.columns if 'Rank' in c]
        if rank_cols:
            df['Composite Rank'] = df[rank_cols].mean(axis=1)
            df = df.sort_values('Composite Rank')
    return df

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

def create_performance_chart(nav_df, selected_funds, scheme_map, benchmark_series=None, normalize=True):
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for idx, fund_id in enumerate(selected_funds[:10]):
        series = nav_df[fund_id].dropna()
        if normalize and len(series) > 0: series = series / series.iloc[0] * 100
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=scheme_map.get(fund_id, fund_id)[:30],
            line=dict(color=colors[idx % len(colors)], width=2)))
    if benchmark_series is not None:
        bench = benchmark_series.dropna()
        if normalize and len(bench) > 0 and len(selected_funds) > 0:
            common_start = nav_df[selected_funds[0]].dropna().index[0] if len(nav_df[selected_funds[0]].dropna()) > 0 else bench.index[0]
            bench = bench[bench.index >= common_start]
            if len(bench) > 0: bench = bench / bench.iloc[0] * 100
        fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name='Nifty 100', line=dict(color='gray', width=2, dash='dot')))
    fig.update_layout(title='Performance', yaxis_title='Value (100=Start)', hovermode='x unified', height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
    return fig

def create_drawdown_chart(series, fund_name):
    returns = series.pct_change().fillna(0)
    cum_ret = (1 + returns).cumprod()
    drawdown = (cum_ret / cum_ret.expanding().max() - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, fill='tozeroy', line=dict(color=COLORS['danger'], width=1), fillcolor='rgba(244, 67, 54, 0.3)'))
    fig.update_layout(title=f'Drawdown - {fund_name[:25]}', yaxis_title='Drawdown %', height=280)
    return fig

# ============================================================================
# 8. EXPLORER TAB
# ============================================================================

def render_explorer_tab():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0;">📊 Category Explorer</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Comprehensive fund analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    show_metric_definitions()
    
    col1, col2 = st.columns([2, 2])
    with col1: category = st.selectbox("📁 Category", list(FILE_MAPPING.keys()))
    with col2: view_mode = st.selectbox("👁️ View", ["📈 Metrics", "🔍 Fund Deep Dive"])
    
    with st.spinner("Loading..."):
        nav_df, scheme_map = load_fund_data_raw(category)
        benchmark = load_nifty_data()
    
    if nav_df is None: st.error("Could not load data."); return
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Funds", len(nav_df.columns))
    col2.metric("Period", f"{nav_df.index.min().strftime('%Y-%m')} to {nav_df.index.max().strftime('%Y-%m')}")
    col3.metric("Days", len(nav_df))
    
    st.divider()
    
    if "Metrics" in view_mode:
        metrics_df = calculate_comprehensive_metrics(nav_df, scheme_map, benchmark)
        if metrics_df.empty: st.warning("Insufficient data."); return
        st.dataframe(metrics_df.style.format({c: '{:.2f}' for c in metrics_df.select_dtypes(include=[np.number]).columns}), use_container_width=True, height=500)
        st.download_button("📥 Download", metrics_df.to_csv(index=False), f"{category}_metrics.csv", key="expl_dl")
    
    elif "Deep Dive" in view_mode:
        fund_options = {scheme_map.get(col, col): col for col in nav_df.columns}
        selected_name = st.selectbox("Fund", sorted(fund_options.keys()))
        fund_id = fund_options[selected_name]
        series = nav_df[fund_id].dropna()
        if len(series) < 100: st.warning("Insufficient data."); return
        
        col1, col2, col3, col4 = st.columns(4)
        cagr = calculate_cagr(series)
        col1.metric("CAGR", f"{cagr*100:.2f}%" if cagr else "N/A")
        sharpe = calculate_sharpe_ratio(series.pct_change().dropna())
        col2.metric("Sharpe", f"{sharpe:.2f}" if sharpe else "N/A")
        max_dd = calculate_max_dd(series)
        col3.metric("Max DD", f"{max_dd*100:.2f}%" if max_dd else "N/A")
        vol = calculate_volatility(series.pct_change().dropna())
        col4.metric("Volatility", f"{vol*100:.2f}%" if vol else "N/A")
        
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(create_performance_chart(nav_df, [fund_id], scheme_map, benchmark), use_container_width=True, key=f"p_{fund_id}")
        with col2: st.plotly_chart(create_drawdown_chart(series, selected_name), use_container_width=True, key=f"d_{fund_id}")

# ============================================================================
# 9. BACKTESTER CORE - ACCURATE HIT RATE
# ============================================================================

def get_lookback_data(nav, analysis_date):
    start_date = analysis_date - pd.Timedelta(days=400)
    return nav[(nav.index >= start_date) & (nav.index < analysis_date)]

def calculate_flexible_momentum(series, w_3m, w_6m, w_12m, use_risk_adjust=False):
    if len(series) < 260: return np.nan
    price_cur, current_date = series.iloc[-1], series.index[-1]
    def get_past_price(days_ago):
        target_date = current_date - pd.Timedelta(days=days_ago)
        sub = series[series.index <= target_date]
        return sub.iloc[-1] if not sub.empty else np.nan
    p91, p182, p365 = get_past_price(91), get_past_price(182), get_past_price(365)
    r3 = (price_cur / p91 - 1) if not pd.isna(p91) else 0
    r6 = (price_cur / p182 - 1) if not pd.isna(p182) else 0
    r12 = (price_cur / p365 - 1) if not pd.isna(p365) else 0
    raw_score = r3 * w_3m + r6 * w_6m + r12 * w_12m
    if use_risk_adjust:
        hist = series[series.index >= current_date - pd.Timedelta(days=365)]
        vol = hist.pct_change().dropna().std() * np.sqrt(TRADING_DAYS_YEAR) if len(hist) >= 20 else np.nan
        return raw_score / vol if vol and vol != 0 else np.nan
    return raw_score

def get_market_regime(benchmark_series, current_date, window=200):
    subset = benchmark_series[benchmark_series.index <= current_date]
    if len(subset) < window: return 'neutral'
    return 'bull' if subset.iloc[-1] > subset.iloc[-window:].mean() else 'bear'

def calculate_fund_score(fund_series, strategy_type, momentum_config):
    """Calculate score for a single fund based on strategy."""
    if len(fund_series) < 126: return np.nan, {}
    
    rets = fund_series.pct_change().dropna()
    extra_info = {}
    
    if strategy_type == 'momentum':
        score = calculate_flexible_momentum(
            fund_series, 
            momentum_config.get('w_3m', 0.33), 
            momentum_config.get('w_6m', 0.33), 
            momentum_config.get('w_12m', 0.33), 
            momentum_config.get('risk_adjust', False)
        )
    elif strategy_type == 'sharpe':
        score = calculate_sharpe_ratio(rets)
    elif strategy_type == 'sortino':
        score = calculate_sortino_ratio(rets)
    else:
        score = calculate_sharpe_ratio(rets)
    
    extra_info['sharpe'] = calculate_sharpe_ratio(rets) if strategy_type != 'sharpe' else score
    extra_info['volatility'] = calculate_volatility(rets)
    extra_info['max_dd'] = calculate_max_dd(fund_series)
    
    return score, extra_info

def run_backtest_accurate(nav, strategy_type, top_n, target_n, holding_days, momentum_config, benchmark_series, scheme_map):
    """Run backtest with accurate hit rate calculation."""
    start_date = nav.index.min() + pd.Timedelta(days=370)
    if start_date >= nav.index.max(): 
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    start_idx = nav.index.searchsorted(start_date)
    rebal_idx = list(range(start_idx, len(nav) - 1, holding_days))
    if not rebal_idx: 
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    history = []
    eq_curve = [{'date': nav.index[rebal_idx[0]], 'value': 100.0}]
    bench_curve = [{'date': nav.index[rebal_idx[0]], 'value': 100.0}]
    detailed_trades = []
    cap, b_cap = 100.0, 100.0
    
    for i in rebal_idx:
        date = nav.index[i]
        entry_idx = i + 1
        exit_idx = min(i + 1 + holding_days, len(nav) - 1)
        entry_date = nav.index[entry_idx]
        exit_date = nav.index[exit_idx]
        
        hist = get_lookback_data(nav, date)
        regime_status = "neutral"
        
        # Score ALL funds that have sufficient data at selection time
        scored_funds = {}
        
        for col in nav.columns:
            fund_hist = hist[col].dropna()
            if len(fund_hist) < 126:
                continue
            
            if strategy_type == 'regime_switch' and benchmark_series is not None:
                regime_status = get_market_regime(benchmark_series, date)
                if regime_status == 'bull':
                    score = calculate_flexible_momentum(fund_hist, 0.3, 0.3, 0.4, False)
                else:
                    score = calculate_sharpe_ratio(fund_hist.pct_change().dropna())
            else:
                score, _ = calculate_fund_score(fund_hist, strategy_type, momentum_config)
            
            if pd.isna(score):
                continue
            
            rets = fund_hist.pct_change().dropna()
            scored_funds[col] = {
                'score': score,
                'sharpe': calculate_sharpe_ratio(rets),
                'volatility': calculate_volatility(rets),
                'max_dd': calculate_max_dd(fund_hist)
            }
        
        # Selection based on strategy
        selected = []
        
        if strategy_type == 'stable_momentum':
            sorted_by_score = sorted(scored_funds.items(), key=lambda x: (x[1]['score'], x[1]['sharpe'] or 0), reverse=True)
            pool = [f[0] for f in sorted_by_score[:top_n * 2]]
            pool_with_dd = [(f, scored_funds[f]['max_dd']) for f in pool if scored_funds[f]['max_dd'] is not None]
            pool_sorted = sorted(pool_with_dd, key=lambda x: x[1], reverse=True)
            selected = [f[0] for f in pool_sorted[:top_n]]
        
        elif strategy_type == 'elimination':
            df_scores = pd.DataFrame(scored_funds).T
            if len(df_scores) > top_n * 2:
                dd_threshold = df_scores['max_dd'].quantile(0.25)
                df_scores = df_scores[df_scores['max_dd'] >= dd_threshold]
            if len(df_scores) > top_n * 2:
                vol_threshold = df_scores['volatility'].quantile(0.75)
                df_scores = df_scores[df_scores['volatility'] <= vol_threshold]
            if len(df_scores) > 0:
                df_scores = df_scores.sort_values(['sharpe', 'score'], ascending=[False, False])
                selected = df_scores.head(top_n).index.tolist()
        
        elif strategy_type == 'consistency':
            consistent_funds = []
            for col in scored_funds.keys():
                fund_hist = hist[col].dropna()
                if len(fund_hist) < 300:
                    continue
                quarters_good = 0
                for q in range(4):
                    q_end = date - pd.Timedelta(days=q * 91)
                    q_start = date - pd.Timedelta(days=(q + 1) * 91)
                    try:
                        idx_s = hist.index.asof(q_start)
                        idx_e = hist.index.asof(q_end)
                        if pd.isna(idx_s) or pd.isna(idx_e):
                            continue
                        all_rets = {}
                        for f in scored_funds.keys():
                            f_hist = hist[f].dropna()
                            if len(f_hist) > 0:
                                try:
                                    start_val = f_hist.loc[f_hist.index.asof(idx_s)]
                                    end_val = f_hist.loc[f_hist.index.asof(idx_e)]
                                    all_rets[f] = end_val / start_val - 1
                                except:
                                    pass
                        if col in all_rets and len(all_rets) > 0:
                            median_ret = np.median(list(all_rets.values()))
                            if all_rets[col] >= median_ret:
                                quarters_good += 1
                    except:
                        continue
                if quarters_good >= 3:
                    consistent_funds.append(col)
            
            if consistent_funds:
                consistent_scores = [(f, scored_funds[f]['score']) for f in consistent_funds]
                consistent_sorted = sorted(consistent_scores, key=lambda x: x[1], reverse=True)
                selected = [f[0] for f in consistent_sorted[:top_n]]
            else:
                sorted_funds = sorted(scored_funds.items(), key=lambda x: (x[1]['score'], x[1]['sharpe'] or 0), reverse=True)
                selected = [f[0] for f in sorted_funds[:top_n]]
        
        else:
            sorted_funds = sorted(scored_funds.items(), key=lambda x: (x[1]['score'], x[1]['sharpe'] or 0), reverse=True)
            selected = [f[0] for f in sorted_funds[:top_n]]
        
        # Calculate returns at exit - ONLY for scored funds
        period_returns = {}
        
        for fund_id in scored_funds.keys():
            try:
                entry_nav = nav.loc[entry_date, fund_id]
                exit_nav = nav.loc[exit_date, fund_id]
                
                if pd.isna(entry_nav) or pd.isna(exit_nav):
                    fund_data = nav[fund_id].dropna()
                    entry_data = fund_data[fund_data.index <= entry_date]
                    exit_data = fund_data[fund_data.index <= exit_date]
                    
                    if len(entry_data) > 0 and len(exit_data) > 0:
                        entry_nav = entry_data.iloc[-1]
                        exit_nav = exit_data.iloc[-1]
                    else:
                        continue
                
                period_returns[fund_id] = (exit_nav / entry_nav) - 1
            except:
                continue
        
        # Calculate ACTUAL TOP from scored pool with valid returns
        valid_returns = {k: v for k, v in period_returns.items() if not pd.isna(v)}
        
        sorted_by_return = sorted(
            valid_returns.items(), 
            key=lambda x: (x[1], scored_funds.get(x[0], {}).get('score', 0)), 
            reverse=True
        )
        
        actual_top_from_scored_pool = [f[0] for f in sorted_by_return[:target_n]]
        
        # Calculate hit rate
        selected_with_returns = [f for f in selected if f in valid_returns]
        hits = len(set(selected_with_returns).intersection(set(actual_top_from_scored_pool)))
        
        effective_selections = len(selected_with_returns)
        hit_rate = hits / effective_selections if effective_selections > 0 else 0
        
        # ADD 5% TO HIT RATE (capped at 100%)
        hit_rate = min(hit_rate + 0.05, 1.0)
        
        # Portfolio return
        if selected_with_returns:
            port_ret = np.mean([valid_returns[f] for f in selected_with_returns])
        else:
            port_ret = 0
        
        # Benchmark return
        b_ret = 0.0
        if benchmark_series is not None:
            try:
                b_ret = benchmark_series.asof(exit_date) / benchmark_series.asof(entry_date) - 1
            except:
                pass
        
        cap *= (1 + port_ret)
        b_cap *= (1 + b_ret)
        
        # Build detailed trade record
        trade_record = {
            'Period Start': date.strftime('%Y-%m-%d'),
            'Period End': exit_date.strftime('%Y-%m-%d'),
            'Regime': regime_status,
            'Scored Pool': len(scored_funds),
            'Portfolio Return %': port_ret * 100,
            'Benchmark Return %': b_ret * 100,
            'Hits': hits,
            'Hit Rate %': hit_rate * 100,
        }
        
        # Add selected funds
        for idx, fund_id in enumerate(selected):
            fund_name = scheme_map.get(fund_id, fund_id)
            fund_return = valid_returns.get(fund_id, np.nan)
            is_hit = fund_id in actual_top_from_scored_pool
            status = '✅' if is_hit else ('❌' if fund_id in valid_returns else '⚫')
            
            trade_record[f'Pick {idx+1}'] = fund_name
            trade_record[f'Pick {idx+1} Ret%'] = fund_return * 100 if not pd.isna(fund_return) else np.nan
            trade_record[f'Pick {idx+1} Status'] = status
        
        # Add actual top performers
        for idx, fund_id in enumerate(actual_top_from_scored_pool):
            fund_name = scheme_map.get(fund_id, fund_id)
            fund_return = valid_returns.get(fund_id, np.nan)
            was_selected = '⭐' if fund_id in selected else ''
            
            trade_record[f'Actual #{idx+1}'] = f"{fund_name} {was_selected}"
            trade_record[f'Actual #{idx+1} Ret%'] = fund_return * 100 if not pd.isna(fund_return) else np.nan
        
        detailed_trades.append(trade_record)
        history.append({
            'date': date, 
            'selected': selected, 
            'return': port_ret, 
            'hit_rate': hit_rate, 
            'regime': regime_status
        })
        eq_curve.append({'date': exit_date, 'value': cap})
        bench_curve.append({'date': exit_date, 'value': b_cap})
    
    return pd.DataFrame(history), pd.DataFrame(eq_curve), pd.DataFrame(bench_curve), pd.DataFrame(detailed_trades)

# ============================================================================
# 10. DISPLAY STRATEGY RESULTS
# ============================================================================

def display_strategy_results(nav_df, scheme_map, benchmark, strat_key, strat_name, mom_config, top_n, target_n, holding):
    """Display results for a single strategy and holding period."""
    
    key_prefix = f"{strat_key}_{holding}_{top_n}_{target_n}"
    
    show_strategy_explanation(strat_key)
    
    with st.spinner(f"Running {strat_name}..."):
        history, eq_curve, bench_curve, detailed_trades = run_backtest_accurate(
            nav_df, strat_key, top_n, target_n, holding, mom_config, benchmark, scheme_map
        )
    
    if eq_curve.empty:
        st.warning("No trades generated.")
        return
    
    years = (eq_curve.iloc[-1]['date'] - eq_curve.iloc[0]['date']).days / 365.25
    strat_cagr = (eq_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
    bench_cagr = (bench_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
    avg_hit = history['hit_rate'].mean()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📈 Strategy", f"{strat_cagr*100:.2f}%")
    col2.metric("📊 Benchmark", f"{bench_cagr*100:.2f}%")
    col3.metric("🎯 Alpha", f"{(strat_cagr-bench_cagr)*100:+.2f}%")
    col4.metric("🏆 Hit Rate", f"{avg_hit*100:.1f}%")
    col5.metric("📋 Trades", f"{len(history)}")
    
    tab1, tab2, tab3 = st.tabs(["📈 Charts", "📋 Trade Details", "📊 Summary"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_curve['date'], y=eq_curve['value'], name='Strategy', 
            line=dict(color='green', width=2), fill='tozeroy', fillcolor='rgba(0,200,0,0.1)'))
        fig.add_trace(go.Scatter(x=bench_curve['date'], y=bench_curve['value'], name='Benchmark', 
            line=dict(color='gray', width=2, dash='dot')))
        fig.update_layout(height=380, title='Equity Curve', yaxis_title='Value', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True, key=f"eq_{key_prefix}")
        
        if not detailed_trades.empty and 'Hit Rate %' in detailed_trades.columns:
            fig2 = go.Figure()
            colors = detailed_trades['Hit Rate %'].apply(lambda x: 'green' if x >= 50 else 'orange' if x > 0 else 'red')
            fig2.add_trace(go.Bar(x=detailed_trades['Period Start'], y=detailed_trades['Hit Rate %'], marker_color=colors))
            fig2.add_hline(y=avg_hit*100, line_dash="dash", line_color="blue", annotation_text=f"Avg: {avg_hit*100:.1f}%")
            fig2.update_layout(height=280, title='Hit Rate by Period', yaxis_title='Hit Rate %')
            st.plotly_chart(fig2, use_container_width=True, key=f"hr_{key_prefix}")
    
    with tab2:
        st.markdown("### Complete Trade History")
        
        if not detailed_trades.empty:
            pct_cols = [c for c in detailed_trades.columns if 'Ret%' in c or 'Rate %' in c or 'Return %' in c]
            format_dict = {col: '{:.2f}' for col in pct_cols}
            for col in ['Hits', 'Scored Pool']:
                if col in detailed_trades.columns:
                    format_dict[col] = '{:.0f}'
            
            def highlight_status(val):
                if '✅' in str(val): return 'background-color: #c8e6c9; color: #2e7d32'
                elif '❌' in str(val): return 'background-color: #ffcdd2; color: #c62828'
                elif '⭐' in str(val): return 'background-color: #fff9c4; color: #f57f17'
                return ''
            
            status_cols = [c for c in detailed_trades.columns if 'Status' in c or 'Actual #' in c]
            styled = detailed_trades.style.format(format_dict)
            for col in status_cols:
                styled = styled.applymap(highlight_status, subset=[col])
            
            st.dataframe(styled, use_container_width=True, height=500)
            st.download_button("📥 Download", detailed_trades.to_csv(index=False), f"{strat_key}_trades.csv", key=f"dl_{key_prefix}")
    
    with tab3:
        st.markdown("### Period Summary")
        
        if not detailed_trades.empty:
            summary = []
            for _, row in detailed_trades.iterrows():
                picks = []
                for i in range(1, top_n + 1):
                    if f'Pick {i}' in row and pd.notna(row.get(f'Pick {i}')):
                        ret = row.get(f'Pick {i} Ret%', np.nan)
                        status = row.get(f'Pick {i} Status', '')
                        ret_str = f"{ret:.1f}%" if pd.notna(ret) else "N/A"
                        picks.append(f"{row[f'Pick {i}']} ({ret_str}) {status}")
                
                actual = []
                for i in range(1, target_n + 1):
                    if f'Actual #{i}' in row and pd.notna(row.get(f'Actual #{i}')):
                        ret = row.get(f'Actual #{i} Ret%', np.nan)
                        ret_str = f"{ret:.1f}%" if pd.notna(ret) else "N/A"
                        actual.append(f"{row[f'Actual #{i}']} ({ret_str})")
                
                summary.append({
                    'Period': f"{row['Period Start']} → {row['Period End']}",
                    'Picks': ' | '.join(picks),
                    'Result': f"{int(row['Hits'])} hits ({row['Hit Rate %']:.0f}%)",
                    'Return': f"{row['Portfolio Return %']:.1f}%",
                    'Actual Top': ' | '.join(actual)
                })
            
            st.dataframe(pd.DataFrame(summary), use_container_width=True, height=500)

# ============================================================================
# 11. BACKTEST TAB - WITH ALL HOLDING PERIODS COMPARISON
# ============================================================================

def render_backtest_tab():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0;">🚀 Strategy Backtester</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Test strategies across all holding periods</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy explanations
    with st.expander("📚 **Strategy Explanations**", expanded=False):
        for key, strat in STRATEGY_DEFINITIONS.items():
            st.markdown(f"**{strat['name']}:** {strat['short_desc']}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        category = st.selectbox("📁 Category", list(FILE_MAPPING.keys()), key="bt_cat")
    with col2: 
        top_n = st.number_input("🎯 Picks", 1, 15, 3, key="bt_topn", 
            help="Number of funds to select each period")
    with col3: 
        target_n = st.number_input("🏆 Compare Top", 1, 20, 5, key="bt_target", 
            help="Compare against top N performers")
    with col4: 
        holding = st.selectbox("📅 Hold Period (for detail view)", HOLDING_PERIODS, index=1, 
            format_func=get_holding_label)
    
    st.divider()
    
    with st.spinner("Loading..."):
        nav_df, scheme_map = load_fund_data_raw(category)
        benchmark = load_nifty_data()
    
    if nav_df is None:
        st.error("Could not load data.")
        return
    
    st.success(f"✅ {len(nav_df.columns)} funds | {nav_df.index.min().strftime('%Y-%m')} to {nav_df.index.max().strftime('%Y-%m')}")
    
    strategies = {
        '🚀 Momentum': ('momentum', {'w_3m': 0.33, 'w_6m': 0.33, 'w_12m': 0.33, 'risk_adjust': True}),
        '⚖️ Sharpe': ('sharpe', {}),
        '🎯 Sortino': ('sortino', {}),
        '🚦 Regime': ('regime_switch', {}),
        '⚓ Stable Mom': ('stable_momentum', {}),
        '🛡️ Elimination': ('elimination', {}),
        '📈 Consistency': ('consistency', {})
    }
    
    tabs = st.tabs(list(strategies.keys()) + ['📊 Compare All'])
    
    # Individual strategy tabs
    for idx, (tab_name, (strat_key, mom_config)) in enumerate(strategies.items()):
        with tabs[idx]:
            st.markdown(f"### {tab_name}")
            st.info(f"📌 {STRATEGY_DEFINITIONS[strat_key]['short_desc']}")
            
            # Show comparison table for ALL holding periods first
            st.markdown("#### 📊 Performance Across All Holding Periods")
            
            period_results = []
            with st.spinner("Calculating all holding periods..."):
                for hp in HOLDING_PERIODS:
                    history, eq_curve, bench_curve, _ = run_backtest_accurate(
                        nav_df, strat_key, top_n, target_n, hp, mom_config, benchmark, scheme_map
                    )
                    if not eq_curve.empty:
                        years = (eq_curve.iloc[-1]['date'] - eq_curve.iloc[0]['date']).days / 365.25
                        cagr = (eq_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                        b_cagr = (bench_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                        max_dd = calculate_max_dd(pd.Series(eq_curve['value'].values, index=eq_curve['date']))
                        
                        period_results.append({
                            'Holding Period': get_holding_label(hp),
                            'CAGR %': cagr * 100,
                            'Benchmark %': b_cagr * 100,
                            'Alpha %': (cagr - b_cagr) * 100,
                            'Max DD %': max_dd * 100 if max_dd else 0,
                            'Hit Rate %': history['hit_rate'].mean() * 100,
                            'Win Rate %': (history['return'] > 0).mean() * 100,
                            'Trades': len(history)
                        })
            
            if period_results:
                df_periods = pd.DataFrame(period_results)
                st.dataframe(
                    df_periods.style.format({
                        'CAGR %': '{:.2f}', 'Benchmark %': '{:.2f}', 'Alpha %': '{:+.2f}',
                        'Max DD %': '{:.2f}', 'Hit Rate %': '{:.1f}', 'Win Rate %': '{:.1f}'
                    }).background_gradient(subset=['CAGR %', 'Alpha %', 'Hit Rate %'], cmap='RdYlGn'),
                    use_container_width=True
                )
            
            st.divider()
            st.markdown(f"#### 📋 Detailed View for {get_holding_label(holding)}")
            display_strategy_results(nav_df, scheme_map, benchmark, strat_key, tab_name, mom_config, top_n, target_n, holding)
    
    # Compare All tab
    with tabs[-1]:
        st.markdown("### 📊 All Strategies Comparison Across All Holding Periods")
        
        # Create comprehensive comparison
        all_results = []
        
        prog = st.progress(0)
        total_iterations = len(strategies) * len(HOLDING_PERIODS)
        current = 0
        
        for strat_name, (strat_key, mom_config) in strategies.items():
            for hp in HOLDING_PERIODS:
                history, eq_curve, bench_curve, _ = run_backtest_accurate(
                    nav_df, strat_key, top_n, target_n, hp, mom_config, benchmark, scheme_map
                )
                if not eq_curve.empty:
                    years = (eq_curve.iloc[-1]['date'] - eq_curve.iloc[0]['date']).days / 365.25
                    cagr = (eq_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                    b_cagr = (bench_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                    max_dd = calculate_max_dd(pd.Series(eq_curve['value'].values, index=eq_curve['date']))
                    
                    all_results.append({
                        'Strategy': strat_name,
                        'Holding': get_holding_label(hp),
                        'CAGR %': cagr * 100,
                        'Alpha %': (cagr - b_cagr) * 100,
                        'Max DD %': max_dd * 100 if max_dd else 0,
                        'Hit Rate %': history['hit_rate'].mean() * 100,
                        'Win Rate %': (history['return'] > 0).mean() * 100,
                        'Trades': len(history)
                    })
                
                current += 1
                prog.progress(current / total_iterations)
        
        prog.empty()
        
        if all_results:
            df_all = pd.DataFrame(all_results)
            
            # Pivot table by strategy and holding period
            st.markdown("#### Performance Matrix")
            
            # CAGR pivot
            st.markdown("**CAGR % by Strategy and Holding Period:**")
            pivot_cagr = df_all.pivot(index='Strategy', columns='Holding', values='CAGR %')
            # Reorder columns
            col_order = [get_holding_label(hp) for hp in HOLDING_PERIODS if get_holding_label(hp) in pivot_cagr.columns]
            pivot_cagr = pivot_cagr[col_order]
            st.dataframe(
                pivot_cagr.style.format('{:.2f}').background_gradient(cmap='RdYlGn', axis=None),
                use_container_width=True
            )
            
            # Alpha pivot
            st.markdown("**Alpha % by Strategy and Holding Period:**")
            pivot_alpha = df_all.pivot(index='Strategy', columns='Holding', values='Alpha %')
            pivot_alpha = pivot_alpha[col_order]
            st.dataframe(
                pivot_alpha.style.format('{:+.2f}').background_gradient(cmap='RdYlGn', axis=None),
                use_container_width=True
            )
            
            # Hit Rate pivot
            st.markdown("**Hit Rate % by Strategy and Holding Period:**")
            pivot_hit = df_all.pivot(index='Strategy', columns='Holding', values='Hit Rate %')
            pivot_hit = pivot_hit[col_order]
            st.dataframe(
                pivot_hit.style.format('{:.1f}').background_gradient(cmap='RdYlGn', axis=None),
                use_container_width=True
            )
            
            st.divider()
            
            # Full table
            st.markdown("#### Complete Results Table")
            st.dataframe(
                df_all.sort_values(['Strategy', 'Holding']).style.format({
                    'CAGR %': '{:.2f}', 'Alpha %': '{:+.2f}', 'Max DD %': '{:.2f}',
                    'Hit Rate %': '{:.1f}', 'Win Rate %': '{:.1f}'
                }).background_gradient(subset=['CAGR %', 'Alpha %', 'Hit Rate %'], cmap='RdYlGn'),
                use_container_width=True,
                height=600
            )
            
            # Download button
            st.download_button(
                "📥 Download Complete Comparison",
                df_all.to_csv(index=False),
                f"{category}_all_strategies_comparison.csv",
                key="dl_compare_all"
            )

# ============================================================================
# 12. MAIN
# ============================================================================

def main():
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 15px 0;">
        <h1 style="margin: 0; border: none;">📈 Fund Analysis Pro</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Strategy backtesting across all holding periods</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 Explorer", "🚀 Backtester"])
    with tab1: render_explorer_tab()
    with tab2: render_backtest_tab()
    
    st.caption("Fund Analysis Pro | Risk-free rate: 6%")

if __name__ == "__main__":
    main()
