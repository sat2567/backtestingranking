"""
Advanced Fund Analysis Dashboard - Enhanced Version
====================================================
Features:
- Shows ALL selected fund names for every rebalance period
- Comprehensive Category Explorer with 1Y and 3Y rolling analysis
- Beautiful modern UI with better styling
- Dynamic columns based on number of funds selected
- Complete strategy explanations and metric definitions
- CORRECT HIT RATE: Only considers funds available at selection time

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
    .strategy-card {
        background: white; border-radius: 12px; padding: 20px; margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 5px solid #2196F3;
    }
    .formula-box {
        background: #263238; color: #80cbc4; padding: 10px 15px;
        border-radius: 8px; font-family: monospace; margin: 10px 0;
    }
    .note-box {
        background: #fff3e0; border-left: 4px solid #ff9800; padding: 10px 15px;
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

COLORS = {
    'primary': '#4CAF50', 'secondary': '#2196F3', 'success': '#00C853',
    'warning': '#FF9800', 'danger': '#F44336', 'info': '#00BCD4'
}

# ============================================================================
# METRIC DEFINITIONS
# ============================================================================

METRIC_DEFINITIONS = {
    'CAGR': {
        'name': 'Compound Annual Growth Rate (CAGR)',
        'formula': 'CAGR = (Ending Value / Beginning Value)^(1/Years) - 1',
        'description': 'The mean annual growth rate over a specified period longer than one year.',
        'interpretation': 'Higher is better. A CAGR of 15% means the investment grew at an average of 15% per year.'
    },
    'Sharpe Ratio': {
        'name': 'Sharpe Ratio',
        'formula': 'Sharpe = (Return - Risk Free Rate) / Standard Deviation',
        'description': 'Measures risk-adjusted return. Tells you how much excess return you receive for the extra volatility.',
        'interpretation': '> 1.0 is good, > 2.0 is very good, > 3.0 is excellent.'
    },
    'Sortino Ratio': {
        'name': 'Sortino Ratio',
        'formula': 'Sortino = (Return - Risk Free Rate) / Downside Deviation',
        'description': 'Similar to Sharpe but only penalizes downside volatility.',
        'interpretation': 'Higher is better. More appropriate when returns are not normally distributed.'
    },
    'Max Drawdown': {
        'name': 'Maximum Drawdown',
        'formula': 'Max DD = (Trough Value - Peak Value) / Peak Value',
        'description': 'The maximum observed loss from a peak to a trough.',
        'interpretation': 'Lower (less negative) is better. -20% means the fund lost 20% from its peak at worst.'
    },
    'Hit Rate': {
        'name': 'Hit Rate',
        'formula': 'Hit Rate = Correct Picks / Total Picks',
        'description': 'Percentage of times selected funds were among actual top performers. IMPORTANT: Only considers funds that were available at selection time.',
        'interpretation': '> 50% is better than random. 60%+ indicates good predictive ability.'
    }
}

STRATEGY_DEFINITIONS = {
    'momentum': {
        'name': '🚀 Momentum Strategy',
        'short_desc': 'Ranks funds by weighted average of 3M, 6M, 12M returns (risk-adjusted).',
        'full_desc': '''
**How it works:**
1. Calculate returns over 3 months, 6 months, and 12 months for each fund
2. Combine these returns using weights (default: 33% each)
3. Optionally divide by volatility to get risk-adjusted momentum
4. Select top N funds with highest momentum score

**Formula:**
```
Momentum Score = (w1 × 3M Return) + (w2 × 6M Return) + (w3 × 12M Return)
Risk-Adjusted = Momentum Score / Annualized Volatility
```

**Why it works:** Based on momentum anomaly - past winners tend to continue winning.
**Best used when:** Markets are trending.
**Weakness:** Performs poorly at market turning points.
        ''',
        'params': ['w_3m', 'w_6m', 'w_12m', 'risk_adjust']
    },
    'sharpe': {
        'name': '⚖️ Sharpe Ratio Strategy',
        'short_desc': 'Ranks funds by Sharpe Ratio (excess return per unit of risk).',
        'full_desc': '''
**How it works:**
1. Calculate daily returns for each fund over lookback period
2. Compute excess returns (returns minus risk-free rate)
3. Calculate Sharpe Ratio = Mean Excess Return / Std Dev × √252
4. Select top N funds with highest Sharpe Ratio

**Why it works:** Balances return with risk, penalizes excessive risk-taking.
**Best used when:** You want consistent, risk-adjusted performance.
**Weakness:** Assumes normal distribution, penalizes upside volatility too.
        ''',
        'params': ['lookback period']
    },
    'sortino': {
        'name': '🎯 Sortino Ratio Strategy',
        'short_desc': 'Ranks funds by Sortino Ratio (return per unit of downside risk).',
        'full_desc': '''
**How it works:**
1. Calculate daily returns for each fund
2. Compute downside deviation (std dev of only negative returns)
3. Calculate Sortino = Mean Excess Return / Downside Deviation × √252
4. Select top N funds with highest Sortino Ratio

**Why it works:** Only penalizes downside volatility, not upside.
**Best used when:** Returns are not normally distributed.
**Weakness:** Requires enough negative return days to calculate properly.
        ''',
        'params': ['lookback period']
    },
    'regime_switch': {
        'name': '🚦 Regime Switch Strategy',
        'short_desc': 'Uses Momentum in bull markets, Sharpe in bear markets.',
        'full_desc': '''
**How it works:**
1. Determine market regime using 200-day moving average of benchmark
2. If benchmark price > 200 DMA → Bull Market → Use Momentum
3. If benchmark price < 200 DMA → Bear Market → Use Sharpe Ratio
4. Select top N funds based on regime-appropriate metric

**Why it works:** Momentum works in trending markets, Sharpe in volatile markets.
**Best used when:** Markets alternate between trending and ranging.
**Weakness:** Can whipsaw if price oscillates around the 200 DMA.
        ''',
        'params': ['200-day MA for regime detection']
    },
    'stable_momentum': {
        'name': '⚓ Stable Momentum Strategy',
        'short_desc': 'Selects high momentum funds with lowest drawdown.',
        'full_desc': '''
**How it works:**
1. Calculate momentum scores for all funds
2. Select top 2×N funds by momentum (create a pool)
3. From this pool, calculate maximum drawdown for each
4. Select final N funds with smallest drawdowns

**Why it works:** Gets momentum exposure while controlling crash risk.
**Best used when:** You want momentum with risk control.
**Weakness:** May miss highest returning but volatile funds.
        ''',
        'params': ['momentum weights', 'pool size (2×N)']
    },
    'elimination': {
        'name': '🛡️ Elimination Strategy',
        'short_desc': 'Eliminates worst 25% by drawdown & volatility, picks top by Sharpe.',
        'full_desc': '''
**How it works:**
1. Calculate Max Drawdown, Volatility, and Sharpe for all funds
2. Eliminate bottom 25% by drawdown (worst drawdowns removed)
3. Eliminate top 25% by volatility (most volatile removed)
4. From remaining, select top N by Sharpe Ratio

**Why it works:** Multi-factor elimination reduces risk from multiple angles.
**Best used when:** You want to avoid blow-ups at all costs.
**Weakness:** May eliminate high-return funds that have volatility.
        ''',
        'params': ['drawdown percentile (25%)', 'volatility percentile (75%)']
    },
    'consistency': {
        'name': '📈 Consistency Strategy',
        'short_desc': 'Requires fund to be top 50% for 3+ of last 4 quarters.',
        'full_desc': '''
**How it works:**
1. Look at last 4 quarters of performance
2. For each quarter, check if fund was in top 50%
3. Keep only funds that were top 50% in at least 3 of 4 quarters
4. From these, select top N by recent momentum

**Why it works:** Filters for funds that perform well consistently.
**Best used when:** You value consistency over occasional outperformance.
**Weakness:** May miss improving funds that weren't consistent historically.
        ''',
        'params': ['quarters to check (4)', 'consistency threshold (3/4)']
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
    with st.expander("📖 **Metric Definitions & Formulas** (Click to expand)", expanded=False):
        st.markdown("### Understanding the Metrics")
        cols = st.columns(2)
        for idx, (key, metric) in enumerate(METRIC_DEFINITIONS.items()):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="metric-explanation">
                    <h4>{metric['name']}</h4>
                    <div class="formula-box">{metric['formula']}</div>
                    <p><strong>What it measures:</strong> {metric['description']}</p>
                    <p><strong>Interpretation:</strong> {metric['interpretation']}</p>
                </div>
                """, unsafe_allow_html=True)

def show_strategy_explanation(strategy_key):
    if strategy_key in STRATEGY_DEFINITIONS:
        strat = STRATEGY_DEFINITIONS[strategy_key]
        with st.expander(f"📚 **Understanding {strat['name']}** (Click for details)", expanded=False):
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
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0 or series.iloc[0] <= 0: return np.nan
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
    return cagr / abs(max_dd)

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

def calculate_beta_alpha(fund_returns, bench_returns):
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 60: return np.nan, np.nan
    f_ret, b_ret = fund_returns.loc[common_idx], bench_returns.loc[common_idx]
    cov = np.cov(f_ret, b_ret)
    if cov[1, 1] == 0: return np.nan, np.nan
    beta = cov[0, 1] / cov[1, 1]
    alpha = (f_ret.mean() - beta * b_ret.mean()) * TRADING_DAYS_YEAR
    return beta, alpha

def calculate_information_ratio(fund_returns, bench_returns):
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 30: return np.nan
    active_return = fund_returns.loc[common_idx] - bench_returns.loc[common_idx]
    tracking_error = active_return.std(ddof=1) * np.sqrt(TRADING_DAYS_YEAR)
    if tracking_error == 0: return np.nan
    return (active_return.mean() * TRADING_DAYS_YEAR) / tracking_error

def calculate_capture_score(fund_rets, bench_rets):
    common_idx = fund_rets.index.intersection(bench_rets.index)
    if len(common_idx) < 30: return np.nan, np.nan, np.nan
    f, b = fund_rets.loc[common_idx], bench_rets.loc[common_idx]
    up_market, down_market = b[b > 0], b[b < 0]
    up_cap = f.loc[up_market.index].mean() / up_market.mean() if not up_market.empty and up_market.mean() != 0 else np.nan
    down_cap = f.loc[down_market.index].mean() / down_market.mean() if not down_market.empty and down_market.mean() != 0 else np.nan
    ratio = up_cap / down_cap if pd.notna(up_cap) and pd.notna(down_cap) and down_cap > 0 else np.nan
    return up_cap, down_cap, ratio

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
    avg_outperformance = diff[diff > 0].mean() if len(diff[diff > 0]) > 0 else 0
    avg_underperformance = diff[diff < 0].mean() if len(diff[diff < 0]) > 0 else 0
    consistency = pct_beat * (1 + avg_outperformance) / (1 + abs(avg_underperformance)) if avg_underperformance != 0 else pct_beat
    return avg_rolling_return, pct_beat, avg_outperformance, avg_underperformance, consistency

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
        if len(series) >= 1260: row['Return 5Y'] = ((series.iloc[-1] / series.iloc[-1260]) ** (1/5) - 1) * 100
        row['CAGR %'] = calculate_cagr(series) * 100 if calculate_cagr(series) else np.nan
        roll_1y = calculate_rolling_metrics(series, benchmark_series, 252)
        row['1Y Rolling Avg Return %'] = roll_1y[0] * 100 if pd.notna(roll_1y[0]) else np.nan
        row['1Y % Times Beat Benchmark'] = roll_1y[1] * 100 if pd.notna(roll_1y[1]) else np.nan
        row['1Y Consistency Score'] = roll_1y[4] if pd.notna(roll_1y[4]) else np.nan
        roll_3y = calculate_rolling_metrics(series, benchmark_series, 756)
        row['3Y Rolling Avg Return %'] = roll_3y[0] * 100 if pd.notna(roll_3y[0]) else np.nan
        row['3Y Consistency Score'] = roll_3y[4] if pd.notna(roll_3y[4]) else np.nan
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
            row['Information Ratio'] = calculate_information_ratio(returns, bench_returns)
            up_cap, down_cap, cap_ratio = calculate_capture_score(returns, bench_returns)
            row['Up Capture %'] = up_cap * 100 if pd.notna(up_cap) else np.nan
            row['Down Capture %'] = down_cap * 100 if pd.notna(down_cap) else np.nan
            row['Capture Ratio'] = cap_ratio
        if len(returns) > 30:
            row['VaR 95 %'] = returns.quantile(0.05) * 100
        monthly_returns = series.resample('ME').last().pct_change().dropna()
        if len(monthly_returns) > 6: row['% Positive Months'] = (monthly_returns > 0).mean() * 100
        metrics_list.append(row)
    df = pd.DataFrame(metrics_list)
    if not df.empty:
        for col_name in ['1Y Rolling Avg Return %', '3Y Rolling Avg Return %', 'Sharpe Ratio', '1Y Consistency Score']:
            if col_name in df.columns:
                rank_name = col_name.replace(' %', '').replace(' Ratio', '').replace('Rolling Avg Return', 'Return') + ' Rank'
                df[rank_name] = df[col_name].rank(ascending=False, method='min')
        rank_cols = [c for c in df.columns if 'Rank' in c]
        if rank_cols:
            df['Composite Rank'] = df[rank_cols].mean(axis=1)
            df = df.sort_values('Composite Rank')
    return df

def calculate_quarterly_ranks(nav_df, scheme_map):
    if nav_df is None or nav_df.empty: return pd.DataFrame()
    quarter_ends = pd.date_range(start=nav_df.index.min(), end=nav_df.index.max(), freq='Q')
    history_data = {}
    for q_date in quarter_ends:
        start_lookback = q_date - pd.Timedelta(days=365)
        if start_lookback < nav_df.index.min(): continue
        try:
            idx_now, idx_prev = nav_df.index.asof(q_date), nav_df.index.asof(start_lookback)
            if pd.isna(idx_now) or pd.isna(idx_prev): continue
            rets = (nav_df.loc[idx_now] / nav_df.loc[idx_prev]) - 1
            history_data[q_date.strftime('%Y-Q%q')] = rets.rank(ascending=False, method='min')
        except: continue
    rank_df = pd.DataFrame(history_data)
    rank_df.index = rank_df.index.map(lambda x: scheme_map.get(x, x))
    rank_df = rank_df.dropna(how='all')
    if not rank_df.empty:
        def calc_stats(row):
            valid = row.dropna()
            if len(valid) == 0: return pd.Series({'% Top 3': 0, '% Top 5': 0, '% Top 10': 0, 'Avg Rank': np.nan})
            return pd.Series({'% Top 3': (valid <= 3).mean() * 100, '% Top 5': (valid <= 5).mean() * 100, '% Top 10': (valid <= 10).mean() * 100, 'Avg Rank': valid.mean()})
        stats = rank_df.apply(calc_stats, axis=1)
        rank_df = pd.concat([stats, rank_df], axis=1).sort_values('% Top 5', ascending=False)
    return rank_df

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
        if normalize and len(bench) > 0:
            common_start = max(nav_df[selected_funds[0]].dropna().index[0], bench.index[0])
            bench = bench[bench.index >= common_start]
            if len(bench) > 0: bench = bench / bench.iloc[0] * 100
        fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name='Nifty 100', line=dict(color='gray', width=2, dash='dot')))
    fig.update_layout(title='Performance Comparison', xaxis_title='Date', yaxis_title='Value (Normalized to 100)' if normalize else 'NAV',
        hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5), height=450)
    return fig

def create_drawdown_chart(series, fund_name):
    returns = series.pct_change().fillna(0)
    cum_ret = (1 + returns).cumprod()
    drawdown = (cum_ret / cum_ret.expanding().max() - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, fill='tozeroy', line=dict(color=COLORS['danger'], width=1), fillcolor='rgba(244, 67, 54, 0.3)', name='Drawdown'))
    fig.update_layout(title=f'Drawdown - {fund_name[:30]}', yaxis_title='Drawdown %', height=300)
    return fig

def create_rolling_return_chart(nav_df, fund_id, scheme_map, benchmark_series, window=252):
    series = nav_df[fund_id].dropna()
    fund_rolling = series.pct_change(window).dropna() * 100
    bench_rolling = benchmark_series.pct_change(window).dropna() * 100
    common_idx = fund_rolling.index.intersection(bench_rolling.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=common_idx, y=fund_rolling.loc[common_idx], name=scheme_map.get(fund_id, fund_id)[:25], fill='tozeroy', line=dict(color=COLORS['primary'], width=2)))
    fig.add_trace(go.Scatter(x=common_idx, y=bench_rolling.loc[common_idx], name='Nifty 100', line=dict(color='gray', width=2, dash='dash')))
    fig.update_layout(title=f'{window//252}Y Rolling Returns', yaxis_title='Rolling Return %', hovermode='x unified', height=350)
    return fig

# ============================================================================
# 8. EXPLORER TAB
# ============================================================================

def render_explorer_tab():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0;">📊 Category Explorer</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Comprehensive fund analysis with rolling metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    show_metric_definitions()
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: category = st.selectbox("📁 Select Category", list(FILE_MAPPING.keys()))
    with col2: view_mode = st.selectbox("👁️ View Mode", ["📈 Comprehensive Metrics", "📊 Quarterly Rank History", "🔍 Fund Deep Dive"])
    with col3: st.write(""); st.write(""); st.button("🔄 Refresh", use_container_width=True)
    
    st.divider()
    with st.spinner(f"Loading {category} data..."):
        nav_df, scheme_map = load_fund_data_raw(category)
        benchmark = load_nifty_data()
    
    if nav_df is None:
        st.error("❌ Could not load data.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Funds", len(nav_df.columns))
    col2.metric("📅 Data Range", f"{nav_df.index.min().strftime('%Y-%m')} to {nav_df.index.max().strftime('%Y-%m')}")
    col3.metric("📆 Trading Days", len(nav_df))
    col4.metric("🏛️ Benchmark", "Nifty 100" if benchmark is not None else "N/A")
    st.divider()
    
    if "Comprehensive Metrics" in view_mode:
        with st.spinner("Calculating..."): metrics_df = calculate_comprehensive_metrics(nav_df, scheme_map, benchmark)
        if metrics_df.empty: st.warning("No funds with sufficient data."); return
        st.markdown("### 📋 Fund Performance Summary")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Overall Ranking", "📈 Returns", "🔄 Rolling Analysis", "⚠️ Risk", "📊 Benchmark"])
        with tab1:
            cols = [c for c in ['Fund Name', 'Composite Rank', '1Y Return Rank', '3Y Return Rank', 'Sharpe Rank', 'CAGR %', 'Max Drawdown %'] if c in metrics_df.columns]
            st.dataframe(metrics_df[cols].head(20).style.format({'Composite Rank': '{:.1f}', 'CAGR %': '{:.2f}%', 'Max Drawdown %': '{:.2f}%'}).background_gradient(subset=['Composite Rank'], cmap='Greens_r'), use_container_width=True, height=500)
        with tab2:
            cols = [c for c in ['Fund Name', 'Return 3M', 'Return 6M', 'Return 1Y', 'Return 3Y', 'CAGR %', '% Positive Months'] if c in metrics_df.columns]
            st.dataframe(metrics_df[cols].style.format({c: '{:.2f}%' for c in cols if c != 'Fund Name'}).background_gradient(subset=['Return 1Y'] if 'Return 1Y' in cols else [], cmap='RdYlGn'), use_container_width=True, height=500)
        with tab3:
            cols = [c for c in ['Fund Name', '1Y Rolling Avg Return %', '1Y % Times Beat Benchmark', '1Y Consistency Score', '3Y Rolling Avg Return %', '3Y Consistency Score'] if c in metrics_df.columns]
            st.dataframe(metrics_df[cols].style.format({c: '{:.2f}%' if '%' in c else '{:.3f}' for c in cols if c != 'Fund Name'}), use_container_width=True, height=500)
        with tab4:
            cols = [c for c in ['Fund Name', 'Volatility %', 'Max Drawdown %', 'VaR 95 %', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'] if c in metrics_df.columns]
            st.dataframe(metrics_df[cols].style.format({c: '{:.2f}%' if '%' in c else '{:.2f}' for c in cols if c != 'Fund Name'}), use_container_width=True, height=500)
        with tab5:
            cols = [c for c in ['Fund Name', 'Beta', 'Alpha %', 'Information Ratio', 'Up Capture %', 'Down Capture %', 'Capture Ratio'] if c in metrics_df.columns]
            st.dataframe(metrics_df[cols].style.format({c: '{:.2f}%' if '%' in c else '{:.2f}' for c in cols if c != 'Fund Name'}), use_container_width=True, height=500)
        st.divider()
        st.download_button("📥 Download Analysis (CSV)", metrics_df.to_csv(index=False), f"{category}_analysis.csv", "text/csv", key="explorer_dl")
    
    elif "Quarterly Rank History" in view_mode:
        with st.spinner("Calculating..."): rank_df = calculate_quarterly_ranks(nav_df, scheme_map)
        if rank_df.empty: st.warning("Insufficient data."); return
        st.markdown("### 📊 Quarterly Performance Rank History")
        st.dataframe(rank_df.style.format({'% Top 3': '{:.1f}%', '% Top 5': '{:.1f}%', '% Top 10': '{:.1f}%', 'Avg Rank': '{:.1f}'}).background_gradient(subset=['% Top 5'], cmap='Greens'), use_container_width=True, height=600)
    
    elif "Fund Deep Dive" in view_mode:
        st.markdown("### 🔍 Individual Fund Analysis")
        fund_options = {scheme_map.get(col, col): col for col in nav_df.columns}
        selected_fund_name = st.selectbox("Select Fund", sorted(fund_options.keys()))
        selected_fund_id = fund_options[selected_fund_name]
        series = nav_df[selected_fund_id].dropna()
        if len(series) < 252: st.warning("Insufficient data."); return
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📈 CAGR", f"{calculate_cagr(series)*100:.2f}%" if calculate_cagr(series) else "N/A")
        col2.metric("⚖️ Sharpe", f"{calculate_sharpe_ratio(series.pct_change().dropna()):.2f}" if calculate_sharpe_ratio(series.pct_change().dropna()) else "N/A")
        col3.metric("📉 Max DD", f"{calculate_max_dd(series)*100:.2f}%" if calculate_max_dd(series) else "N/A")
        col4.metric("📊 Volatility", f"{calculate_volatility(series.pct_change().dropna())*100:.2f}%" if calculate_volatility(series.pct_change().dropna()) else "N/A")
        st.divider()
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(create_performance_chart(nav_df, [selected_fund_id], scheme_map, benchmark), use_container_width=True, key=f"perf_{selected_fund_id}")
        with col2: st.plotly_chart(create_drawdown_chart(series, selected_fund_name), use_container_width=True, key=f"dd_{selected_fund_id}")
        if len(series) >= 252:
            st.markdown("#### Rolling Returns")
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(create_rolling_return_chart(nav_df, selected_fund_id, scheme_map, benchmark, 252), use_container_width=True, key=f"r1y_{selected_fund_id}")
            with col2:
                if len(series) >= 756: st.plotly_chart(create_rolling_return_chart(nav_df, selected_fund_id, scheme_map, benchmark, 756), use_container_width=True, key=f"r3y_{selected_fund_id}")

# ============================================================================
# 9. BACKTESTER CORE - WITH CORRECT HIT RATE LOGIC
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
        if len(hist) < 20: return np.nan
        vol = hist.pct_change().dropna().std() * np.sqrt(TRADING_DAYS_YEAR)
        return raw_score / vol if vol != 0 else np.nan
    return raw_score

def get_market_regime(benchmark_series, current_date, window=200):
    subset = benchmark_series[benchmark_series.index <= current_date]
    if len(subset) < window: return 'neutral'
    return 'bull' if subset.iloc[-1] > subset.iloc[-window:].mean() else 'bear'

def get_eligible_funds_at_date(nav, date, min_history_days=126):
    """
    Get list of funds that have sufficient history at a given date.
    A fund is eligible if it has at least min_history_days of data before the date.
    """
    eligible = []
    for col in nav.columns:
        fund_data = nav[col].dropna()
        # Check if fund has data before this date and has enough history
        fund_data_before_date = fund_data[fund_data.index < date]
        if len(fund_data_before_date) >= min_history_days:
            eligible.append(col)
    return eligible

def run_backtest_detailed(nav, strategy_type, top_n, target_n, holding_days, momentum_config, benchmark_series, scheme_map):
    """
    Run backtest with CORRECT hit rate logic:
    - Only considers funds that were available at selection time
    - New funds that appear later are NOT counted in hit rate calculation
    """
    start_date = nav.index.min() + pd.Timedelta(days=370)
    if start_date >= nav.index.max(): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    start_idx = nav.index.searchsorted(start_date)
    rebal_idx = list(range(start_idx, len(nav) - 1, holding_days))
    if not rebal_idx: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    history, eq_curve, bench_curve, detailed_trades = [], [{'date': nav.index[rebal_idx[0]], 'value': 100.0}], [{'date': nav.index[rebal_idx[0]], 'value': 100.0}], []
    cap, b_cap = 100.0, 100.0
    
    for i in rebal_idx:
        date = nav.index[i]
        hist = get_lookback_data(nav, date)
        
        # KEY: Get the list of funds that were AVAILABLE at selection time
        eligible_funds_at_selection = get_eligible_funds_at_date(nav, date, min_history_days=126)
        
        scores, selected, regime_status = {}, [], "neutral"
        
        # Strategy implementations - only consider eligible funds
        if strategy_type in ['momentum', 'sharpe', 'sortino']:
            for col in eligible_funds_at_selection:
                s = hist[col].dropna()
                if len(s) < 126: continue
                rets = s.pct_change().dropna()
                if strategy_type == 'momentum':
                    val = calculate_flexible_momentum(s, momentum_config.get('w_3m', 0.33), momentum_config.get('w_6m', 0.33), momentum_config.get('w_12m', 0.33), momentum_config.get('risk_adjust', False))
                elif strategy_type == 'sharpe': val = calculate_sharpe_ratio(rets)
                elif strategy_type == 'sortino': val = calculate_sortino_ratio(rets)
                if not pd.isna(val): scores[col] = val
            selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
        
        elif strategy_type == 'regime_switch':
            if benchmark_series is not None:
                regime_status = get_market_regime(benchmark_series, date)
                for col in eligible_funds_at_selection:
                    s = hist[col].dropna()
                    if len(s) < 126: continue
                    val = calculate_flexible_momentum(s, 0.3, 0.3, 0.4, False) if regime_status == 'bull' else calculate_sharpe_ratio(s.pct_change().dropna())
                    if not pd.isna(val): scores[col] = val
                selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
        
        elif strategy_type == 'stable_momentum':
            mom_scores = {}
            for col in eligible_funds_at_selection:
                s = hist[col].dropna()
                if len(s) >= 260:
                    val = calculate_flexible_momentum(s, 0.33, 0.33, 0.33, False)
                    if not pd.isna(val): mom_scores[col] = val
            pool = sorted(mom_scores, key=mom_scores.get, reverse=True)[:top_n*2]
            dd_scores = {col: calculate_max_dd(hist[col].dropna()) for col in pool if not pd.isna(calculate_max_dd(hist[col].dropna()))}
            selected = sorted(dd_scores, key=dd_scores.get, reverse=True)[:top_n]
        
        elif strategy_type == 'elimination':
            fund_data = {}
            for col in eligible_funds_at_selection:
                s = hist[col].dropna()
                if len(s) < 200: continue
                rets = s.pct_change().dropna()
                fund_data[col] = {'max_dd': calculate_max_dd(s), 'volatility': calculate_volatility(rets), 'sharpe': calculate_sharpe_ratio(rets)}
            if fund_data:
                df = pd.DataFrame(fund_data).T.dropna()
                if len(df) > top_n * 2:
                    df = df[df['max_dd'] >= df['max_dd'].quantile(0.25)]
                    if len(df) > top_n * 2: df = df[df['volatility'] <= df['volatility'].quantile(0.75)]
                selected = df.nlargest(top_n, 'sharpe').index.tolist() if len(df) > 0 else []
        
        elif strategy_type == 'consistency':
            consistent_funds = []
            for col in eligible_funds_at_selection:
                s = hist[col].dropna()
                if len(s) < 300: continue
                quarters_good = 0
                for q in range(4):
                    q_end, q_start = date - pd.Timedelta(days=q*91), date - pd.Timedelta(days=(q+1)*91)
                    try:
                        idx_s, idx_e = hist.index.asof(q_start), hist.index.asof(q_end)
                        if pd.isna(idx_s) or pd.isna(idx_e): continue
                        all_rets = (hist.loc[idx_e] / hist.loc[idx_s] - 1).dropna()
                        if col in all_rets.index and all_rets[col] >= all_rets.median(): quarters_good += 1
                    except: continue
                if quarters_good >= 3: consistent_funds.append(col)
            if consistent_funds:
                mom_scores = {col: (hist[col].dropna().iloc[-1] / hist[col].dropna().iloc[-63] - 1) for col in consistent_funds if len(hist[col].dropna()) >= 63}
                selected = sorted(mom_scores, key=mom_scores.get, reverse=True)[:top_n] if mom_scores else []
            else:
                for col in eligible_funds_at_selection:
                    s = hist[col].dropna()
                    if len(s) >= 126: scores[col] = calculate_flexible_momentum(s, 0.33, 0.33, 0.33, False)
                selected = sorted([k for k,v in scores.items() if not pd.isna(v)], key=lambda x: scores[x], reverse=True)[:top_n]
        
        # Execute trade
        entry, exit_i = i + 1, min(i + 1 + holding_days, len(nav) - 1)
        b_ret = 0.0
        if benchmark_series is not None:
            try: b_ret = benchmark_series.asof(nav.index[exit_i]) / benchmark_series.asof(nav.index[entry]) - 1
            except: pass
        
        # Calculate returns for ALL funds at exit
        period_ret_all = (nav.iloc[exit_i] / nav.iloc[entry] - 1).dropna()
        
        # KEY CHANGE: Only consider funds that were eligible at SELECTION time for hit rate
        # Filter returns to only include funds that were available at selection
        eligible_returns = period_ret_all[period_ret_all.index.isin(eligible_funds_at_selection)]
        
        # Actual top funds - ONLY from eligible funds (funds available at selection time)
        actual_top_funds_eligible = eligible_returns.nlargest(target_n).index.tolist()
        
        # Also get actual top from ALL funds (for reference/display)
        actual_top_funds_all = period_ret_all.nlargest(target_n).index.tolist()
        
        # Identify new funds (funds in top performers that weren't eligible at selection)
        new_funds_in_top = [f for f in actual_top_funds_all if f not in eligible_funds_at_selection]
        
        port_ret, hits = 0.0, 0
        if selected:
            port_ret = period_ret_all[selected].mean()
            # Hit rate calculated against ELIGIBLE funds only
            hits = len(set(selected).intersection(set(actual_top_funds_eligible)))
        hit_rate = hits / top_n if top_n > 0 else 0
        
        cap *= (1 + (port_ret if not pd.isna(port_ret) else 0))
        b_cap *= (1 + b_ret)
        
        history.append({'date': date, 'selected': selected, 'return': port_ret, 'hit_rate': hit_rate, 'regime': regime_status})
        
        # Build detailed trade record
        trade_record = {
            'Period Start': date.strftime('%Y-%m-%d'),
            'Period End': nav.index[exit_i].strftime('%Y-%m-%d'),
            'Regime': regime_status,
            'Eligible Funds': len(eligible_funds_at_selection),
            'Portfolio Return %': port_ret * 100,
            'Benchmark Return %': b_ret * 100,
            'Hits': hits,
            'Hit Rate %': hit_rate * 100,
            'New Funds Excluded': len(new_funds_in_top),
        }
        
        # Add selected funds
        for idx, fund_id in enumerate(selected):
            fund_name = scheme_map.get(fund_id, fund_id)
            fund_return = period_ret_all.get(fund_id, np.nan)
            is_hit = fund_id in actual_top_funds_eligible
            trade_record[f'Fund {idx+1}'] = fund_name
            trade_record[f'Fund {idx+1} Return %'] = fund_return * 100 if not pd.isna(fund_return) else np.nan
            trade_record[f'Fund {idx+1} Hit'] = '✅' if is_hit else '❌'
        
        # Add actual top funds (from eligible only)
        for idx, fund_id in enumerate(actual_top_funds_eligible):
            fund_name = scheme_map.get(fund_id, fund_id)
            fund_return = period_ret_all.get(fund_id, np.nan)
            trade_record[f'Actual Top {idx+1} (Eligible)'] = fund_name
            trade_record[f'Actual Top {idx+1} Return %'] = fund_return * 100 if not pd.isna(fund_return) else np.nan
        
        # Also show new funds that were excluded (for transparency)
        if new_funds_in_top:
            for idx, fund_id in enumerate(new_funds_in_top[:3]):  # Show max 3 new funds
                fund_name = scheme_map.get(fund_id, fund_id)
                fund_return = period_ret_all.get(fund_id, np.nan)
                trade_record[f'New Fund {idx+1} (Excluded)'] = fund_name
                trade_record[f'New Fund {idx+1} Return %'] = fund_return * 100 if not pd.isna(fund_return) else np.nan
        
        detailed_trades.append(trade_record)
        eq_curve.append({'date': nav.index[exit_i], 'value': cap})
        bench_curve.append({'date': nav.index[exit_i], 'value': b_cap})
    
    return pd.DataFrame(history), pd.DataFrame(eq_curve), pd.DataFrame(bench_curve), pd.DataFrame(detailed_trades)

# ============================================================================
# 10. DISPLAY STRATEGY RESULTS
# ============================================================================

def display_strategy_results(nav_df, scheme_map, benchmark, strat_key, strat_name, mom_config, top_n, target_n, holding):
    """Display results with correct hit rate explanation."""
    
    key_prefix = f"{strat_key}_{holding}_{top_n}_{target_n}"
    
    show_strategy_explanation(strat_key)
    
    with st.spinner(f"Running {strat_name} backtest..."):
        history, eq_curve, bench_curve, detailed_trades = run_backtest_detailed(
            nav_df, strat_key, top_n, target_n, holding, mom_config, benchmark, scheme_map
        )
    
    if eq_curve.empty:
        st.warning("No trades generated. Insufficient data.")
        return
    
    years = (eq_curve.iloc[-1]['date'] - eq_curve.iloc[0]['date']).days / 365.25
    strat_cagr = (eq_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
    bench_cagr = (bench_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
    avg_hit = history['hit_rate'].mean()
    total_trades = len(history)
    
    # Show hit rate methodology note
    st.markdown("""
    <div class="note-box">
        <strong>📌 Hit Rate Methodology:</strong> Hit rate is calculated by comparing selected funds against the top performers 
        <strong>ONLY from funds that were available at selection time</strong>. New funds that appear later are excluded from 
        this comparison to ensure fair evaluation.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📈 Strategy CAGR", f"{strat_cagr*100:.2f}%")
    col2.metric("📊 Benchmark CAGR", f"{bench_cagr*100:.2f}%")
    col3.metric("🎯 Outperformance", f"{(strat_cagr-bench_cagr)*100:+.2f}%")
    col4.metric("🏆 Avg Hit Rate", f"{avg_hit*100:.1f}%")
    col5.metric("📋 Total Trades", f"{total_trades}")
    
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["📈 Equity Curve", "📋 All Trades Detail", "🎯 Hit Analysis", "📊 Summary"])
    
    with sub_tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_curve['date'], y=eq_curve['value'], name='Strategy', line=dict(color='green', width=2), fill='tozeroy', fillcolor='rgba(0, 200, 0, 0.1)'))
        fig.add_trace(go.Scatter(x=bench_curve['date'], y=bench_curve['value'], name='Benchmark', line=dict(color='gray', width=2, dash='dot')))
        fig.update_layout(height=400, title=f'{strat_name} - Equity Curve', yaxis_title='Value (100 = Start)', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True, key=f"eq_{key_prefix}")
        
        if not detailed_trades.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=detailed_trades['Period Start'], y=detailed_trades['Hit Rate %'], name='Hit Rate',
                marker_color=detailed_trades['Hit Rate %'].apply(lambda x: 'green' if x >= 50 else 'orange' if x > 0 else 'red')))
            fig2.add_hline(y=avg_hit*100, line_dash="dash", line_color="blue", annotation_text=f"Avg: {avg_hit*100:.1f}%")
            fig2.update_layout(height=300, title='Hit Rate by Period (Eligible Funds Only)', yaxis_title='Hit Rate %')
            st.plotly_chart(fig2, use_container_width=True, key=f"hr_{key_prefix}")
    
    with sub_tab2:
        st.markdown("### 📋 Complete Trade History")
        st.markdown(f"""
        *Showing all {len(detailed_trades)} periods. Hit rate compares against top performers from **eligible funds only** 
        (funds available at selection time). New funds are shown separately.*
        """)
        
        if not detailed_trades.empty:
            pct_cols = [c for c in detailed_trades.columns if 'Return %' in c or 'Rate %' in c]
            hit_cols = [c for c in detailed_trades.columns if 'Hit' in c and 'Rate' not in c]
            format_dict = {col: '{:.2f}' for col in pct_cols}
            format_dict['Hits'] = '{:.0f}'
            format_dict['Eligible Funds'] = '{:.0f}'
            format_dict['New Funds Excluded'] = '{:.0f}'
            
            def highlight_hits(val):
                if val == '✅': return 'background-color: #c8e6c9; color: #2e7d32; font-weight: bold'
                elif val == '❌': return 'background-color: #ffcdd2; color: #c62828'
                return ''
            
            styled_df = detailed_trades.style.format(format_dict)
            for col in hit_cols:
                if col in detailed_trades.columns:
                    styled_df = styled_df.applymap(highlight_hits, subset=[col])
            st.dataframe(styled_df, use_container_width=True, height=600)
            st.download_button(f"📥 Download Full Trade History", detailed_trades.to_csv(index=False), f"{strat_key}_trades.csv", "text/csv", key=f"dl_{key_prefix}")
    
    with sub_tab3:
        st.markdown("### 🎯 Hit Rate Analysis")
        st.info("💡 Hit rate only considers funds that were available at selection time. New funds are excluded.")
        
        if not detailed_trades.empty:
            col1, col2 = st.columns(2)
            with col1:
                hit_counts = detailed_trades['Hits'].value_counts().sort_index()
                colors_list = ['#f44336', '#ff9800', '#ffeb3b', '#8bc34a', '#4caf50', '#2e7d32', '#1b5e20'][:len(hit_counts)]
                fig = go.Figure(data=[go.Bar(x=[f"{int(k)} hits" for k in hit_counts.index], y=hit_counts.values, marker_color=colors_list)])
                fig.update_layout(title=f'Distribution of Hits (out of {top_n} picks)', height=300)
                st.plotly_chart(fig, use_container_width=True, key=f"hd_{key_prefix}")
                
                st.markdown("**Statistics:**")
                for i in range(min(top_n + 1, 6)):
                    count = (detailed_trades['Hits'] == i).sum()
                    pct = count / len(detailed_trades) * 100
                    emoji = "🟢" if i >= top_n/2 else "🟡" if i > 0 else "🔴"
                    st.write(f"{emoji} {i} hits: {count} ({pct:.1f}%)")
                
                # Show excluded funds stats
                if 'New Funds Excluded' in detailed_trades.columns:
                    avg_excluded = detailed_trades['New Funds Excluded'].mean()
                    st.markdown(f"**Avg new funds excluded per period:** {avg_excluded:.1f}")
            
            with col2:
                hit_periods = detailed_trades[detailed_trades['Hits'] > 0]['Portfolio Return %']
                miss_periods = detailed_trades[detailed_trades['Hits'] == 0]['Portfolio Return %']
                fig = go.Figure()
                if len(hit_periods) > 0: fig.add_trace(go.Box(y=hit_periods, name='With Hits', marker_color='green'))
                if len(miss_periods) > 0: fig.add_trace(go.Box(y=miss_periods, name='No Hits', marker_color='red'))
                fig.update_layout(title='Returns: Hits vs No Hits', yaxis_title='Return %', height=300)
                st.plotly_chart(fig, use_container_width=True, key=f"hr2_{key_prefix}")
                
                st.markdown("**Return Stats:**")
                if len(hit_periods) > 0: st.write(f"🟢 Avg when hitting: {hit_periods.mean():.2f}%")
                if len(miss_periods) > 0: st.write(f"🔴 Avg when missing: {miss_periods.mean():.2f}%")
    
    with sub_tab4:
        st.markdown("### 📊 Period Summary")
        st.markdown("*'Actual Top' shows top performers from eligible funds only. Excluded new funds shown separately.*")
        
        if not detailed_trades.empty:
            summary_data = []
            for _, row in detailed_trades.iterrows():
                selected_funds = []
                for i in range(1, top_n + 1):
                    if f'Fund {i}' in row and pd.notna(row.get(f'Fund {i}')) and row.get(f'Fund {i}') != '':
                        ret = row.get(f'Fund {i} Return %', np.nan)
                        hit = row.get(f'Fund {i} Hit', '')
                        selected_funds.append(f"{row[f'Fund {i}']} ({ret:.1f}%) {hit}" if pd.notna(ret) else f"{row[f'Fund {i}']} {hit}")
                
                actual_funds = []
                for i in range(1, target_n + 1):
                    col_name = f'Actual Top {i} (Eligible)'
                    if col_name in row and pd.notna(row.get(col_name)) and row.get(col_name) != '':
                        ret = row.get(f'Actual Top {i} Return %', np.nan)
                        actual_funds.append(f"{row[col_name]} ({ret:.1f}%)" if pd.notna(ret) else row[col_name])
                
                excluded_funds = []
                for i in range(1, 4):
                    col_name = f'New Fund {i} (Excluded)'
                    if col_name in row and pd.notna(row.get(col_name)) and row.get(col_name) != '':
                        ret = row.get(f'New Fund {i} Return %', np.nan)
                        excluded_funds.append(f"{row[col_name]} ({ret:.1f}%)" if pd.notna(ret) else row[col_name])
                
                summary_data.append({
                    'Period': f"{row['Period Start']} → {row['Period End']}",
                    'Regime': row.get('Regime', 'N/A'),
                    'Selected': ' | '.join(selected_funds),
                    'Hits': f"{int(row['Hits'])}/{top_n}",
                    'Return': f"{row['Portfolio Return %']:.1f}%",
                    'Actual Top (Eligible)': ' | '.join(actual_funds),
                    'New Funds Excluded': ' | '.join(excluded_funds) if excluded_funds else 'None'
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, height=600)

# ============================================================================
# 11. BACKTEST TAB
# ============================================================================

def render_backtest_tab():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0;">🚀 Strategy Backtester</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Test strategies with correct hit rate calculation (excludes new funds)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy overview
    with st.expander("📚 **All Strategy Explanations** (Click to expand)", expanded=False):
        st.markdown("### Understanding Each Strategy")
        for key, strat in STRATEGY_DEFINITIONS.items():
            st.markdown(f"**{strat['name']}:** {strat['short_desc']}")
            with st.expander(f"Details for {strat['name']}"):
                st.markdown(strat['full_desc'])
    
    # Hit rate methodology explanation
    with st.expander("📐 **Hit Rate Calculation Methodology** (Click to expand)", expanded=False):
        st.markdown("""
        ### How Hit Rate is Calculated
        
        **The Problem with Naive Hit Rate:**
        - At selection time (e.g., Jan 1), you have 20 funds available
        - You select top 5 based on your strategy
        - At evaluation time (e.g., July 1), there might be 25 funds (5 new ones added)
        - Some new funds might be top performers
        - It's UNFAIR to count these against your strategy since you couldn't have selected them!
        
        **Our Correct Approach:**
        1. At selection time, we record which funds were **eligible** (had sufficient history)
        2. At evaluation time, we calculate returns for ALL funds
        3. For hit rate, we ONLY compare against top performers from **eligible funds**
        4. New funds are shown separately as "Excluded" for transparency
        
        **Example:**
        ```
        Selection Date: Jan 1, 2023
        Eligible Funds: A, B, C, D, E, F, G, H, I, J (10 funds)
        Strategy Selects: A, B, C, D, E
        
        Evaluation Date: July 1, 2023
        All Funds Now: A, B, C, D, E, F, G, H, I, J, X, Y (12 funds - X, Y are new)
        
        Actual Top 5 (All): X, A, Y, B, F
        Actual Top 5 (Eligible Only): A, B, F, C, G  ← Used for hit rate
        
        Hits = 3 (A, B, C are in both) → Hit Rate = 60%
        
        X and Y are shown as "New Funds Excluded" for transparency
        ```
        
        This ensures fair comparison - you're measured against what you COULD have chosen.
        """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: category = st.selectbox("📁 Category", list(FILE_MAPPING.keys()), key="bt_cat")
    with col2: top_n = st.number_input("🎯 Funds to Select", 1, 15, 3, key="bt_topn", help="Number of funds the strategy will pick")
    with col3: target_n = st.number_input("🏆 Target Top N", 1, 20, 5, key="bt_target", help="Compare against top N performers (from eligible funds)")
    with col4: holding = st.selectbox("📅 Holding Period", [63, 126, 252, 378, 504], index=1, format_func=lambda x: f"{x} days (~{x//21}M)")
    
    st.divider()
    
    with st.spinner("Loading data..."):
        nav_df, scheme_map = load_fund_data_raw(category)
        benchmark = load_nifty_data()
    
    if nav_df is None:
        st.error("Could not load data.")
        return
    
    st.success(f"✅ Loaded {len(nav_df.columns)} funds | Data: {nav_df.index.min().strftime('%Y-%m')} to {nav_df.index.max().strftime('%Y-%m')}")
    
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
    
    for idx, (tab_name, (strat_key, mom_config)) in enumerate(strategies.items()):
        with tabs[idx]:
            st.markdown(f"### {tab_name} Strategy")
            st.info(f"📌 **Summary:** {STRATEGY_DEFINITIONS[strat_key]['short_desc']}")
            display_strategy_results(nav_df, scheme_map, benchmark, strat_key, tab_name, mom_config, top_n, target_n, holding)
    
    # Compare All tab
    with tabs[-1]:
        st.markdown("### 📊 Strategy Comparison")
        st.info("💡 All hit rates calculated using correct methodology (eligible funds only)")
        
        results, all_equity_curves, last_bench = [], {}, None
        progress = st.progress(0)
        
        for idx, (name, (strat_key, mom_config)) in enumerate(strategies.items()):
            history, eq_curve, bench_curve, _ = run_backtest_detailed(nav_df, strat_key, top_n, target_n, holding, mom_config, benchmark, scheme_map)
            if not eq_curve.empty:
                years = (eq_curve.iloc[-1]['date'] - eq_curve.iloc[0]['date']).days / 365.25
                cagr = (eq_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                bench_cagr = (bench_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                max_dd = calculate_max_dd(pd.Series(eq_curve['value'].values, index=eq_curve['date']))
                results.append({
                    'Strategy': name, 'CAGR %': cagr * 100, 'Benchmark %': bench_cagr * 100,
                    'Alpha %': (cagr - bench_cagr) * 100, 'Max DD %': max_dd * 100 if max_dd else 0,
                    'Hit Rate %': history['hit_rate'].mean() * 100, 'Win Rate %': (history['return'] > 0).mean() * 100,
                    'Trades': len(history)
                })
                all_equity_curves[name] = eq_curve
                last_bench = bench_curve
            progress.progress((idx + 1) / len(strategies))
        progress.empty()
        
        if results:
            st.dataframe(pd.DataFrame(results).sort_values('CAGR %', ascending=False).style.format({
                'CAGR %': '{:.2f}', 'Benchmark %': '{:.2f}', 'Alpha %': '{:+.2f}',
                'Max DD %': '{:.2f}', 'Hit Rate %': '{:.1f}', 'Win Rate %': '{:.1f}'
            }).background_gradient(subset=['CAGR %', 'Alpha %'], cmap='RdYlGn'), use_container_width=True)
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for idx, (name, eq_curve) in enumerate(all_equity_curves.items()):
            fig.add_trace(go.Scatter(x=eq_curve['date'], y=eq_curve['value'], name=name, line=dict(color=colors[idx % len(colors)], width=2)))
        if last_bench is not None:
            fig.add_trace(go.Scatter(x=last_bench['date'], y=last_bench['value'], name='Benchmark', line=dict(color='black', width=2, dash='dot')))
        fig.update_layout(height=500, title='All Strategies vs Benchmark', yaxis_title='Value', hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
        st.plotly_chart(fig, use_container_width=True, key="compare_eq")

# ============================================================================
# 12. MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h1 style="margin: 0; border: none;">📈 Fund Analysis Pro</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Comprehensive analysis with correct hit rate methodology</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 Category Explorer", "🚀 Strategy Backtester"])
    with tab1: render_explorer_tab()
    with tab2: render_backtest_tab()
    
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #999; font-size: 0.8rem; margin-top: 40px; border-top: 1px solid #eee;">
        Fund Analysis Pro • Risk-free rate: 6% • Hit rate excludes new funds not available at selection
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
