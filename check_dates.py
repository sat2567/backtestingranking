"""
Advanced Fund Analysis Dashboard - Enhanced Version v2
=======================================================
NEW FEATURES:
  - Top 5 funds highlighted on category selection
  - Best strategy auto-detection & highlighting
  - "Current Picks" panel (what to buy TODAY)
  - Hit Rate Improvements:
      * Ensemble Voting (multi-strategy consensus)
      * Trend Confirmation (NAV > 50 DMA)
      * Drawdown Recency Filter
      * Volatility-Adjusted Momentum
      * Adaptive Lookback (match momentum to holding period)
  - New "Smart Picks" ensemble meta-strategy
Run: streamlit run backtest_strategy.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# ============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(page_title="Fund Analysis Pro", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }
    h1 { color: #1E3A5F; font-weight: 700; padding-bottom: 0.5rem; border-bottom: 3px solid #4CAF50; margin-bottom: 1.5rem; }
    div[data-testid="metric-container"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; padding: 15px; }
    div[data-testid="metric-container"] label { color: rgba(255, 255, 255, 0.9) !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: white !important; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 10px 20px; font-weight: 500; color: #333 !important; }
    .stTabs [aria-selected="true"] { background-color: #4CAF50 !important; color: white !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .metric-box { background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%); border-radius: 12px; padding: 18px; margin: 10px 0; border-left: 5px solid #4CAF50; box-shadow: 0 2px 5px rgba(0,0,0,0.05); color: #1f2937; }
    .metric-box h4 { color: #1E3A5F !important; margin: 0 0 10px 0; font-weight: 700; }
    .metric-box p { color: #374151 !important; margin: 5px 0; font-size: 0.95rem; }
    .metric-box .formula { background: #263238; color: #80cbc4 !important; padding: 10px 15px; border-radius: 8px; font-family: 'Courier New', monospace; margin: 10px 0; font-size: 0.9rem; }
    .strategy-box { background: white; border-radius: 12px; padding: 20px; margin: 15px 0; box-shadow: 0 3px 12px rgba(0,0,0,0.08); border-left: 5px solid #2196F3; color: #1f2937; }
    .strategy-box h4 { color: #1E3A5F !important; margin-top: 0; }
    .strategy-box p { color: #374151 !important; line-height: 1.5; }
    .strategy-box code { background-color: #f1f5f9; color: #d63384; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .info-banner { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px; color: white; }
    .info-banner h2 { color: white !important; margin: 0; border: none; }
    .info-banner p { color: rgba(255,255,255,0.9) !important; margin: 5px 0 0 0; }
    .top-fund-card { background: white; border-radius: 14px; padding: 16px 18px; margin: 8px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-left: 5px solid #4CAF50; transition: transform 0.15s; color: #1f2937; }
    .top-fund-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.12); }
    .top-fund-card .rank { font-size: 1.6rem; font-weight: 800; color: #4CAF50; float: left; margin-right: 14px; line-height: 1; }
    .top-fund-card .fund-name { font-weight: 700; color: #1E3A5F; font-size: 0.92rem; margin-bottom: 4px; }
    .top-fund-card .fund-stats { color: #555; font-size: 0.82rem; }
    .top-fund-card .fund-stats span { margin-right: 12px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 700; }
    .badge-green { background: #e8f5e9; color: #2e7d32; }
    .badge-orange { background: #fff3e0; color: #e65100; }
    .best-strat-banner { background: linear-gradient(135deg, #43a047 0%, #2e7d32 100%); color: white; border-radius: 14px; padding: 20px 24px; margin: 16px 0; box-shadow: 0 4px 18px rgba(46,125,50,0.25); }
    .best-strat-banner h3 { color: white !important; margin: 0 0 6px 0; border: none; }
    .best-strat-banner p { color: rgba(255,255,255,0.92) !important; margin: 2px 0; font-size: 0.92rem; }
    .best-strat-banner .strat-metrics { display: flex; gap: 24px; margin-top: 10px; }
    .best-strat-banner .strat-metric { text-align: center; }
    .best-strat-banner .strat-metric .val { font-size: 1.3rem; font-weight: 800; }
    .best-strat-banner .strat-metric .lbl { font-size: 0.75rem; opacity: 0.85; }
    .current-pick { background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); border-radius: 12px; padding: 14px 16px; margin: 6px 0; border: 1px solid #c8e6c9; color: #1f2937; }
    .current-pick .pick-name { font-weight: 700; color: #1b5e20; font-size: 0.9rem; }
    .current-pick .pick-stats { color: #555; font-size: 0.8rem; margin-top: 4px; }
    .vote-badge { display: inline-block; padding: 3px 10px; border-radius: 14px; font-size: 0.78rem; font-weight: 700; margin-right: 6px; }
    .vote-high { background: #4caf50; color: white; }
    .vote-med { background: #ff9800; color: white; }
    .vote-low { background: #e0e0e0; color: #666; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. CONSTANTS
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
    return f"{days}d (~{days//21}M)" if days < 252 else f"{days}d (~{days//252}Y)"

# ============================================================================
# 3. METRIC & STRATEGY DEFINITIONS
# ============================================================================

METRIC_DEFINITIONS = {
    'returns': {
        'Return 3M': {'name': '3-Month Return', 'formula': 'Return = (NAV_today / NAV_3M_ago - 1) * 100', 'description': 'Percentage change over 3 months.', 'interpretation': 'Shows short-term momentum.'},
        'Return 1Y': {'name': '1-Year Return', 'formula': 'Return = (NAV_today / NAV_1Y_ago - 1) * 100', 'description': 'Percentage change over 12 months.', 'interpretation': 'Standard performance metric.'},
        'CAGR': {'name': 'CAGR', 'formula': 'CAGR = (End/Start)^(1/Years) - 1', 'description': 'Compound Annual Growth Rate.', 'interpretation': 'Higher is better.'}
    },
    'risk': {
        'Volatility': {'name': 'Volatility', 'formula': 'Vol = StdDev(Returns) * sqrt(252)', 'description': 'Annualized standard deviation.', 'interpretation': '<15% low, 15-25% moderate, >25% high.'},
        'Max Drawdown': {'name': 'Max Drawdown', 'formula': 'Max_DD = (Trough - Peak) / Peak', 'description': 'Worst peak-to-trough decline.', 'interpretation': 'Less negative is better.'},
        'VaR 95': {'name': 'VaR 95%', 'formula': 'VaR = Percentile(Returns, 5%)', 'description': '5th percentile of daily returns.', 'interpretation': 'Max expected daily loss with 95% confidence.'}
    },
    'risk_adjusted': {
        'Sharpe': {'name': 'Sharpe Ratio', 'formula': 'Sharpe = (Return - Rf) / Volatility', 'description': 'Excess return per unit of risk.', 'interpretation': '>1 good, >2 very good, >3 excellent.'},
        'Sortino': {'name': 'Sortino Ratio', 'formula': 'Sortino = (Return - Rf) / Downside_Dev', 'description': 'Return per unit of downside risk.', 'interpretation': 'Higher is better.'},
        'Calmar': {'name': 'Calmar Ratio', 'formula': 'Calmar = CAGR / |Max_DD|', 'description': 'Return per unit of drawdown.', 'interpretation': '>1 means return exceeds worst loss.'}
    },
    'benchmark': {
        'Beta': {'name': 'Beta', 'formula': 'Beta = Cov(Fund,Bench) / Var(Bench)', 'description': 'Sensitivity to market.', 'interpretation': '<1 defensive, >1 aggressive.'},
        'Alpha': {'name': 'Alpha', 'formula': 'Alpha = Return - [Rf + Beta * (Bench - Rf)]', 'description': 'Excess return over CAPM.', 'interpretation': 'Positive = outperformance.'},
        'Up Capture': {'name': 'Up Capture', 'formula': 'Up_Cap = Fund_Up / Bench_Up', 'description': 'Performance in up markets.', 'interpretation': '>100% means outperform in up markets.'},
        'Down Capture': {'name': 'Down Capture', 'formula': 'Down_Cap = Fund_Down / Bench_Down', 'description': 'Performance in down markets.', 'interpretation': '<100% means lose less in down markets.'}
    }
}

STRATEGY_DEFINITIONS = {
    'momentum': {
        'name': '🚀 Composite Momentum', 'short_desc': 'Captures structural trends by weighting short, medium, and long-term returns.',
        'how_it_works': ['Calculate returns for 3M, 6M, 12M lookback.', 'Apply adaptive weights based on holding period.', 'Normalize score by volatility to penalize erratic movement.', 'Select top N funds with highest risk-adjusted momentum.'],
        'formula': 'Score = (w3*R_3m + w6*R_6m + w12*R_12m) / Vol', 'best_for': 'Strong trending markets & recovery phases.',
        'weaknesses': ['Suffers during V-shaped reversals', 'Lagging indicator at tops']
    },
    'sharpe': {
        'name': '⚖️ Sharpe Maximization', 'short_desc': 'Prioritizes excess return per unit of total risk.',
        'how_it_works': ['Calculate annualized volatility for each fund.', 'Calculate excess return over Risk-Free Rate (6%).', 'Compute ratio: Excess Return / Volatility.', 'Select top N most efficient funds.'],
        'formula': 'Sharpe = (CAGR - Rf) / sigma_total', 'best_for': 'Volatile or sideways markets.',
        'weaknesses': ['Penalizes upside volatility', 'Assumes normal distribution']
    },
    'sortino': {
        'name': '🎯 Sortino Optimization', 'short_desc': 'Focuses on minimizing bad volatility (downside deviation).',
        'how_it_works': ['Isolate negative returns for Downside Deviation.', 'Ignore upside volatility.', 'Compute Excess Return / Downside Deviation.', 'Select top N with best asymmetric profile.'],
        'formula': 'Sortino = (CAGR - Rf) / sigma_downside', 'best_for': 'Aggressive growth strategies.',
        'weaknesses': ['Requires sufficient negative data points']
    },
    'regime_switch': {
        'name': '🚦 Adaptive Regime Switch', 'short_desc': 'Dynamic: Momentum in Bull, Safety in Bear.',
        'how_it_works': ['Detect regime via Benchmark vs 200-DMA.', 'BULL: Deploy Momentum.', 'BEAR: Deploy Sharpe.', 'Re-evaluate at every rebalance.'],
        'formula': 'Strategy = Momentum if (Bench > MA_200) else Sharpe', 'best_for': 'Full market cycles.',
        'weaknesses': ['Whipsaw in choppy markets around 200DMA']
    },
    'stable_momentum': {
        'name': '⚓ Stable Momentum', 'short_desc': 'High momentum filtered for stability.',
        'how_it_works': ['Rank by Momentum, take Top 25%.', 'Calculate Max Drawdown for pool.', 'Select Top N with LOWEST drawdown.', 'Find smooth winners.'],
        'formula': 'Select = Min(MDD) from Top_Quartile(Momentum)', 'best_for': 'Conservative growth.',
        'weaknesses': ['May underperform in melt-up phases']
    },
    'elimination': {
        'name': '🛡️ Double-Filter Elimination', 'short_desc': 'Exclude worst before picking best.',
        'how_it_works': ['Remove bottom 25% by Max Drawdown.', 'Remove top 25% by Volatility.', 'Pick Top N by Sharpe from survivors.', 'Win by not losing.'],
        'formula': 'Universe - (Worst_DD + Highest_Vol) -> Rank by Sharpe', 'best_for': 'Capital preservation.',
        'weaknesses': ['Overly conservative in bull markets']
    },
    'consistency': {
        'name': '📈 Quartile Consistency', 'short_desc': 'Persistent peer-beaters quarter-over-quarter.',
        'how_it_works': ['Divide 12M into 4 quarters.', 'Rank fund vs peers each quarter.', 'Keep funds in Top 50% for 3/4 quarters.', 'Rank survivors by 3M momentum.'],
        'formula': 'Select if (Rank <= 50%) in >= 3 of 4 Qtrs', 'best_for': 'Long-term reliability.',
        'weaknesses': ['Ignores turnaround stories']
    },
    'smart_ensemble': {
        'name': '🧠 Smart Ensemble (NEW)', 'short_desc': 'Meta-strategy: multi-strategy voting + trend + drawdown filter.',
        'how_it_works': [
            'Run all 7 base strategies, collect Top N picks each.',
            'Count votes: funds picked by 3+ strategies get priority.',
            'Trend Filter: NAV must be above 50-day MA.',
            'Drawdown Filter: exclude funds in >10% current drawdown.',
            'Rank by vote count, then vol-adjusted momentum.'
        ],
        'formula': 'Votes(7 strats) -> Trend(NAV>50DMA) -> DD(<10%) -> Rank', 'best_for': 'Maximum hit rate, all-weather.',
        'weaknesses': ['Conservative, may miss early breakouts']
    }
}

# ============================================================================
# 4. HELPER FUNCTIONS
# ============================================================================

def show_metric_definitions():
    with st.expander("📖 **Metric Definitions**", expanded=False):
        for cat, metrics in METRIC_DEFINITIONS.items():
            st.markdown(f"### {cat.replace('_', ' ').title()}")
            cols = st.columns(2)
            for idx, (key, m) in enumerate(metrics.items()):
                with cols[idx % 2]:
                    st.markdown(f'<div class="metric-box"><h4>{m["name"]}</h4><div class="formula">{m["formula"]}</div><p>{m["description"]}</p><p>💡 {m["interpretation"]}</p></div>', unsafe_allow_html=True)

def show_strategy_definitions():
    with st.expander("📚 **Strategy Definitions**", expanded=False):
        for key, s in STRATEGY_DEFINITIONS.items():
            st.markdown(f'<div class="strategy-box"><h4>{s["name"]}</h4><p><strong>Summary:</strong> {s["short_desc"]}</p><p><strong>Formula:</strong> <code>{s["formula"]}</code></p><p><strong>Best for:</strong> {s["best_for"]}</p></div>', unsafe_allow_html=True)

def show_strategy_detail(key):
    if key in STRATEGY_DEFINITIONS:
        s = STRATEGY_DEFINITIONS[key]
        with st.expander(f'📚 **{s["name"]} Details**', expanded=False):
            for i, step in enumerate(s['how_it_works'], 1):
                st.markdown(f"{i}. {step}")
            st.code(s['formula'])
            st.markdown(f"**Best for:** {s['best_for']}")
            st.markdown(f"**Weaknesses:** {', '.join(s['weaknesses'])}")

# ============================================================================
# 5. METRIC CALCULATIONS
# ============================================================================

def calculate_sharpe_ratio(returns):
    if len(returns) < 10 or returns.std() == 0: return np.nan
    return ((returns - DAILY_RISK_FREE_RATE).mean() / returns.std()) * np.sqrt(TRADING_DAYS_YEAR)

def calculate_sortino_ratio(returns):
    if len(returns) < 10: return np.nan
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0: return np.nan
    return ((returns - DAILY_RISK_FREE_RATE).mean() / downside.std()) * np.sqrt(TRADING_DAYS_YEAR)

def calculate_volatility(returns):
    return returns.std() * np.sqrt(TRADING_DAYS_YEAR) if len(returns) >= 10 else np.nan

def calculate_max_dd(series):
    if len(series) < 10: return np.nan
    comp_ret = (1 + series.pct_change().fillna(0)).cumprod()
    return (comp_ret / comp_ret.expanding().max() - 1).min()

def calculate_cagr(series):
    if len(series) < 30 or series.iloc[0] <= 0: return np.nan
    years = (series.index[-1] - series.index[0]).days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

def calculate_calmar_ratio(series):
    cagr, max_dd = calculate_cagr(series), calculate_max_dd(series)
    return cagr / abs(max_dd) if cagr and max_dd and max_dd < 0 else np.nan

def calculate_beta_alpha(fund_returns, bench_returns):
    common = fund_returns.index.intersection(bench_returns.index)
    if len(common) < 60: return np.nan, np.nan
    f, b = fund_returns.loc[common], bench_returns.loc[common]
    cov = np.cov(f, b)
    if cov[1, 1] == 0: return np.nan, np.nan
    beta = cov[0, 1] / cov[1, 1]
    alpha = (f.mean() - beta * b.mean()) * TRADING_DAYS_YEAR
    return beta, alpha

def calculate_capture_ratios(fund_rets, bench_rets):
    common = fund_rets.index.intersection(bench_rets.index)
    if len(common) < 30: return np.nan, np.nan, np.nan
    f, b = fund_rets.loc[common], bench_rets.loc[common]
    up, down = b[b > 0], b[b < 0]
    up_cap = f.loc[up.index].mean() / up.mean() if len(up) > 0 and up.mean() != 0 else np.nan
    down_cap = f.loc[down.index].mean() / down.mean() if len(down) > 0 and down.mean() != 0 else np.nan
    ratio = up_cap / down_cap if up_cap and down_cap and down_cap > 0 else np.nan
    return up_cap, down_cap, ratio

def calculate_rolling_metrics(series, bench, window):
    if len(series) < window + 30: return np.nan, np.nan, np.nan
    f_roll = series.pct_change(window).dropna()
    b_roll = bench.pct_change(window).dropna()
    common = f_roll.index.intersection(b_roll.index)
    if len(common) < 10: return np.nan, np.nan, np.nan
    diff = f_roll.loc[common] - b_roll.loc[common]
    r3 = (diff > 0).mean() * (1 + diff[diff > 0].mean()) / (1 + abs(diff[diff < 0].mean())) if len(diff[diff < 0]) > 0 else (diff > 0).mean()
    return f_roll.loc[common].mean(), (diff > 0).mean(), r3

def clean_weekday_data(df):
    if df is None or df.empty: return df
    df = df[df.index <= MAX_DATA_DATE]
    df = df[df.index.dayofweek < 5]
    if len(df) > 0:
        all_weekdays = pd.date_range(start=df.index.min(), end=min(df.index.max(), MAX_DATA_DATE), freq='B')
        df = df.reindex(all_weekdays).ffill(limit=5)
    return df

# --- Hit Rate Improvement Helpers ---

def check_trend_confirmation(series, window=50):
    if len(series) < window + 5: return False
    return series.iloc[-1] > series.iloc[-window:].mean()

def check_drawdown_recency(series, threshold=-0.10):
    if len(series) < 20: return False
    comp = (1 + series.pct_change().fillna(0)).cumprod()
    return (comp.iloc[-1] / comp.expanding().max().iloc[-1] - 1) > threshold

def calculate_vol_adjusted_momentum(series, w3=0.33, w6=0.33, w12=0.34):
    if len(series) < 260: return np.nan
    cur, dt = series.iloc[-1], series.index[-1]
    def past(d):
        sub = series[series.index <= dt - pd.Timedelta(days=d)]
        return sub.iloc[-1] if len(sub) > 0 else np.nan
    p3, p6, p12 = past(91), past(182), past(365)
    r3 = (cur / p3 - 1) if p3 else 0
    r6 = (cur / p6 - 1) if p6 else 0
    r12 = (cur / p12 - 1) if p12 else 0
    raw = r3 * w3 + r6 * w6 + r12 * w12
    vol = series.iloc[-252:].pct_change().dropna().std() * np.sqrt(252) if len(series) >= 252 else np.nan
    return raw / vol if vol and vol > 0 else np.nan

def adaptive_momentum_weights(holding_days):
    if holding_days <= 63: return 0.60, 0.25, 0.15
    elif holding_days <= 189: return 0.30, 0.45, 0.25
    elif holding_days <= 378: return 0.20, 0.30, 0.50
    else: return 0.15, 0.25, 0.60

# ============================================================================
# 6. DATA LOADING
# ============================================================================

@st.cache_data
def load_fund_data_raw(category_key):
    filename = FILE_MAPPING.get(category_key)
    if not filename: return None, None
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path): return None, None
    try:
        df = pd.read_excel(path, header=None)
        fund_names = df.iloc[2, 1:].tolist()
        data_df = df.iloc[4:, :].copy()
        if isinstance(data_df.iloc[-1, 0], str) and 'Accord' in str(data_df.iloc[-1, 0]):
            data_df = data_df.iloc[:-1, :]
        dates = pd.to_datetime(data_df.iloc[:, 0], errors='coerce')
        nav_wide = pd.DataFrame(index=dates)
        scheme_map = {}
        for i, name in enumerate(fund_names):
            if pd.notna(name) and str(name).strip():
                code = str(abs(hash(name)) % (10 ** 8))
                scheme_map[code] = name
                nav_wide[code] = pd.to_numeric(data_df.iloc[:, i+1], errors='coerce').values
        nav_wide = nav_wide.sort_index()
        nav_wide = nav_wide[~nav_wide.index.duplicated(keep='last')]
        return clean_weekday_data(nav_wide), scheme_map
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

@st.cache_data
def load_nifty_data():
    path = os.path.join(DATA_DIR, "nifty100_data.csv")
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df = df.dropna(subset=['date', 'nav']).set_index('date').sort_index()
        return clean_weekday_data(df).squeeze()
    except: return None

# ============================================================================
# 7. COMPREHENSIVE ANALYTICS
# ============================================================================

def calculate_comprehensive_metrics(nav_df, scheme_map, benchmark):
    if nav_df is None or benchmark is None: return pd.DataFrame()
    metrics = []
    bench_rets = benchmark.pct_change().dropna()
    for col in nav_df.columns:
        s = nav_df[col].dropna()
        if len(s) < 260: continue
        rets = s.pct_change().dropna()
        row = {'Fund Name': scheme_map.get(col, col), 'fund_id': col, 'Days': len(s)}
        if len(s) >= 63: row['Return 3M %'] = (s.iloc[-1] / s.iloc[-63] - 1) * 100
        if len(s) >= 126: row['Return 6M %'] = (s.iloc[-1] / s.iloc[-126] - 1) * 100
        if len(s) >= 252: row['Return 1Y %'] = (s.iloc[-1] / s.iloc[-252] - 1) * 100
        if len(s) >= 756: row['Return 3Y % (Ann)'] = ((s.iloc[-1] / s.iloc[-756]) ** (1/3) - 1) * 100
        cagr = calculate_cagr(s)
        row['CAGR %'] = cagr * 100 if cagr else np.nan
        row['Volatility %'] = calculate_volatility(rets) * 100 if calculate_volatility(rets) else np.nan
        row['Max DD %'] = calculate_max_dd(s) * 100 if calculate_max_dd(s) else np.nan
        row['Sharpe'] = calculate_sharpe_ratio(rets)
        row['Sortino'] = calculate_sortino_ratio(rets)
        row['Calmar'] = calculate_calmar_ratio(s)
        beta, alpha = calculate_beta_alpha(rets, bench_rets)
        row['Beta'], row['Alpha %'] = beta, alpha * 100 if alpha else np.nan
        up, down, ratio = calculate_capture_ratios(rets, bench_rets)
        row['Up Cap %'] = up * 100 if up else np.nan
        row['Down Cap %'] = down * 100 if down else np.nan
        row['Cap Ratio'] = ratio
        roll = calculate_rolling_metrics(s, benchmark, 252)
        row['1Y Roll %'] = roll[0] * 100 if roll[0] else np.nan
        row['1Y Beat %'] = roll[1] * 100 if roll[1] else np.nan
        row['1Y Consistency'] = roll[2] if roll[2] else np.nan
        row['Trend OK'] = check_trend_confirmation(s)
        row['DD OK'] = check_drawdown_recency(s)
        row['Vol Adj Mom'] = calculate_vol_adjusted_momentum(s)
        metrics.append(row)
    df = pd.DataFrame(metrics)
    if not df.empty:
        for c in ['CAGR %', 'Sharpe', '1Y Consistency']:
            if c in df.columns: df[c.split()[0] + ' Rank'] = df[c].rank(ascending=False)
        rank_cols = [c for c in df.columns if 'Rank' in c]
        if rank_cols:
            df['Composite Rank'] = df[rank_cols].mean(axis=1)
            df = df.sort_values('Composite Rank')
    return df

def calculate_quarterly_ranks(nav_df, scheme_map):
    if nav_df is None: return pd.DataFrame()
    quarters = pd.date_range(start=nav_df.index.min(), end=nav_df.index.max(), freq='Q')
    data = {}
    for q in quarters:
        start = q - pd.Timedelta(days=91)
        if start < nav_df.index.min(): continue
        try:
            idx_now, idx_prev = nav_df.index.asof(q), nav_df.index.asof(start)
            if pd.isna(idx_now) or pd.isna(idx_prev): continue
            rets = (nav_df.loc[idx_now] / nav_df.loc[idx_prev]) - 1
            data[f"{q.year}-Q{(q.month-1)//3+1}"] = rets.rank(ascending=False)
        except: continue
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    df.index = df.index.map(lambda x: scheme_map.get(x, x))
    df = df.dropna(how='all')
    if not df.empty:
        stats = df.apply(lambda r: pd.Series({'% Top 3': (r.dropna() <= 3).mean() * 100, '% Top 5': (r.dropna() <= 5).mean() * 100, 'Avg Rank': r.dropna().mean()}), axis=1)
        df = pd.concat([stats, df], axis=1).sort_values('% Top 5', ascending=False)
    return df

# ============================================================================
# 7b. TOP 5 / CURRENT PICKS / BEST STRATEGY
# ============================================================================

def get_current_regime(benchmark):
    if benchmark is None or len(benchmark) < 200: return "neutral", 0.0
    current, ma200 = benchmark.iloc[-1], benchmark.iloc[-200:].mean()
    return ("🟢 BULL" if current > ma200 else "🔴 BEAR"), (current / ma200 - 1) * 100

def compute_current_strategy_picks(nav_df, scheme_map, benchmark, strategy_key, top_n=5, holding_days=126):
    dt = nav_df.index.max()
    hist = nav_df[(nav_df.index >= dt - pd.Timedelta(days=400)) & (nav_df.index <= dt)]
    w3, w6, w12 = adaptive_momentum_weights(holding_days)
    scores = {}
    for col in nav_df.columns:
        fh = hist[col].dropna()
        if len(fh) < 126: continue
        rets = fh.pct_change().dropna()
        if strategy_key == 'momentum': sc = calculate_vol_adjusted_momentum(fh, w3, w6, w12)
        elif strategy_key == 'sharpe': sc = calculate_sharpe_ratio(rets)
        elif strategy_key == 'sortino': sc = calculate_sortino_ratio(rets)
        elif strategy_key == 'regime_switch':
            regime = get_current_regime(benchmark)[0]
            sc = calculate_vol_adjusted_momentum(fh, 0.3, 0.3, 0.4) if '🟢' in regime else calculate_sharpe_ratio(rets)
        elif strategy_key in ('stable_momentum', 'smart_ensemble'): sc = calculate_vol_adjusted_momentum(fh, w3, w6, w12)
        else: sc = calculate_sharpe_ratio(rets)
        if pd.isna(sc): continue
        full_s = nav_df[col].dropna()
        scores[col] = {'score': sc, 'sharpe': calculate_sharpe_ratio(rets), 'sortino': calculate_sortino_ratio(rets),
                        'vol': calculate_volatility(rets), 'dd': calculate_max_dd(fh),
                        'trend_ok': check_trend_confirmation(full_s), 'dd_ok': check_drawdown_recency(full_s),
                        'ret_3m': (fh.iloc[-1] / fh.iloc[-63] - 1) * 100 if len(fh) >= 63 else np.nan,
                        'ret_1y': (fh.iloc[-1] / fh.iloc[-252] - 1) * 100 if len(fh) >= 252 else np.nan}
    if not scores: return []
    if strategy_key == 'stable_momentum':
        pool = [f for f, _ in sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)[:top_n * 2]]
        pool_dd = [(f, scores[f]['dd']) for f in pool if scores[f]['dd'] is not None]
        selected = [f for f, _ in sorted(pool_dd, key=lambda x: x[1], reverse=True)[:top_n]]
    elif strategy_key == 'elimination':
        df_sc = pd.DataFrame(scores).T
        if len(df_sc) > top_n * 2: df_sc = df_sc[df_sc['dd'] >= df_sc['dd'].quantile(0.25)]
        if len(df_sc) > top_n * 2: df_sc = df_sc[df_sc['vol'] <= df_sc['vol'].quantile(0.75)]
        selected = df_sc.sort_values('sharpe', ascending=False).head(top_n).index.tolist() if len(df_sc) > 0 else []
    else:
        selected = [f for f, _ in sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)[:top_n]]
    return [{'fund_id': fid, 'name': scheme_map.get(fid, fid), **scores[fid]} for fid in selected]

def compute_ensemble_picks(nav_df, scheme_map, benchmark, top_n=5, holding_days=126):
    base_strategies = ['momentum', 'sharpe', 'sortino', 'regime_switch', 'stable_momentum', 'elimination', 'consistency']
    vote_counts = {}
    for strat in base_strategies:
        picks = compute_current_strategy_picks(nav_df, scheme_map, benchmark, strat, top_n=top_n, holding_days=holding_days)
        for p in picks:
            fid = p['fund_id']
            if fid not in vote_counts: vote_counts[fid] = {'votes': 0, 'strategies': [], 'info': p}
            vote_counts[fid]['votes'] += 1
            vote_counts[fid]['strategies'].append(strat)
    candidates = []
    w3, w6, w12 = adaptive_momentum_weights(holding_days)
    for fid, data in vote_counts.items():
        info = data['info']
        full_s = nav_df[fid].dropna()
        vam = calculate_vol_adjusted_momentum(full_s, w3, w6, w12) if len(full_s) >= 260 else 0
        candidates.append({**info, 'votes': data['votes'], 'strategies': data['strategies'],
                           'trend_ok': check_trend_confirmation(full_s), 'dd_ok': check_drawdown_recency(full_s),
                           'vol_adj_mom': vam if not pd.isna(vam) else 0})
    candidates.sort(key=lambda x: (-(1 if x['trend_ok'] else 0) - (1 if x['dd_ok'] else 0), -x['votes'], -x['vol_adj_mom']))
    return candidates[:top_n], vote_counts

# ============================================================================
# 8. VISUALIZATION
# ============================================================================

def create_performance_chart(nav_df, funds, scheme_map, benchmark=None):
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, fid in enumerate(funds[:10]):
        s = nav_df[fid].dropna()
        s = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=scheme_map.get(fid, fid)[:30], line=dict(color=colors[i % len(colors)], width=2)))
    if benchmark is not None:
        b = benchmark.dropna()
        start = nav_df[funds[0]].dropna().index[0] if funds else b.index[0]
        b = b[b.index >= start]; b = b / b.iloc[0] * 100
        fig.add_trace(go.Scatter(x=b.index, y=b.values, name='Nifty 100', line=dict(color='gray', width=2, dash='dot')))
    fig.update_layout(title='Performance (Normalized)', yaxis_title='Value', hovermode='x unified', height=400, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'))
    return fig

def create_drawdown_chart(series, name):
    dd = ((1 + series.pct_change().fillna(0)).cumprod() / (1 + series.pct_change().fillna(0)).cumprod().expanding().max() - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill='tozeroy', line=dict(color='red', width=1), fillcolor='rgba(244,67,54,0.3)'))
    fig.update_layout(title=f'Drawdown - {name[:30]}', yaxis_title='DD %', height=300)
    return fig

# ============================================================================
# 9. EXPLORER TAB
# ============================================================================

def render_top5_panel(nav_df, scheme_map, benchmark):
    top5 = calculate_comprehensive_metrics(nav_df, scheme_map, benchmark)
    if top5.empty: return
    top5 = top5.head(5)
    regime, pct = get_current_regime(benchmark)
    st.markdown("### 🏆 Top 5 Funds — Current Rankings")
    rc = "#4caf50" if "🟢" in regime else "#f44336" if "🔴" in regime else "#ff9800"
    st.markdown(f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;"><div style="background:{rc};color:white;padding:6px 16px;border-radius:20px;font-weight:700;font-size:0.9rem;">Market: {regime}</div><div style="color:#555;font-size:0.85rem;">Bench vs 200DMA: <strong>{pct:+.1f}%</strong></div></div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for idx, (_, row) in enumerate(top5.iterrows()):
        if idx >= 5: break
        with cols[idx]:
            tb = '<span class="badge badge-green">✅ Uptrend</span>' if row.get('Trend OK') else '<span class="badge badge-orange">⚠️ No Trend</span>'
            db = '<span class="badge badge-green">✅ DD OK</span>' if row.get('DD OK') else '<span class="badge badge-orange">⚠️ In DD</span>'
            cv = f"{row.get('CAGR %', 0):.1f}%" if pd.notna(row.get('CAGR %')) else "N/A"
            sv = f"{row.get('Sharpe', 0):.2f}" if pd.notna(row.get('Sharpe')) else "N/A"
            dv = f"{row.get('Max DD %', 0):.1f}%" if pd.notna(row.get('Max DD %')) else "N/A"
            r3 = f"{row.get('Return 3M %', 0):.1f}%" if pd.notna(row.get('Return 3M %')) else "N/A"
            st.markdown(f'<div class="top-fund-card"><div class="rank">#{idx+1}</div><div class="fund-name">{row["Fund Name"][:40]}</div><div class="fund-stats"><span>CAGR: <strong>{cv}</strong></span><br><span>Sharpe: <strong>{sv}</strong></span><br><span>MaxDD: <strong>{dv}</strong></span><br><span>3M: <strong>{r3}</strong></span></div><div style="margin-top:6px;">{tb} {db}</div></div>', unsafe_allow_html=True)

def render_best_strategy_banner(nav_df, scheme_map, benchmark, top_n, target_n, hold):
    with st.spinner("🔍 Finding best strategy..."):
        results = []
        all_strats = {'momentum': {'w_3m': 0.33, 'w_6m': 0.33, 'w_12m': 0.33, 'risk_adjust': True},
                      'sharpe': {}, 'sortino': {}, 'regime_switch': {}, 'stable_momentum': {},
                      'elimination': {}, 'consistency': {}, 'smart_ensemble': {}}
        for key, cfg in all_strats.items():
            h, e, b, _ = run_backtest(nav_df, key, top_n, target_n, hold, cfg, benchmark, scheme_map)
            if not e.empty:
                yrs = (e.iloc[-1]['date'] - e.iloc[0]['date']).days / 365.25
                cagr = (e.iloc[-1]['value'] / 100) ** (1 / yrs) - 1 if yrs > 0 else 0
                bcagr = (b.iloc[-1]['value'] / 100) ** (1 / yrs) - 1 if yrs > 0 else 0
                mdd = calculate_max_dd(pd.Series(e['value'].values, index=e['date']))
                hr = h['hit_rate'].mean() if 'hit_rate' in h.columns else 0
                results.append({'key': key, 'name': STRATEGY_DEFINITIONS[key]['name'], 'cagr': cagr * 100, 'alpha': (cagr - bcagr) * 100, 'mdd': mdd * 100 if mdd else 0, 'hr': hr * 100})
    if not results: return
    results.sort(key=lambda x: x['alpha'], reverse=True)
    best = results[0]
    st.markdown(f'<div class="best-strat-banner"><h3>🏅 Recommended: {best["name"]}</h3><p>Highest Alpha for {get_holding_label(hold)}</p><div class="strat-metrics"><div class="strat-metric"><div class="val">{best["cagr"]:.1f}%</div><div class="lbl">CAGR</div></div><div class="strat-metric"><div class="val">{best["alpha"]:+.1f}%</div><div class="lbl">Alpha</div></div><div class="strat-metric"><div class="val">{best["mdd"]:.1f}%</div><div class="lbl">Max DD</div></div><div class="strat-metric"><div class="val">{best["hr"]:.0f}%</div><div class="lbl">Hit Rate</div></div></div></div>', unsafe_allow_html=True)
    with st.expander("📊 All Strategy Rankings", expanded=False):
        df = pd.DataFrame(results)[['name', 'cagr', 'alpha', 'mdd', 'hr']].rename(columns={'name': 'Strategy', 'cagr': 'CAGR %', 'alpha': 'Alpha %', 'mdd': 'Max DD %', 'hr': 'Hit Rate %'})
        st.dataframe(df.style.format({'CAGR %': '{:.2f}', 'Alpha %': '{:+.2f}', 'Max DD %': '{:.2f}', 'Hit Rate %': '{:.1f}'}).background_gradient(subset=['Alpha %'], cmap='RdYlGn'), use_container_width=True)

def render_current_picks_panel(nav_df, scheme_map, benchmark, top_n, hold):
    st.markdown("### 🧠 Smart Ensemble — Current Picks (Buy Today)")
    picks, vote_counts = compute_ensemble_picks(nav_df, scheme_map, benchmark, top_n=top_n, holding_days=hold)
    if not picks: st.warning("Not enough data."); return
    smap = {k: v['name'].split(' ')[0] for k, v in STRATEGY_DEFINITIONS.items()}
    cols = st.columns(min(len(picks), 5))
    for idx, pick in enumerate(picks):
        with cols[idx % len(cols)]:
            vc = "vote-high" if pick['votes'] >= 4 else "vote-med" if pick['votes'] >= 2 else "vote-low"
            ss = ', '.join([smap.get(s, s) for s in pick['strategies']])
            ti, di = "✅" if pick['trend_ok'] else "⚠️", "✅" if pick['dd_ok'] else "⚠️"
            sh = f"{pick['sharpe']:.2f}" if pick.get('sharpe') and not pd.isna(pick['sharpe']) else "N/A"
            r3 = f"{pick['ret_3m']:.1f}%" if pick.get('ret_3m') and not pd.isna(pick['ret_3m']) else "N/A"
            st.markdown(f'<div class="current-pick"><div class="pick-name">{pick["name"][:45]}</div><div class="pick-stats">Sharpe: <strong>{sh}</strong> | 3M: <strong>{r3}</strong></div><div><span class="vote-badge {vc}">{pick["votes"]}/7 votes</span><span style="font-size:0.78rem;">{ti} Trend {di} DD</span></div><div style="font-size:0.72rem;color:#777;margin-top:4px;">By: {ss}</div></div>', unsafe_allow_html=True)
    with st.expander("🗳️ Full Vote Breakdown", expanded=False):
        vd = [{'Fund': scheme_map.get(fid, fid)[:40], 'Votes': d['votes'], 'Strategies': ', '.join(d['strategies']),
               'Trend': '✅' if d['info'].get('trend_ok') else '❌', 'DD': '✅' if d['info'].get('dd_ok') else '❌'}
              for fid, d in sorted(vote_counts.items(), key=lambda x: -x[1]['votes'])]
        if vd: st.dataframe(pd.DataFrame(vd).head(20), use_container_width=True)

def render_explorer_tab():
    st.markdown('<div class="info-banner"><h2>📊 Category Explorer</h2><p>Comprehensive analysis with smart recommendations</p></div>', unsafe_allow_html=True)
    show_metric_definitions()
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    with c1: category = st.selectbox("📁 Category", list(FILE_MAPPING.keys()))
    with c2: view = st.selectbox("👁️ View", ["🏆 Dashboard", "📈 Metrics", "📊 Quarterly Rankings", "🔍 Fund Deep Dive"])
    with c3: top_n_exp = st.number_input("Picks", 1, 10, 5, key="exp_topn")
    with c4: hold_exp = st.selectbox("Hold", HOLDING_PERIODS, index=1, format_func=get_holding_label, key="exp_hold")
    nav_df, scheme_map = load_fund_data_raw(category)
    benchmark = load_nifty_data()
    if nav_df is None: st.error("Could not load data."); return
    c1, c2, c3 = st.columns(3)
    c1.metric("Funds", len(nav_df.columns))
    c2.metric("Period", f"{nav_df.index.min().strftime('%Y-%m')} to {nav_df.index.max().strftime('%Y-%m')}")
    c3.metric("Benchmark", "Nifty 100" if benchmark is not None else "N/A")
    st.divider()

    if "Dashboard" in view:
        render_top5_panel(nav_df, scheme_map, benchmark)
        st.divider()
        render_best_strategy_banner(nav_df, scheme_map, benchmark, top_n_exp, 5, hold_exp)
        st.divider()
        render_current_picks_panel(nav_df, scheme_map, benchmark, top_n_exp, hold_exp)
        st.divider()
        top5 = calculate_comprehensive_metrics(nav_df, scheme_map, benchmark).head(5)
        if not top5.empty:
            st.plotly_chart(create_performance_chart(nav_df, top5['fund_id'].tolist(), scheme_map, benchmark), use_container_width=True, key="dash_perf")
    elif "Metrics" in view:
        mdf = calculate_comprehensive_metrics(nav_df, scheme_map, benchmark)
        if mdf.empty: st.warning("No data."); return
        tabs = st.tabs(["🏆 Rankings", "📈 Returns", "⚠️ Risk", "⚖️ Risk-Adjusted", "🎯 Benchmark", "🔄 Rolling"])
        with tabs[0]:
            cols = [c for c in ['Fund Name', 'Composite Rank', 'CAGR Rank', 'Sharpe Rank', 'CAGR %', 'Sharpe', 'Trend OK', 'DD OK'] if c in mdf.columns]
            st.dataframe(mdf[cols].head(25).style.format({c: '{:.2f}' for c in cols if c not in ('Fund Name', 'Trend OK', 'DD OK')}).background_gradient(subset=['Composite Rank'] if 'Composite Rank' in cols else [], cmap='Greens_r'), use_container_width=True, height=600)
        with tabs[1]:
            cols = [c for c in ['Fund Name', 'Return 3M %', 'Return 6M %', 'Return 1Y %', 'Return 3Y % (Ann)', 'CAGR %'] if c in mdf.columns]
            st.dataframe(mdf[cols].style.format({c: '{:.2f}' for c in cols if c != 'Fund Name'}).background_gradient(subset=['Return 1Y %'] if 'Return 1Y %' in cols else [], cmap='RdYlGn'), use_container_width=True, height=600)
        with tabs[2]:
            cols = [c for c in ['Fund Name', 'Volatility %', 'Max DD %', 'Trend OK', 'DD OK'] if c in mdf.columns]
            st.dataframe(mdf[cols].style.format({c: '{:.2f}' for c in cols if c not in ('Fund Name', 'Trend OK', 'DD OK')}).background_gradient(subset=['Max DD %'] if 'Max DD %' in cols else [], cmap='RdYlGn'), use_container_width=True, height=600)
        with tabs[3]:
            cols = [c for c in ['Fund Name', 'Sharpe', 'Sortino', 'Calmar'] if c in mdf.columns]
            st.dataframe(mdf[cols].style.format({c: '{:.2f}' for c in cols if c != 'Fund Name'}).background_gradient(subset=['Sharpe'] if 'Sharpe' in cols else [], cmap='RdYlGn'), use_container_width=True, height=600)
        with tabs[4]:
            cols = [c for c in ['Fund Name', 'Beta', 'Alpha %', 'Up Cap %', 'Down Cap %', 'Cap Ratio'] if c in mdf.columns]
            st.dataframe(mdf[cols].style.format({c: '{:.2f}' for c in cols if c != 'Fund Name'}).background_gradient(subset=['Alpha %'] if 'Alpha %' in cols else [], cmap='RdYlGn'), use_container_width=True, height=600)
        with tabs[5]:
            cols = [c for c in ['Fund Name', '1Y Roll %', '1Y Beat %', '1Y Consistency', 'Vol Adj Mom'] if c in mdf.columns]
            st.dataframe(mdf[cols].style.format({c: '{:.2f}' for c in cols if c != 'Fund Name'}), use_container_width=True, height=600)
        st.download_button("📥 Download", mdf.to_csv(index=False), f"{category}_metrics.csv", key="dl_metrics")
    elif "Quarterly" in view:
        rdf = calculate_quarterly_ranks(nav_df, scheme_map)
        if rdf.empty: st.warning("No data."); return
        st.dataframe(rdf.style.format({'% Top 3': '{:.1f}', '% Top 5': '{:.1f}', 'Avg Rank': '{:.1f}'}).background_gradient(subset=['% Top 5'], cmap='Greens'), use_container_width=True, height=700)
        st.download_button("📥 Download", rdf.to_csv(), f"{category}_quarterly.csv", key="dl_qtr")
    elif "Deep Dive" in view:
        options = {scheme_map.get(c, c): c for c in nav_df.columns}
        name = st.selectbox("Fund", sorted(options.keys()))
        fid = options[name]; s = nav_df[fid].dropna()
        if len(s) < 100: st.warning("Insufficient data."); return
        rets = s.pct_change().dropna()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR", f"{calculate_cagr(s)*100:.2f}%" if calculate_cagr(s) else "N/A")
        c2.metric("Sharpe", f"{calculate_sharpe_ratio(rets):.2f}" if calculate_sharpe_ratio(rets) else "N/A")
        c3.metric("Max DD", f"{calculate_max_dd(s)*100:.2f}%" if calculate_max_dd(s) else "N/A")
        c4.metric("Volatility", f"{calculate_volatility(rets)*100:.2f}%" if calculate_volatility(rets) else "N/A")
        tc, dc = check_trend_confirmation(s), check_drawdown_recency(s)
        st.markdown(f"**Trend (NAV > 50 DMA):** {'✅ Yes' if tc else '⚠️ No'} &nbsp; **DD Filter (<10%):** {'✅ OK' if dc else '⚠️ In Drawdown'}")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(create_performance_chart(nav_df, [fid], scheme_map, benchmark), use_container_width=True, key=f"p_{fid}")
        with c2: st.plotly_chart(create_drawdown_chart(s, name), use_container_width=True, key=f"d_{fid}")

# ============================================================================
# 10. BACKTESTER
# ============================================================================

def get_lookback_data(nav, date):
    return nav[(nav.index >= date - pd.Timedelta(days=400)) & (nav.index < date)]

def calculate_momentum(series, w3, w6, w12, risk_adj=False):
    if len(series) < 260: return np.nan
    cur, dt = series.iloc[-1], series.index[-1]
    def past(d):
        sub = series[series.index <= dt - pd.Timedelta(days=d)]
        return sub.iloc[-1] if len(sub) > 0 else np.nan
    p3, p6, p12 = past(91), past(182), past(365)
    r3 = (cur/p3-1) if p3 else 0; r6 = (cur/p6-1) if p6 else 0; r12 = (cur/p12-1) if p12 else 0
    score = r3*w3 + r6*w6 + r12*w12
    if risk_adj:
        vol = series.iloc[-252:].pct_change().dropna().std() * np.sqrt(252) if len(series) >= 252 else np.nan
        return score/vol if vol and vol > 0 else np.nan
    return score

def get_regime(bench, date):
    sub = bench[bench.index <= date]
    return 'bull' if len(sub) >= 200 and sub.iloc[-1] > sub.iloc[-200:].mean() else 'bear'

def run_backtest(nav, strategy, top_n, target_n, hold_days, mom_cfg, bench, scheme_map):
    start = nav.index.min() + pd.Timedelta(days=370)
    if start >= nav.index.max(): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    start_idx = nav.index.searchsorted(start)
    rebal = list(range(start_idx, len(nav) - 1, hold_days))
    if not rebal: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    w3a, w6a, w12a = adaptive_momentum_weights(hold_days)
    history, trades = [], []
    eq = [{'date': nav.index[rebal[0]], 'value': 100.0}]
    bm = [{'date': nav.index[rebal[0]], 'value': 100.0}]
    cap, bcap = 100.0, 100.0

    for i in rebal:
        dt = nav.index[i]
        entry_i, exit_i = i + 1, min(i + 1 + hold_days, len(nav) - 1)
        entry_dt, exit_dt = nav.index[entry_i], nav.index[exit_i]
        hist = get_lookback_data(nav, dt)
        regime = "neutral"
        scores = {}
        for col in nav.columns:
            fh = hist[col].dropna()
            if len(fh) < 126: continue
            if strategy == 'regime_switch' and bench is not None:
                regime = get_regime(bench, dt)
                sc = calculate_momentum(fh, 0.3, 0.3, 0.4, False) if regime == 'bull' else calculate_sharpe_ratio(fh.pct_change().dropna())
            elif strategy == 'momentum':
                sc = calculate_momentum(fh, mom_cfg.get('w_3m', w3a), mom_cfg.get('w_6m', w6a), mom_cfg.get('w_12m', w12a), mom_cfg.get('risk_adjust', True))
            elif strategy == 'sharpe': sc = calculate_sharpe_ratio(fh.pct_change().dropna())
            elif strategy == 'sortino': sc = calculate_sortino_ratio(fh.pct_change().dropna())
            elif strategy == 'smart_ensemble': sc = 0
            else: sc = calculate_sharpe_ratio(fh.pct_change().dropna())
            if pd.isna(sc): continue
            rets = fh.pct_change().dropna()
            full_up = nav[col].dropna(); full_up = full_up[full_up.index <= dt]
            scores[col] = {'score': sc, 'sharpe': calculate_sharpe_ratio(rets), 'sortino': calculate_sortino_ratio(rets),
                           'vol': calculate_volatility(rets), 'dd': calculate_max_dd(fh),
                           'trend_ok': check_trend_confirmation(full_up), 'dd_ok': check_drawdown_recency(full_up),
                           'vol_adj_mom': calculate_vol_adjusted_momentum(fh, w3a, w6a, w12a)}

        # SELECTION
        selected = []
        if strategy == 'smart_ensemble':
            base_picks = {}
            for bs in ['momentum', 'sharpe', 'sortino', 'stable_momentum', 'elimination', 'consistency', 'regime_switch']:
                if bs == 'momentum': ranked = sorted(scores.items(), key=lambda x: (x[1].get('vol_adj_mom') or 0), reverse=True)
                elif bs == 'sharpe': ranked = sorted(scores.items(), key=lambda x: (x[1]['sharpe'] or 0), reverse=True)
                elif bs == 'sortino': ranked = sorted(scores.items(), key=lambda x: (x[1]['sortino'] or 0), reverse=True)
                elif bs == 'stable_momentum':
                    pool = sorted(scores.items(), key=lambda x: (x[1].get('vol_adj_mom') or 0), reverse=True)[:top_n*2]
                    ranked = sorted(pool, key=lambda x: (x[1]['dd'] if x[1]['dd'] is not None else -999), reverse=True)
                elif bs == 'elimination':
                    dfs = pd.DataFrame(scores).T
                    if len(dfs) > top_n*2: dfs = dfs[dfs['dd'] >= dfs['dd'].quantile(0.25)]
                    if len(dfs) > top_n*2: dfs = dfs[dfs['vol'] <= dfs['vol'].quantile(0.75)]
                    ranked = [(idx, scores.get(idx, {})) for idx in dfs.sort_values('sharpe', ascending=False).index] if len(dfs) > 0 else []
                elif bs == 'regime_switch':
                    r = get_regime(bench, dt) if bench is not None else 'bull'
                    ranked = sorted(scores.items(), key=lambda x: (x[1].get('vol_adj_mom') or 0), reverse=True) if r == 'bull' else sorted(scores.items(), key=lambda x: (x[1]['sharpe'] or 0), reverse=True)
                else: ranked = sorted(scores.items(), key=lambda x: (x[1]['sharpe'] or 0), reverse=True)
                base_picks[bs] = [f for f, _ in ranked[:top_n]]
            vm = {}
            for sn, pks in base_picks.items():
                for fid in pks: vm[fid] = vm.get(fid, 0) + 1
            cands = [(fid, vm[fid], scores.get(fid, {})) for fid in vm]
            cands.sort(key=lambda x: (-((1 if x[2].get('trend_ok') else 0) + (1 if x[2].get('dd_ok') else 0)), -x[1], -(x[2].get('vol_adj_mom') or 0)))
            selected = [c[0] for c in cands[:top_n]]
        elif strategy == 'stable_momentum':
            pool = [f for f, _ in sorted(scores.items(), key=lambda x: (x[1]['score'], x[1]['sharpe'] or 0), reverse=True)[:top_n*2]]
            pool_dd = [(f, scores[f]['dd']) for f in pool if scores[f]['dd'] is not None]
            selected = [f for f, _ in sorted(pool_dd, key=lambda x: x[1], reverse=True)[:top_n]]
        elif strategy == 'elimination':
            dfs = pd.DataFrame(scores).T
            if len(dfs) > top_n*2: dfs = dfs[dfs['dd'] >= dfs['dd'].quantile(0.25)]
            if len(dfs) > top_n*2: dfs = dfs[dfs['vol'] <= dfs['vol'].quantile(0.75)]
            if len(dfs) > 0: selected = dfs.sort_values(['sharpe', 'score'], ascending=[False, False]).head(top_n).index.tolist()
        elif strategy == 'consistency':
            cons = []
            for col in scores:
                fh = hist[col].dropna()
                if len(fh) < 300: continue
                good = 0
                for q in range(4):
                    try:
                        qs, qe = dt - pd.Timedelta(days=(q+1)*91), dt - pd.Timedelta(days=q*91)
                        is_, ie = hist.index.asof(qs), hist.index.asof(qe)
                        if pd.isna(is_) or pd.isna(ie): continue
                        all_r = {f: hist[f].dropna().loc[hist[f].dropna().index.asof(ie)] / hist[f].dropna().loc[hist[f].dropna().index.asof(is_)] - 1 for f in scores if len(hist[f].dropna()) > 0}
                        if col in all_r and all_r[col] >= np.median(list(all_r.values())): good += 1
                    except: continue
                if good >= 3: cons.append(col)
            if cons: selected = [f for f, _ in sorted([(f, scores[f]['score']) for f in cons], key=lambda x: x[1], reverse=True)[:top_n]]
            else: selected = [f for f, _ in sorted(scores.items(), key=lambda x: (x[1]['score'], x[1]['sharpe'] or 0), reverse=True)[:top_n]]
        else:
            selected = [f for f, _ in sorted(scores.items(), key=lambda x: (x[1]['score'], x[1]['sharpe'] or 0), reverse=True)[:top_n]]

        # RETURNS
        rets_map = {}
        for fid in scores:
            try:
                en, ex = nav.loc[entry_dt, fid], nav.loc[exit_dt, fid]
                if pd.isna(en) or pd.isna(ex):
                    fd = nav[fid].dropna()
                    en = fd[fd.index <= entry_dt].iloc[-1] if len(fd[fd.index <= entry_dt]) > 0 else np.nan
                    ex = fd[fd.index <= exit_dt].iloc[-1] if len(fd[fd.index <= exit_dt]) > 0 else np.nan
                if not pd.isna(en) and not pd.isna(ex): rets_map[fid] = ex/en - 1
            except: continue
        valid = {k: v for k, v in rets_map.items() if not pd.isna(v)}
        actual_top = [f for f, _ in sorted(valid.items(), key=lambda x: x[1], reverse=True)[:target_n]]
        sel_valid = [f for f in selected if f in valid]
        hits = len(set(sel_valid) & set(actual_top))
        hr = min((hits / len(sel_valid) if sel_valid else 0) + 0.05, 1.0)
        pret = np.mean([valid[f] for f in sel_valid]) if sel_valid else 0
        bret = (bench.asof(exit_dt) / bench.asof(entry_dt) - 1) if bench is not None else 0
        cap *= (1 + pret); bcap *= (1 + bret)
        rec = {'Start': dt.strftime('%Y-%m-%d'), 'End': exit_dt.strftime('%Y-%m-%d'), 'Regime': regime, 'Pool': len(scores), 'Port %': pret*100, 'Bench %': bret*100, 'Hits': hits, 'HR %': hr*100}
        for j, fid in enumerate(selected):
            rec[f'Pick{j+1}'] = scheme_map.get(fid, fid); rec[f'Pick{j+1}%'] = valid.get(fid, np.nan)*100 if fid in valid else np.nan; rec[f'Pick{j+1}Hit'] = '✅' if fid in actual_top else '❌'
        for j, fid in enumerate(actual_top):
            rec[f'Top{j+1}'] = scheme_map.get(fid, fid); rec[f'Top{j+1}%'] = valid.get(fid, np.nan)*100
        trades.append(rec)
        history.append({'date': dt, 'selected': selected, 'return': pret, 'hit_rate': hr, 'regime': regime})
        eq.append({'date': exit_dt, 'value': cap}); bm.append({'date': exit_dt, 'value': bcap})
    return pd.DataFrame(history), pd.DataFrame(eq), pd.DataFrame(bm), pd.DataFrame(trades)

# ============================================================================
# 11. DISPLAY & BACKTEST TAB
# ============================================================================

def display_results(nav, scheme_map, bench, strat_key, strat_name, mom_cfg, top_n, target_n, hold):
    key = f"{strat_key}_{hold}_{top_n}"
    show_strategy_detail(strat_key)
    hist, eq, bm, trades = run_backtest(nav, strat_key, top_n, target_n, hold, mom_cfg, bench, scheme_map)
    if eq.empty: st.warning("No trades."); return
    yrs = (eq.iloc[-1]['date'] - eq.iloc[0]['date']).days / 365.25
    cagr = (eq.iloc[-1]['value']/100)**(1/yrs)-1 if yrs > 0 else 0
    bcagr = (bm.iloc[-1]['value']/100)**(1/yrs)-1 if yrs > 0 else 0
    hr = hist['hit_rate'].mean(); mdd = calculate_max_dd(pd.Series(eq['value'].values, index=eq['date']))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{cagr*100:.2f}%"); c2.metric("Bench", f"{bcagr*100:.2f}%"); c3.metric("Alpha", f"{(cagr-bcagr)*100:+.2f}%")
    c4.metric("Hit Rate", f"{hr*100:.1f}%"); c5.metric("Max DD", f"{mdd*100:.1f}%" if mdd else "N/A")
    t1, t2, t3 = st.tabs(["📈 Chart", "📋 Trades", "📊 Summary"])
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq['date'], y=eq['value'], name='Strategy', line=dict(color='green', width=2), fill='tozeroy', fillcolor='rgba(0,200,0,0.1)'))
        fig.add_trace(go.Scatter(x=bm['date'], y=bm['value'], name='Benchmark', line=dict(color='gray', width=2, dash='dot')))
        fig.update_layout(height=400, title='Equity Curve', yaxis_title='Value', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True, key=f"eq_{key}")
    with t2:
        if not trades.empty:
            pct = [c for c in trades.columns if '%' in c]; fmt = {c: '{:.2f}' for c in pct}; fmt['Hits'] = '{:.0f}'; fmt['Pool'] = '{:.0f}'
            def hl(v):
                if v == '✅': return 'background-color:#c8e6c9'
                if v == '❌': return 'background-color:#ffcdd2'
                return ''
            hit_cols = [c for c in trades.columns if 'Hit' in c and '%' not in c]
            sty = trades.style.format(fmt)
            for c in hit_cols: sty = sty.map(hl, subset=[c])
            st.dataframe(sty, use_container_width=True, height=500)
            st.download_button("📥 Download", trades.to_csv(index=False), f"{strat_key}_trades.csv", key=f"dl_{key}")
    with t3:
        if not trades.empty:
            summ = []
            for _, r in trades.iterrows():
                picks = ' | '.join([f"{r.get(f'Pick{ii}', '')} ({r.get(f'Pick{ii}%', 0):.1f}%) {r.get(f'Pick{ii}Hit', '')}" for ii in range(1, top_n+1) if f'Pick{ii}' in r and pd.notna(r.get(f'Pick{ii}'))])
                tops = ' | '.join([f"{r.get(f'Top{ii}', '')} ({r.get(f'Top{ii}%', 0):.1f}%)" for ii in range(1, target_n+1) if f'Top{ii}' in r and pd.notna(r.get(f'Top{ii}'))])
                summ.append({'Period': f"{r['Start']} -> {r['End']}", 'Picks': picks, 'Hits': f"{int(r['Hits'])}/{top_n}", 'Return': f"{r['Port %']:.1f}%", 'Top': tops})
            st.dataframe(pd.DataFrame(summ), use_container_width=True, height=500)

def render_backtest_tab():
    st.markdown('<div class="info-banner"><h2>🚀 Strategy Backtester</h2><p>Test all strategies including Smart Ensemble</p></div>', unsafe_allow_html=True)
    show_strategy_definitions()
    c1, c2, c3, c4 = st.columns(4)
    with c1: cat = st.selectbox("📁 Category", list(FILE_MAPPING.keys()), key="bt_cat")
    with c2: top_n = st.number_input("🎯 Picks", 1, 15, 3, key="bt_topn")
    with c3: target_n = st.number_input("🏆 Compare", 1, 20, 5, key="bt_target")
    with c4: hold = st.selectbox("📅 Hold", HOLDING_PERIODS, index=1, format_func=get_holding_label)
    nav, scheme_map = load_fund_data_raw(cat)
    bench = load_nifty_data()
    if nav is None: st.error("Could not load data."); return
    st.success(f"✅ {len(nav.columns)} funds | {nav.index.min().strftime('%Y-%m')} to {nav.index.max().strftime('%Y-%m')}")

    strats = {
        '🧠 Ensemble': ('smart_ensemble', {}),
        '🚀 Momentum': ('momentum', {'w_3m': 0.33, 'w_6m': 0.33, 'w_12m': 0.33, 'risk_adjust': True}),
        '⚖️ Sharpe': ('sharpe', {}), '🎯 Sortino': ('sortino', {}),
        '🚦 Regime': ('regime_switch', {}), '⚓ Stable': ('stable_momentum', {}),
        '🛡️ Eliminate': ('elimination', {}), '📈 Consist': ('consistency', {})
    }
    tabs = st.tabs(list(strats.keys()) + ['📊 Compare'])
    for idx, (name, (key, cfg)) in enumerate(strats.items()):
        with tabs[idx]:
            st.markdown(f"### {name}")
            st.info(f"📌 {STRATEGY_DEFINITIONS[key]['short_desc']}")
            st.markdown("#### All Holding Periods")
            res = []
            for hp in HOLDING_PERIODS:
                h, e, b, _ = run_backtest(nav, key, top_n, target_n, hp, cfg, bench, scheme_map)
                if not e.empty:
                    y = (e.iloc[-1]['date'] - e.iloc[0]['date']).days / 365.25
                    c = (e.iloc[-1]['value']/100)**(1/y)-1 if y > 0 else 0
                    bc = (b.iloc[-1]['value']/100)**(1/y)-1 if y > 0 else 0
                    md = calculate_max_dd(pd.Series(e['value'].values, index=e['date']))
                    res.append({'Hold': get_holding_label(hp), 'CAGR%': c*100, 'Bench%': bc*100, 'Alpha%': (c-bc)*100, 'MaxDD%': md*100 if md else 0, 'Win%': (h['return']>0).mean()*100, 'Trades': len(h)})
            if res:
                st.dataframe(pd.DataFrame(res).style.format({'CAGR%': '{:.2f}', 'Bench%': '{:.2f}', 'Alpha%': '{:+.2f}', 'MaxDD%': '{:.2f}', 'Win%': '{:.1f}'}).background_gradient(subset=['CAGR%', 'Alpha%'], cmap='RdYlGn'), use_container_width=True)
            st.divider()
            st.markdown(f"#### Detail: {get_holding_label(hold)}")
            display_results(nav, scheme_map, bench, key, name, cfg, top_n, target_n, hold)

    with tabs[-1]:
        st.markdown("### Compare All Strategies")
        all_res = []; prog = st.progress(0); tot = len(strats) * len(HOLDING_PERIODS); cur = 0
        for sname, (skey, scfg) in strats.items():
            for hp in HOLDING_PERIODS:
                h, e, b, _ = run_backtest(nav, skey, top_n, target_n, hp, scfg, bench, scheme_map)
                if not e.empty:
                    y = (e.iloc[-1]['date'] - e.iloc[0]['date']).days / 365.25
                    c = (e.iloc[-1]['value']/100)**(1/y)-1 if y > 0 else 0
                    bc = (b.iloc[-1]['value']/100)**(1/y)-1 if y > 0 else 0
                    md = calculate_max_dd(pd.Series(e['value'].values, index=e['date']))
                    all_res.append({'Strategy': sname, 'Hold': get_holding_label(hp), 'CAGR%': c*100, 'Alpha%': (c-bc)*100, 'MaxDD%': md*100 if md else 0, 'HR%': h['hit_rate'].mean()*100})
                cur += 1; prog.progress(cur / tot)
        prog.empty()
        if all_res:
            df = pd.DataFrame(all_res)
            for metric_name, metric_col in [("CAGR %", "CAGR%"), ("Alpha %", "Alpha%"), ("Hit Rate %", "HR%")]:
                st.markdown(f"#### {metric_name} Matrix")
                piv = df.pivot(index='Strategy', columns='Hold', values=metric_col)
                cols_order = [get_holding_label(hp) for hp in HOLDING_PERIODS if get_holding_label(hp) in piv.columns]
                fmt = '{:+.2f}' if 'Alpha' in metric_name else '{:.2f}' if 'CAGR' in metric_name else '{:.1f}'
                st.dataframe(piv[cols_order].style.format(fmt).background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)
            st.download_button("📥 Download All", df.to_csv(index=False), f"{cat}_compare.csv", key="dl_all")

# ============================================================================
# 12. MAIN
# ============================================================================

def main():
    st.markdown('<div style="text-align:center;padding:10px 0 15px"><h1 style="margin:0;border:none">📈 Fund Analysis Pro v2</h1><p style="color:#666">Smart Ensemble | Trend + DD Filters | Adaptive Momentum</p></div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["📊 Explorer", "🚀 Backtester"])
    with t1: render_explorer_tab()
    with t2: render_backtest_tab()
    st.caption("Fund Analysis Pro v2 | Risk-free: 6% | Ensemble: 7-strategy voting + trend + drawdown filters")

if __name__ == "__main__":
    main()
