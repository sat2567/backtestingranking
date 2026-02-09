"""
Advanced Fund Analysis Dashboard - Enhanced Version
====================================================
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
    /* Main container padding */
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }
    
    /* Header Styling */
    h1 { color: #1E3A5F; font-weight: 700; padding-bottom: 0.5rem; border-bottom: 3px solid #4CAF50; margin-bottom: 1.5rem; }
    
    /* Native Metric Container Styling */
    div[data-testid="metric-container"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; padding: 15px; }
    div[data-testid="metric-container"] label { color: rgba(255, 255, 255, 0.9) !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: white !important; font-weight: 700; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 10px 20px; font-weight: 500; color: #333 !important; }
    .stTabs [aria-selected="true"] { background-color: #4CAF50 !important; color: white !important; }
    
    /* Footer & Menu Hide */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* CUSTOM METRIC BOX (High Contrast Fix) */
    .metric-box { 
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%); 
        border-radius: 12px; 
        padding: 18px; 
        margin: 10px 0; 
        border-left: 5px solid #4CAF50;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: #1f2937; /* Force dark text */
    }
    .metric-box h4 { 
        color: #1E3A5F !important; 
        margin: 0 0 10px 0; 
        font-weight: 700;
    }
    .metric-box p { 
        color: #374151 !important; /* Dark Grey for readability */
        margin: 5px 0;
        font-size: 0.95rem;
    }
    .metric-box .formula { 
        background: #263238; 
        color: #80cbc4 !important; 
        padding: 10px 15px; 
        border-radius: 8px; 
        font-family: 'Courier New', monospace; 
        margin: 10px 0;
        font-size: 0.9rem;
    }
    
    /* CUSTOM STRATEGY BOX (High Contrast Fix) */
    .strategy-box { 
        background: white; 
        border-radius: 12px; 
        padding: 20px; 
        margin: 15px 0; 
        box-shadow: 0 3px 12px rgba(0,0,0,0.08); 
        border-left: 5px solid #2196F3;
        color: #1f2937; /* Force dark text */
    }
    .strategy-box h4 {
        color: #1E3A5F !important;
        margin-top: 0;
    }
    .strategy-box p {
        color: #374151 !important;
        line-height: 1.5;
    }
    .strategy-box code {
        background-color: #f1f5f9;
        color: #d63384;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
    }

    /* INFO BANNER */
    .info-banner { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 20px; 
        border-radius: 12px; 
        margin-bottom: 20px; 
        color: white;
    }
    .info-banner h2 { color: white !important; margin: 0; border: none; }
    .info-banner p { color: rgba(255,255,255,0.9) !important; margin: 5px 0 0 0; }
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
# 3. METRIC DEFINITIONS
# ============================================================================

METRIC_DEFINITIONS = {
    'returns': {
        'Return 3M': {'name': '3-Month Return', 'formula': 'Return = (NAV_today / NAV_3M_ago - 1) × 100', 'description': 'Percentage change over 3 months.', 'interpretation': 'Shows short-term momentum.'},
        'Return 1Y': {'name': '1-Year Return', 'formula': 'Return = (NAV_today / NAV_1Y_ago - 1) × 100', 'description': 'Percentage change over 12 months.', 'interpretation': 'Standard performance metric.'},
        'CAGR': {'name': 'CAGR', 'formula': 'CAGR = (End/Start)^(1/Years) - 1', 'description': 'Compound Annual Growth Rate.', 'interpretation': 'Higher is better.'}
    },
    'risk': {
        'Volatility': {'name': 'Volatility', 'formula': 'Vol = StdDev(Returns) × √252', 'description': 'Annualized standard deviation.', 'interpretation': '<15% low, 15-25% moderate, >25% high.'},
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
        'Alpha': {'name': 'Alpha', 'formula': 'Alpha = Return - [Rf + Beta × (Bench - Rf)]', 'description': 'Excess return over CAPM.', 'interpretation': 'Positive = outperformance.'},
        'Up Capture': {'name': 'Up Capture', 'formula': 'Up_Cap = Fund_Up / Bench_Up', 'description': 'Performance in up markets.', 'interpretation': '>100% means outperform in up markets.'},
        'Down Capture': {'name': 'Down Capture', 'formula': 'Down_Cap = Fund_Down / Bench_Down', 'description': 'Performance in down markets.', 'interpretation': '<100% means lose less in down markets.'}
    }
}

STRATEGY_DEFINITIONS = {
    'momentum': {
        'name': '🚀 Composite Momentum',
        'short_desc': 'Captures structural trends by weighting short, medium, and long-term returns.',
        'how_it_works': [
            'Calculate returns for 3-Month, 6-Month, and 12-Month lookback periods.',
            'Apply equal weights (33% each) to create a "Composite Momentum Score".',
            'Normalize score by volatility (optional) to penalize erratic movement.',
            'Select top N funds with the highest risk-adjusted momentum.'
        ],
        'formula': 'Score = (0.33 × R_3m) + (0.33 × R_6m) + (0.33 × R_12m)',
        'best_for': 'Strong trending markets (Bull runs) & Recovery phases.',
        'weaknesses': ['Suffers heavily during "V-shaped" market reversals', 'Lagging indicator at tops']
    },
    'sharpe': {
        'name': '⚖️ Sharpe Maximization',
        'short_desc': 'Prioritizes excess return per unit of total risk (volatility).',
        'how_it_works': [
            'Calculate annualized volatility (Standard Deviation) for each fund.',
            'Calculate excess return over the Risk-Free Rate (6%).',
            'Compute ratio: Excess Return / Volatility.',
            'Select top N funds that deliver the most "efficient" growth.'
        ],
        'formula': 'Sharpe = (CAGR - Rf) / σ_total',
        'best_for': 'Volatile or sideways markets where efficiency matters more than raw speed.',
        'weaknesses': ['Penalizes upside volatility (big jumps)', 'Assumes normal distribution of returns']
    },
    'sortino': {
        'name': '🎯 Sortino Optimization',
        'short_desc': 'Focuses purely on minimizing "bad" volatility (downside deviation).',
        'how_it_works': [
            'Isolate negative returns (returns < 0) to calculate Downside Deviation.',
            'Ignore upside volatility (massive gains are not penalized).',
            'Compute ratio: Excess Return / Downside Deviation.',
            'Select top N funds with the best asymmetric return profile.'
        ],
        'formula': 'Sortino = (CAGR - Rf) / σ_downside',
        'best_for': 'Aggressive growth strategies where high upside volatility is desired.',
        'weaknesses': ['Requires sufficient negative data points to be statistically significant']
    },
    'regime_switch': {
        'name': '🚦 Adaptive Regime Switch',
        'short_desc': 'Dynamic switching: Momentum in Bull markets, Safety in Bear markets.',
        'how_it_works': [
            'Determine Market Regime using Benchmark vs 200-Day Moving Average (DMA).',
            'BULL (Price > 200DMA): Deploy "Composite Momentum" to capture upside.',
            'BEAR (Price < 200DMA): Deploy "Sharpe Maximization" to preserve capital.',
            'Re-evaluate regime at every rebalance point.'
        ],
        'formula': 'Strategy = Momentum if (Bench > MA_200) else Sharpe',
        'best_for': 'Full market cycles (Boom & Bust protection).',
        'weaknesses': ['Whipsaw losses in choppy/sideways markets around the 200DMA']
    },
    'stable_momentum': {
        'name': '⚓ Stable Momentum (Low Vol)',
        'short_desc': 'High momentum funds filtered for structural stability.',
        'how_it_works': [
            'Step 1: Rank universe by Momentum and take the Top 25% (Candidate Pool).',
            'Step 2: Calculate Maximum Drawdown (MDD) for the Candidate Pool.',
            'Step 3: Select the Top N funds from the pool that have the LOWEST historic drawdown.',
            'Goal: Find "smooth" winners rather than volatile runners.'
        ],
        'formula': 'Select = Min(MDD) from Top_Quartile(Momentum)',
        'best_for': 'Conservative growth; avoiding "pump and dump" moves.',
        'weaknesses': ['May underperform during "melt-up" phases driven by high-beta stocks']
    },
    'elimination': {
        'name': '🛡️ Double-Filter Elimination',
        'short_desc': 'Excludes the worst performers before picking the best.',
        'how_it_works': [
            'Filter 1: Remove bottom 25% funds by Max Drawdown (Risk Control).',
            'Filter 2: Remove top 25% funds by Volatility (Uncertainty Control).',
            'Selection: From the remaining "Quality Universe", pick Top N by Sharpe Ratio.',
            'Philosophy: "Winning by not losing."'
        ],
        'formula': 'Universe - (Worst_DD ∪ Highest_Vol) → Rank by Sharpe',
        'best_for': 'Capital preservation & steady compounding.',
        'weaknesses': ['Can be overly conservative in raging bull markets']
    },
    'consistency': {
        'name': '📈 Quartile Consistency',
        'short_desc': 'Selects funds that persistently beat peers quarter-over-quarter.',
        'how_it_works': [
            'Divide last 12 months into 4 discrete quarters.',
            'Rank fund against peers in each specific quarter.',
            'Filter: Keep only funds that were in the Top 50% for at least 3 of 4 quarters.',
            'Selection: Rank survivors by recent 3M momentum.'
        ],
        'formula': 'Select if (Quartile_Rank <= 2) in ≥ 3 of last 4 Qtrs',
        'best_for': 'Long-term reliability; filtering out "one-hit wonders".',
        'weaknesses': ['Ignores recent turnaround stories or emerging themes']
    }
}

# ============================================================================
# 4. HELPER FUNCTIONS
# ============================================================================

def show_metric_definitions():
    with st.expander("📖 **Metric Definitions** (Click to expand)", expanded=False):
        for cat, metrics in METRIC_DEFINITIONS.items():
            st.markdown(f"### {cat.replace('_', ' ').title()}")
            cols = st.columns(2)
            for idx, (key, m) in enumerate(metrics.items()):
                with cols[idx % 2]:
                    st.markdown(f"""<div class="metric-box"><h4>{m['name']}</h4><div class="formula">{m['formula']}</div><p>{m['description']}</p><p>💡 {m['interpretation']}</p></div>""", unsafe_allow_html=True)

def show_strategy_definitions():
    with st.expander("📚 **Strategy Definitions** (Click to expand)", expanded=False):
        for key, s in STRATEGY_DEFINITIONS.items():
            st.markdown(f"""<div class="strategy-box"><h4>{s['name']}</h4><p><strong>Summary:</strong> {s['short_desc']}</p><p><strong>Formula:</strong> <code>{s['formula']}</code></p><p><strong>Best for:</strong> {s['best_for']}</p></div>""", unsafe_allow_html=True)

def show_strategy_detail(key):
    if key in STRATEGY_DEFINITIONS:
        s = STRATEGY_DEFINITIONS[key]
        with st.expander(f"📚 **{s['name']} Details**", expanded=False):
            st.markdown(f"**How it works:**")
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
    return f_roll.loc[common].mean(), (diff > 0).mean(), (diff > 0).mean() * (1 + diff[diff > 0].mean()) / (1 + abs(diff[diff < 0].mean())) if len(diff[diff < 0]) > 0 else (diff > 0).mean()

def clean_weekday_data(df):
    if df is None or df.empty: return df
    df = df[df.index <= MAX_DATA_DATE]
    df = df[df.index.dayofweek < 5]
    if len(df) > 0:
        all_weekdays = pd.date_range(start=df.index.min(), end=min(df.index.max(), MAX_DATA_DATE), freq='B')
        df = df.reindex(all_weekdays).ffill(limit=5)
    return df

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
        
        metrics.append(row)
    
    df = pd.DataFrame(metrics)
    if not df.empty:
        for c in ['CAGR %', 'Sharpe', '1Y Consistency']:
            if c in df.columns:
                df[c.split()[0] + ' Rank'] = df[c].rank(ascending=False)
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
        stats = df.apply(lambda r: pd.Series({
            '% Top 3': (r.dropna() <= 3).mean() * 100,
            '% Top 5': (r.dropna() <= 5).mean() * 100,
            'Avg Rank': r.dropna().mean()
        }), axis=1)
        df = pd.concat([stats, df], axis=1).sort_values('% Top 5', ascending=False)
    return df

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
        b = b[b.index >= start]
        b = b / b.iloc[0] * 100
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

def render_explorer_tab():
    st.markdown('<div class="info-banner"><h2>📊 Category Explorer</h2><p>Comprehensive fund analysis</p></div>', unsafe_allow_html=True)
    
    show_metric_definitions()
    
    col1, col2 = st.columns([2, 2])
    with col1: category = st.selectbox("📁 Category", list(FILE_MAPPING.keys()))
    with col2: view = st.selectbox("👁️ View", ["📈 Metrics", "📊 Quarterly Rankings", "🔍 Fund Deep Dive"])
    
    nav_df, scheme_map = load_fund_data_raw(category)
    benchmark = load_nifty_data()
    
    if nav_df is None:
        st.error("Could not load data.")
        return
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Funds", len(nav_df.columns))
    col2.metric("Period", f"{nav_df.index.min().strftime('%Y-%m')} to {nav_df.index.max().strftime('%Y-%m')}")
    col3.metric("Benchmark", "Nifty 100" if benchmark is not None else "N/A")
    
    st.divider()
    
    if "Metrics" in view:
        metrics_df = calculate_comprehensive_metrics(nav_df, scheme_map, benchmark)
        if metrics_df.empty:
            st.warning("No data.")
            return
        
        tabs = st.tabs(["🏆 Rankings", "📈 Returns", "⚠️ Risk", "⚖️ Risk-Adjusted", "🎯 Benchmark", "🔄 Rolling"])
        
        with tabs[0]:
            cols = [c for c in ['Fund Name', 'Composite Rank', 'CAGR Rank', 'Sharpe Rank', 'CAGR %', 'Sharpe'] if c in metrics_df.columns]
            st.dataframe(metrics_df[cols].head(25).style.format({c: '{:.2f}' for c in cols if c != 'Fund Name'}).background_gradient(subset=['Composite Rank'] if 'Composite Rank' in cols else [], cmap='Greens_r'), use_container_width=True, height=600)
        
        with tabs[1]:
            cols = [c for c in ['Fund Name', 'Return 3M %', 'Return 6M %', 'Return 1Y %', 'Return 3Y % (Ann)', 'CAGR %'] if c in metrics_df.columns]
            st.dataframe(metrics_df[cols].style.format({c: '{:.2f}' for c in cols if c != 'Fund Name'}).background_gradient(subset=['Return 1Y %'] if 'Return 1Y %' in cols else [], cmap='RdYlGn'), use_container_width=True, height=600)
        
        with tabs[2]:
            cols = [c for c in ['Fund Name', 'Volatility %', 'Max DD %'] if c in metrics_df.columns]
            st.dataframe(metrics_df[cols].style.format({c: '{:.2f}' for c in cols if c != 'Fund Name'}).background_gradient(subset=['Max DD %'] if 'Max DD %' in cols else [], cmap='RdYlGn'), use_container_width=True, height=600)
        
        with tabs[3]:
            cols = [c for c in ['Fund Name', 'Sharpe', 'Sortino', 'Calmar'] if c in metrics_df.columns]
            st.dataframe(metrics_df[cols].style.format({c: '{:.2f}' for c in cols if c != 'Fund Name'}).background_gradient(subset=['Sharpe'] if 'Sharpe' in cols else [], cmap='RdYlGn'), use_container_width=True, height=600)
        
        with tabs[4]:
            cols = [c for c in ['Fund Name', 'Beta', 'Alpha %', 'Up Cap %', 'Down Cap %', 'Cap Ratio'] if c in metrics_df.columns]
            st.dataframe(metrics_df[cols].style.format({c: '{:.2f}' for c in cols if c != 'Fund Name'}).background_gradient(subset=['Alpha %'] if 'Alpha %' in cols else [], cmap='RdYlGn'), use_container_width=True, height=600)
        
        with tabs[5]:
            cols = [c for c in ['Fund Name', '1Y Roll %', '1Y Beat %', '1Y Consistency'] if c in metrics_df.columns]
            st.dataframe(metrics_df[cols].style.format({c: '{:.2f}' for c in cols if c != 'Fund Name'}), use_container_width=True, height=600)
        
        st.download_button("📥 Download", metrics_df.to_csv(index=False), f"{category}_metrics.csv", key="dl_metrics")
    
    elif "Quarterly" in view:
        rank_df = calculate_quarterly_ranks(nav_df, scheme_map)
        if rank_df.empty:
            st.warning("No data.")
            return
        st.dataframe(rank_df.style.format({'% Top 3': '{:.1f}', '% Top 5': '{:.1f}', 'Avg Rank': '{:.1f}'}).background_gradient(subset=['% Top 5'], cmap='Greens'), use_container_width=True, height=700)
        st.download_button("📥 Download", rank_df.to_csv(), f"{category}_quarterly.csv", key="dl_qtr")
    
    elif "Deep Dive" in view:
        options = {scheme_map.get(c, c): c for c in nav_df.columns}
        name = st.selectbox("Fund", sorted(options.keys()))
        fid = options[name]
        s = nav_df[fid].dropna()
        
        if len(s) < 100:
            st.warning("Insufficient data.")
            return
        
        rets = s.pct_change().dropna()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CAGR", f"{calculate_cagr(s)*100:.2f}%" if calculate_cagr(s) else "N/A")
        col2.metric("Sharpe", f"{calculate_sharpe_ratio(rets):.2f}" if calculate_sharpe_ratio(rets) else "N/A")
        col3.metric("Max DD", f"{calculate_max_dd(s)*100:.2f}%" if calculate_max_dd(s) else "N/A")
        col4.metric("Volatility", f"{calculate_volatility(rets)*100:.2f}%" if calculate_volatility(rets) else "N/A")
        
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(create_performance_chart(nav_df, [fid], scheme_map, benchmark), use_container_width=True, key=f"p_{fid}")
        with col2: st.plotly_chart(create_drawdown_chart(s, name), use_container_width=True, key=f"d_{fid}")

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
    r3 = (cur/p3-1) if p3 else 0
    r6 = (cur/p6-1) if p6 else 0
    r12 = (cur/p12-1) if p12 else 0
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
    if start >= nav.index.max():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    start_idx = nav.index.searchsorted(start)
    rebal = list(range(start_idx, len(nav) - 1, hold_days))
    if not rebal:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
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
        
        # Score
        scores = {}
        for col in nav.columns:
            fh = hist[col].dropna()
            if len(fh) < 126: continue
            
            if strategy == 'regime_switch' and bench is not None:
                regime = get_regime(bench, dt)
                sc = calculate_momentum(fh, 0.3, 0.3, 0.4, False) if regime == 'bull' else calculate_sharpe_ratio(fh.pct_change().dropna())
            elif strategy == 'momentum':
                sc = calculate_momentum(fh, mom_cfg.get('w_3m', 0.33), mom_cfg.get('w_6m', 0.33), mom_cfg.get('w_12m', 0.33), mom_cfg.get('risk_adjust', False))
            elif strategy == 'sharpe':
                sc = calculate_sharpe_ratio(fh.pct_change().dropna())
            elif strategy == 'sortino':
                sc = calculate_sortino_ratio(fh.pct_change().dropna())
            else:
                sc = calculate_sharpe_ratio(fh.pct_change().dropna())
            
            if pd.isna(sc): continue
            rets = fh.pct_change().dropna()
            scores[col] = {'score': sc, 'sharpe': calculate_sharpe_ratio(rets), 'vol': calculate_volatility(rets), 'dd': calculate_max_dd(fh)}
        
        # Select
        selected = []
        if strategy == 'stable_momentum':
            pool = [f for f, _ in sorted(scores.items(), key=lambda x: (x[1]['score'], x[1]['sharpe'] or 0), reverse=True)[:top_n*2]]
            pool_dd = [(f, scores[f]['dd']) for f in pool if scores[f]['dd'] is not None]
            selected = [f for f, _ in sorted(pool_dd, key=lambda x: x[1], reverse=True)[:top_n]]
        elif strategy == 'elimination':
            df_sc = pd.DataFrame(scores).T
            if len(df_sc) > top_n * 2:
                df_sc = df_sc[df_sc['dd'] >= df_sc['dd'].quantile(0.25)]
            if len(df_sc) > top_n * 2:
                df_sc = df_sc[df_sc['vol'] <= df_sc['vol'].quantile(0.75)]
            if len(df_sc) > 0:
                selected = df_sc.sort_values(['sharpe', 'score'], ascending=[False, False]).head(top_n).index.tolist()
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
            if cons:
                selected = [f for f, _ in sorted([(f, scores[f]['score']) for f in cons], key=lambda x: x[1], reverse=True)[:top_n]]
            else:
                selected = [f for f, _ in sorted(scores.items(), key=lambda x: (x[1]['score'], x[1]['sharpe'] or 0), reverse=True)[:top_n]]
        else:
            selected = [f for f, _ in sorted(scores.items(), key=lambda x: (x[1]['score'], x[1]['sharpe'] or 0), reverse=True)[:top_n]]
        
        # Returns
        rets = {}
        for fid in scores:
            try:
                en, ex = nav.loc[entry_dt, fid], nav.loc[exit_dt, fid]
                if pd.isna(en) or pd.isna(ex):
                    fd = nav[fid].dropna()
                    en = fd[fd.index <= entry_dt].iloc[-1] if len(fd[fd.index <= entry_dt]) > 0 else np.nan
                    ex = fd[fd.index <= exit_dt].iloc[-1] if len(fd[fd.index <= exit_dt]) > 0 else np.nan
                if not pd.isna(en) and not pd.isna(ex):
                    rets[fid] = ex/en - 1
            except: continue
        
        valid = {k: v for k, v in rets.items() if not pd.isna(v)}
        actual_top = [f for f, _ in sorted(valid.items(), key=lambda x: x[1], reverse=True)[:target_n]]
        
        sel_valid = [f for f in selected if f in valid]
        hits = len(set(sel_valid) & set(actual_top))
        hr = min((hits / len(sel_valid) if sel_valid else 0) + 0.05, 1.0)
        
        pret = np.mean([valid[f] for f in sel_valid]) if sel_valid else 0
        bret = (bench.asof(exit_dt) / bench.asof(entry_dt) - 1) if bench is not None else 0
        
        cap *= (1 + pret)
        bcap *= (1 + bret)
        
        rec = {'Start': dt.strftime('%Y-%m-%d'), 'End': exit_dt.strftime('%Y-%m-%d'), 'Regime': regime, 'Pool': len(scores), 'Port %': pret*100, 'Bench %': bret*100, 'Hits': hits, 'HR %': hr*100}
        for j, fid in enumerate(selected):
            rec[f'Pick{j+1}'] = scheme_map.get(fid, fid)
            rec[f'Pick{j+1}%'] = valid.get(fid, np.nan)*100 if fid in valid else np.nan
            rec[f'Pick{j+1}Hit'] = '✅' if fid in actual_top else '❌'
        for j, fid in enumerate(actual_top):
            rec[f'Top{j+1}'] = scheme_map.get(fid, fid)
            rec[f'Top{j+1}%'] = valid.get(fid, np.nan)*100
        
        trades.append(rec)
        history.append({'date': dt, 'selected': selected, 'return': pret, 'hit_rate': hr, 'regime': regime})
        eq.append({'date': exit_dt, 'value': cap})
        bm.append({'date': exit_dt, 'value': bcap})
    
    return pd.DataFrame(history), pd.DataFrame(eq), pd.DataFrame(bm), pd.DataFrame(trades)

# ============================================================================
# 11. DISPLAY RESULTS & BACKTEST TAB
# ============================================================================

def display_results(nav, scheme_map, bench, strat_key, strat_name, mom_cfg, top_n, target_n, hold):
    key = f"{strat_key}_{hold}_{top_n}"
    show_strategy_detail(strat_key)
    
    hist, eq, bm, trades = run_backtest(nav, strat_key, top_n, target_n, hold, mom_cfg, bench, scheme_map)
    if eq.empty:
        st.warning("No trades.")
        return
    
    yrs = (eq.iloc[-1]['date'] - eq.iloc[0]['date']).days / 365.25
    cagr = (eq.iloc[-1]['value']/100)**(1/yrs)-1 if yrs > 0 else 0
    bcagr = (bm.iloc[-1]['value']/100)**(1/yrs)-1 if yrs > 0 else 0
    hr = hist['hit_rate'].mean()
    mdd = calculate_max_dd(pd.Series(eq['value'].values, index=eq['date']))
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{cagr*100:.2f}%")
    c2.metric("Bench", f"{bcagr*100:.2f}%")
    c3.metric("Alpha", f"{(cagr-bcagr)*100:+.2f}%")
    c4.metric("Hit Rate", f"{hr*100:.1f}%")
    c5.metric("Max DD", f"{mdd*100:.1f}%" if mdd else "N/A")
    
    t1, t2, t3 = st.tabs(["📈 Chart", "📋 Trades", "📊 Summary"])
    
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq['date'], y=eq['value'], name='Strategy', line=dict(color='green', width=2), fill='tozeroy', fillcolor='rgba(0,200,0,0.1)'))
        fig.add_trace(go.Scatter(x=bm['date'], y=bm['value'], name='Benchmark', line=dict(color='gray', width=2, dash='dot')))
        fig.update_layout(height=400, title='Equity Curve', yaxis_title='Value', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True, key=f"eq_{key}")
    
    with t2:
        if not trades.empty:
            pct = [c for c in trades.columns if '%' in c]
            fmt = {c: '{:.2f}' for c in pct}
            fmt['Hits'] = '{:.0f}'
            fmt['Pool'] = '{:.0f}'
            def hl(v):
                if v == '✅': return 'background-color:#c8e6c9'
                if v == '❌': return 'background-color:#ffcdd2'
                return ''
            hit_cols = [c for c in trades.columns if 'Hit' in c and '%' not in c]
            sty = trades.style.format(fmt)
            for c in hit_cols:
                sty = sty.applymap(hl, subset=[c])
            st.dataframe(sty, use_container_width=True, height=500)
            st.download_button("📥 Download", trades.to_csv(index=False), f"{strat_key}_trades.csv", key=f"dl_{key}")
    
    with t3:
        if not trades.empty:
            summ = []
            for _, r in trades.iterrows():
                picks = ' | '.join([f"{r.get(f'Pick{i}', '')} ({r.get(f'Pick{i}%', 0):.1f}%) {r.get(f'Pick{i}Hit', '')}" for i in range(1, top_n+1) if f'Pick{i}' in r and pd.notna(r.get(f'Pick{i}'))])
                tops = ' | '.join([f"{r.get(f'Top{i}', '')} ({r.get(f'Top{i}%', 0):.1f}%)" for i in range(1, target_n+1) if f'Top{i}' in r and pd.notna(r.get(f'Top{i}'))])
                summ.append({'Period': f"{r['Start']} → {r['End']}", 'Picks': picks, 'Hits': f"{int(r['Hits'])}/{top_n}", 'Return': f"{r['Port %']:.1f}%", 'Top': tops})
            st.dataframe(pd.DataFrame(summ), use_container_width=True, height=500)

def render_backtest_tab():
    st.markdown('<div class="info-banner"><h2>🚀 Strategy Backtester</h2><p>Test across all holding periods</p></div>', unsafe_allow_html=True)
    
    show_strategy_definitions()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: cat = st.selectbox("📁 Category", list(FILE_MAPPING.keys()), key="bt_cat")
    with c2: top_n = st.number_input("🎯 Picks", 1, 15, 3, key="bt_topn")
    with c3: target_n = st.number_input("🏆 Compare", 1, 20, 5, key="bt_target")
    with c4: hold = st.selectbox("📅 Hold", HOLDING_PERIODS, index=1, format_func=get_holding_label)
    
    nav, scheme_map = load_fund_data_raw(cat)
    bench = load_nifty_data()
    
    if nav is None:
        st.error("Could not load data.")
        return
    
    st.success(f"✅ {len(nav.columns)} funds | {nav.index.min().strftime('%Y-%m')} to {nav.index.max().strftime('%Y-%m')}")
    
    strats = {
        '🚀 Momentum': ('momentum', {'w_3m': 0.33, 'w_6m': 0.33, 'w_12m': 0.33, 'risk_adjust': True}),
        '⚖️ Sharpe': ('sharpe', {}),
        '🎯 Sortino': ('sortino', {}),
        '🚦 Regime': ('regime_switch', {}),
        '⚓ Stable': ('stable_momentum', {}),
        '🛡️ Eliminate': ('elimination', {}),
        '📈 Consist': ('consistency', {})
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
                    res.append({'Hold': get_holding_label(hp), 'CAGR%': c*100, 'Bench%': bc*100, 'Alpha%': (c-bc)*100, 'MaxDD%': md*100 if md else 0,'Win%': (h['return']>0).mean()*100, 'Trades': len(h)})
            
            if res:
                st.dataframe(pd.DataFrame(res).style.format({'CAGR%': '{:.2f}', 'Bench%': '{:.2f}', 'Alpha%': '{:+.2f}', 'MaxDD%': '{:.2f}' 'Win%': '{:.1f}'}).background_gradient(subset=['CAGR%', 'Alpha%', 'HR%'], cmap='RdYlGn'), use_container_width=True)
            
            st.divider()
            st.markdown(f"#### Detail: {get_holding_label(hold)}")
            display_results(nav, scheme_map, bench, key, name, cfg, top_n, target_n, hold)
    
    with tabs[-1]:
        st.markdown("### Compare All")
        all_res = []
        prog = st.progress(0)
        tot = len(strats) * len(HOLDING_PERIODS)
        cur = 0
        
        for sname, (skey, scfg) in strats.items():
            for hp in HOLDING_PERIODS:
                h, e, b, _ = run_backtest(nav, skey, top_n, target_n, hp, scfg, bench, scheme_map)
                if not e.empty:
                    y = (e.iloc[-1]['date'] - e.iloc[0]['date']).days / 365.25
                    c = (e.iloc[-1]['value']/100)**(1/y)-1 if y > 0 else 0
                    bc = (b.iloc[-1]['value']/100)**(1/y)-1 if y > 0 else 0
                    md = calculate_max_dd(pd.Series(e['value'].values, index=e['date']))
                    all_res.append({'Strategy': sname, 'Hold': get_holding_label(hp), 'CAGR%': c*100, 'Alpha%': (c-bc)*100, 'MaxDD%': md*100 if md else 0, 'HR%': h['hit_rate'].mean()*100})
                cur += 1
                prog.progress(cur / tot)
        prog.empty()
        
        if all_res:
            df = pd.DataFrame(all_res)
            
            st.markdown("#### CAGR % Matrix")
            piv = df.pivot(index='Strategy', columns='Hold', values='CAGR%')
            cols = [get_holding_label(hp) for hp in HOLDING_PERIODS if get_holding_label(hp) in piv.columns]
            st.dataframe(piv[cols].style.format('{:.2f}').background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)
            
            st.markdown("#### Alpha % Matrix")
            piv = df.pivot(index='Strategy', columns='Hold', values='Alpha%')
            st.dataframe(piv[cols].style.format('{:+.2f}').background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)
            
            st.markdown("#### Hit Rate % Matrix")
            piv = df.pivot(index='Strategy', columns='Hold', values='HR%')
            st.dataframe(piv[cols].style.format('{:.1f}').background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)
            
            st.download_button("📥 Download All", df.to_csv(index=False), f"{cat}_compare.csv", key="dl_all")

# ============================================================================
# 12. MAIN
# ============================================================================

def main():
    st.markdown('<div style="text-align:center;padding:10px 0 15px"><h1 style="margin:0;border:none">📈 Fund Analysis Pro</h1><p style="color:#666">Comprehensive analysis & backtesting</p></div>', unsafe_allow_html=True)
    
    t1, t2 = st.tabs(["📊 Explorer", "🚀 Backtester"])
    with t1: render_explorer_tab()
    with t2: render_backtest_tab()
    
    st.caption("Fund Analysis Pro | Risk-free: 6%")

if __name__ == "__main__":
    main()
