import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import linregress

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="Advanced Fund Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
DEFAULT_HOLDING = 504  # <--- CHANGED TO 504 AS REQUESTED
DEFAULT_TOP_N = 2
DEFAULT_TARGET_N = 4
RISK_FREE_RATE = 0.06
TRADING_DAYS_YEAR = 252
DAILY_RISK_FREE_RATE = (1 + RISK_FREE_RATE) ** (1/TRADING_DAYS_YEAR) - 1
DATA_DIR = "data"
MAX_DATA_DATE = pd.Timestamp('2025-12-05')

# --- File Mapping ---
FILE_MAPPING = {
    "Large Cap": "largecap_merged.xlsx",
    "Mid Cap": "midcap.xlsx",
    "Small Cap": "smallcap.xlsx",
    "Large & Mid Cap": "large_and_midcap_fund.xlsx",
    "Multi Cap": "MULTICAP.xlsx",
    "International": "international_merged.xlsx"
}

# ============================================================================
# 2. HELPER FUNCTIONS
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

# --- METRICS ---

def calculate_sharpe_ratio(returns):
    if len(returns) < 10 or returns.std() == 0: return np.nan
    excess_returns = returns - DAILY_RISK_FREE_RATE
    return (excess_returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS_YEAR)

def calculate_sortino_ratio(returns):
    if len(returns) < 10: return np.nan
    downside = returns[returns < 0]
    if len(downside) == 0: return np.nan
    downside_std = downside.std()
    if downside_std == 0: return np.nan
    mean_return = (returns - DAILY_RISK_FREE_RATE).mean()
    return (mean_return / downside_std) * np.sqrt(TRADING_DAYS_YEAR)

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
    drawdowns = drawdowns.fillna(0)
    ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
    if ulcer_index == 0: return np.nan
    start_val, end_val = series.iloc[0], series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0 or start_val <= 0: return np.nan
    cagr = (end_val / start_val) ** (1 / years) - 1
    return (cagr - RISK_FREE_RATE) / ulcer_index

def calculate_capture_score(fund_rets, bench_rets):
    common_idx = fund_rets.index.intersection(bench_rets.index)
    if len(common_idx) < 30: return np.nan
    f = fund_rets.loc[common_idx]
    b = bench_rets.loc[common_idx]
    up_market = b[b > 0]
    if up_market.empty or up_market.mean() == 0: return np.nan
    up_cap = f.loc[up_market.index].mean() / up_market.mean()
    down_market = b[b < 0]
    if down_market.empty or down_market.mean() == 0: return np.nan
    down_cap = f.loc[down_market.index].mean() / down_market.mean()
    if down_cap <= 0: return up_cap * 2 
    return up_cap / down_cap

def calculate_volatility(returns):
    if len(returns) < 10: return np.nan
    return returns.std() * np.sqrt(TRADING_DAYS_YEAR)

def calculate_max_dd(series):
    if len(series) < 10 or series.isna().all(): return np.nan
    comp_ret = (1 + series.pct_change().fillna(0)).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret / peak) - 1
    return dd.min()

def calculate_vol_adj_return(series):
    if series is None or len(series) < 63: return np.nan
    cleaned = series.dropna()
    if len(cleaned) < 63: return np.nan
    returns = cleaned.pct_change().dropna()
    total_days = (cleaned.index[-1] - cleaned.index[0]).days
    if total_days <= 0: return np.nan
    years = total_days / 365.25
    if years <= 0: return np.nan
    start_val, end_val = cleaned.iloc[0], cleaned.iloc[-1]
    if start_val <= 0 or end_val <= 0: return np.nan
    annual_return = (end_val / start_val) ** (1 / years) - 1
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS_YEAR)
    if annual_vol == 0: return np.nan
    return annual_return / annual_vol

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
        
        # Calculate volatility
        vol = hist_vol_data.pct_change().dropna().std() * np.sqrt(TRADING_DAYS_YEAR)
        
        # --- FIX FOR INFINITY: Volatility Clamp ---
        # If volatility is extremely low (e.g., < 0.000001), treat it as invalid or clamp it
        if vol < 1e-6: return np.nan
        
        return raw_score / vol
    return raw_score

def calculate_information_ratio(fund_returns, bench_returns):
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 30: return np.nan
    f_ret = fund_returns.loc[common_idx]
    b_ret = bench_returns.loc[common_idx]
    active_return = f_ret - b_ret
    tracking_error = active_return.std(ddof=1) * np.sqrt(TRADING_DAYS_YEAR)
    if tracking_error == 0: return np.nan
    return (active_return.mean() * TRADING_DAYS_YEAR) / tracking_error

def calculate_rolling_win_rate(fund_series, bench_series, window=252):
    if len(fund_series) < window + 10: return np.nan
    common_idx = fund_series.index.intersection(bench_series.index)
    if len(common_idx) < window + 10: return np.nan
    f_s = fund_series.loc[common_idx]
    b_s = bench_series.loc[common_idx]
    f_roll = f_s.pct_change(window).dropna()
    b_roll = b_s.pct_change(window).dropna()
    common_roll = f_roll.index.intersection(b_roll.index)
    if len(common_roll) < 10: return np.nan
    wins = (f_roll.loc[common_roll] > b_roll.loc[common_roll]).sum()
    return wins / len(common_roll)

# --- NEW QUANT FUNCTIONS (FIXED) ---

def calculate_residual_momentum(series, benchmark_series):
    """
    Calculates Idiosyncratic Momentum.
    FIXED: Handles constant data and zero residuals to avoid 0 or Inf scores.
    """
    # 1. Clean Data
    s_ret = series.pct_change().dropna()
    b_ret = benchmark_series.pct_change().dropna()
    
    # 2. Strict Alignment
    aligned_data = pd.concat([s_ret, b_ret], axis=1, join='inner').dropna()
    
    # 3. Check history
    if len(aligned_data) < 60: return np.nan
    
    y = aligned_data.iloc[:, 0] # Fund
    x = aligned_data.iloc[:, 1] # Bench
    
    # 3.5 Check variance to prevent regression errors
    if x.std() < 1e-6 or y.std() < 1e-6: return np.nan
    
    # 4. Regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    if not np.isfinite(slope) or not np.isfinite(intercept): return np.nan

    # 5. Residuals
    expected_return = (x * slope) + intercept
    residuals = y - expected_return
    
    # 6. Score
    res_std = residuals.std()
    
    # --- FIX FOR ZERO SCORE ---
    # If res_std is 0 (perfect fit) or extremely close, return NaN
    if res_std < 1e-6 or pd.isna(res_std): return np.nan 
    
    score = residuals.mean() / res_std
    
    if not np.isfinite(score): return np.nan
    
    return score

def get_market_regime(benchmark_series, current_date, window=200):
    """Determines if Market is Bull (Price > 200 DMA) or Bear."""
    subset = benchmark_series[benchmark_series.index <= current_date]
    if len(subset) < window: return 'neutral'
    current_price = subset.iloc[-1]
    dma = subset.iloc[-window:].mean()
    return 'bull' if current_price > dma else 'bear'

# ============================================================================
# 3. DATA LOADING
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
# 4. EXPLORER LOGIC
# ============================================================================

def calculate_analytics_metrics(nav_df, scheme_map, benchmark_series=None):
    if nav_df is None or nav_df.empty or benchmark_series is None: return pd.DataFrame()
    
    bench_rolling_1y = benchmark_series.pct_change(252).dropna()
    bench_daily = benchmark_series.pct_change().dropna()
    
    b_comp = (1 + bench_daily).cumprod()
    b_peak = b_comp.expanding(min_periods=1).max()
    bench_dd = (b_comp / b_peak) - 1
    
    metrics = []
    
    for col in nav_df.columns:
        series = nav_df[col].dropna()
        if len(series) < 260: continue
        
        fund_roll = series.pct_change(252).dropna()
        common_roll = fund_roll.index.intersection(bench_rolling_1y.index)
        
        if len(common_roll) < 10: continue
        
        f_r = fund_roll.loc[common_roll]
        b_r = bench_rolling_1y.loc[common_roll]
        
        percent_rolling_avg = f_r.mean() * 100
        
        diff = f_r - b_r
        beats = diff[diff > 0]
        perc_times_beated = (len(beats) / len(diff)) * 100
        beated_by_percent_avg = beats.mean() * 100 if not beats.empty else 0
        
        daily_ret = series.pct_change().dropna()
        comp = (1 + daily_ret).cumprod()
        peak = comp.expanding(min_periods=1).max()
        fund_dd = (comp / peak) - 1
        max_drawdown = fund_dd.min() * 100
        
        metrics.append({
            'fund_name': scheme_map.get(col, col),
            'percent_rolling_avg': percent_rolling_avg,
            'perc_times_beated': perc_times_beated,
            'beated_by_percent_avg': beated_by_percent_avg,
            'MaxDrawdown': max_drawdown
        })
        
    df = pd.DataFrame(metrics)
    if not df.empty:
        df['return_rank'] = df['percent_rolling_avg'].rank(ascending=False, method='min')
        return df.sort_values('return_rank')
    return df

def calculate_quarterly_ranks(nav_df, scheme_map):
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
            history_data[q_date.strftime('%Y-%m-%d')] = ranks
        except: continue
    rank_df = pd.DataFrame(history_data)
    rank_df.index = rank_df.index.map(lambda x: scheme_map.get(x, x))
    rank_df = rank_df.dropna(how='all')
    
    if not rank_df.empty:
        def calc_top5_perc(row):
            valid = row.dropna()
            if len(valid) == 0: return 0.0
            return (valid <= 5).sum() / len(valid) * 100
        
        rank_df['% Time in Top 5'] = rank_df.apply(calc_top5_perc, axis=1)
        rank_df = rank_df.sort_values('% Time in Top 5', ascending=False)
        
    return rank_df

def render_explorer_tab():
    st.header("📂 Category Explorer")
    col1, col2 = st.columns([1, 3])
    with col1:
        cat = st.selectbox("Select Category", list(FILE_MAPPING.keys()))
        view = st.radio("View Mode", ["Performance Metrics", "Quarterly Ranking History"])
    with col2:
        with st.spinner(f"Processing {cat}..."):
            nav, maps = load_fund_data_raw(cat)
            if nav is None: st.error("Data not found."); return
            if view == "Performance Metrics":
                nifty = load_nifty_data()
                df = calculate_analytics_metrics(nav, maps, nifty)
                st.dataframe(df, use_container_width=True, height=600)
            else:
                df = calculate_quarterly_ranks(nav, maps)
                st.dataframe(df.style.format({'% Time in Top 5': '{:.1f}%'}, na_rep=""), use_container_width=True, height=600)

# ============================================================================
# 5. BACKTESTER LOGIC (UPDATED WITH TARGET N HIT RATE)
# ============================================================================

def get_lookback_data(nav, analysis_date):
    max_days = 400 
    start_date = analysis_date - pd.Timedelta(days=max_days)
    return nav[(nav.index >= start_date) & (nav.index < analysis_date)]

def run_backtest(nav, strategy_type, top_n, target_n, holding_days, custom_weights, momentum_config, benchmark_series, ensemble_weights=None):
    
    # --- 1. DYNAMIC TIMELINE HANDLING ---
    start_date = nav.index.min() + pd.Timedelta(days=370)
    if start_date >= nav.index.max(): 
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    start_idx = nav.index.searchsorted(start_date)
    # Loop until end of data - 1 day to allow at least 1 day trade
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

        # --- STRATEGY SELECTION LOGIC ---

        # A. RESIDUAL MOMENTUM
        if strategy_type == 'residual_momentum':
            if benchmark_series is not None:
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) > 126:
                        # Pass raw price series slice of benchmark
                        b_slice_price = get_lookback_data(benchmark_series.to_frame(), date)['nav']
                        val = calculate_residual_momentum(s, b_slice_price)
                        # Updated check: exclude NaN, Inf, and Zero
                        if not pd.isna(val) and np.isfinite(val) and abs(val) > 1e-9:
                            scores[col] = val
                selected = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # B. STABLE MOMENTUM
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

        # C. REGIME SWITCH
        elif strategy_type == 'regime_switch':
            if benchmark_series is not None:
                regime = get_market_regime(benchmark_series, date)
                regime_status = regime
                
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) < 126: continue
                    val = np.nan
                    
                    if regime == 'bull':
                        # Use Risk Adjust = True to match logic but we fixed the math inside function
                        val = calculate_flexible_momentum(s, 0.3, 0.3, 0.4, True)
                    else:
                        val = calculate_sharpe_ratio(s.pct_change().dropna())
                    
                    # Updated check: exclude NaN, Inf, and Zero
                    if not pd.isna(val) and np.isfinite(val) and abs(val) > 1e-9:
                         scores[col] =
