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
                         scores[col] = val
                selected = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # D. ENSEMBLE
        elif strategy_type == 'ensemble':
            fund_ranks = pd.DataFrame(index=nav.columns)
            
            def dict_to_norm_rank(s_dict):
                if not s_dict: return pd.Series(0, index=fund_ranks.index)
                s = pd.Series(s_dict)
                s = s.reindex(fund_ranks.index)
                return s.rank(pct=True, ascending=True).fillna(0)

            if ensemble_weights.get('momentum', 0) > 0:
                temp = {}
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) > 70: temp[col] = calculate_flexible_momentum(s, momentum_config['w_3m'], momentum_config['w_6m'], momentum_config['w_12m'], momentum_config['risk_adjust'])
                fund_ranks['momentum'] = dict_to_norm_rank(temp)

            if ensemble_weights.get('sharpe', 0) > 0:
                temp = {}
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) > 126: temp[col] = calculate_sharpe_ratio(s.pct_change().dropna())
                fund_ranks['sharpe'] = dict_to_norm_rank(temp)
                
            if ensemble_weights.get('martin', 0) > 0:
                temp = {}
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) > 126: temp[col] = calculate_martin_ratio(s)
                fund_ranks['martin'] = dict_to_norm_rank(temp)

            if ensemble_weights.get('omega', 0) > 0:
                temp = {}
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) > 126: temp[col] = calculate_omega_ratio(s.pct_change().dropna())
                fund_ranks['omega'] = dict_to_norm_rank(temp)
                
            if ensemble_weights.get('capture', 0) > 0:
                temp = {}
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) > 126 and bench_rets is not None: temp[col] = calculate_capture_score(s.pct_change().dropna(), bench_rets)
                fund_ranks['capture'] = dict_to_norm_rank(temp)
                
            if ensemble_weights.get('sortino', 0) > 0:
                temp = {}
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) > 126: temp[col] = calculate_sortino_ratio(s.pct_change().dropna())
                fund_ranks['sortino'] = dict_to_norm_rank(temp)
            
            if ensemble_weights.get('var', 0) > 0:
                temp = {}
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) > 126: temp[col] = calculate_vol_adj_return(s)
                fund_ranks['var'] = dict_to_norm_rank(temp)

            if ensemble_weights.get('consistent_alpha', 0) > 0:
                temp = {}
                for col in nav.columns:
                    s = hist[col].dropna()
                    if len(s) > 252:
                         wr = calculate_rolling_win_rate(s, benchmark_series.loc[:date]) if benchmark_series is not None else 0.5
                         rets = s.pct_change().dropna()
                         ir = calculate_information_ratio(rets, bench_rets) if bench_rets is not None else 0
                         temp[col] = (wr * 0.6) + (ir * 0.4) if not pd.isna(ir) else 0
                fund_ranks['consistent_alpha'] = dict_to_norm_rank(temp)

            final_score = pd.Series(0.0, index=fund_ranks.index)
            if sum(ensemble_weights.values()) > 0:
                for k, w in ensemble_weights.items():
                    if k in fund_ranks.columns:
                        final_score += fund_ranks[k] * w
            scores = final_score.to_dict()
            selected = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # E. CONSISTENT ALPHA
        elif strategy_type == 'consistent_alpha':
            temp_rows = []
            for col in nav.columns:
                s = hist[col].dropna()
                if len(s) < 252: continue
                wr = calculate_rolling_win_rate(s, benchmark_series.loc[:date]) if benchmark_series is not None else 0.5
                rets = s.pct_change().dropna()
                ir = calculate_information_ratio(rets, bench_rets) if bench_rets is not None else 0
                temp_rows.append({'id': col, 'win_rate': wr, 'ir': ir if not pd.isna(ir) else 0})
            if temp_rows:
                df_sc = pd.DataFrame(temp_rows).set_index('id')
                final_score = (df_sc['win_rate'].rank(pct=True) * 0.6) + (df_sc['ir'].rank(pct=True) * 0.4)
                scores = final_score.to_dict()
                selected = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # F. SINGLE METRICS
        elif strategy_type in ['martin', 'omega', 'capture_ratio', 'info_ratio', 'momentum', 'sharpe', 'sortino', 'var']:
             for col in nav.columns:
                s = hist[col].dropna()
                if len(s) < 126: continue
                val = np.nan
                rets = s.pct_change().dropna()
                
                if strategy_type == 'martin': val = calculate_martin_ratio(s)
                elif strategy_type == 'omega': val = calculate_omega_ratio(rets)
                elif strategy_type == 'capture_ratio': val = calculate_capture_score(rets, bench_rets) if bench_rets is not None else 0
                elif strategy_type == 'info_ratio': val = calculate_information_ratio(rets, bench_rets) if bench_rets is not None else 0
                elif strategy_type == 'momentum': val = calculate_flexible_momentum(s, momentum_config['w_3m'], momentum_config['w_6m'], momentum_config['w_12m'], momentum_config['risk_adjust'])
                elif strategy_type == 'sharpe': val = calculate_sharpe_ratio(rets)
                elif strategy_type == 'sortino': val = calculate_sortino_ratio(rets)
                elif strategy_type == 'var': val = calculate_vol_adj_return(s)
                
                if not pd.isna(val): scores[col] = val
             selected = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # G. CUSTOM WEIGHTS
        elif strategy_type == 'custom':
             temp_data = []
             for col in nav.columns:
                s = hist[col].dropna()
                if len(s) < 126: continue
                rets = s.pct_change().dropna()
                row = {'id': col}
                if custom_weights.get('sharpe',0): row['sharpe'] = calculate_sharpe_ratio(rets)
                if custom_weights.get('momentum',0): row['momentum'] = calculate_flexible_momentum(s, 0.33, 0.33, 0.33, True)
                if custom_weights.get('sortino',0): row['sortino'] = calculate_sortino_ratio(rets)
                if custom_weights.get('volatility',0): row['volatility'] = calculate_volatility(rets)
                if custom_weights.get('maxdd',0): row['maxdd'] = calculate_max_dd(s)
                if custom_weights.get('calmar',0): row['calmar'] = calculate_calmar_ratio(s, date)
                if custom_weights.get('info_ratio',0): row['info_ratio'] = calculate_information_ratio(rets, bench_rets) if bench_rets is not None else 0
                if custom_weights.get('beta',0) or custom_weights.get('alpha',0) or custom_weights.get('treynor',0):
                    if bench_rets is not None:
                        b, a, t = calculate_beta_alpha_treynor(rets, bench_rets)
                        row['beta'] = b
                        row['alpha'] = a
                        row['treynor'] = t

                temp_data.append(row)
             
             if temp_data:
                 df_met = pd.DataFrame(temp_data).set_index('id')
                 f_sc = pd.Series(0.0, index=df_met.index)
                 for k, w in custom_weights.items():
                     if k in df_met.columns:
                         if k in ['volatility', 'maxdd', 'beta']: rank = df_met[k].abs().rank(pct=True, ascending=False)
                         else: rank = df_met[k].rank(pct=True, ascending=True)
                         f_sc = f_sc.add(rank*w, fill_value=0)
                 scores = f_sc.to_dict()
                 selected = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # --- EXECUTION ---
        entry = i + 1
        exit_i = min(i + 1 + holding_days, len(nav)-1)
        
        # Benchmark Return
        b_ret = 0.0
        if benchmark_series is not None:
             try: b_ret = (benchmark_series.asof(nav.index[exit_i]) / benchmark_series.asof(nav.index[entry])) - 1
             except: pass
        
        # Portfolio Return
        port_ret = 0.0
        hit_rate = 0.0
        
        if selected:
            # If strategy found funds, calculate their return
            period_ret_all_funds = (nav.iloc[exit_i] / nav.iloc[entry]) - 1
            port_ret = period_ret_all_funds[selected].mean()
            
            # --- UPDATED HIT RATE LOGIC (MATCH AGAINST TOP TARGET N) ---
            actual_top_target_funds = period_ret_all_funds.dropna().nlargest(target_n).index.tolist()
            matches = len(set(selected).intersection(set(actual_top_target_funds)))
            hit_rate = matches / top_n if top_n > 0 else 0
        else:
            port_ret = 0.0
            hit_rate = 0.0
        
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

def generate_snapshot_table(nav_wide, analysis_date, holding_days, strategy_type, names_map, custom_weights, momentum_config, benchmark_series, ensemble_weights=None):
    try: idx = nav_wide.index.get_loc(analysis_date)
    except: return pd.DataFrame()
    
    entry_idx = idx + 1
    exit_idx = entry_idx + holding_days
    has_future = (exit_idx < len(nav_wide))
    hist_data = get_lookback_data(nav_wide, analysis_date)
    temp = []
    
    bench_rets = None
    if benchmark_series is not None:
        try:
            b_slice = get_lookback_data(benchmark_series.to_frame(), analysis_date)
            bench_rets = b_slice['nav'].pct_change().dropna()
        except: pass

    for col in nav_wide.columns:
        s = hist_data[col].dropna()
        if len(s) < 126: continue
        row = {'id': col, 'name': names_map.get(col, col)}
        val = np.nan
        rets = s.pct_change().dropna()
        
        if strategy_type == 'consistent_alpha':
            wr = calculate_rolling_win_rate(s, benchmark_series.loc[:analysis_date]) if benchmark_series is not None else 0.5
            ir = calculate_information_ratio(rets, bench_rets) if bench_rets is not None else 0
            val = (wr * 0.6) + (ir * 0.4) 
        
        elif strategy_type == 'omega':
            val = calculate_omega_ratio(rets)
            
        elif strategy_type == 'capture_ratio':
            val = calculate_capture_score(rets, bench_rets) if bench_rets is not None else 0
            
        elif strategy_type == 'martin':
            val = calculate_martin_ratio(s)
            
        elif strategy_type == 'info_ratio':
            val = calculate_information_ratio(rets, bench_rets) if bench_rets is not None else 0

        elif strategy_type == 'momentum':
            val = calculate_flexible_momentum(s, momentum_config['w_3m'], momentum_config['w_6m'], momentum_config['w_12m'], momentum_config['risk_adjust'])
        elif strategy_type == 'var': val = calculate_vol_adj_return(s)
        elif strategy_type == 'sharpe': val = calculate_sharpe_ratio(rets)
        elif strategy_type == 'sortino': val = calculate_sortino_ratio(rets)
        elif strategy_type == 'custom':
            score = 0
            if custom_weights.get('sharpe',0): score += calculate_sharpe_ratio(rets) * custom_weights['sharpe']
            val = score
        elif strategy_type == 'ensemble':
            val = 0 
        
        elif strategy_type == 'stable_momentum':
             if len(s) > 200:
                ret_12m = (s.iloc[-1] / s.iloc[0]) - 1
                vol_12m = s.pct_change().std() * np.sqrt(252)
                val = ret_12m / vol_12m if vol_12m > 0 else 0
        
        elif strategy_type == 'residual_momentum':
             if benchmark_series is not None:
                 b_slice_price = get_lookback_data(benchmark_series.to_frame(), analysis_date)['nav']
                 val = calculate_residual_momentum(s, b_slice_price)

        # Rank Fix: If Val is NaN, put it at bottom (-Inf) so rank works
        row['Score'] = val if not np.isnan(val) else -float('inf')
        
        fwd_ret = np.nan
        if has_future:
            try: fwd_ret = (nav_wide[col].iloc[exit_idx] / nav_wide[col].iloc[entry_idx]) - 1
            except: pass
        row['Forward Return %'] = fwd_ret * 100
        temp.append(row)
        
    df = pd.DataFrame(temp)
    if df.empty: return df
    
    df['Strategy Rank'] = df['Score'].rank(ascending=False, method='min')
    if has_future: df['Actual Rank'] = df['Forward Return %'].rank(ascending=False, method='min')
    else: df['Actual Rank'] = np.nan
    return df.sort_values('Strategy Rank')

# ============================================================================
# 6. UI COMPONENTS
# ============================================================================

def display_backtest_results(nav, maps, nifty, strat_key, top_n, target_n, hold, cust_w, mom_cfg, ens_w=None):
    hist, eq, ben = run_backtest(nav, strat_key, top_n, target_n, hold, cust_w, mom_cfg, nifty, ens_w)
    
    if not eq.empty:
        start_date = eq.iloc[0]['date']
        end_date = eq.iloc[-1]['date']
        years = (end_date - start_date).days / 365.25
        
        strat_fin = eq.iloc[-1]['value']
        strat_cagr = (strat_fin/100)**(1/years)-1 if years>0 else 0
        
        bench_fin = ben.iloc[-1]['value'] if not ben.empty else 100
        bench_cagr = (bench_fin/100)**(1/years)-1 if years>0 and not ben.empty else 0
        
        avg_hit_rate = hist['hit_rate'].mean() if 'hit_rate' in hist.columns else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Strategy CAGR", f"{strat_cagr:.2%}")
        c2.metric("Benchmark CAGR", f"{bench_cagr:.2%}")
        c3.metric("Strategy Tot Ret", f"{strat_fin-100:.1f}%")
        c4.metric("Benchmark Tot Ret", f"{bench_fin-100:.1f}%")
        c5.metric("Avg Hit Rate", f"{avg_hit_rate:.1%}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq['date'], y=eq['value'], name='Strategy', line=dict(color='green')))
        if not ben.empty: 
            fig.add_trace(go.Scatter(x=ben['date'], y=ben['value'], name='Benchmark (Nifty 100)', line=dict(color='red', dash='dot')))
        
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{strat_key}")
        
        st.divider()
        st.subheader("🔍 Deep Dive Snapshot")
        
        max_idx_possible = len(nav) - hold - 1
        if max_idx_possible > 0:
            max_valid_date = nav.index[max_idx_possible]
            start_snapshot_date = nav.index.min() + pd.Timedelta(days=370)
            
            if start_snapshot_date < max_valid_date:
                quarterly_dates = pd.date_range(start=start_snapshot_date, end=max_valid_date, freq='Q')
                valid_dates = []
                for d in quarterly_dates:
                    trading_day = nav.index.asof(d)
                    if pd.notna(trading_day):
                        valid_dates.append(trading_day)
                valid_dates = sorted(list(set(valid_dates)), reverse=True)
                d_str = [d.strftime('%Y-%m-%d') for d in valid_dates]
                
                sel_d = st.selectbox("Snapshot Date (Quarterly)", d_str, key=f"dd_{strat_key}")
                
                if sel_d:
                    df_snap = generate_snapshot_table(nav, pd.to_datetime(sel_d), hold, strat_key, maps, cust_w, mom_cfg, nifty, ens_w)
                    if not df_snap.empty:
                        df_snap = df_snap[['name', 'Strategy Rank', 'Forward Return %', 'Actual Rank', 'Score']]
                        st.dataframe(
                            df_snap.style.format({
                                'Forward Return %': "{:.2f}%", 
                                'Strategy Rank':"{:.0f}", 
                                'Actual Rank':"{:.0f}",
                                'Score': "{:.4f}"
                            }).background_gradient(subset=['Forward Return %'], cmap='RdYlGn'), 
                            use_container_width=True
                        )
            else:
                st.warning("Not enough data history to generate quarterly snapshots.")
        else:
            st.warning("Holding period is too long relative to the available data history.")
    else:
        st.warning("No trades generated or insufficient data.")

def render_comparison_tab(nav, maps, nifty, top_n, target_n, hold):
    st.markdown("### 🏆 Strategy Leaderboard")
    
    # ADDED NEW STRATEGIES HERE
    strategies = {
        'Stable Momentum': ('stable_momentum', {}, {}, None),
        'Regime Switch (Smart)': ('regime_switch', {}, {}, None),
        'Residual Momentum': ('residual_momentum', {}, {}, None),
        'Consistent Alpha': ('consistent_alpha', {}, {}, None),
        'Martin Ratio': ('martin', {}, {}, None),
        'Ensemble': ('ensemble', {}, {'w_3m':0.33,'w_6m':0.33,'w_12m':0.33,'risk_adjust':True}, {'momentum':0.4, 'capture':0.6}),
        'Momentum': ('momentum', {}, {'w_3m':0.33,'w_6m':0.33,'w_12m':0.33,'risk_adjust':True}, None),
        'Sharpe': ('sharpe', {}, {}, None)
    }
    
    results = []
    fig = go.Figure()
    
    progress = st.progress(0)
    for idx, (name, (key, cust, mom, ens)) in enumerate(strategies.items()):
        hist, eq, ben = run_backtest(nav, key, top_n, target_n, hold, cust, mom, nifty, ens)
        if not eq.empty:
            start_date = eq.iloc[0]['date']
            end_date = eq.iloc[-1]['date']
            years = (end_date - start_date).days / 365.25
            total_ret = eq.iloc[-1]['value'] - 100
            cagr = (eq.iloc[-1]['value']/100)**(1/years)-1 if years>0 else 0
            avg_acc = hist['hit_rate'].mean() * 100 if 'hit_rate' in hist.columns else 0
            max_dd = calculate_max_dd(eq.set_index('date')['value'])
            
            results.append({
                'Strategy': name,
                'CAGR %': cagr * 100,
                'Max Drawdown %': max_dd * 100,
                'Hit Rate %': avg_acc
            })
            fig.add_trace(go.Scatter(x=eq['date'], y=eq['value'], name=name))
        progress.progress((idx + 1) / len(strategies))

    if 'ben' in locals() and not ben.empty:
         fig.add_trace(go.Scatter(x=ben['date'], y=ben['value'], name='Benchmark (Nifty)', line=dict(color='black', dash='dot', width=2)))

    if results:
        df_res = pd.DataFrame(results).set_index('Strategy').sort_values('CAGR %', ascending=False)
        st.caption(f"Hit Rate Logic: How many of the Top {top_n} selected funds landed in the Top {target_n} actual performers.")
        st.dataframe(df_res.style.format("{:.2f}").background_gradient(subset=['CAGR %'], cmap='Greens'), use_container_width=True)
        st.plotly_chart(fig, use_container_width=True, key="comparison_chart")

def render_backtest_tab():
    st.header("🚀 Strategy Backtester")
    c1, c2, c3, c4 = st.columns(4)
    cat = c1.selectbox("Category", list(FILE_MAPPING.keys()))
    top_n = c2.number_input("Top N Funds", 1, 20, DEFAULT_TOP_N, help="Funds to Buy")
    target_n = c3.number_input("Target N Funds", 1, 20, DEFAULT_TARGET_N, help="Hit Rate Success Target")
    hold = c4.number_input("Holding Period (Days)", 20, 504, DEFAULT_HOLDING) # Max 504 days
    
    with st.spinner("Loading Data..."):
        nav, maps = load_fund_data_raw(cat)
        nifty = load_nifty_data()
    
    if nav is None: st.error("Data not found."); return

    # ADDED TABS FOR NEW STRATEGIES
    tabs = st.tabs([
        "🏆 Compare All", "🚦 Regime Switch", "⚓ Stable Mom", "📉 Residual Mom", 
        "🧩 Ensemble", "🧠 Consistent Alpha", "🛡️ Martin", "🚀 Momentum", "⚖️ Sharpe", "Custom"
    ])
    
    with tabs[0]:
        # RUNS AUTOMATICALLY NOW (Button Removed)
        render_comparison_tab(nav, maps, nifty, top_n, target_n, hold)

    with tabs[1]:
        st.info("🚦 **Regime Switch Strategy**")
        st.markdown("**Logic:** Bull Market (>200 DMA) = **Momentum**. Bear Market (<200 DMA) = **Sharpe/Quality**.")
        display_backtest_results(nav, maps, nifty, 'regime_switch', top_n, target_n, hold, {}, {})

    with tabs[2]:
        st.info("⚓ **Stable Momentum**")
        st.markdown("**Logic:** 1. Get Top 2x funds by **Momentum**. 2. Filter that pool for **Lowest Drawdown**.")
        display_backtest_results(nav, maps, nifty, 'stable_momentum', top_n, target_n, hold, {}, {})

    with tabs[3]:
        st.info("📉 **Residual Momentum**")
        st.markdown("**Logic:** Ranking by the **Alpha Residuals** of a regression (Fund vs Benchmark). Removes market beta noise.")
        display_backtest_results(nav, maps, nifty, 'residual_momentum', top_n, target_n, hold, {}, {})

    with tabs[4]:
        st.info("🧩 **Ensemble Strategy**")
        display_backtest_results(nav, maps, nifty, 'ensemble', top_n, target_n, hold, {}, {'w_3m':0.33, 'w_6m':0.33, 'w_12m':0.33, 'risk_adjust':True}, {'momentum':0.4, 'capture':0.6})

    with tabs[5]:
        st.info("🧠 **Consistent Alpha**")
        display_backtest_results(nav, maps, nifty, 'consistent_alpha', top_n, target_n, hold, {}, {})

    with tabs[6]:
        display_backtest_results(nav, maps, nifty, 'martin', top_n, target_n, hold, {}, {})
        
    with tabs[7]:
        display_backtest_results(nav, maps, nifty, 'momentum', top_n, target_n, hold, {}, {'w_3m':0.33, 'w_6m':0.33, 'w_12m':0.33, 'risk_adjust':True})

    with tabs[8]:
        display_backtest_results(nav, maps, nifty, 'sharpe', top_n, target_n, hold, {}, {})

    with tabs[9]:
        st.write("Custom weights...")
        col_c1, col_c2, col_c3 = st.columns(3)
        cw_sh = col_c1.slider("Sharpe Ratio", 0.0, 1.0, 0.5, key="c_sh")
        cw_so = col_c2.slider("Sortino Ratio", 0.0, 1.0, 0.0, key="c_so")
        cw_mo = col_c3.slider("Momentum", 0.0, 1.0, 0.5, key="c_mo")
        col_c4, col_c5, col_c6 = st.columns(3)
        cw_vo = col_c4.slider("Low Volatility", 0.0, 1.0, 0.0, key="c_vo")
        cw_ir = col_c5.slider("Information Ratio", 0.0, 1.0, 0.0, key="c_ir")
        cw_maxdd = col_c6.slider("Low Max Drawdown", 0.0, 1.0, 0.0, key="c_maxdd")
        col_c7, col_c8, col_c9 = st.columns(3)
        cw_ca = col_c7.slider("Calmar Ratio", 0.0, 1.0, 0.0, key="c_ca")
        cw_alpha = col_c8.slider("Jensen's Alpha", 0.0, 1.0, 0.0, key="c_al")
        cw_treynor = col_c9.slider("Treynor Ratio", 0.0, 1.0, 0.0, key="c_tr")
        col_c10 = st.columns(1)[0]
        cw_beta = col_c10.slider("Low Beta", 0.0, 1.0, 0.0, key="c_be")
        
        raw_weights = {
            'sharpe': cw_sh, 'sortino': cw_so, 'momentum': cw_mo, 
            'volatility': cw_vo, 'info_ratio': cw_ir, 'maxdd': cw_maxdd,
            'calmar': cw_ca, 'alpha': cw_alpha, 'treynor': cw_treynor, 'beta': cw_beta
        }
        total_weight = sum(raw_weights.values())
        
        if total_weight > 0:
            final_weights = {k: v / total_weight for k, v in raw_weights.items()}
            m_c = {'w_3m':0.33, 'w_6m':0.33, 'w_12m':0.33, 'risk_adjust':True}
            display_backtest_results(nav, maps, nifty, 'custom', top_n, target_n, hold, final_weights, m_c)
        else:
            st.warning("Please select at least one weight > 0")

def main():
    t1, t2 = st.tabs(["📂 Category Explorer", "🚀 Strategy Backtester"])
    with t1: render_explorer_tab()
    with t2: render_backtest_tab()

if __name__ == "__main__":
    main()
