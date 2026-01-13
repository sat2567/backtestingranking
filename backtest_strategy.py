import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="Fund Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
DEFAULT_HOLDING = 126
DEFAULT_TOP_N = 5
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
# 2. HELPER FUNCTIONS: DATA CLEANING & MATH
# ============================================================================

def clean_weekday_data(df):
    """Ensures data is Mon-Fri and forward filled."""
    if df is None or df.empty: return df
    df = df[df.index <= MAX_DATA_DATE]
    df = df[df.index.dayofweek < 5] # Remove weekends
    if len(df) > 0:
        start_date = df.index.min()
        end_date = min(df.index.max(), MAX_DATA_DATE)
        all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B') 
        df = df.reindex(all_weekdays)
        df = df.ffill(limit=5)
    return df

# --- Strategy Metrics ---
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

def calculate_calmar_ratio(series, analysis_date):
    if len(series) < 10: return np.nan
    start_date = series.index[0]
    end_val = series.iloc[-1]
    start_val = series.iloc[0]
    years = (analysis_date - start_date).days / 365.25
    if years <= 0 or start_val <= 0: return np.nan
    cagr = (end_val / start_val) ** (1 / years) - 1
    max_dd = calculate_max_dd(series)
    if max_dd >= 0: return np.nan 
    return cagr / abs(max_dd)

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
    if len(series) < 70: return np.nan 
    price_cur = series.iloc[-1]
    current_date = series.index[-1]
    def get_past_price(days_ago):
        target_date = current_date - pd.Timedelta(days=days_ago)
        sub_series = series[series.index <= target_date]
        if sub_series.empty: return np.nan
        return sub_series.iloc[-1]
    r3 = (price_cur / get_past_price(91)) - 1 if not pd.isna(get_past_price(91)) else 0
    r6 = (price_cur / get_past_price(182)) - 1 if not pd.isna(get_past_price(182)) else 0
    r12 = (price_cur / get_past_price(365)) - 1 if not pd.isna(get_past_price(365)) else 0
    raw_score = (r3 * w_3m) + (r6 * w_6m) + (r12 * w_12m)
    if use_risk_adjust:
        date_1y_ago = current_date - pd.Timedelta(days=365)
        hist_vol_data = series[series.index >= date_1y_ago]
        if len(hist_vol_data) < 20: return np.nan
        vol = hist_vol_data.pct_change().dropna().std() * np.sqrt(TRADING_DAYS_YEAR)
        if vol == 0: return np.nan
        return raw_score / vol
    return raw_score

def calculate_beta_alpha_treynor(fund_returns, bench_returns):
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 30: return np.nan, np.nan, np.nan
    f_ret = fund_returns.loc[common_idx]
    b_ret = bench_returns.loc[common_idx]
    excess_fund = f_ret - DAILY_RISK_FREE_RATE
    excess_bench = b_ret - DAILY_RISK_FREE_RATE
    cov_matrix = np.cov(excess_fund, excess_bench)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else np.nan
    if np.isnan(beta): return np.nan, np.nan, np.nan
    annual_excess_fund = excess_fund.mean() * TRADING_DAYS_YEAR
    annual_excess_bench = excess_bench.mean() * TRADING_DAYS_YEAR
    alpha = annual_excess_fund - beta * annual_excess_bench
    treynor = annual_excess_fund / beta if beta != 0 else np.nan
    return beta, alpha, treynor

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
    bench_monthly = benchmark_series.resample('M').last().pct_change().dropna()
    bench_quarterly = benchmark_series.resample('Q').last().pct_change().dropna()
    
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
        losses = diff[diff <= 0]
        
        perc_times_beated = (len(beats) / len(diff)) * 100
        beated_by_percent_avg = beats.mean() * 100 if not beats.empty else 0
        lost_by_percent_avg = losses.mean() * 100 if not losses.empty else 0
        
        beated_by_percent_max = beats.max() * 100 if not beats.empty else 0
        beated_by_percent_min = beats.min() * 100 if not beats.empty else 0
        lost_by_percent_max = losses.min() * 100 if not losses.empty else 0
        lost_by_percent_min = losses.max() * 100 if not losses.empty else 0
        
        win_rate = perc_times_beated / 100
        loss_rate = 1 - win_rate
        wtd_avg_out = (win_rate * beated_by_percent_avg) + (loss_rate * lost_by_percent_avg)
        
        daily_ret = series.pct_change().dropna()
        comp = (1 + daily_ret).cumprod()
        peak = comp.expanding(min_periods=1).max()
        fund_dd = (comp / peak) - 1
        max_drawdown = fund_dd.min() * 100
        
        common_dd_idx = fund_dd.index.intersection(bench_dd.index)
        if len(common_dd_idx) > 10:
            f_dd = fund_dd.loc[common_dd_idx]
            b_dd = bench_dd.loc[common_dd_idx]
            dd_diff = f_dd - b_dd 
            less_dd = dd_diff[dd_diff > 0]
            more_dd = dd_diff[dd_diff < 0]
            
            perc_times_dd_less = (len(less_dd) / len(dd_diff)) * 100
            dd_less_by_percent_avg = less_dd.mean() * 100 if not less_dd.empty else 0
            dd_greater_by_percent_avg = more_dd.mean() * 100 if not more_dd.empty else 0
            dd_greater_max = more_dd.min() * 100 if not more_dd.empty else 0
            dd_less_min = less_dd.min() * 100 if not less_dd.empty else 0
        else:
            perc_times_dd_less, dd_less_by_percent_avg, dd_greater_by_percent_avg = 0,0,0
            dd_greater_max, dd_less_min = 0,0
            
        common_daily = daily_ret.index.intersection(bench_daily.index)
        daily_corr = daily_ret.loc[common_daily].corr(bench_daily.loc[common_daily]) if len(common_daily)>10 else np.nan
        
        fund_monthly = series.resample('M').last().pct_change().dropna()
        common_monthly = fund_monthly.index.intersection(bench_monthly.index)
        monthly_corr = fund_monthly.loc[common_monthly].corr(bench_monthly.loc[common_monthly]) if len(common_monthly)>5 else np.nan
        
        fund_quarterly = series.resample('Q').last().pct_change().dropna()
        common_quarterly = fund_quarterly.index.intersection(bench_quarterly.index)
        quart_corr = fund_quarterly.loc[common_quarterly].corr(bench_quarterly.loc[common_quarterly]) if len(common_quarterly)>3 else np.nan

        metrics.append({
            'fund_name': scheme_map.get(col, col),
            'percent_rolling_avg': percent_rolling_avg,
            'beated_by_percent_avg': beated_by_percent_avg,
            'lost_by_percent_avg': lost_by_percent_avg,
            'perc_times_beated': perc_times_beated,
            'beated_by_percent_max': beated_by_percent_max,
            'beated_by_percent_min': beated_by_percent_min,
            'lost_by_percent_max': lost_by_percent_max,
            'lost_by_percent_min': lost_by_percent_min,
            'Wtd_avg_outperformance': wtd_avg_out,
            'MaxDrawdown': max_drawdown,
            'dd_less_by_percent_avg': dd_less_by_percent_avg,
            'dd_greater_by_percent_avg': dd_greater_by_percent_avg,
            'perc_times_dd_less': perc_times_dd_less,
            'dd_greater_max': dd_greater_max,
            'dd_less_min': dd_less_min,
            'Daily_Return_Corr': daily_corr,
            'Monthly_Return_Corr': monthly_corr,
            'Quaterly_Return_Corr': quart_corr
        })
        
    df = pd.DataFrame(metrics)
    if not df.empty:
        df['return_rank'] = df['percent_rolling_avg'].rank(ascending=False, method='min')
        cols = [
            'return_rank', 'fund_name', 'percent_rolling_avg', 'Wtd_avg_outperformance',
            'perc_times_beated', 'beated_by_percent_avg', 'lost_by_percent_avg',
            'MaxDrawdown', 'perc_times_dd_less', 'dd_less_by_percent_avg',
            'Daily_Return_Corr', 'Monthly_Return_Corr', 'Quaterly_Return_Corr'
        ]
        remaining = [c for c in df.columns if c not in cols]
        df = df[cols + remaining].sort_values('return_rank')
        
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

# ============================================================================
# 5. BACKTESTER LOGIC
# ============================================================================

def get_lookback_data(nav_wide, analysis_date):
    max_days = 400 
    start_date = analysis_date - pd.Timedelta(days=max_days)
    return nav_wide[(nav_wide.index >= start_date) & (nav_wide.index < analysis_date)]

def run_backtest(nav_wide, strategy_type, top_n, holding_days, custom_weights, momentum_config, benchmark_series, ensemble_weights=None):
    last_possible_rebal_idx = len(nav_wide) - holding_days - 1
    if last_possible_rebal_idx < 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    last_possible_rebal_date = nav_wide.index[last_possible_rebal_idx]

    start_date = nav_wide.index.min() + pd.Timedelta(days=370)
    try: start_idx = nav_wide.index.searchsorted(start_date)
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    max_idx = nav_wide.index.searchsorted(last_possible_rebal_date, side='right')
    max_idx = min(max_idx, len(nav_wide)-holding_days-1)
    
    if max_idx < start_idx: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rebal_idx = list(range(start_idx, max_idx + 1, holding_days))
    if not rebal_idx: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    history, eq_curve, bench_curve = [], [{'date': nav_wide.index[rebal_idx[0]], 'value': 100.0}], [{'date': nav_wide.index[rebal_idx[0]], 'value': 100.0}]
    cap, b_cap = 100.0, 100.0
    
    for i in rebal_idx:
        date = nav_wide.index[i]
        hist = get_lookback_data(nav_wide, date)
        
        bench_rets = None
        if benchmark_series is not None:
            try:
                b_slice = get_lookback_data(benchmark_series.to_frame(), date)
                bench_rets = b_slice['nav'].pct_change().dropna()
            except: pass

        scores = {}
        
        # --- STRATEGY SCORING ---
        if strategy_type == 'ensemble':
            fund_ranks = pd.DataFrame(index=nav_wide.columns)
            def dict_to_norm_rank(s_dict):
                if not s_dict: return pd.Series(0, index=fund_ranks.index)
                s = pd.Series(s_dict)
                s = s.reindex(fund_ranks.index)
                return s.rank(pct=True, ascending=True).fillna(0)

            if ensemble_weights.get('momentum', 0) > 0:
                temp = {}
                for col in nav_wide.columns:
                    s = hist[col].dropna()
                    if len(s) > 70:
                        temp[col] = calculate_flexible_momentum(s, momentum_config['w_3m'], momentum_config['w_6m'], momentum_config['w_12m'], momentum_config['risk_adjust'])
                fund_ranks['momentum'] = dict_to_norm_rank(temp)

            if ensemble_weights.get('sharpe', 0) > 0:
                temp = {}
                for col in nav_wide.columns:
                    s = hist[col].dropna()
                    if len(s) > 126:
                        temp[col] = calculate_sharpe_ratio(s.pct_change().dropna())
                fund_ranks['sharpe'] = dict_to_norm_rank(temp)

            if ensemble_weights.get('martin', 0) > 0:
                temp = {}
                for col in nav_wide.columns:
                    s = hist[col].dropna()
                    if len(s) > 126:
                        temp[col] = calculate_martin_ratio(s)
                fund_ranks['martin'] = dict_to_norm_rank(temp)
                
            if ensemble_weights.get('info_ratio', 0) > 0:
                temp = {}
                for col in nav_wide.columns:
                    s = hist[col].dropna()
                    if len(s) > 126 and bench_rets is not None:
                        temp[col] = calculate_information_ratio(s.pct_change().dropna(), bench_rets)
                fund_ranks['info_ratio'] = dict_to_norm_rank(temp)
            
            if ensemble_weights.get('omega', 0) > 0:
                temp = {}
                for col in nav_wide.columns:
                    s = hist[col].dropna()
                    if len(s) > 126:
                        temp[col] = calculate_omega_ratio(s.pct_change().dropna())
                fund_ranks['omega'] = dict_to_norm_rank(temp)

            if ensemble_weights.get('capture', 0) > 0:
                temp = {}
                for col in nav_wide.columns:
                    s = hist[col].dropna()
                    if len(s) > 126 and bench_rets is not None:
                        temp[col] = calculate_capture_score(s.pct_change().dropna(), bench_rets)
                fund_ranks['capture'] = dict_to_norm_rank(temp)

            if ensemble_weights.get('sortino', 0) > 0:
                temp = {}
                for col in nav_wide.columns:
                    s = hist[col].dropna()
                    if len(s) > 126:
                        temp[col] = calculate_sortino_ratio(s.pct_change().dropna())
                fund_ranks['sortino'] = dict_to_norm_rank(temp)
            
            if ensemble_weights.get('var', 0) > 0:
                temp = {}
                for col in nav_wide.columns:
                    s = hist[col].dropna()
                    if len(s) > 126:
                        temp[col] = calculate_vol_adj_return(s)
                fund_ranks['var'] = dict_to_norm_rank(temp)
            
            if ensemble_weights.get('consistent_alpha', 0) > 0:
                temp = {}
                for col in nav_wide.columns:
                    s = hist[col].dropna()
                    if len(s) > 252:
                         wr = calculate_rolling_win_rate(s, benchmark_series.loc[:date]) if benchmark_series is not None else 0.5
                         rets = s.pct_change().dropna()
                         ir = calculate_information_ratio(rets, bench_rets) if bench_rets is not None else 0
                         temp[col] = (wr * 0.6) + (ir * 0.4) if not pd.isna(ir) else 0
                fund_ranks['consistent_alpha'] = dict_to_norm_rank(temp)

            final_score = pd.Series(0.0, index=fund_ranks.index)
            total_w = sum(ensemble_weights.values())
            if total_w > 0:
                for k, w in ensemble_weights.items():
                    if k in fund_ranks.columns:
                        final_score += fund_ranks[k] * w
            scores = final_score.to_dict()

        elif strategy_type == 'consistent_alpha':
            temp_rows = []
            for col in nav_wide.columns:
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
        
        elif strategy_type == 'omega':
             for col in nav_wide.columns:
                s = hist[col].dropna()
                if len(s) < 126: continue
                val = calculate_omega_ratio(s.pct_change().dropna())
                if not pd.isna(val): scores[col] = val
        
        elif strategy_type == 'capture_ratio':
             for col in nav_wide.columns:
                s = hist[col].dropna()
                if len(s) < 126: continue
                val = calculate_capture_score(s.pct_change().dropna(), bench_rets) if bench_rets is not None else 0
                if not pd.isna(val): scores[col] = val

        elif strategy_type == 'martin':
             for col in nav_wide.columns:
                s = hist[col].dropna()
                if len(s) < 126: continue
                val = calculate_martin_ratio(s)
                if not pd.isna(val): scores[col] = val
                
        elif strategy_type == 'info_ratio':
             for col in nav_wide.columns:
                s = hist[col].dropna()
                if len(s) < 126: continue
                rets = s.pct_change().dropna()
                val = calculate_information_ratio(rets, bench_rets) if bench_rets is not None else 0
                if not pd.isna(val): scores[col] = val

        else:
            needed_metrics = set()
            if strategy_type == 'momentum': needed_metrics = {'momentum'}
            elif strategy_type == 'var': needed_metrics = {'var'}
            elif strategy_type == 'sharpe': needed_metrics = {'sharpe'}
            elif strategy_type == 'sortino': needed_metrics = {'sortino'}
            elif strategy_type == 'custom': needed_metrics = set(k for k,v in custom_weights.items() if v > 0)
            
            temp_data = []
            for col in nav_wide.columns:
                s = hist[col].dropna()
                if len(s) < 126: continue
                rets = s.pct_change().dropna()
                row = {'id': col}
                
                if 'momentum' in needed_metrics:
                    row['momentum'] = calculate_flexible_momentum(s, momentum_config['w_3m'], momentum_config['w_6m'], momentum_config['w_12m'], momentum_config['risk_adjust'])
                if 'var' in needed_metrics:
                    row['var'] = calculate_vol_adj_return(s)
                if 'sharpe' in needed_metrics:
                    row['sharpe'] = calculate_sharpe_ratio(rets)
                if 'sortino' in needed_metrics:
                    row['sortino'] = calculate_sortino_ratio(rets)
                if 'volatility' in needed_metrics:
                    row['volatility'] = calculate_volatility(rets)
                if 'maxdd' in needed_metrics:
                    row['maxdd'] = calculate_max_dd(s)
                if 'calmar' in needed_metrics:
                    row['calmar'] = calculate_calmar_ratio(s, date)
                if any(x in needed_metrics for x in ['info_ratio', 'alpha', 'beta', 'treynor']) and bench_rets is not None and not bench_rets.empty:
                    b, a, t = calculate_beta_alpha_treynor(rets, bench_rets)
                    ir = calculate_information_ratio(rets, bench_rets)
                    if 'beta' in needed_metrics: row['beta'] = b
                    if 'alpha' in needed_metrics: row['alpha'] = a
                    if 'treynor' in needed_metrics: row['treynor'] = t
                    if 'info_ratio' in needed_metrics: row['info_ratio'] = ir
                
                temp_data.append(row)
                
            if temp_data:
                df_metrics = pd.DataFrame(temp_data).set_index('id')
                final_score = pd.Series(0.0, index=df_metrics.index)
                active_weights = custom_weights if strategy_type == 'custom' else {list(needed_metrics)[0]: 1.0}
                for metric, w in active_weights.items():
                    if metric not in df_metrics.columns: continue
                    if metric in ['volatility', 'maxdd', 'beta']:
                        rank_score = df_metrics[metric].abs().rank(pct=True, ascending=False)
                    else:
                        rank_score = df_metrics[metric].rank(pct=True, ascending=True)
                    final_score = final_score.add(rank_score * w, fill_value=0)
                scores = final_score.to_dict()

        selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
        if not selected: continue

        entry_i = i + 1
        exit_i = min(i + 1 + holding_days, len(nav_wide)-1)
        
        # Calculate Strategy Return
        period_ret_all_funds = (nav_wide.iloc[exit_i] / nav_wide.iloc[entry_i]) - 1
        port_ret = period_ret_all_funds[selected].mean()
        
        # --- Selection Accuracy Logic ---
        actual_top_n_funds = period_ret_all_funds.dropna().nlargest(top_n).index.tolist()
        matches = len(set(selected).intersection(set(actual_top_n_funds)))
        hit_rate = matches / top_n if top_n > 0 else 0
        
        cap *= (1 + (port_ret if not pd.isna(port_ret) else 0))
        b_ret = 0.0
        if benchmark_series is not None:
            try: b_ret = (benchmark_series.asof(nav_wide.index[exit_i]) / benchmark_series.asof(nav_wide.index[entry_i])) - 1
            except: pass
        b_cap *= (1 + b_ret)
        
        history.append({
            'date': date, 
            'selected': selected, 
            'return': port_ret,
            'hit_rate': hit_rate 
        })
        eq_curve.append({'date': nav_wide.index[exit_i], 'value': cap})
        bench_curve.append({'date': nav_wide.index[exit_i], 'value': b_cap})

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
        
        row['Score'] = val if not np.isnan(val) else -999
        
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

def display_backtest_results(nav, maps, nifty, strat_key, top_n, hold, cust_w, mom_cfg, ens_w=None):
    hist, eq, ben = run_backtest(nav, strat_key, top_n, hold, cust_w, mom_cfg, nifty, ens_w)
    
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
                        df_snap = df_snap[['name', 'Strategy Rank', 'Forward Return %', 'Actual Rank']]
                        st.dataframe(
                            df_snap.style.format({
                                'Forward Return %': "{:.2f}%", 
                                'Strategy Rank':"{:.0f}", 
                                'Actual Rank':"{:.0f}"
                            }).background_gradient(subset=['Forward Return %'], cmap='RdYlGn'), 
                            use_container_width=True
                        )
            else:
                st.warning("Not enough data history to generate quarterly snapshots.")
        else:
            st.warning("Holding period is too long relative to the available data history.")
    else:
        st.warning("No trades generated or insufficient data.")

def render_comparison_tab(nav, maps, nifty, top_n, hold):
    st.markdown("### 🏆 Strategy Leaderboard")
    st.info("Comparing all strategies over the selected period.")
    
    strategies = {
        'Martin Ratio (Ulcer)': ('martin', {}, {}, None),
        'Information Ratio': ('info_ratio', {}, {}, None),
        'Omega Ratio': ('omega', {}, {}, None),
        'Capture Ratio': ('capture_ratio', {}, {}, None),
        'Momentum': ('momentum', {}, {'w_3m':0.33,'w_6m':0.33,'w_12m':0.33,'risk_adjust':True}, None),
        'Sharpe': ('sharpe', {}, {}, None),
        'Sortino': ('sortino', {}, {}, None),
        'VAR': ('var', {}, {}, None),
        'Consistent Alpha': ('consistent_alpha', {}, {}, None)
    }
    
    results = []
    fig = go.Figure()
    
    for name, (key, cust, mom, ens) in strategies.items():
        hist, eq, ben = run_backtest(nav, key, top_n, hold, cust, mom, nifty, ens)
        if not eq.empty:
            start_date = eq.iloc[0]['date']
            end_date = eq.iloc[-1]['date']
            years = (end_date - start_date).days / 365.25
            total_ret = eq.iloc[-1]['value'] - 100
            cagr = (eq.iloc[-1]['value']/100)**(1/years)-1 if years>0 else 0
            
            avg_acc = hist['hit_rate'].mean() * 100 if 'hit_rate' in hist.columns else 0

            strat_rets = eq.set_index('date')['value'].pct_change().dropna()
            bench_rets = ben.set_index('date')['value'].pct_change().dropna()
            
            common = strat_rets.index.intersection(bench_rets.index)
            if not common.empty:
                wins = (strat_rets.loc[common] > bench_rets.loc[common]).sum()
                win_rate = wins / len(common)
                max_dd = calculate_max_dd(eq.set_index('date')['value'])
                
                results.append({
                    'Strategy': name,
                    'CAGR %': cagr * 100,
                    'Total Return %': total_ret,
                    'Max Drawdown %': max_dd * 100,
                    'Avg Selection Accuracy %': avg_acc, 
                    'Win Rate vs Nifty %': win_rate * 100
                })
                
                fig.add_trace(go.Scatter(x=eq['date'], y=eq['value'], name=name))

    if 'ben' in locals() and not ben.empty:
         fig.add_trace(go.Scatter(x=ben['date'], y=ben['value'], name='Benchmark (Nifty)', line=dict(color='black', dash='dot', width=2)))

    if results:
        df_res = pd.DataFrame(results).set_index('Strategy').sort_values('CAGR %', ascending=False)
        st.dataframe(df_res.style.format("{:.2f}"), use_container_width=True)
        st.plotly_chart(fig, use_container_width=True, key="comparison_chart")
    else:
        st.error("Could not run backtests. Check data availability.")

def render_backtest_tab():
    st.header("🚀 Strategy Backtester")
    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("Category", list(FILE_MAPPING.keys()))
    top_n = c2.number_input("Top N Funds", 1, 20, DEFAULT_TOP_N)
    hold = c3.number_input("Holding Period (Days)", 20, 252, DEFAULT_HOLDING)
    
    with st.spinner("Loading Data..."):
        nav, maps = load_fund_data_raw(cat)
        nifty = load_nifty_data()
    
    if nav is None: st.error("Data not found."); return

    # Total 13 Tabs now
    tabs = st.tabs(["🏆 Compare All", "🧩 Ensemble", "🛡️ Martin Ratio", "🎯 Info Ratio", "Ω Omega", "📉 Capture", "🚀 Momentum", "⚖️ Sharpe", "📉 Sortino", "📊 VAR", "🧠 Consistent Alpha", "🛠️ Custom", "📚 Formulas & Logic"])
    
    with tabs[0]:
        if st.button("Run Comparison Analysis"):
            render_comparison_tab(nav, maps, nifty, top_n, hold)
        else:
            st.info("Click the button to run a comparison of all strategies.")

    with tabs[1]:
        st.info("🧩 **Ensemble Strategy (The 'Master Strategy')**")
        st.markdown(r"""
        * **What it is:** Combines the rankings of multiple strategies into one final score.
        * **Formula:** $\sum (Weight_i \times Rank_i)$ for each chosen metric.
        * **Why use it:** No single metric works in all markets. Momentum works in bull markets, Capture Ratio in choppy markets, and Martin Ratio in bear markets. Combining them creates a robust 'All-Weather' strategy.
        """)
        ec1, ec2, ec3 = st.columns(3)
        ew_mom = ec1.slider("Momentum Weight", 0.0, 1.0, 0.4, key="ew_mom")
        ew_shp = ec2.slider("Sharpe Weight", 0.0, 1.0, 0.0, key="ew_shp")
        ew_alp = ec3.slider("Consistent Alpha Weight", 0.0, 1.0, 0.0, key="ew_alp")
        
        ec4, ec5, ec6 = st.columns(3)
        ew_sor = ec4.slider("Sortino Weight", 0.0, 1.0, 0.0, key="ew_sor")
        ew_var = ec5.slider("VAR Weight", 0.0, 1.0, 0.0, key="ew_var")
        ew_cap = ec6.slider("Capture Ratio Weight", 0.0, 1.0, 0.6, key="ew_cap")
        
        ec7, ec8 = st.columns(2)
        ew_omg = ec7.slider("Omega Weight", 0.0, 1.0, 0.0, key="ew_omg")
        ew_mar = ec8.slider("Martin Ratio Weight", 0.0, 1.0, 0.0, key="ew_mar")
        
        ens_weights = {
            'momentum': ew_mom, 'sharpe': ew_shp, 'consistent_alpha': ew_alp,
            'sortino': ew_sor, 'var': ew_var, 'capture': ew_cap, 'omega': ew_omg, 'martin': ew_mar
        }
        
        ens_mom_cfg = {'w_3m':0.33, 'w_6m':0.33, 'w_12m':0.33, 'risk_adjust':True}
        
        if sum(ens_weights.values()) > 0:
            display_backtest_results(nav, maps, nifty, 'ensemble', top_n, hold, {}, ens_mom_cfg, ens_weights)
        else:
            st.warning("Please select at least one weight.")

    with tabs[2]:
        st.info("🛡️ **Martin Ratio (Ulcer Index)**")
        st.markdown(r"""
        * **Formula:** $\frac{R_p - R_f}{\text{Ulcer Index}}$ where Ulcer Index = $\sqrt{\frac{\sum D_i^2}{n}}$
        * **Logic:**  Unlike Standard Deviation (which treats upside jumps as risk), Ulcer Index only measures **downside** volatility. Crucially, it penalizes **depth** AND **duration** of drawdowns.
        * **Why use it:** If you hate seeing your portfolio in the red for months, this is your metric. It optimizes for 'Sleep Well At Night'.
        """)
        display_backtest_results(nav, maps, nifty, 'martin', top_n, hold, {}, {})

    with tabs[3]:
        st.info("🎯 **Information Ratio (Manager Skill)**")
        st.markdown(r"""
        * **Formula:** $\frac{R_p - R_b}{\sigma_{(R_p - R_b)}}$ (Active Return / Tracking Error)
        * **Logic:**  It measures pure active management skill. It asks: "Is the manager consistently beating the benchmark, or just taking wild random bets?"
        * **Tracking Error:** The standard deviation of the difference between fund and benchmark returns.
        * **Why use it:** To find consistent outperformers who don't just hug the index.
        """)
        display_backtest_results(nav, maps, nifty, 'info_ratio', top_n, hold, {}, {})

    with tabs[4]:
        st.info("Ω **Omega Ratio (Fat Tail Risk)**")
        st.markdown(r"""
        * **Formula:** $\frac{\sum P(Win)}{\sum P(Loss)}$ relative to a threshold (Risk Free Rate).
        * **Logic:**  Standard metrics assume returns follow a normal Bell Curve (they don't!). Omega captures the 'Fat Tails'—the extreme crash risks and the massive upside jackpots.
        * **Why use it:** If you want to capture massive upside potential while strictly limiting the chance of catastrophic loss.
        """)
        display_backtest_results(nav, maps, nifty, 'omega', top_n, hold, {}, {})

    with tabs[5]:
        st.info("📉 **Upside/Downside Capture Ratio**")
        st.markdown(r"""
        * **Formula:** $\frac{\text{Upside Capture \%}}{\text{Downside Capture \%}}$
        * **Logic:** 
            * **Upside Capture:** How much of the market's rally did the fund capture? (e.g., Market +10%, Fund +12% = 120%)
            * **Downside Capture:** How much of the market's crash did the fund suffer? (e.g., Market -10%, Fund -5% = 50%)
        * **Why use it:** It identifies 'efficient' funds. You want funds that participate in rallies but resist crashes.
        """)
        display_backtest_results(nav, maps, nifty, 'capture_ratio', top_n, hold, {}, {})

    with tabs[6]:
        st.info("🚀 **Momentum Strategy**")
        st.markdown(r"""
        * **Formula:** Weighted average of returns over past 3M, 6M, and 12M. $Score = \frac{w_1 R_{3m} + w_2 R_{6m} + w_3 R_{12m}}{Volatility}$
        * **Logic:** 'Winners tend to keep winning.' Funds that have outperformed recently often continue to do so in the near term due to investor inflow and trend strength.
        * **Risk Adjustment:** We divide the score by volatility to penalize funds that just got lucky with one massive spike.
        * **Why use it:** Aggressive growth. Works best in trending bull markets.
        """)
        mc1, mc2, mc3 = st.columns(3)
        w3 = mc1.slider("3M Weight", 0.0, 1.0, 1.0, key="m_w3")
        w6 = mc2.slider("6M Weight", 0.0, 1.0, 1.0, key="m_w6")
        w12 = mc3.slider("12M Weight", 0.0, 1.0, 1.0, key="m_w12")
        adj = st.checkbox("Risk Adjust?", True, key="m_adj")
        tot = w3+w6+w12
        mom_cfg = {'w_3m':w3/tot if tot else 0, 'w_6m':w6/tot if tot else 0, 'w_12m':w12/tot if tot else 0, 'risk_adjust':adj}
        display_backtest_results(nav, maps, nifty, 'momentum', top_n, hold, {}, mom_cfg)

    with tabs[7]:
        st.info("⚖️ **Sharpe Ratio**")
        st.markdown(r"""
        * **Formula:** $\frac{R_p - R_f}{\sigma_p}$ (Excess Return / Volatility)
        * **Logic:** The most famous risk metric. It tells you return per unit of total risk.
        * **The Flaw:** It punishes *upside* volatility. If a fund jumps +50%, Sharpe thinks that's 'risky' and lowers the score.
        * **Why use it:** Good for general comparison, but Sortino or Omega are often better for growth funds.
        """)
        display_backtest_results(nav, maps, nifty, 'sharpe', top_n, hold, {}, {})

    with tabs[8]:
        st.info("📉 **Sortino Ratio**")
        st.markdown(r"""
        * **Formula:** $\frac{R_p - R_f}{\sigma_d}$ (Excess Return / Downside Deviation)
        * **Logic:** Like Sharpe, but it only cares about **bad volatility** (losses). It ignores upside volatility (big gains).
        * **Why use it:** Perfect for growth investors who don't mind a bumpy ride as long as the bumps are upward.
        """)
        display_backtest_results(nav, maps, nifty, 'sortino', top_n, hold, {}, {})

    with tabs[9]:
        st.info("📊 **VAR (Volatility Adjusted Return)**")
        st.markdown(r"""
        * **Formula:** $\frac{R_p}{\sigma_p}$ (Return / Volatility)
        * **Logic:** A simplified efficiency metric. How much return am I getting for every 1% of wiggle in the price?
        * **Why use it:** When the Risk-Free rate is negligible or you just want raw efficiency without accounting for a benchmark.
        """)
        display_backtest_results(nav, maps, nifty, 'var', top_n, hold, {}, {})

    with tabs[10]:
        st.info("🧠 **Consistent Alpha**")
        st.markdown(r"""
        * **Formula:** $0.6 \times Rank(WinRate) + 0.4 \times Rank(InfoRatio)$
        * **Win Rate:** Frequency of beating the benchmark (Consistency).
        * **Info Ratio:** Magnitude of beating the benchmark (Skill).
        * **Why use it:** To find funds that are 'steady Eddies'—they may not be the top 1% every month, but they rarely underperform.
        """)
        display_backtest_results(nav, maps, nifty, 'consistent_alpha', top_n, hold, {}, {})

    with tabs[11]:
        st.info("🛠️ **Custom Strategy**")
        st.write("Mix and match standard metrics.")
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
            with st.expander("📊 View Normalized Weights"):
                st.write(pd.DataFrame([final_weights]).T.rename(columns={0: 'Weight'}).style.format("{:.1%}"))
            m_c = {'w_3m':0.33, 'w_6m':0.33, 'w_12m':0.33, 'risk_adjust':True}
            display_backtest_results(nav, maps, nifty, 'custom', top_n, hold, final_weights, m_c)
        else:
            st.warning("Please select at least one weight > 0")

    # --- NEW: METHODOLOGY & FORMULAS TAB ---
    with tabs[12]:
        st.markdown("# 📚 Strategy Methodologies & Formulas")
        st.markdown("---")
        
        st.markdown("### 🌍 Global Backtesting Settings")
        st.markdown(f"""
        * **Lookback Period (Window):** 400 Days. 
          * *Why?* To calculate 12-month momentum or 1-year rolling volatility, we need at least 252 trading days. 400 days gives us a safe buffer (approx 1.6 years) to ensure all metrics are populated for every rebalance.
        * **Holding Period:** {hold} Days (User Selectable).
          * *Why?* This simulates how often you 'check' your portfolio. If set to 126 days, the backtester 'buys' the top funds, holds them for 6 months, sells, and repeats.
        * **Risk-Free Rate ($R_f$):** {RISK_FREE_RATE*100}% Annually.
          * *Why?* This is the standard 'hurdle rate' (e.g., a 10-year G-Sec bond yield). Returns below this are considered 'risk-free' and should not be rewarded.
        """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 1. Sharpe Ratio")
            st.latex(r"Sharpe = \frac{R_p - R_f}{\sigma_p}")
            st.markdown("""
            * **$R_p$**: Portfolio Annual Return
            * **$R_f$**: Risk-Free Rate (6%)
            * **$\sigma_p$**: Annualized Standard Deviation (Total Volatility)
            * **Logic:** Reward returns, punish ALL volatility.
            """)
            
            st.markdown("### 2. Sortino Ratio")
            st.latex(r"Sortino = \frac{R_p - R_f}{\sigma_d}")
            st.markdown("""
            * **$\sigma_d$**: Downside Deviation (Vol of only negative returns)
            * **Logic:** Reward returns, punish only BAD volatility (crashes).
            """)
            
            st.markdown("### 3. Omega Ratio")
            st.latex(r"\Omega = \frac{\sum P(Gain)}{\sum P(Loss)}")
            st.markdown("""
            * **Threshold:** Risk-Free Rate
            * **Logic:** Ratio of the 'area' of gains above 6% vs. area of losses below 6%. Captures 'Fat Tails' (extreme events).
            """)
            
            st.markdown("### 4. Martin Ratio (Ulcer)")
            st.latex(r"Martin = \frac{R_p - R_f}{\text{Ulcer Index}}")
            st.latex(r"Ulcer = \sqrt{\frac{\sum D_i^2}{n}}")
            st.markdown("""
            * **$D_i$**: % Drawdown on Day i
            * **Logic:** Penalizes duration. Long, deep drawdowns destroy this score.
            """)

        with col2:
            st.markdown("### 5. Information Ratio")
            st.latex(r"IR = \frac{R_p - R_b}{\text{Tracking Error}}")
            st.markdown("""
            * **$R_b$**: Benchmark Return (Nifty 100)
            * **Tracking Error:** Std Dev of $(R_p - R_b)$
            * **Logic:** Measures pure manager skill (Active Return) per unit of active risk.
            """)
            
            st.markdown("### 6. Capture Ratio")
            st.latex(r"Score = \frac{\text{Up Capture \%}}{\text{Down Capture \%}}")
            st.markdown("""
            * **Logic:** If Market +10% and Fund +12%, Up Cap = 120%. If Market -10% and Fund -5%, Down Cap = 50%. Score = 2.4.
            """)
            
            st.markdown("### 7. Momentum")
            st.latex(r"Mom = \frac{w_1 R_{3m} + w_2 R_{6m} + w_3 R_{12m}}{Volatility}")
            st.markdown("""
            * **Logic:** Recent winners tend to keep winning. Adjusted by volatility to avoid 'lucky spikes'.
            """)
            
            st.markdown("### 8. Consistent Alpha")
            st.latex(r"Score = 0.6 \times Rank(WinRate) + 0.4 \times Rank(IR)")
            st.markdown("""
            * **Win Rate:** % of days Fund > Benchmark.
            * **Logic:** Rewards funds that beat the index OFTEN (Win Rate) and EFFICIENTLY (IR).
            """)

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
                
                column_configuration = {
                    "fund_name": st.column_config.TextColumn("Fund Name", help="Name of the mutual fund scheme."),
                    "return_rank": st.column_config.NumberColumn("Rank", help="The fund's rank based purely on returns (1 is best).", format="%d"),
                    "percent_rolling_avg": st.column_config.NumberColumn("Rolling Avg %", help="The average return of the fund calculated on a rolling basis. This is a measure of the fund's 'typical' performance.", format="%.2f%%"),
                    "Wtd_avg_outperformance": st.column_config.NumberColumn("Wtd Avg Outperf", help="Weighted Average Outperformance. A composite score combining frequency of beating the index and the magnitude of the win. Higher is better.", format="%.2f%%"),
                    "perc_times_beated": st.column_config.NumberColumn("% Times Beaten", help="The consistency score. It shows the percentage of time intervals where the fund's return was higher than the benchmark's return.", format="%.1f%%"),
                    "beated_by_percent_avg": st.column_config.NumberColumn("Avg Win %", help="On days/periods where the fund beat the benchmark, this is the average percentage outperformance.", format="%.2f%%"),
                    "lost_by_percent_avg": st.column_config.NumberColumn("Avg Loss %", help="On days/periods where the fund lagged the benchmark, this is the average percentage underperformance.", format="%.2f%%"),
                    "MaxDrawdown": st.column_config.NumberColumn("Max Drawdown", help="The maximum observed loss from a peak to a trough over the period analyzed.", format="%.2f%%"),
                    "perc_times_dd_less": st.column_config.NumberColumn("% DD Less", help="The percentage of time the fund had a smaller drawdown (fell less) than the benchmark. High numbers indicate good defensive characteristics.", format="%.1f%%"),
                    "dd_less_by_percent_avg": st.column_config.NumberColumn("Avg DD Saved %", help="Measures downside protection. When the market falls, this is the average % by which the fund fell less than the benchmark.", format="%.2f%%"),
                    "Daily_Return_Corr": st.column_config.NumberColumn("Daily Corr", help="Correlation of daily returns. 1.0 means perfectly in sync with benchmark.", format="%.2f"),
                    "Monthly_Return_Corr": st.column_config.NumberColumn("Monthly Corr", help="Correlation of monthly returns. Active funds often have lower correlations.", format="%.2f"),
                    "Quaterly_Return_Corr": st.column_config.NumberColumn("Quarterly Corr", help="Correlation of quarterly returns.", format="%.2f")
                }
                
                subset_cols = [
                    'return_rank', 'fund_name', 'percent_rolling_avg', 'Wtd_avg_outperformance',
                    'perc_times_beated', 'beated_by_percent_avg', 'lost_by_percent_avg',
                    'MaxDrawdown', 'perc_times_dd_less', 'dd_less_by_percent_avg',
                    'Daily_Return_Corr', 'Monthly_Return_Corr', 'Quaterly_Return_Corr'
                ]
                
                valid_subset = [c for c in subset_cols if c in df.columns]
                
                st.dataframe(
                    df[valid_subset],
                    column_config=column_configuration,
                    use_container_width=True, 
                    height=600
                )
            else:
                df = calculate_quarterly_ranks(nav, maps)
                st.dataframe(df.style.format({'% Time in Top 5': '{:.1f}%'}, na_rep=""), use_container_width=True, height=600)

def main():
    t1, t2 = st.tabs(["📂 Category Explorer", "🚀 Strategy Backtester"])
    with t1: render_explorer_tab()
    with t2: render_backtest_tab()

if __name__ == "__main__":
    main()
