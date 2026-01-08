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
LAST_REBALANCE_DATE = pd.Timestamp('2025-05-01')
QUARTERLY_STEP = 63

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
    """Calculates % of times rolling 1Y fund return > rolling 1Y benchmark return."""
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

def calculate_analytics_metrics(nav_df, scheme_map, benchmark_series=None, top_n_for_freq=None):
    if nav_df is None or nav_df.empty: return pd.DataFrame()
    cat_avg_nav = nav_df.mean(axis=1)
    window = 252
    rolling_rets = nav_df.pct_change(window).dropna(how='all')
    benchmark_rolling = cat_avg_nav.pct_change(window).dropna()
    
    bench_daily = benchmark_series.pct_change().dropna() if benchmark_series is not None else None
    metrics = []
    
    for col in nav_df.columns:
        series = nav_df[col].dropna()
        if len(series) < window + 30: continue
        
        daily_rets = series.pct_change().dropna()
        fund_roll = rolling_rets[col].dropna()
        common_idx = fund_roll.index.intersection(benchmark_rolling.index)
        
        avg_rolling = fund_roll.loc[common_idx].mean() if not common_idx.empty else np.nan
        max_dd = calculate_max_dd(series)
        
        beats, losses = pd.Series(dtype=float), pd.Series(dtype=float)
        if not common_idx.empty:
            excess = fund_roll.loc[common_idx] - benchmark_rolling.loc[common_idx]
            beats = excess[excess > 0]
            losses = excess[excess <= 0]
        
        beta, alpha, treynor, ir = np.nan, np.nan, np.nan, np.nan
        if bench_daily is not None:
            beta, alpha, treynor = calculate_beta_alpha_treynor(daily_rets, bench_daily)
            ir = calculate_information_ratio(daily_rets, bench_daily)

        metrics.append({
            'id': col,
            'Fund Name': scheme_map.get(col, col),
            'Avg Rolling Return (1Y)': avg_rolling * 100 if not np.isnan(avg_rolling) else 0,
            'Avg Outperformance %': beats.mean()*100 if not beats.empty else 0,
            'Avg Underperformance %': losses.mean()*100 if not losses.empty else 0,
            'Win Rate vs Nifty %': (len(beats)/len(common_idx))*100 if len(common_idx)>0 else 0,
            'Max Drawdown': max_dd * 100 if not np.isnan(max_dd) else 0,
            'Beta': beta,
            'Jensen Alpha': alpha * 100 if not np.isnan(alpha) else np.nan,
            'Treynor': treynor,
            'Info Ratio': ir
        })
    df = pd.DataFrame(metrics)
    if not df.empty:
        df['Rank'] = df['Avg Rolling Return (1Y)'].rank(ascending=False, method='min')
        cols = ['Rank', 'Fund Name', 'Avg Rolling Return (1Y)', 'Max Drawdown', 'Win Rate vs Nifty %', 'Avg Outperformance %', 'Avg Underperformance %', 'Beta', 'Jensen Alpha', 'Treynor', 'Info Ratio']
        df = df[[c for c in cols if c in df.columns]].sort_values('Rank')
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
        rank_df = rank_df.sort_values(rank_df.columns[-1])
    return rank_df

# ============================================================================
# 5. BACKTESTER LOGIC
# ============================================================================

def get_lookback_data(nav_wide, analysis_date):
    max_days = 400 
    start_date = analysis_date - pd.Timedelta(days=max_days)
    return nav_wide[(nav_wide.index >= start_date) & (nav_wide.index < analysis_date)]

def run_backtest(nav_wide, strategy_type, top_n, holding_days, custom_weights, momentum_config, benchmark_series):
    start_date = nav_wide.index.min() + pd.Timedelta(days=370)
    try: start_idx = nav_wide.index.searchsorted(start_date)
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    max_idx = nav_wide.index.searchsorted(LAST_REBALANCE_DATE, side='right') - 1
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
        
        # --- NEW: Consistent Alpha Strategy Logic ---
        if strategy_type == 'consistent_alpha':
            temp_rows = []
            for col in nav_wide.columns:
                s = hist[col].dropna()
                if len(s) < 252: continue
                
                # Win Rate
                wr = calculate_rolling_win_rate(s, benchmark_series.loc[:date]) if benchmark_series is not None else 0.5
                
                # Info Ratio
                rets = s.pct_change().dropna()
                ir = calculate_information_ratio(rets, bench_rets) if bench_rets is not None else 0
                
                temp_rows.append({'id': col, 'win_rate': wr, 'ir': ir if not pd.isna(ir) else 0})
            
            if temp_rows:
                df_sc = pd.DataFrame(temp_rows).set_index('id')
                # 60% Win Rate, 40% IR
                final_score = (df_sc['win_rate'].rank(pct=True) * 0.6) + (df_sc['ir'].rank(pct=True) * 0.4)
                scores = final_score.to_dict()

        # --- Standard & Custom Logic ---
        else:
            needed_metrics = set()
            combo_weights = {}
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
        
        period_ret = (nav_wide.iloc[exit_i] / nav_wide.iloc[entry_i]) - 1
        port_ret = period_ret[selected].mean()
        cap *= (1 + (port_ret if not pd.isna(port_ret) else 0))
        
        b_ret = 0.0
        if benchmark_series is not None:
            try: b_ret = (benchmark_series.asof(nav_wide.index[exit_i]) / benchmark_series.asof(nav_wide.index[entry_i])) - 1
            except: pass
        b_cap *= (1 + b_ret)
        
        history.append({'date': date, 'selected': selected, 'return': port_ret})
        eq_curve.append({'date': nav_wide.index[exit_i], 'value': cap})
        bench_curve.append({'date': nav_wide.index[exit_i], 'value': b_cap})

    return pd.DataFrame(history), pd.DataFrame(eq_curve), pd.DataFrame(bench_curve)

def generate_snapshot_table(nav_wide, analysis_date, holding_days, strategy_type, names_map, custom_weights, momentum_config, benchmark_series):
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
            # Return raw score as simple sum for display, rank determines actual pos
            val = (wr * 0.6) + (ir * 0.4) 
        
        elif strategy_type == 'momentum':
            val = calculate_flexible_momentum(s, momentum_config['w_3m'], momentum_config['w_6m'], momentum_config['w_12m'], momentum_config['risk_adjust'])
        elif strategy_type == 'var': val = calculate_vol_adj_return(s)
        elif strategy_type == 'sharpe': val = calculate_sharpe_ratio(rets)
        elif strategy_type == 'sortino': val = calculate_sortino_ratio(rets)
        elif strategy_type == 'custom':
            score = 0
            # Simplified for display
            if custom_weights.get('sharpe',0): score += calculate_sharpe_ratio(rets) * custom_weights['sharpe']
            val = score 
        
        row['Score'] = val if not np.isnan(val) else -999
        
        fwd_ret = np.nan
        if has_future:
            try: fwd_ret = (nav_wide[col].iloc[exit_idx] / nav_wide[col].iloc[entry_idx]) - 1
            except: pass
        row['Forward Return %'] = fwd_ret * 100
        temp.append(row)
        
    df = pd.DataFrame(temp)
    if df.empty: return df
    
    # Recalculate Rank based on Score
    df['Strategy Rank'] = df['Score'].rank(ascending=False, method='min')
    if has_future: df['Actual Rank'] = df['Forward Return %'].rank(ascending=False, method='min')
    else: df['Actual Rank'] = np.nan
    return df.sort_values('Strategy Rank')

# ============================================================================
# 6. UI COMPONENTS
# ============================================================================

def display_backtest_results(nav, maps, nifty, strat_key, top_n, hold, cust_w, mom_cfg):
    hist, eq, ben = run_backtest(nav, strat_key, top_n, hold, cust_w, mom_cfg, nifty)
    if not eq.empty:
        start_date = eq.iloc[0]['date']
        end_date = eq.iloc[-1]['date']
        years = (end_date - start_date).days / 365.25
        
        strat_fin = eq.iloc[-1]['value']
        strat_cagr = (strat_fin/100)**(1/years)-1 if years>0 else 0
        
        bench_fin = ben.iloc[-1]['value'] if not ben.empty else 100
        bench_cagr = (bench_fin/100)**(1/years)-1 if years>0 and not ben.empty else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Strategy CAGR", f"{strat_cagr:.2%}")
        c2.metric("Benchmark CAGR", f"{bench_cagr:.2%}")
        c3.metric("Strategy Total Return", f"{strat_fin-100:.1f}%")
        c4.metric("Benchmark Total Return", f"{bench_fin-100:.1f}%")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq['date'], y=eq['value'], name='Strategy', line=dict(color='green')))
        if not ben.empty: 
            fig.add_trace(go.Scatter(x=ben['date'], y=ben['value'], name='Benchmark (Nifty 100)', line=dict(color='red', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("🔍 Deep Dive Snapshot")
        s_date = nav.index.min() + pd.Timedelta(days=370)
        s_idx = nav.index.searchsorted(s_date)
        idxs = list(range(s_idx, len(nav), QUARTERLY_STEP))
        dates = [nav.index[i] for i in idxs if i < len(nav)-1 and nav.index[i] <= LAST_REBALANCE_DATE]
        d_str = [d.strftime('%Y-%m-%d') for d in dates]
        sel_d = st.selectbox("Snapshot Date", d_str, key=f"dd_{strat_key}")
        
        if sel_d:
            df_snap = generate_snapshot_table(nav, pd.to_datetime(sel_d), hold, strat_key, maps, cust_w, mom_cfg, nifty)
            if not df_snap.empty:
                df_snap = df_snap[['name', 'Strategy Rank', 'Forward Return %', 'Actual Rank']]
                st.dataframe(df_snap.style.format({'Forward Return %': "{:.2f}%", 'Strategy Rank':"{:.0f}", 'Actual Rank':"{:.0f}"}).background_gradient(subset=['Forward Return %'], cmap='RdYlGn'), use_container_width=True)
    else:
        st.warning("No trades generated or insufficient data.")

def render_backtest_tab():
    st.header("🚀 Strategy Backtester")
    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("Category", list(FILE_MAPPING.keys()))
    top_n = c2.number_input("Top N Funds", 1, 20, DEFAULT_TOP_N)
    hold = c3.number_input("Holding Period (Days)", 20, 252, DEFAULT_HOLDING)
    
    tabs = st.tabs(["🚀 Momentum", "⚖️ Sharpe", "📉 Sortino", "📊 VAR", "🧠 Consistent Alpha", "🛠️ Custom"])
    
    with st.spinner("Loading Data..."):
        nav, maps = load_fund_data_raw(cat)
        nifty = load_nifty_data()
    
    if nav is None: st.error("Data not found."); return

    # 1. Momentum
    with tabs[0]:
        mc1, mc2, mc3 = st.columns(3)
        w3 = mc1.slider("3M Weight", 0.0, 1.0, 1.0, key="m_w3")
        w6 = mc2.slider("6M Weight", 0.0, 1.0, 1.0, key="m_w6")
        w12 = mc3.slider("12M Weight", 0.0, 1.0, 1.0, key="m_w12")
        adj = st.checkbox("Risk Adjust?", True, key="m_adj")
        tot = w3+w6+w12
        mom_cfg = {'w_3m':w3/tot if tot else 0, 'w_6m':w6/tot if tot else 0, 'w_12m':w12/tot if tot else 0, 'risk_adjust':adj}
        display_backtest_results(nav, maps, nifty, 'momentum', top_n, hold, {}, mom_cfg)

    # 2. Sharpe
    with tabs[1]:
        st.caption("Ranks funds by Sharpe Ratio (Excess Return / Volatility).")
        display_backtest_results(nav, maps, nifty, 'sharpe', top_n, hold, {}, {})

    # 3. Sortino
    with tabs[2]:
        st.caption("Ranks funds by Sortino Ratio (Excess Return / Downside Deviation).")
        display_backtest_results(nav, maps, nifty, 'sortino', top_n, hold, {}, {})

    # 4. VAR
    with tabs[3]:
        st.caption("Ranks funds by Volatility Adjusted Return (Annual Return / Annual Volatility).")
        display_backtest_results(nav, maps, nifty, 'var', top_n, hold, {}, {})

    # 5. Consistent Alpha (New Tab)
    with tabs[4]:
        st.info("🧠 **Consistent Alpha Strategy**")
        st.markdown("""
        **Formula:** Score = 60% × Rank(Rolling Win Rate) + 40% × Rank(Information Ratio)
        * **Win Rate:** Frequency of beating Nifty 100 on a rolling 1-year basis.
        * **Goal:** Finds funds that consistently outperform, not just those with one lucky spike.
        """)
        display_backtest_results(nav, maps, nifty, 'consistent_alpha', top_n, hold, {}, {})

    # 6. Custom
    with tabs[5]:
        st.info("Weights will be automatically normalized to sum to 1.0 (100%)")
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
                subset_cols = ['Avg Rolling Return (1Y)', 'Max Drawdown', 'Win Rate vs Nifty %', 'Avg Outperformance %', 'Avg Underperformance %']
                valid_subset = [c for c in subset_cols if c in df.columns]
                st.dataframe(df.style.format("{:.2f}", subset=valid_subset), use_container_width=True, height=600)
            else:
                df = calculate_quarterly_ranks(nav, maps)
                st.dataframe(df, use_container_width=True, height=600)

def main():
    t1, t2 = st.tabs(["📂 Category Explorer", "🚀 Strategy Backtester"])
    with t1: render_explorer_tab()
    with t2: render_backtest_tab()

if __name__ == "__main__":
    main()
