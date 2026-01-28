"""
Advanced Fund Analysis Dashboard with Factor Discovery Integration
===================================================================
This dashboard combines:
1. Factor Discovery Results (pre-computed)
2. Strategy Backtester with Data-Driven Factor Weights
3. Live Fund Selection Based on Discovered Factors

Run: streamlit run main_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats
import json

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Advanced Fund Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

RISK_FREE_RATE = 0.06
TRADING_DAYS_YEAR = 252
DAILY_RISK_FREE_RATE = (1 + RISK_FREE_RATE) ** (1/TRADING_DAYS_YEAR) - 1
DATA_DIR = "data"
RESULTS_DIR = "factor_discovery_results"
MAX_DATA_DATE = pd.Timestamp('2025-12-05')

FILE_MAPPING = {
    "Large Cap": "largecap_merged.xlsx",
    "Mid Cap": "midcap.xlsx",
    "Small Cap": "smallcap.xlsx",
    "Large & Mid Cap": "large_and_midcap_fund.xlsx",
    "Multi Cap": "MULTICAP.xlsx",
    "International": "international_merged.xlsx"
}

# ============================================================================
# KEY INSIGHTS FROM FACTOR DISCOVERY (Hardcoded from analysis)
# ============================================================================

# Top factors by category (from factor_importance_by_category.csv)
CATEGORY_TOP_FACTORS = {
    "Large Cap": ["information_ratio", "batting_avg", "alpha", "return_high_vol", "ma50_vs_ma200"],
    "Mid Cap": ["price_vs_ma200", "rank_persistence", "ma50_vs_ma200", "momentum_6m", "win_rate_down_days"],
    "Small Cap": ["cvar_95", "price_vs_ma200", "momentum_12m", "ma50_vs_ma200", "trend_r_squared"],
    "Large & Mid Cap": ["capture_ratio", "cvar_95", "var_95", "pct_positive_months", "return_low_vol"],
    "Multi Cap": ["momentum_3m", "avg_drawdown", "max_drawdown", "price_vs_ma200", "momentum_6m"],
    "International": ["return_low_vol", "sortino", "recovery_speed", "information_ratio", "sharpe"]
}

# Universal top factors (from factor_rankings_cross_category.csv - lowest avg rank)
UNIVERSAL_TOP_FACTORS = [
    ("information_ratio", 2.5),
    ("recovery_speed", 3.0),
    ("return_low_vol", 3.0),
    ("return_high_vol", 4.0),
    ("alpha", 4.5),
    ("cvar_95", 4.67),
    ("pct_positive_months", 5.5),
    ("batting_avg", 5.5),
    ("var_95", 5.67),
    ("price_vs_ma200", 6.4)
]

# Factor interpretation (higher is better or lower is better)
FACTOR_DIRECTION = {
    'momentum_3m': 1, 'momentum_6m': 1, 'momentum_12m': 1, 'momentum_12m_ex_1m': 1,
    'sharpe': 1, 'sortino': 1, 'calmar': 1, 'information_ratio': 1,
    'alpha': 1, 'batting_avg': 1, 'capture_ratio': 1, 'up_capture': 1,
    'rank_persistence': 1, 'pct_positive_months': 1, 'recovery_speed': 1,
    'trend_r_squared': 1, 'trend_slope': 1, 'price_vs_ma200': 1, 'price_vs_ma50': 1,
    'ma50_vs_ma200': 1, 'return_low_vol': 1, 'return_high_vol': 1,
    'regime_adaptability': 1, 'sharpe_consistency': 1, 'win_rate_down_days': 1,
    'skewness': 1,
    # Lower is better
    'volatility': -1, 'downside_vol': -1, 'vol_of_vol': -1,
    'max_drawdown': -1, 'avg_drawdown': -1, 'drawdown_duration_pct': -1,
    'down_capture': -1, 'beta': -1, 'kurtosis': -1,
    'var_95': -1, 'cvar_95': -1
}

# ============================================================================
# 2. DATA LOADING
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

@st.cache_data
def load_fund_data_raw(category_key):
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

@st.cache_data
def load_factor_discovery_results():
    """Load pre-computed factor discovery results."""
    results = {}
    
    try:
        fi_path = os.path.join(RESULTS_DIR, 'factor_importance_by_category.csv')
        if os.path.exists(fi_path):
            results['factor_importance'] = pd.read_csv(fi_path, index_col=0)
        
        ranks_path = os.path.join(RESULTS_DIR, 'factor_rankings_cross_category.csv')
        if os.path.exists(ranks_path):
            results['cross_category_ranks'] = pd.read_csv(ranks_path, index_col=0)
        
        summary_path = os.path.join(RESULTS_DIR, 'category_summary.csv')
        if os.path.exists(summary_path):
            results['category_summary'] = pd.read_csv(summary_path)
        
        winners_path = os.path.join(RESULTS_DIR, 'period_winners.csv')
        if os.path.exists(winners_path):
            results['period_winners'] = pd.read_csv(winners_path)
        
        meta_path = os.path.join(RESULTS_DIR, 'run_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                results['metadata'] = json.load(f)
        
        results['loaded'] = True
    except:
        results['loaded'] = False
    
    return results

# ============================================================================
# 3. FACTOR CALCULATIONS
# ============================================================================

def calculate_max_dd(series):
    if len(series) < 10 or series.isna().all(): return None
    try:
        comp_ret = (1 + series.pct_change().fillna(0)).cumprod()
        peak = comp_ret.expanding(min_periods=1).max()
        dd = (comp_ret / peak) - 1
        return dd.min()
    except: return None

def calculate_recovery_speed(series):
    if len(series) < 126: return None
    cum_max = series.expanding(min_periods=1).max()
    drawdowns = (series / cum_max) - 1
    in_drawdown = False
    dd_start = None
    recovery_times = []
    for i, (date, dd) in enumerate(drawdowns.items()):
        if not in_drawdown and dd < -0.05:
            in_drawdown = True
            dd_start = i
        elif in_drawdown and dd >= -0.01:
            recovery_time = i - dd_start
            recovery_times.append(recovery_time)
            in_drawdown = False
    if not recovery_times: return None
    avg_recovery = np.mean(recovery_times)
    return 1.0 / avg_recovery * 100 if avg_recovery > 0 else None

def calculate_rank_persistence(series, nav_df, lookback_quarters=4):
    if len(series) < 252: return None
    current_date = series.index[-1]
    persist_score = 0
    valid_quarters = 0
    fund_id = series.name if hasattr(series, 'name') else None
    if fund_id is None: return None
    
    for q in range(1, lookback_quarters + 1):
        q_start = current_date - pd.Timedelta(days=91 * q)
        q_end = current_date - pd.Timedelta(days=91 * (q - 1))
        try:
            idx_start = nav_df.index.asof(q_start)
            idx_end = nav_df.index.asof(q_end)
            if pd.isna(idx_start) or pd.isna(idx_end): continue
            q_returns = (nav_df.loc[idx_end] / nav_df.loc[idx_start]) - 1
            q_returns = q_returns.dropna()
            if len(q_returns) < 5 or fund_id not in q_returns.index: continue
            rank_pct = q_returns.rank(pct=True, ascending=True)[fund_id]
            if rank_pct >= 0.75: persist_score += 1.0
            elif rank_pct >= 0.5: persist_score += 0.5
            elif rank_pct >= 0.25: persist_score += 0.0
            else: persist_score -= 0.5
            valid_quarters += 1
        except: continue
    return persist_score / valid_quarters if valid_quarters > 0 else None

def calculate_all_factors(series, bench_series=None, nav_df=None):
    """Calculate ALL factors for a fund."""
    factors = {}
    if len(series) < 126: return factors
    
    returns = series.pct_change().dropna()
    bench_rets = None
    if bench_series is not None:
        common_idx = returns.index.intersection(bench_series.index)
        if len(common_idx) > 30:
            bench_rets = bench_series.pct_change().dropna()
            bench_rets = bench_rets.loc[bench_rets.index.intersection(returns.index)]
    
    # Momentum
    if len(series) >= 63:
        factors['momentum_3m'] = (series.iloc[-1] / series.iloc[-63]) - 1 if series.iloc[-63] > 0 else np.nan
    if len(series) >= 126:
        factors['momentum_6m'] = (series.iloc[-1] / series.iloc[-126]) - 1 if series.iloc[-126] > 0 else np.nan
    if len(series) >= 252:
        factors['momentum_12m'] = (series.iloc[-1] / series.iloc[-252]) - 1 if series.iloc[-252] > 0 else np.nan
        factors['momentum_12m_ex_1m'] = (series.iloc[-21] / series.iloc[-252]) - 1 if series.iloc[-252] > 0 else np.nan
    
    # Risk-adjusted
    if len(returns) >= 126 and returns.std() > 0:
        excess_ret = returns - DAILY_RISK_FREE_RATE
        factors['sharpe'] = (excess_ret.mean() / returns.std()) * np.sqrt(TRADING_DAYS_YEAR)
    
    if len(returns) >= 126:
        downside = returns[returns < 0]
        if len(downside) > 10 and downside.std() > 0:
            factors['sortino'] = ((returns - DAILY_RISK_FREE_RATE).mean() / downside.std()) * np.sqrt(TRADING_DAYS_YEAR)
    
    if len(series) >= 252:
        max_dd = calculate_max_dd(series)
        if max_dd is not None and max_dd < 0:
            years = len(series) / TRADING_DAYS_YEAR
            cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1 if years > 0 else 0
            factors['calmar'] = cagr / abs(max_dd)
    
    # Volatility
    if len(returns) >= 63:
        factors['volatility'] = returns.std() * np.sqrt(TRADING_DAYS_YEAR)
        downside = returns[returns < 0]
        if len(downside) > 10:
            factors['downside_vol'] = downside.std() * np.sqrt(TRADING_DAYS_YEAR)
    
    if len(returns) >= 126:
        rolling_vol = returns.rolling(21).std()
        factors['vol_of_vol'] = rolling_vol.std() if rolling_vol.std() > 0 else np.nan
    
    # Drawdown
    max_dd = calculate_max_dd(series)
    if max_dd is not None:
        factors['max_drawdown'] = max_dd
    
    if len(series) >= 126:
        cum_max = series.expanding(min_periods=1).max()
        drawdowns = (series / cum_max) - 1
        factors['avg_drawdown'] = drawdowns.mean()
        factors['drawdown_duration_pct'] = (series < cum_max).sum() / len(series)
    
    recovery = calculate_recovery_speed(series)
    if recovery is not None:
        factors['recovery_speed'] = recovery
    
    # Benchmark relative
    if bench_rets is not None and len(bench_rets) > 60:
        common_idx = returns.index.intersection(bench_rets.index)
        f_ret = returns.loc[common_idx]
        b_ret = bench_rets.loc[common_idx]
        
        if len(common_idx) > 30:
            cov_matrix = np.cov(f_ret, b_ret)
            if cov_matrix[1, 1] > 0:
                factors['beta'] = cov_matrix[0, 1] / cov_matrix[1, 1]
                mean_fund = f_ret.mean() * TRADING_DAYS_YEAR
                mean_bench = b_ret.mean() * TRADING_DAYS_YEAR
                factors['alpha'] = mean_fund - (RISK_FREE_RATE + factors['beta'] * (mean_bench - RISK_FREE_RATE))
        
        active_ret = f_ret - b_ret
        tracking_error = active_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
        if tracking_error > 0:
            factors['information_ratio'] = (active_ret.mean() * TRADING_DAYS_YEAR) / tracking_error
        
        up_market = b_ret[b_ret > 0]
        if len(up_market) > 10 and up_market.mean() != 0:
            factors['up_capture'] = f_ret.loc[up_market.index].mean() / up_market.mean()
        
        down_market = b_ret[b_ret < 0]
        if len(down_market) > 10 and down_market.mean() != 0:
            factors['down_capture'] = f_ret.loc[down_market.index].mean() / down_market.mean()
        
        if 'up_capture' in factors and 'down_capture' in factors and factors['down_capture'] > 0:
            factors['capture_ratio'] = factors['up_capture'] / factors['down_capture']
        
        factors['batting_avg'] = (f_ret > b_ret).sum() / len(common_idx)
        
        if len(down_market) > 10:
            factors['win_rate_down_days'] = (f_ret.loc[down_market.index] > down_market).sum() / len(down_market)
    
    # Consistency
    if len(returns) >= 126:
        monthly_rets = series.resample('ME').last().pct_change().dropna()
        if len(monthly_rets) > 3:
            factors['pct_positive_months'] = (monthly_rets > 0).sum() / len(monthly_rets)
    
    if len(returns) >= 252:
        rolling_sharpe = returns.rolling(63).apply(
            lambda x: (x.mean() - DAILY_RISK_FREE_RATE) / x.std() * np.sqrt(TRADING_DAYS_YEAR) if x.std() > 0 else 0
        )
        if rolling_sharpe.mean() != 0:
            factors['sharpe_consistency'] = 1 - (rolling_sharpe.std() / abs(rolling_sharpe.mean()))
    
    if nav_df is not None and len(series) >= 252:
        rank_persist = calculate_rank_persistence(series, nav_df)
        if rank_persist is not None:
            factors['rank_persistence'] = rank_persist
    
    # Trend quality
    if len(series) >= 126:
        y = np.log(series.iloc[-126:].values)
        x = np.arange(len(y))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            factors['trend_r_squared'] = r_value ** 2
            factors['trend_slope'] = slope * TRADING_DAYS_YEAR
        except: pass
    
    if len(series) >= 200:
        ma_50 = series.rolling(50).mean().iloc[-1]
        ma_200 = series.rolling(200).mean().iloc[-1]
        current = series.iloc[-1]
        if ma_50 > 0: factors['price_vs_ma50'] = (current / ma_50) - 1
        if ma_200 > 0:
            factors['price_vs_ma200'] = (current / ma_200) - 1
            factors['ma50_vs_ma200'] = (ma_50 / ma_200) - 1
    
    # Higher moments
    if len(returns) >= 126:
        factors['skewness'] = returns.skew()
        factors['kurtosis'] = returns.kurtosis()
        factors['var_95'] = returns.quantile(0.05)
        var_threshold = returns.quantile(0.05)
        factors['cvar_95'] = returns[returns <= var_threshold].mean()
    
    # Regime
    if len(returns) >= 252 and bench_rets is not None:
        bench_vol = bench_rets.rolling(21).std()
        median_vol = bench_vol.median()
        common_idx = returns.index.intersection(bench_vol.index)
        if len(common_idx) > 100:
            ret_common = returns.loc[common_idx]
            high_vol_idx = bench_vol.loc[common_idx][bench_vol.loc[common_idx] > median_vol].index
            low_vol_idx = bench_vol.loc[common_idx][bench_vol.loc[common_idx] <= median_vol].index
            if len(high_vol_idx) > 20 and len(low_vol_idx) > 20:
                factors['return_high_vol'] = ret_common.loc[high_vol_idx].mean() * TRADING_DAYS_YEAR
                factors['return_low_vol'] = ret_common.loc[low_vol_idx].mean() * TRADING_DAYS_YEAR
                factors['regime_adaptability'] = min(factors['return_high_vol'], factors['return_low_vol'])
    
    return factors

# ============================================================================
# 4. DATA-DRIVEN FUND SELECTION
# ============================================================================

def select_funds_data_driven(nav_df, benchmark_series, category, scheme_map, top_n=3):
    """
    Select top funds using discovered factors for the specific category.
    Returns ranked funds with their scores and factor values.
    """
    # Get category-specific top factors
    cat_factors = CATEGORY_TOP_FACTORS.get(category, CATEGORY_TOP_FACTORS["Large Cap"])
    
    # Calculate factors for all funds
    fund_data = []
    
    for col in nav_df.columns:
        series = nav_df[col].dropna()
        if len(series) < 200:
            continue
        
        series.name = col
        factors = calculate_all_factors(series, benchmark_series, nav_df)
        
        if not factors:
            continue
        
        # Calculate composite score using category-specific factors
        score = 0
        valid_factors = 0
        factor_values = {}
        
        for i, factor in enumerate(cat_factors):
            if factor in factors and not pd.isna(factors[factor]):
                # Weight: first factor gets highest weight
                weight = 1.0 / (i + 1)
                direction = FACTOR_DIRECTION.get(factor, 1)
                
                # Normalize by storing raw value
                factor_values[factor] = factors[factor]
                score += factors[factor] * weight * direction
                valid_factors += 1
        
        if valid_factors >= 2:
            fund_data.append({
                'fund_id': col,
                'fund_name': scheme_map.get(col, col),
                'composite_score': score,
                'valid_factors': valid_factors,
                **factor_values
            })
    
    if not fund_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(fund_data)
    
    # Rank by composite score
    df['rank'] = df['composite_score'].rank(ascending=False, method='min')
    df = df.sort_values('rank')
    
    return df

# ============================================================================
# 5. BACKTEST ENGINE
# ============================================================================

def run_data_driven_backtest(nav_df, benchmark_series, category, holding_period=252, top_n=3, scheme_map=None):
    """Run backtest using category-specific discovered factors."""
    
    cat_factors = CATEGORY_TOP_FACTORS.get(category, CATEGORY_TOP_FACTORS["Large Cap"])
    
    min_history = 370
    start_date = nav_df.index.min() + pd.Timedelta(days=min_history)
    
    if start_date >= nav_df.index.max():
        return None, None, None
    
    start_idx = nav_df.index.searchsorted(start_date)
    rebal_idx = list(range(start_idx, len(nav_df) - 1, holding_period))
    
    if not rebal_idx:
        return None, None, None
    
    history = []
    eq_curve = [{'date': nav_df.index[rebal_idx[0]], 'value': 100.0}]
    bench_curve = [{'date': nav_df.index[rebal_idx[0]], 'value': 100.0}]
    cap = 100.0
    b_cap = 100.0
    
    for i in rebal_idx:
        date = nav_df.index[i]
        hist_data = nav_df.loc[:date]
        bench_hist = benchmark_series.loc[:date] if benchmark_series is not None else None
        
        # Score all funds using discovered factors
        fund_scores = {}
        
        for col in nav_df.columns:
            series = hist_data[col].dropna()
            if len(series) < 200:
                continue
            series.name = col
            factors = calculate_all_factors(series, bench_hist, hist_data)
            
            if not factors:
                continue
            
            score = 0
            valid = 0
            for idx, factor in enumerate(cat_factors):
                if factor in factors and not pd.isna(factors[factor]):
                    weight = 1.0 / (idx + 1)
                    direction = FACTOR_DIRECTION.get(factor, 1)
                    score += factors[factor] * weight * direction
                    valid += 1
            
            if valid >= 2:
                fund_scores[col] = score
        
        # Select top N
        if fund_scores:
            selected = sorted(fund_scores, key=fund_scores.get, reverse=True)[:top_n]
        else:
            selected = []
        
        # Calculate returns
        entry = i + 1
        exit_i = min(i + 1 + holding_period, len(nav_df) - 1)
        
        b_ret = 0.0
        if benchmark_series is not None:
            try:
                b_ret = (benchmark_series.asof(nav_df.index[exit_i]) / benchmark_series.asof(nav_df.index[entry])) - 1
            except: pass
        
        port_ret = 0.0
        hit_rate = 0.0
        
        if selected:
            period_ret_all = (nav_df.iloc[exit_i] / nav_df.iloc[entry]) - 1
            port_ret = period_ret_all[selected].mean()
            
            actual_top = period_ret_all.dropna().nlargest(top_n).index.tolist()
            matches = len(set(selected).intersection(set(actual_top)))
            hit_rate = matches / top_n
        
        cap *= (1 + (port_ret if not pd.isna(port_ret) else 0))
        b_cap *= (1 + b_ret)
        
        history.append({
            'date': date,
            'selected': [scheme_map.get(s, s) for s in selected] if scheme_map else selected,
            'return': port_ret,
            'hit_rate': hit_rate
        })
        
        eq_curve.append({'date': nav_df.index[exit_i], 'value': cap})
        bench_curve.append({'date': nav_df.index[exit_i], 'value': b_cap})
    
    return pd.DataFrame(history), pd.DataFrame(eq_curve), pd.DataFrame(bench_curve)

# ============================================================================
# 6. UI PAGES
# ============================================================================

def render_home_page():
    """Main dashboard home with key insights."""
    st.title("📈 Advanced Fund Analysis Dashboard")
    
    st.markdown("""
    ### 🧬 Powered by Factor Discovery Analysis
    
    This dashboard uses **data-driven insights** from analyzing 17+ years of mutual fund data 
    to identify which factors actually predict winning funds.
    """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Categories Analyzed", "6")
    col2.metric("Factors Tested", "37")
    col3.metric("Time Periods", "69 quarters")
    col4.metric("Holding Period", "252 days")
    
    st.divider()
    
    # Universal top factors
    st.subheader("🎯 Top Universal Winning Factors")
    st.markdown("*These factors consistently predicted top performers across ALL categories:*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (factor, avg_rank) in enumerate(UNIVERSAL_TOP_FACTORS[:5], 1):
            direction = "📈 Higher is better" if FACTOR_DIRECTION.get(factor, 1) == 1 else "📉 Lower is better"
            st.markdown(f"**{i}. {factor}** (avg rank: {avg_rank:.1f}) - {direction}")
    
    with col2:
        for i, (factor, avg_rank) in enumerate(UNIVERSAL_TOP_FACTORS[5:10], 6):
            direction = "📈 Higher is better" if FACTOR_DIRECTION.get(factor, 1) == 1 else "📉 Lower is better"
            st.markdown(f"**{i}. {factor}** (avg rank: {avg_rank:.1f}) - {direction}")
    
    st.divider()
    
    # Category-specific insights
    st.subheader("📊 Top Factors by Category")
    
    cat_data = []
    for cat, factors in CATEGORY_TOP_FACTORS.items():
        cat_data.append({
            'Category': cat,
            'Factor 1': factors[0],
            'Factor 2': factors[1],
            'Factor 3': factors[2]
        })
    
    st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)
    
    st.info("""
    💡 **How to use this dashboard:**
    1. **Live Selection**: Get current fund recommendations based on discovered factors
    2. **Backtest**: See how the data-driven strategy would have performed historically
    3. **Factor Analysis**: Deep dive into factor importance by category
    """)

def render_live_selection_page():
    """Live fund selection based on discovered factors."""
    st.title("🎯 Live Fund Selection")
    
    st.markdown("""
    Select funds using the **discovered winning factors** for each category.
    The model uses category-specific factor weights learned from historical analysis.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category = st.selectbox("Select Category", list(FILE_MAPPING.keys()))
    with col2:
        top_n = st.number_input("Number of Funds to Select", 1, 10, 3)
    with col3:
        run_selection = st.button("🔍 Find Top Funds", type="primary", use_container_width=True)
    
    # Show category-specific factors
    st.markdown(f"**Factors used for {category}:** {', '.join(CATEGORY_TOP_FACTORS[category])}")
    
    if run_selection:
        with st.spinner("Loading data and calculating factors..."):
            nav_df, scheme_map = load_fund_data_raw(category)
            benchmark = load_nifty_data()
        
        if nav_df is None:
            st.error("Could not load data")
            return
        
        with st.spinner("Ranking funds..."):
            results = select_funds_data_driven(nav_df, benchmark, category, scheme_map, top_n)
        
        if results.empty:
            st.warning("No funds meet the criteria")
            return
        
        st.success(f"✅ Analyzed {len(nav_df.columns)} funds")
        
        # Display top picks
        st.subheader(f"🏆 Top {top_n} Recommended Funds")
        
        top_picks = results.head(top_n)
        
        for idx, row in top_picks.iterrows():
            with st.expander(f"#{int(row['rank'])} - {row['fund_name']}", expanded=True):
                cols = st.columns(len(CATEGORY_TOP_FACTORS[category]))
                for i, factor in enumerate(CATEGORY_TOP_FACTORS[category]):
                    if factor in row:
                        value = row[factor]
                        if not pd.isna(value):
                            # Format based on factor type
                            if 'momentum' in factor or 'return' in factor or 'alpha' in factor:
                                cols[i].metric(factor, f"{value:.2%}")
                            elif 'ratio' in factor or 'sharpe' in factor or 'sortino' in factor:
                                cols[i].metric(factor, f"{value:.2f}")
                            else:
                                cols[i].metric(factor, f"{value:.3f}")
        
        st.divider()
        
        # Full rankings
        st.subheader("📋 Complete Fund Rankings")
        
        display_cols = ['rank', 'fund_name', 'composite_score'] + CATEGORY_TOP_FACTORS[category]
        display_cols = [c for c in display_cols if c in results.columns]
        
        st.dataframe(
            results[display_cols].style.format({
                'composite_score': '{:.4f}',
                'rank': '{:.0f}'
            }),
            use_container_width=True,
            height=400
        )

def render_backtest_page():
    """Backtest the data-driven strategy."""
    st.title("📈 Strategy Backtest")
    
    st.markdown("""
    Backtest the **data-driven fund selection strategy** using discovered factors.
    See how selecting funds based on category-specific winning factors would have performed.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        category = st.selectbox("Category", list(FILE_MAPPING.keys()), key="bt_cat")
    with col2:
        holding_period = st.selectbox("Holding Period", [126, 252, 378, 504], index=1,
                                       format_func=lambda x: f"{x} days (~{x//21} months)")
    with col3:
        top_n = st.number_input("Funds to Select", 1, 10, 3, key="bt_topn")
    with col4:
        run_bt = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    st.markdown(f"**Strategy factors for {category}:** {', '.join(CATEGORY_TOP_FACTORS[category])}")
    
    if run_bt:
        with st.spinner("Loading data..."):
            nav_df, scheme_map = load_fund_data_raw(category)
            benchmark = load_nifty_data()
        
        if nav_df is None:
            st.error("Could not load data")
            return
        
        with st.spinner("Running backtest..."):
            history, eq_curve, bench_curve = run_data_driven_backtest(
                nav_df, benchmark, category, holding_period, top_n, scheme_map
            )
        
        if history is None or eq_curve.empty:
            st.error("Backtest failed - insufficient data")
            return
        
        # Results
        start_date = eq_curve.iloc[0]['date']
        end_date = eq_curve.iloc[-1]['date']
        years = (end_date - start_date).days / 365.25
        
        strat_fin = eq_curve.iloc[-1]['value']
        strat_cagr = (strat_fin/100)**(1/years)-1 if years > 0 else 0
        
        bench_fin = bench_curve.iloc[-1]['value'] if not bench_curve.empty else 100
        bench_cagr = (bench_fin/100)**(1/years)-1 if years > 0 else 0
        
        avg_hit_rate = history['hit_rate'].mean()
        
        st.success("✅ Backtest Complete!")
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Strategy CAGR", f"{strat_cagr:.2%}")
        col2.metric("Benchmark CAGR", f"{bench_cagr:.2%}")
        col3.metric("Outperformance", f"{(strat_cagr - bench_cagr):.2%}")
        col4.metric("Avg Hit Rate", f"{avg_hit_rate:.1%}")
        col5.metric("Total Return", f"{strat_fin - 100:.1f}%")
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_curve['date'], y=eq_curve['value'],
            name='Data-Driven Strategy', line=dict(color='green', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=bench_curve['date'], y=bench_curve['value'],
            name='Benchmark (Nifty 100)', line=dict(color='red', dash='dot', width=2)
        ))
        fig.update_layout(
            title=f'{category} - Data-Driven Strategy vs Benchmark',
            yaxis_title='Value (100 = Start)',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade history
        st.subheader("📋 Trade History")
        
        history_display = history.copy()
        history_display['return'] = history_display['return'].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "")
        history_display['hit_rate'] = history_display['hit_rate'].apply(lambda x: f"{x:.0%}")
        history_display['selected'] = history_display['selected'].apply(lambda x: ', '.join([s[:25] for s in x]) if x else '')
        
        st.dataframe(history_display, use_container_width=True, height=300)

def render_factor_analysis_page():
    """Deep dive into factor importance."""
    st.title("🔬 Factor Analysis")
    
    # Load pre-computed results
    results = load_factor_discovery_results()
    
    if not results.get('loaded') or 'factor_importance' not in results:
        st.warning("Pre-computed factor discovery results not found. Showing hardcoded insights.")
        
        # Show hardcoded insights
        st.subheader("Category-Specific Top Factors")
        
        for cat, factors in CATEGORY_TOP_FACTORS.items():
            with st.expander(f"📁 {cat}"):
                for i, factor in enumerate(factors, 1):
                    direction = "Higher is better" if FACTOR_DIRECTION.get(factor, 1) == 1 else "Lower is better"
                    st.markdown(f"{i}. **{factor}** - {direction}")
        
        return
    
    # Category selector
    category = st.selectbox("Select Category", list(FILE_MAPPING.keys()))
    
    # Filter factor importance for category
    fi = results['factor_importance']
    cat_fi = fi[fi['category'] == category].copy()
    cat_fi = cat_fi.sort_values('Predictive_Score', ascending=False)
    
    st.subheader(f"Factor Importance - {category}")
    
    # Top factors table
    display_cols = ['Predictive_Score', 'Pct_Winners_Higher', 'Avg_Winners_TopQuartile', 
                    'Pct_Significant', 'Avg_Effect_Size', 'Num_Periods']
    
    st.dataframe(
        cat_fi[display_cols].head(20).style.format({
            'Predictive_Score': '{:.3f}',
            'Pct_Winners_Higher': '{:.1%}',
            'Avg_Winners_TopQuartile': '{:.1%}',
            'Pct_Significant': '{:.1%}',
            'Avg_Effect_Size': '{:.3f}'
        }).background_gradient(subset=['Predictive_Score'], cmap='Greens'),
        use_container_width=True,
        height=500
    )
    
    # Chart
    top_15 = cat_fi.head(15)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='% Winners Higher',
        x=top_15.index,
        y=top_15['Pct_Winners_Higher'] * 100,
        marker_color='steelblue'
    ))
    fig.add_trace(go.Bar(
        name='% Winners in Top Quartile',
        x=top_15.index,
        y=top_15['Avg_Winners_TopQuartile'] * 100,
        marker_color='forestgreen'
    ))
    fig.update_layout(
        barmode='group',
        title=f'Factor Predictive Power - {category}',
        xaxis_tickangle=-45,
        yaxis_title='Percentage (%)',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Winner DNA profile
    st.subheader("🧬 Winner DNA Profile")
    
    col1, col2 = st.columns(2)
    
    high_factors = cat_fi[cat_fi['Pct_Winners_Higher'] > 0.55].head(5)
    low_factors = cat_fi[cat_fi['Pct_Winners_Higher'] < 0.45].head(5)
    
    with col1:
        st.markdown("**Winners typically had HIGHER:**")
        for factor, row in high_factors.iterrows():
            pct = row['Pct_Winners_Higher']
            st.markdown(f"✅ **{factor}** ({pct:.0%} of time)")
    
    with col2:
        st.markdown("**Winners typically had LOWER:**")
        for factor, row in low_factors.iterrows():
            pct = 1 - row['Pct_Winners_Higher']
            st.markdown(f"📉 **{factor}** ({pct:.0%} of time)")

def render_historical_winners_page():
    """Show historical period winners."""
    st.title("📅 Historical Winners")
    
    results = load_factor_discovery_results()
    
    if not results.get('loaded') or 'period_winners' not in results:
        st.warning("Period winners data not found.")
        return
    
    winners_df = results['period_winners']
    
    # Category filter
    category = st.selectbox("Select Category", winners_df['category'].unique().tolist())
    
    filtered = winners_df[winners_df['category'] == category].copy()
    filtered = filtered.sort_values('period', ascending=False)
    
    st.subheader(f"Historical Top 5 Winners - {category}")
    st.markdown("*These are the actual top performing funds for each period*")
    
    st.dataframe(filtered[['period', 'top_funds', 'top_returns']], use_container_width=True, height=500)
    
    # Download
    csv = filtered.to_csv(index=False)
    st.download_button(
        "📥 Download Historical Winners",
        csv,
        f"historical_winners_{category}.csv",
        "text/csv"
    )

# ============================================================================
# 7. MAIN APP
# ============================================================================

def main():
    # Sidebar navigation
    st.sidebar.title("📈 Fund Analysis")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Home", "🎯 Live Selection", "📈 Backtest", "🔬 Factor Analysis", "📅 Historical Winners"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This dashboard uses **factor discovery** to find which metrics actually predict winning mutual funds.
    
    Based on analysis of:
    - 6 fund categories
    - 37 factors tested
    - 17+ years of data
    """)
    
    # Render selected page
    if page == "🏠 Home":
        render_home_page()
    elif page == "🎯 Live Selection":
        render_live_selection_page()
    elif page == "📈 Backtest":
        render_backtest_page()
    elif page == "🔬 Factor Analysis":
        render_factor_analysis_page()
    elif page == "📅 Historical Winners":
        render_historical_winners_page()

if __name__ == "__main__":
    main()
