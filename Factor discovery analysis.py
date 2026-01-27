import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="Factor Discovery - Winner DNA Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
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
# 3. ALL FACTOR CALCULATIONS
# ============================================================================

def calculate_all_factors(series, bench_series=None, nav_df=None):
    """
    Calculate ALL possible factors for a fund at a given point in time.
    Returns a dictionary of factor_name: value
    """
    factors = {}
    
    if len(series) < 126:
        return factors
    
    returns = series.pct_change().dropna()
    
    # Get benchmark returns if available
    bench_rets = None
    if bench_series is not None:
        common_idx = returns.index.intersection(bench_series.index)
        if len(common_idx) > 30:
            bench_rets = bench_series.pct_change().dropna()
            bench_rets = bench_rets.loc[bench_rets.index.intersection(returns.index)]
    
    # ==========================================
    # RETURN-BASED FACTORS
    # ==========================================
    
    # 1. Momentum (various windows)
    if len(series) >= 63:
        factors['momentum_3m'] = (series.iloc[-1] / series.iloc[-63]) - 1 if series.iloc[-63] > 0 else np.nan
    
    if len(series) >= 126:
        factors['momentum_6m'] = (series.iloc[-1] / series.iloc[-126]) - 1 if series.iloc[-126] > 0 else np.nan
    
    if len(series) >= 252:
        factors['momentum_12m'] = (series.iloc[-1] / series.iloc[-252]) - 1 if series.iloc[-252] > 0 else np.nan
    
    # 2. Momentum excluding recent month (avoids short-term reversal)
    if len(series) >= 252:
        factors['momentum_12m_ex_1m'] = (series.iloc[-21] / series.iloc[-252]) - 1 if series.iloc[-252] > 0 else np.nan
    
    # ==========================================
    # RISK-ADJUSTED FACTORS
    # ==========================================
    
    # 3. Sharpe Ratio
    if len(returns) >= 126 and returns.std() > 0:
        excess_ret = returns - DAILY_RISK_FREE_RATE
        factors['sharpe'] = (excess_ret.mean() / returns.std()) * np.sqrt(TRADING_DAYS_YEAR)
    
    # 4. Sortino Ratio
    if len(returns) >= 126:
        downside = returns[returns < 0]
        if len(downside) > 10 and downside.std() > 0:
            excess_ret = (returns - DAILY_RISK_FREE_RATE).mean()
            factors['sortino'] = (excess_ret / downside.std()) * np.sqrt(TRADING_DAYS_YEAR)
    
    # 5. Calmar Ratio
    if len(series) >= 252:
        max_dd = calculate_max_dd(series)
        if max_dd is not None and max_dd < 0:
            years = len(series) / TRADING_DAYS_YEAR
            cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1 if years > 0 else 0
            factors['calmar'] = cagr / abs(max_dd)
    
    # ==========================================
    # VOLATILITY FACTORS
    # ==========================================
    
    # 6. Volatility (annualized)
    if len(returns) >= 63:
        factors['volatility'] = returns.std() * np.sqrt(TRADING_DAYS_YEAR)
    
    # 7. Downside Volatility
    if len(returns) >= 63:
        downside = returns[returns < 0]
        if len(downside) > 10:
            factors['downside_vol'] = downside.std() * np.sqrt(TRADING_DAYS_YEAR)
    
    # 8. Volatility of Volatility (stability of risk)
    if len(returns) >= 126:
        rolling_vol = returns.rolling(21).std()
        factors['vol_of_vol'] = rolling_vol.std() if rolling_vol.std() > 0 else np.nan
    
    # ==========================================
    # DRAWDOWN FACTORS
    # ==========================================
    
    # 9. Max Drawdown
    max_dd = calculate_max_dd(series)
    if max_dd is not None:
        factors['max_drawdown'] = max_dd
    
    # 10. Average Drawdown
    if len(series) >= 126:
        cum_max = series.expanding(min_periods=1).max()
        drawdowns = (series / cum_max) - 1
        factors['avg_drawdown'] = drawdowns.mean()
    
    # 11. Drawdown Duration (avg days in drawdown)
    if len(series) >= 126:
        cum_max = series.expanding(min_periods=1).max()
        in_drawdown = series < cum_max
        factors['drawdown_duration_pct'] = in_drawdown.sum() / len(series)
    
    # 12. Recovery Speed (inverse of avg recovery time)
    recovery_speed = calculate_recovery_speed(series)
    if recovery_speed is not None:
        factors['recovery_speed'] = recovery_speed
    
    # ==========================================
    # BENCHMARK-RELATIVE FACTORS
    # ==========================================
    
    if bench_rets is not None and len(bench_rets) > 60:
        common_idx = returns.index.intersection(bench_rets.index)
        f_ret = returns.loc[common_idx]
        b_ret = bench_rets.loc[common_idx]
        
        # 13. Beta
        if len(common_idx) > 30:
            cov_matrix = np.cov(f_ret, b_ret)
            if cov_matrix[1, 1] > 0:
                factors['beta'] = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        # 14. Alpha (Jensen's)
        if 'beta' in factors:
            mean_fund = f_ret.mean() * TRADING_DAYS_YEAR
            mean_bench = b_ret.mean() * TRADING_DAYS_YEAR
            factors['alpha'] = mean_fund - (RISK_FREE_RATE + factors['beta'] * (mean_bench - RISK_FREE_RATE))
        
        # 15. Information Ratio
        active_ret = f_ret - b_ret
        tracking_error = active_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
        if tracking_error > 0:
            factors['information_ratio'] = (active_ret.mean() * TRADING_DAYS_YEAR) / tracking_error
        
        # 16. Up Capture
        up_market = b_ret[b_ret > 0]
        if len(up_market) > 10 and up_market.mean() != 0:
            factors['up_capture'] = f_ret.loc[up_market.index].mean() / up_market.mean()
        
        # 17. Down Capture
        down_market = b_ret[b_ret < 0]
        if len(down_market) > 10 and down_market.mean() != 0:
            factors['down_capture'] = f_ret.loc[down_market.index].mean() / down_market.mean()
        
        # 18. Capture Ratio
        if 'up_capture' in factors and 'down_capture' in factors:
            if factors['down_capture'] > 0:
                factors['capture_ratio'] = factors['up_capture'] / factors['down_capture']
        
        # 19. Batting Average (% of periods beating benchmark)
        factors['batting_avg'] = (f_ret > b_ret).sum() / len(common_idx)
        
        # 20. Win Rate on Down Days
        if len(down_market) > 10:
            fund_on_down = f_ret.loc[down_market.index]
            factors['win_rate_down_days'] = (fund_on_down > down_market).sum() / len(down_market)
    
    # ==========================================
    # CONSISTENCY FACTORS
    # ==========================================
    
    # 21. Return Consistency (% of positive months)
    if len(returns) >= 126:
        monthly_rets = series.resample('M').last().pct_change().dropna()
        if len(monthly_rets) > 3:
            factors['pct_positive_months'] = (monthly_rets > 0).sum() / len(monthly_rets)
    
    # 22. Rolling Sharpe Consistency
    if len(returns) >= 252:
        rolling_sharpe = returns.rolling(63).apply(
            lambda x: (x.mean() - DAILY_RISK_FREE_RATE) / x.std() * np.sqrt(TRADING_DAYS_YEAR) if x.std() > 0 else 0
        )
        factors['sharpe_consistency'] = 1 - (rolling_sharpe.std() / abs(rolling_sharpe.mean())) if rolling_sharpe.mean() != 0 else 0
    
    # 23. Rank Persistence (if nav_df provided)
    if nav_df is not None and len(series) >= 252:
        rank_persist = calculate_rank_persistence(series, nav_df)
        if rank_persist is not None:
            factors['rank_persistence'] = rank_persist
    
    # ==========================================
    # TREND QUALITY FACTORS
    # ==========================================
    
    # 24. Trend R-squared (quality of trend)
    if len(series) >= 126:
        y = np.log(series.iloc[-126:].values)
        x = np.arange(len(y))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            factors['trend_r_squared'] = r_value ** 2
            factors['trend_slope'] = slope * TRADING_DAYS_YEAR  # Annualized
        except:
            pass
    
    # 25. Price vs Moving Averages
    if len(series) >= 200:
        ma_50 = series.rolling(50).mean().iloc[-1]
        ma_200 = series.rolling(200).mean().iloc[-1]
        current = series.iloc[-1]
        
        factors['price_vs_ma50'] = (current / ma_50) - 1 if ma_50 > 0 else np.nan
        factors['price_vs_ma200'] = (current / ma_200) - 1 if ma_200 > 0 else np.nan
        factors['ma50_vs_ma200'] = (ma_50 / ma_200) - 1 if ma_200 > 0 else np.nan
    
    # ==========================================
    # HIGHER MOMENT FACTORS
    # ==========================================
    
    # 26. Skewness (positive = more upside surprises)
    if len(returns) >= 126:
        factors['skewness'] = returns.skew()
    
    # 27. Kurtosis (lower = fewer extreme events)
    if len(returns) >= 126:
        factors['kurtosis'] = returns.kurtosis()
    
    # 28. VaR 95% (Value at Risk)
    if len(returns) >= 126:
        factors['var_95'] = returns.quantile(0.05)
    
    # 29. CVaR / Expected Shortfall
    if len(returns) >= 126:
        var_threshold = returns.quantile(0.05)
        factors['cvar_95'] = returns[returns <= var_threshold].mean()
    
    # ==========================================
    # REGIME FACTORS
    # ==========================================
    
    # 30. Performance in High Vol vs Low Vol periods
    if len(returns) >= 252 and bench_rets is not None:
        bench_vol = bench_rets.rolling(21).std()
        median_vol = bench_vol.median()
        
        high_vol_mask = bench_vol > median_vol
        low_vol_mask = bench_vol <= median_vol
        
        common_idx = returns.index.intersection(bench_vol.index)
        if len(common_idx) > 100:
            ret_common = returns.loc[common_idx]
            high_vol_idx = high_vol_mask.loc[common_idx][high_vol_mask.loc[common_idx]].index
            low_vol_idx = low_vol_mask.loc[common_idx][low_vol_mask.loc[common_idx]].index
            
            if len(high_vol_idx) > 20 and len(low_vol_idx) > 20:
                factors['return_high_vol'] = ret_common.loc[high_vol_idx].mean() * TRADING_DAYS_YEAR
                factors['return_low_vol'] = ret_common.loc[low_vol_idx].mean() * TRADING_DAYS_YEAR
                factors['regime_adaptability'] = min(factors['return_high_vol'], factors['return_low_vol'])
    
    return factors


def calculate_max_dd(series):
    if len(series) < 10 or series.isna().all(): 
        return None
    try:
        comp_ret = (1 + series.pct_change().fillna(0)).cumprod()
        peak = comp_ret.expanding(min_periods=1).max()
        dd = (comp_ret / peak) - 1
        return dd.min()
    except:
        return None


def calculate_recovery_speed(series):
    if len(series) < 126:
        return None
    
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
    
    if not recovery_times:
        return None
    
    avg_recovery = np.mean(recovery_times)
    if avg_recovery == 0:
        return None
    
    return 1.0 / avg_recovery * 100


def calculate_rank_persistence(series, nav_df, lookback_quarters=4):
    if len(series) < 252:
        return None
    
    current_date = series.index[-1]
    persist_score = 0
    valid_quarters = 0
    
    fund_id = series.name if hasattr(series, 'name') else None
    if fund_id is None:
        return None
    
    for q in range(1, lookback_quarters + 1):
        q_start = current_date - pd.Timedelta(days=91 * q)
        q_end = current_date - pd.Timedelta(days=91 * (q - 1))
        
        try:
            idx_start = nav_df.index.asof(q_start)
            idx_end = nav_df.index.asof(q_end)
            
            if pd.isna(idx_start) or pd.isna(idx_end):
                continue
            
            q_returns = (nav_df.loc[idx_end] / nav_df.loc[idx_start]) - 1
            q_returns = q_returns.dropna()
            
            if len(q_returns) < 5:
                continue
            
            if fund_id not in q_returns.index:
                continue
            
            rank_pct = q_returns.rank(pct=True, ascending=True)[fund_id]
            
            if rank_pct >= 0.75:
                persist_score += 1.0
            elif rank_pct >= 0.5:
                persist_score += 0.5
            elif rank_pct >= 0.25:
                persist_score += 0.0
            else:
                persist_score -= 0.5
            
            valid_quarters += 1
        except:
            continue
    
    if valid_quarters == 0:
        return None
    
    return persist_score / valid_quarters


# ============================================================================
# 4. FACTOR DISCOVERY ENGINE
# ============================================================================

def run_factor_discovery(nav_df, benchmark_series, holding_period=252, top_n=5, scheme_map=None):
    """
    Main function: Discovers which factors predicted top performers.
    
    Returns:
    - factor_importance: DataFrame showing how predictive each factor was
    - period_analysis: Detailed analysis for each period
    - winner_profiles: What top funds looked like
    """
    
    # Need at least 1 year of lookback + holding period
    min_history = 370 + holding_period
    start_date = nav_df.index.min() + pd.Timedelta(days=min_history)
    
    if start_date >= nav_df.index.max():
        return None, None, None
    
    # Get rebalancing dates (quarterly for analysis)
    rebal_dates = pd.date_range(start=start_date, end=nav_df.index.max() - pd.Timedelta(days=holding_period), freq='Q')
    
    all_period_data = []
    factor_winner_comparison = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, analysis_date in enumerate(rebal_dates):
        status_text.text(f"Analyzing period {idx+1}/{len(rebal_dates)}: {analysis_date.strftime('%Y-%m-%d')}")
        progress_bar.progress((idx + 1) / len(rebal_dates))
        
        # Get analysis date index
        try:
            analysis_idx = nav_df.index.get_loc(nav_df.index.asof(analysis_date))
        except:
            continue
        
        # Get forward return period
        entry_idx = analysis_idx + 1
        exit_idx = entry_idx + holding_period
        
        if exit_idx >= len(nav_df):
            continue
        
        # Calculate ACTUAL forward returns for all funds
        forward_returns = (nav_df.iloc[exit_idx] / nav_df.iloc[entry_idx]) - 1
        forward_returns = forward_returns.dropna()
        
        if len(forward_returns) < 10:
            continue
        
        # Identify actual top N and bottom N
        actual_top_n = forward_returns.nlargest(top_n).index.tolist()
        actual_bottom_n = forward_returns.nsmallest(top_n).index.tolist()
        
        # Get historical data for factor calculation (before analysis_date)
        hist_data = nav_df.loc[:nav_df.index[analysis_idx]]
        
        # Get benchmark data
        bench_series_hist = None
        if benchmark_series is not None:
            bench_series_hist = benchmark_series.loc[:nav_df.index[analysis_idx]]
        
        # Calculate ALL factors for ALL funds at analysis_date
        fund_factors = {}
        for col in nav_df.columns:
            series = hist_data[col].dropna()
            if len(series) < 126:
                continue
            series.name = col
            factors = calculate_all_factors(series, bench_series_hist, hist_data)
            if factors:
                fund_factors[col] = factors
        
        if len(fund_factors) < 10:
            continue
        
        # Convert to DataFrame
        factors_df = pd.DataFrame(fund_factors).T
        
        # For each factor, compare winners vs losers vs all
        for factor in factors_df.columns:
            factor_values = factors_df[factor].dropna()
            
            if len(factor_values) < 10:
                continue
            
            # Get factor values for winners and losers
            winner_values = factor_values.loc[factor_values.index.intersection(actual_top_n)]
            loser_values = factor_values.loc[factor_values.index.intersection(actual_bottom_n)]
            all_values = factor_values
            
            if len(winner_values) < 2 or len(loser_values) < 2:
                continue
            
            # Statistical comparison
            winner_mean = winner_values.mean()
            loser_mean = loser_values.mean()
            all_mean = all_values.mean()
            all_std = all_values.std()
            
            # Normalized difference (effect size)
            if all_std > 0:
                effect_size = (winner_mean - all_mean) / all_std
            else:
                effect_size = 0
            
            # T-test: are winners significantly different from rest?
            rest_values = factor_values.loc[~factor_values.index.isin(actual_top_n)]
            if len(rest_values) > 5:
                try:
                    t_stat, p_value = stats.ttest_ind(winner_values, rest_values)
                except:
                    t_stat, p_value = np.nan, np.nan
            else:
                t_stat, p_value = np.nan, np.nan
            
            # Were winners in top quartile for this factor?
            top_quartile_threshold = factor_values.quantile(0.75)
            winners_in_top_quartile = (winner_values >= top_quartile_threshold).sum() / len(winner_values)
            
            factor_winner_comparison.append({
                'period': analysis_date,
                'factor': factor,
                'winner_mean': winner_mean,
                'loser_mean': loser_mean,
                'all_mean': all_mean,
                'effect_size': effect_size,
                't_statistic': t_stat,
                'p_value': p_value,
                'winners_higher': winner_mean > all_mean,
                'winners_in_top_quartile': winners_in_top_quartile
            })
        
        # Store period data
        all_period_data.append({
            'period': analysis_date,
            'holding_period': holding_period,
            'top_n_funds': actual_top_n,
            'top_n_returns': forward_returns.loc[actual_top_n].tolist(),
            'factors_df': factors_df,
            'forward_returns': forward_returns
        })
    
    progress_bar.empty()
    status_text.empty()
    
    if not factor_winner_comparison:
        return None, None, None
    
    # Aggregate factor importance across all periods
    comparison_df = pd.DataFrame(factor_winner_comparison)
    
    factor_importance = comparison_df.groupby('factor').agg({
        'effect_size': ['mean', 'std'],
        'winners_higher': 'mean',  # % of time winners had higher value
        'winners_in_top_quartile': 'mean',  # Avg % of winners in top quartile
        'p_value': lambda x: (x < 0.05).mean(),  # % of time statistically significant
        'period': 'count'
    }).round(3)
    
    factor_importance.columns = ['Avg_Effect_Size', 'Effect_Size_Std', 'Pct_Winners_Higher', 
                                  'Avg_Winners_TopQuartile', 'Pct_Significant', 'Num_Periods']
    
    # Sort by predictive power
    factor_importance['Predictive_Score'] = (
        factor_importance['Pct_Winners_Higher'] * 0.3 +
        factor_importance['Avg_Winners_TopQuartile'] * 0.3 +
        factor_importance['Pct_Significant'] * 0.2 +
        (factor_importance['Avg_Effect_Size'].abs() / factor_importance['Avg_Effect_Size'].abs().max()) * 0.2
    )
    
    factor_importance = factor_importance.sort_values('Predictive_Score', ascending=False)
    
    return factor_importance, comparison_df, all_period_data


# ============================================================================
# 5. DATA-DRIVEN STRATEGY
# ============================================================================

def run_data_driven_backtest(nav_df, benchmark_series, factor_importance, holding_period=252, top_n=3, scheme_map=None):
    """
    Uses discovered factor weights to pick funds.
    """
    
    # Get top factors
    top_factors = factor_importance.head(10).index.tolist()
    
    # Get factor weights based on importance
    weights = factor_importance.loc[top_factors, 'Predictive_Score'].to_dict()
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Check which factors favor HIGHER vs LOWER values
    factor_direction = {}
    for factor in top_factors:
        # If winners consistently had higher values, we want higher
        # Some factors like volatility, max_drawdown - lower is better
        pct_higher = factor_importance.loc[factor, 'Pct_Winners_Higher']
        factor_direction[factor] = 1 if pct_higher > 0.5 else -1
    
    # Run backtest
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
        
        # Get historical data
        hist_data = nav_df.loc[:date]
        
        bench_series_hist = None
        if benchmark_series is not None:
            bench_series_hist = benchmark_series.loc[:date]
        
        # Calculate factors for all funds
        fund_scores = {}
        
        for col in nav_df.columns:
            series = hist_data[col].dropna()
            if len(series) < 126:
                continue
            series.name = col
            
            factors = calculate_all_factors(series, bench_series_hist, hist_data)
            
            if not factors:
                continue
            
            # Calculate weighted score using discovered factors
            score = 0
            valid_factors = 0
            
            for factor, weight in weights.items():
                if factor in factors and not pd.isna(factors[factor]):
                    # Normalize factor value (rank-based)
                    normalized_value = factors[factor]
                    # Apply direction
                    score += normalized_value * weight * factor_direction.get(factor, 1)
                    valid_factors += 1
            
            if valid_factors >= 3:
                fund_scores[col] = score
        
        # Rank and select top N
        if not fund_scores:
            selected = []
        else:
            # Convert to ranks for fair comparison
            score_series = pd.Series(fund_scores)
            selected = score_series.nlargest(top_n).index.tolist()
        
        # Calculate returns
        entry = i + 1
        exit_i = min(i + 1 + holding_period, len(nav_df) - 1)
        
        # Benchmark return
        b_ret = 0.0
        if benchmark_series is not None:
            try:
                b_ret = (benchmark_series.asof(nav_df.index[exit_i]) / benchmark_series.asof(nav_df.index[entry])) - 1
            except:
                pass
        
        # Portfolio return
        port_ret = 0.0
        hit_rate = 0.0
        
        if selected:
            period_ret_all = (nav_df.iloc[exit_i] / nav_df.iloc[entry]) - 1
            port_ret = period_ret_all[selected].mean()
            
            # Hit rate
            actual_top = period_ret_all.dropna().nlargest(top_n).index.tolist()
            matches = len(set(selected).intersection(set(actual_top)))
            hit_rate = matches / top_n
        
        cap *= (1 + (port_ret if not pd.isna(port_ret) else 0))
        b_cap *= (1 + b_ret)
        
        history.append({
            'date': date,
            'selected': selected,
            'selected_names': [scheme_map.get(s, s) for s in selected] if scheme_map else selected,
            'return': port_ret,
            'hit_rate': hit_rate
        })
        
        eq_curve.append({'date': nav_df.index[exit_i], 'value': cap})
        bench_curve.append({'date': nav_df.index[exit_i], 'value': b_cap})
    
    return pd.DataFrame(history), pd.DataFrame(eq_curve), pd.DataFrame(bench_curve)


# ============================================================================
# 6. UI COMPONENTS
# ============================================================================

def render_factor_discovery_page():
    st.title("🧬 Factor Discovery: Winner DNA Analysis")
    
    st.markdown("""
    ### What This Does
    
    Instead of **assuming** which factors matter, this tool **discovers** what top-performing funds actually had in common.
    
    **Process:**
    1. At each historical period, identify the ACTUAL top 5 funds (by forward returns)
    2. Look at what factors these winners had BEFORE they won
    3. Compare winners vs rest of funds
    4. Find patterns that consistently predict winners
    """)
    
    st.divider()
    
    # Controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        category = st.selectbox("Category", list(FILE_MAPPING.keys()))
    
    with col2:
        holding_period = st.selectbox("Holding Period", [126, 252, 378, 504], index=1, 
                                       format_func=lambda x: f"{x} days (~{x//21} months)")
    
    with col3:
        top_n = st.number_input("Top N to Analyze", 3, 10, 5, 
                                help="How many top performers to study")
    
    with col4:
        run_analysis = st.button("🔍 Run Factor Discovery", type="primary", use_container_width=True)
    
    if run_analysis:
        with st.spinner("Loading data..."):
            nav_df, scheme_map = load_fund_data_raw(category)
            benchmark = load_nifty_data()
        
        if nav_df is None:
            st.error("Could not load data")
            return
        
        st.info(f"Loaded {len(nav_df.columns)} funds from {category}")
        
        with st.spinner("Running factor discovery... This may take a minute."):
            factor_importance, comparison_df, period_data = run_factor_discovery(
                nav_df, benchmark, holding_period, top_n, scheme_map
            )
        
        if factor_importance is None:
            st.error("Not enough data for analysis")
            return
        
        # ==========================================
        # RESULTS
        # ==========================================
        
        st.success("✅ Factor Discovery Complete!")
        
        # Tab layout for results
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Factor Importance", 
            "🧬 Winner DNA Profile",
            "📈 Data-Driven Strategy",
            "🔬 Detailed Analysis"
        ])
        
        with tab1:
            st.subheader("Which Factors Predicted Winners?")
            
            st.markdown("""
            **How to Read This:**
            - **Pct_Winners_Higher**: % of time winners had higher values for this factor
            - **Avg_Winners_TopQuartile**: How often were winners in top 25% for this factor?
            - **Pct_Significant**: Statistical significance (p < 0.05)
            - **Predictive_Score**: Combined score (higher = more predictive)
            """)
            
            # Display top factors
            display_df = factor_importance.head(20).copy()
            display_df = display_df.round(3)
            
            st.dataframe(
                display_df.style.background_gradient(subset=['Predictive_Score'], cmap='Greens')
                .background_gradient(subset=['Pct_Winners_Higher'], cmap='Blues')
                .format({
                    'Pct_Winners_Higher': '{:.1%}',
                    'Avg_Winners_TopQuartile': '{:.1%}',
                    'Pct_Significant': '{:.1%}',
                    'Predictive_Score': '{:.3f}'
                }),
                use_container_width=True
            )
            
            # Visualization
            st.subheader("Factor Predictive Power")
            
            top_factors = factor_importance.head(15)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='% Winners Higher',
                x=top_factors.index,
                y=top_factors['Pct_Winners_Higher'] * 100,
                marker_color='steelblue'
            ))
            
            fig.add_trace(go.Bar(
                name='% Winners in Top Quartile',
                x=top_factors.index,
                y=top_factors['Avg_Winners_TopQuartile'] * 100,
                marker_color='forestgreen'
            ))
            
            fig.update_layout(
                barmode='group',
                title='Factor Predictive Power (Top 15 Factors)',
                xaxis_tickangle=-45,
                yaxis_title='Percentage (%)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("🧬 Winner DNA Profile")
            
            st.markdown("""
            **What did winning funds look like?**
            
            Based on historical analysis, here's the "fingerprint" of funds that ended up in the Top 5:
            """)
            
            # Get top predictive factors
            top_predictive = factor_importance[factor_importance['Predictive_Score'] > 0.5].head(10)
            
            if len(top_predictive) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### ✅ Factors Where Winners Scored HIGH")
                    high_factors = top_predictive[top_predictive['Pct_Winners_Higher'] > 0.55]
                    for factor in high_factors.index:
                        pct = high_factors.loc[factor, 'Pct_Winners_Higher']
                        st.markdown(f"- **{factor}**: Winners higher {pct:.0%} of the time")
                
                with col2:
                    st.markdown("### 📉 Factors Where Winners Scored LOW")
                    low_factors = top_predictive[top_predictive['Pct_Winners_Higher'] < 0.45]
                    for factor in low_factors.index:
                        pct = 1 - low_factors.loc[factor, 'Pct_Winners_Higher']
                        st.markdown(f"- **{factor}**: Winners lower {pct:.0%} of the time")
                
                st.divider()
                
                st.markdown("### 🎯 Recommended Selection Criteria")
                st.markdown("Based on factor discovery, look for funds with:")
                
                criteria = []
                for factor in top_predictive.head(5).index:
                    pct_higher = top_predictive.loc[factor, 'Pct_Winners_Higher']
                    quartile_pct = top_predictive.loc[factor, 'Avg_Winners_TopQuartile']
                    
                    if pct_higher > 0.55:
                        criteria.append(f"✅ **High {factor}** (winners had higher {pct_higher:.0%} of time)")
                    elif pct_higher < 0.45:
                        criteria.append(f"✅ **Low {factor}** (winners had lower {1-pct_higher:.0%} of time)")
                
                for c in criteria:
                    st.markdown(c)
            
            else:
                st.warning("No strongly predictive factors found. Market may be highly efficient in this category.")
        
        with tab3:
            st.subheader("📈 Data-Driven Strategy Backtest")
            
            st.markdown("""
            Now let's use the discovered factors to build a strategy and test it.
            
            **Strategy:** Weight factors by their predictive power, then rank funds and pick top N.
            """)
            
            # Show factor weights being used
            st.markdown("**Factor Weights (based on discovery):**")
            top_factors_used = factor_importance.head(10)
            weights_df = top_factors_used[['Predictive_Score']].copy()
            weights_df['Weight %'] = (weights_df['Predictive_Score'] / weights_df['Predictive_Score'].sum() * 100).round(1)
            st.dataframe(weights_df[['Weight %']], use_container_width=True)
            
            # Run backtest
            with st.spinner("Running data-driven backtest..."):
                hist, eq, bench = run_data_driven_backtest(
                    nav_df, benchmark, factor_importance, holding_period, min(top_n, 3), scheme_map
                )
            
            if hist is not None and not eq.empty:
                # Results
                start_date = eq.iloc[0]['date']
                end_date = eq.iloc[-1]['date']
                years = (end_date - start_date).days / 365.25
                
                strat_fin = eq.iloc[-1]['value']
                strat_cagr = (strat_fin/100)**(1/years)-1 if years > 0 else 0
                
                bench_fin = bench.iloc[-1]['value'] if not bench.empty else 100
                bench_cagr = (bench_fin/100)**(1/years)-1 if years > 0 and not bench.empty else 0
                
                avg_hit_rate = hist['hit_rate'].mean()
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Strategy CAGR", f"{strat_cagr:.2%}")
                c2.metric("Benchmark CAGR", f"{bench_cagr:.2%}")
                c3.metric("Outperformance", f"{(strat_cagr - bench_cagr):.2%}")
                c4.metric("Avg Hit Rate", f"{avg_hit_rate:.1%}")
                
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=eq['date'], y=eq['value'], name='Data-Driven Strategy', line=dict(color='green', width=2)))
                fig.add_trace(go.Scatter(x=bench['date'], y=bench['value'], name='Benchmark', line=dict(color='red', dash='dot')))
                fig.update_layout(title='Data-Driven Strategy vs Benchmark', yaxis_title='Value (100 = Start)')
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade history
                st.subheader("Trade History")
                st.dataframe(hist[['date', 'selected_names', 'return', 'hit_rate']].style.format({
                    'return': '{:.2%}',
                    'hit_rate': '{:.0%}'
                }), use_container_width=True)
            
            else:
                st.warning("Could not run backtest")
        
        with tab4:
            st.subheader("🔬 Detailed Period-by-Period Analysis")
            
            if period_data:
                period_select = st.selectbox(
                    "Select Period",
                    [p['period'].strftime('%Y-%m-%d') for p in period_data],
                    key="period_select"
                )
                
                selected_period = next(p for p in period_data if p['period'].strftime('%Y-%m-%d') == period_select)
                
                st.markdown(f"**Period:** {period_select}")
                st.markdown(f"**Holding Period:** {selected_period['holding_period']} days")
                
                # Show top funds and their returns
                st.markdown("**Actual Top 5 Winners:**")
                for idx, (fund_id, ret) in enumerate(zip(selected_period['top_n_funds'], selected_period['top_n_returns'])):
                    fund_name = scheme_map.get(fund_id, fund_id)
                    st.markdown(f"{idx+1}. **{fund_name}**: {ret:.2%} return")
                
                # Show their factors
                st.markdown("**Factor Values for Winners:**")
                factors_df = selected_period['factors_df']
                winner_factors = factors_df.loc[factors_df.index.intersection(selected_period['top_n_funds'])]
                winner_factors.index = winner_factors.index.map(lambda x: scheme_map.get(x, x))
                
                st.dataframe(winner_factors.T.style.format("{:.3f}"), use_container_width=True, height=400)


def main():
    render_factor_discovery_page()


if __name__ == "__main__":
    main()
