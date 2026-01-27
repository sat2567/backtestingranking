import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats

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
DEFAULT_HOLDING = 504  # Default set to 504 days
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

# ============================================================================
# 3. EXISTING METRICS (keeping all your originals)
# ============================================================================

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
        vol = hist_vol_data.pct_change().dropna().std() * np.sqrt(TRADING_DAYS_YEAR)
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

def calculate_beta_alpha_treynor(fund_returns, bench_returns):
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 30: return np.nan, np.nan, np.nan
    y = fund_returns.loc[common_idx]
    x = bench_returns.loc[common_idx]
    cov_matrix = np.cov(y, x)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    mean_fund = y.mean() * TRADING_DAYS_YEAR
    mean_bench = x.mean() * TRADING_DAYS_YEAR
    alpha = mean_fund - (RISK_FREE_RATE + beta * (mean_bench - RISK_FREE_RATE))
    treynor = (mean_fund - RISK_FREE_RATE) / beta if beta != 0 else np.nan
    return beta, alpha, treynor

def calculate_calmar_ratio(series, current_date):
    if len(series) < 252: return np.nan
    three_years_ago = current_date - pd.Timedelta(days=365*3)
    sub = series[series.index >= three_years_ago]
    if len(sub) < 100: sub = series 
    max_dd = calculate_max_dd(sub)
    if max_dd == 0 or pd.isna(max_dd): return np.nan
    start_val = sub.iloc[0]
    end_val = sub.iloc[-1]
    years = (sub.index[-1] - sub.index[0]).days / 365.25
    if years <= 0: return np.nan
    cagr = (end_val / start_val) ** (1/years) - 1
    return cagr / abs(max_dd)

# ============================================================================
# 4. NEW HIGH-HIT-RATE STRATEGIES
# ============================================================================

def calculate_rank_persistence(series, nav_df, lookback_quarters=4):
    """
    Funds that consistently rank in top quartile tend to persist.
    This captures "skill" vs "luck" - skilled managers maintain ranks.
    """
    if len(series) < 252: return np.nan
    
    current_date = series.index[-1]
    persist_score = 0
    valid_quarters = 0
    
    for q in range(1, lookback_quarters + 1):
        q_start = current_date - pd.Timedelta(days=91 * q)
        q_end = current_date - pd.Timedelta(days=91 * (q - 1))
        
        try:
            # Get returns for all funds in that quarter
            idx_start = nav_df.index.asof(q_start)
            idx_end = nav_df.index.asof(q_end)
            
            if pd.isna(idx_start) or pd.isna(idx_end): continue
            
            q_returns = (nav_df.loc[idx_end] / nav_df.loc[idx_start]) - 1
            q_returns = q_returns.dropna()
            
            if len(q_returns) < 5: continue
            
            # Get rank for this fund
            fund_id = series.name
            if fund_id not in q_returns.index: continue
            
            rank_pct = q_returns.rank(pct=True, ascending=True)[fund_id]
            
            # Score: top quartile = 1, second = 0.5, third = 0, bottom = -0.5
            if rank_pct >= 0.75: persist_score += 1.0
            elif rank_pct >= 0.5: persist_score += 0.5
            elif rank_pct >= 0.25: persist_score += 0.0
            else: persist_score -= 0.5
            
            valid_quarters += 1
        except:
            continue
    
    if valid_quarters == 0: return np.nan
    return persist_score / valid_quarters


def calculate_drawdown_recovery_speed(series):
    """
    Funds that recover quickly from drawdowns are often better managed.
    Fast recovery = strong fund selection/risk management.
    """
    if len(series) < 252: return np.nan
    
    cum_max = series.expanding(min_periods=1).max()
    drawdowns = (series / cum_max) - 1
    
    # Find significant drawdowns (> 5%)
    in_drawdown = False
    dd_start = None
    recovery_times = []
    
    for i, (date, dd) in enumerate(drawdowns.items()):
        if not in_drawdown and dd < -0.05:
            in_drawdown = True
            dd_start = i
        elif in_drawdown and dd >= -0.01:  # Recovered
            recovery_time = i - dd_start
            recovery_times.append(recovery_time)
            in_drawdown = False
    
    if not recovery_times: return np.nan
    
    # Faster recovery = higher score (inverse of avg recovery time)
    avg_recovery = np.mean(recovery_times)
    if avg_recovery == 0: return np.nan
    
    return 1.0 / avg_recovery * 100  # Scale up


def calculate_consistency_score(series, window=63):
    """
    Measures how consistently a fund outperforms its own rolling average.
    Consistent funds are more predictable.
    """
    if len(series) < window * 2: return np.nan
    
    returns = series.pct_change().dropna()
    rolling_avg = returns.rolling(window=window).mean()
    
    # Count how often daily return exceeds rolling average
    aligned = pd.concat([returns, rolling_avg], axis=1).dropna()
    if len(aligned) < window: return np.nan
    
    aligned.columns = ['ret', 'avg']
    above_avg = (aligned['ret'] > aligned['avg']).sum()
    
    return above_avg / len(aligned)


def calculate_downside_consistency(fund_rets, bench_rets):
    """
    Measures how consistently fund protects in down markets.
    More consistent protection = more predictable future protection.
    """
    common_idx = fund_rets.index.intersection(bench_rets.index)
    if len(common_idx) < 60: return np.nan
    
    f = fund_rets.loc[common_idx]
    b = bench_rets.loc[common_idx]
    
    # Look at worst 20% of benchmark days
    threshold = b.quantile(0.2)
    worst_days = b[b <= threshold]
    
    if len(worst_days) < 10: return np.nan
    
    # For each worst day, did fund do better than benchmark?
    protection_count = 0
    for date in worst_days.index:
        if f.loc[date] > b.loc[date]:
            protection_count += 1
    
    return protection_count / len(worst_days)


def calculate_momentum_quality(series):
    """
    Not just momentum, but QUALITY of momentum.
    High R-squared trend = more likely to continue.
    """
    if len(series) < 126: return np.nan
    
    # Use last 6 months
    recent = series.iloc[-126:]
    
    # Fit linear regression
    x = np.arange(len(recent))
    y = np.log(recent.values + 1e-10)  # Log prices
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    except:
        return np.nan
    
    r_squared = r_value ** 2
    
    # Score = slope * r_squared (strong uptrend with high fit)
    # Annualize slope
    annual_slope = slope * 252
    
    return annual_slope * r_squared


def calculate_smart_beta_tilt(series, bench_series):
    """
    Funds with positive but moderate beta tend to outperform.
    Too high beta = too much market dependency.
    Too low beta = missing market returns.
    Optimal is around 0.8-1.1
    """
    if len(series) < 126: return np.nan
    
    fund_ret = series.pct_change().dropna()
    bench_ret = bench_series.pct_change().dropna()
    
    common_idx = fund_ret.index.intersection(bench_ret.index)
    if len(common_idx) < 60: return np.nan
    
    f = fund_ret.loc[common_idx]
    b = bench_ret.loc[common_idx]
    
    # Calculate beta
    cov = np.cov(f, b)
    beta = cov[0, 1] / cov[1, 1]
    
    # Optimal beta range penalty
    # Best score at beta = 0.95, penalize deviation
    optimal_beta = 0.95
    beta_penalty = abs(beta - optimal_beta)
    
    # Also factor in alpha
    alpha = f.mean() - beta * b.mean()
    
    # Combined score: high alpha, beta close to optimal
    score = alpha * 252 - beta_penalty * 0.5
    
    return score


def calculate_volatility_regime_adaptability(series, lookback=252):
    """
    Funds that adapt well to changing volatility regimes 
    tend to perform better in the future.
    """
    if len(series) < lookback + 63: return np.nan
    
    returns = series.pct_change().dropna()
    
    # Split into high-vol and low-vol periods
    rolling_vol = returns.rolling(21).std()
    median_vol = rolling_vol.median()
    
    high_vol_periods = returns[rolling_vol > median_vol]
    low_vol_periods = returns[rolling_vol <= median_vol]
    
    if len(high_vol_periods) < 30 or len(low_vol_periods) < 30:
        return np.nan
    
    # Sharpe in each regime
    high_vol_sharpe = high_vol_periods.mean() / high_vol_periods.std() if high_vol_periods.std() > 0 else 0
    low_vol_sharpe = low_vol_periods.mean() / low_vol_periods.std() if low_vol_periods.std() > 0 else 0
    
    # Good funds perform well in BOTH regimes
    # Use geometric mean to penalize poor performance in either
    if high_vol_sharpe <= 0 or low_vol_sharpe <= 0:
        return (high_vol_sharpe + low_vol_sharpe) / 2
    
    return np.sqrt(high_vol_sharpe * low_vol_sharpe)


def calculate_peer_relative_strength(series, nav_df, lookback=126):
    """
    Relative strength vs peers, not just absolute momentum.
    Funds consistently in top half of peers tend to stay there.
    """
    if len(series) < lookback: return np.nan
    
    current_date = series.index[-1]
    
    # Calculate returns for all funds
    try:
        idx_start = nav_df.index.asof(current_date - pd.Timedelta(days=lookback))
        idx_end = current_date
        
        period_returns = (nav_df.loc[idx_end] / nav_df.loc[idx_start]) - 1
        period_returns = period_returns.dropna()
        
        if len(period_returns) < 5: return np.nan
        
        fund_id = series.name
        if fund_id not in period_returns.index: return np.nan
        
        # Percentile rank among peers
        rank_pct = period_returns.rank(pct=True, ascending=True)[fund_id]
        
        return rank_pct
    except:
        return np.nan


def calculate_max_gain_capture(fund_rets, bench_rets):
    """
    Focus on capturing BIG up days, not average up days.
    Funds that capture best days often have better stock selection.
    """
    common_idx = fund_rets.index.intersection(bench_rets.index)
    if len(common_idx) < 60: return np.nan
    
    f = fund_rets.loc[common_idx]
    b = bench_rets.loc[common_idx]
    
    # Look at best 10% of benchmark days
    threshold = b.quantile(0.9)
    best_days = b[b >= threshold]
    
    if len(best_days) < 5: return np.nan
    
    # How much did fund capture on these best days?
    fund_on_best = f.loc[best_days.index]
    bench_on_best = best_days
    
    capture_ratio = fund_on_best.sum() / bench_on_best.sum() if bench_on_best.sum() != 0 else 0
    
    return capture_ratio


def calculate_trend_following_score(series, short_window=50, long_window=200):
    """
    Funds above their moving averages tend to continue outperforming.
    Multiple timeframe confirmation = stronger signal.
    """
    if len(series) < long_window + 20: return np.nan
    
    current_price = series.iloc[-1]
    ma_short = series.rolling(short_window).mean().iloc[-1]
    ma_long = series.rolling(long_window).mean().iloc[-1]
    
    score = 0
    
    # Price above short MA
    if current_price > ma_short:
        score += 0.4
    
    # Price above long MA
    if current_price > ma_long:
        score += 0.3
    
    # Short MA above long MA (golden cross territory)
    if ma_short > ma_long:
        score += 0.3
    
    # Bonus: both MAs trending up
    ma_short_prev = series.rolling(short_window).mean().iloc[-21]
    ma_long_prev = series.rolling(long_window).mean().iloc[-21]
    
    if ma_short > ma_short_prev:
        score += 0.1
    if ma_long > ma_long_prev:
        score += 0.1
    
    return score


def calculate_information_coefficient(series, nav_df, forward_window=63):
    """
    Historical predictive accuracy - how well did past rankings
    predict future performance? Higher IC = more predictable fund.
    """
    if len(series) < 252 + forward_window: return np.nan
    
    # This requires looking at historical predictions
    # We'll compute rolling rank correlations
    
    current_date = series.index[-1]
    correlations = []
    
    for lookback_months in range(3, 12):
        try:
            past_date = current_date - pd.Timedelta(days=30 * lookback_months)
            
            # Ranks at past_date based on prior 6m return
            prior_start = past_date - pd.Timedelta(days=126)
            
            idx_prior_start = nav_df.index.asof(prior_start)
            idx_past = nav_df.index.asof(past_date)
            idx_future = nav_df.index.asof(past_date + pd.Timedelta(days=forward_window))
            
            if any(pd.isna([idx_prior_start, idx_past, idx_future])): continue
            
            prior_returns = (nav_df.loc[idx_past] / nav_df.loc[idx_prior_start]) - 1
            future_returns = (nav_df.loc[idx_future] / nav_df.loc[idx_past]) - 1
            
            # Correlation between prior and future returns
            common = prior_returns.dropna().index.intersection(future_returns.dropna().index)
            if len(common) < 5: continue
            
            corr = prior_returns[common].corr(future_returns[common])
            if not pd.isna(corr):
                correlations.append(corr)
        except:
            continue
    
    if not correlations: return np.nan
    
    # Average correlation (higher = more predictable category)
    avg_ic = np.mean(correlations)
    
    # Fund's relative strength in predictable category
    fund_id = series.name
    try:
        recent_6m_ret = (series.iloc[-1] / series.iloc[-126]) - 1
        all_6m_ret = (nav_df.iloc[-1] / nav_df.iloc[-126]) - 1
        rank_pct = all_6m_ret.dropna().rank(pct=True, ascending=True).get(fund_id, 0.5)
    except:
        rank_pct = 0.5
    
    # High IC category + high rank = good signal
    return avg_ic * rank_pct if avg_ic > 0 else rank_pct * 0.5


# ============================================================================
# 5. NEW COMPOSITE STRATEGIES
# ============================================================================

def strategy_quality_momentum(hist_data, nav_df, bench_rets, top_n):
    """
    Strategy: Quality Momentum
    Combines momentum quality (R-squared trend) with consistency.
    Rationale: Smooth uptrends more likely to continue than volatile ones.
    """
    scores = {}
    
    for col in nav_df.columns:
        s = hist_data[col].dropna()
        if len(s) < 200: continue
        
        mom_quality = calculate_momentum_quality(s)
        consistency = calculate_consistency_score(s)
        
        if pd.isna(mom_quality) or pd.isna(consistency): continue
        
        # Combined score
        scores[col] = mom_quality * 0.6 + consistency * 0.4
    
    return sorted(scores, key=scores.get, reverse=True)[:top_n]


def strategy_persistent_alpha(hist_data, nav_df, bench_series, bench_rets, top_n):
    """
    Strategy: Persistent Alpha
    Funds with consistent outperformance across multiple periods.
    Rationale: Luck doesn't persist; skill does.
    """
    scores = {}
    
    for col in nav_df.columns:
        s = hist_data[col].dropna()
        s.name = col
        if len(s) < 252: continue
        
        # Rank persistence
        persist = calculate_rank_persistence(s, hist_data)
        
        # Downside consistency
        if bench_rets is not None and len(bench_rets) > 60:
            rets = s.pct_change().dropna()
            down_consist = calculate_downside_consistency(rets, bench_rets)
        else:
            down_consist = 0.5
        
        if pd.isna(persist): continue
        
        scores[col] = persist * 0.5 + (down_consist if not pd.isna(down_consist) else 0.5) * 0.5
    
    return sorted(scores, key=scores.get, reverse=True)[:top_n]


def strategy_regime_adaptive(hist_data, nav_df, bench_series, bench_rets, top_n):
    """
    Strategy: Regime Adaptive
    Funds that perform in both high and low volatility environments.
    Rationale: Robust across market conditions = better future performance.
    """
    scores = {}
    
    for col in nav_df.columns:
        s = hist_data[col].dropna()
        if len(s) < 252: continue
        
        adaptability = calculate_volatility_regime_adaptability(s)
        
        if pd.isna(adaptability): continue
        
        # Also factor in recovery speed
        recovery = calculate_drawdown_recovery_speed(s)
        
        if pd.isna(recovery):
            scores[col] = adaptability
        else:
            scores[col] = adaptability * 0.6 + min(recovery, 10) / 10 * 0.4
    
    return sorted(scores, key=scores.get, reverse=True)[:top_n]


def strategy_trend_confirmation(hist_data, nav_df, bench_series, bench_rets, top_n):
    """
    Strategy: Multi-Timeframe Trend Confirmation
    Strong trend across multiple timeframes.
    Rationale: Confirmed trends have higher probability of continuation.
    """
    scores = {}
    
    for col in nav_df.columns:
        s = hist_data[col].dropna()
        if len(s) < 252: continue
        
        trend_score = calculate_trend_following_score(s)
        
        # Also check peer relative strength
        s.name = col
        peer_strength = calculate_peer_relative_strength(s, hist_data)
        
        if pd.isna(trend_score): continue
        
        if pd.isna(peer_strength):
            scores[col] = trend_score
        else:
            scores[col] = trend_score * 0.5 + peer_strength * 0.5
    
    return sorted(scores, key=scores.get, reverse=True)[:top_n]


def strategy_best_day_capture(hist_data, nav_df, bench_series, bench_rets, top_n):
    """
    Strategy: Best Day Capture
    Funds that capture the biggest up days.
    Rationale: Missing best days kills returns; funds that capture them outperform.
    """
    if bench_rets is None: return []
    
    scores = {}
    
    for col in nav_df.columns:
        s = hist_data[col].dropna()
        if len(s) < 126: continue
        
        rets = s.pct_change().dropna()
        
        max_gain = calculate_max_gain_capture(rets, bench_rets)
        
        if pd.isna(max_gain): continue
        
        # Also factor in overall capture ratio
        capture = calculate_capture_score(rets, bench_rets)
        
        if pd.isna(capture):
            scores[col] = max_gain
        else:
            scores[col] = max_gain * 0.6 + capture * 0.4
    
    return sorted(scores, key=scores.get, reverse=True)[:top_n]


def strategy_low_volatility_momentum(hist_data, nav_df, bench_rets, top_n):
    """
    Strategy: Low Volatility Momentum
    Momentum but filtered for lower volatility funds.
    Rationale: Low vol + momentum = more sustainable outperformance.
    """
    # First pass: get momentum scores
    mom_scores = {}
    vol_scores = {}
    
    for col in nav_df.columns:
        s = hist_data[col].dropna()
        if len(s) < 200: continue
        
        mom = calculate_flexible_momentum(s, 0.4, 0.4, 0.2, False)
        vol = calculate_volatility(s.pct_change().dropna())
        
        if not pd.isna(mom): mom_scores[col] = mom
        if not pd.isna(vol): vol_scores[col] = vol
    
    if not mom_scores or not vol_scores: return []
    
    # Get top 2x by momentum
    top_mom = sorted(mom_scores, key=mom_scores.get, reverse=True)[:top_n * 3]
    
    # Filter for lowest volatility among momentum leaders
    vol_filtered = {k: vol_scores.get(k, float('inf')) for k in top_mom if k in vol_scores}
    
    return sorted(vol_filtered, key=vol_filtered.get)[:top_n]


def strategy_composite_predictor(hist_data, nav_df, bench_series, bench_rets, top_n):
    """
    Strategy: Composite Predictor
    Combines multiple predictive signals with empirical weights.
    Uses ensemble of best performing indicators.
    """
    fund_scores = pd.DataFrame(index=nav_df.columns)
    
    for col in nav_df.columns:
        s = hist_data[col].dropna()
        s.name = col
        if len(s) < 252: continue
        
        rets = s.pct_change().dropna()
        
        scores = {}
        
        # 1. Quality Momentum (20%)
        qm = calculate_momentum_quality(s)
        if not pd.isna(qm): scores['qm'] = qm
        
        # 2. Rank Persistence (20%)
        rp = calculate_rank_persistence(s, hist_data)
        if not pd.isna(rp): scores['rp'] = rp
        
        # 3. Regime Adaptability (15%)
        ra = calculate_volatility_regime_adaptability(s)
        if not pd.isna(ra): scores['ra'] = ra
        
        # 4. Trend Score (15%)
        ts = calculate_trend_following_score(s)
        if not pd.isna(ts): scores['ts'] = ts
        
        # 5. Downside Consistency (15%)
        if bench_rets is not None:
            dc = calculate_downside_consistency(rets, bench_rets)
            if not pd.isna(dc): scores['dc'] = dc
        
        # 6. Sharpe (15%)
        sh = calculate_sharpe_ratio(rets)
        if not pd.isna(sh): scores['sh'] = sh
        
        fund_scores.loc[col, 'count'] = len(scores)
        
        for k, v in scores.items():
            fund_scores.loc[col, k] = v
    
    # Normalize and combine
    fund_scores = fund_scores.dropna(subset=['count'])
    if fund_scores.empty: return []
    
    # Only keep funds with at least 4 valid signals
    fund_scores = fund_scores[fund_scores['count'] >= 4]
    
    # Rank-based combination
    final_score = pd.Series(0.0, index=fund_scores.index)
    
    weights = {'qm': 0.20, 'rp': 0.20, 'ra': 0.15, 'ts': 0.15, 'dc': 0.15, 'sh': 0.15}
    
    for col, w in weights.items():
        if col in fund_scores.columns:
            ranks = fund_scores[col].rank(pct=True, ascending=True, na_option='bottom')
            final_score += ranks * w
    
    return final_score.nlargest(top_n).index.tolist()


# ============================================================================
# 6. DATA LOADING (Same as original)
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
# 7. EXPLORER LOGIC (Same as original)
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
# 8. UPDATED BACKTESTER WITH NEW STRATEGIES
# ============================================================================

def get_lookback_data(nav, analysis_date):
    max_days = 400 
    start_date = analysis_date - pd.Timedelta(days=max_days)
    return nav[(nav.index >= start_date) & (nav.index < analysis_date)]


def run_backtest(nav, strategy_type, top_n, target_n, holding_days, custom_weights, momentum_config, benchmark_series, ensemble_weights=None):
    
    start_date = nav.index.min() + pd.Timedelta(days=370)
    if start_date >= nav.index.max(): 
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    start_idx = nav.index.searchsorted(start_date)
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

        # =============================================
        # NEW STRATEGIES
        # =============================================
        
        if strategy_type == 'quality_momentum':
            selected = strategy_quality_momentum(hist, nav, bench_rets, top_n)
        
        elif strategy_type == 'persistent_alpha':
            selected = strategy_persistent_alpha(hist, nav, benchmark_series, bench_rets, top_n)
        
        elif strategy_type == 'regime_adaptive':
            selected = strategy_regime_adaptive(hist, nav, benchmark_series, bench_rets, top_n)
        
        elif strategy_type == 'trend_confirmation':
            selected = strategy_trend_confirmation(hist, nav, benchmark_series, bench_rets, top_n)
        
        elif strategy_type == 'best_day_capture':
            selected = strategy_best_day_capture(hist, nav, benchmark_series, bench_rets, top_n)
        
        elif strategy_type == 'low_vol_momentum':
            selected = strategy_low_volatility_momentum(hist, nav, bench_rets, top_n)
        
        elif strategy_type == 'composite_predictor':
            selected = strategy_composite_predictor(hist, nav, benchmark_series, bench_rets, top_n)

        # =============================================
        # EXISTING STRATEGIES (unchanged)
        # =============================================

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
            period_ret_all_funds = (nav.iloc[exit_i] / nav.iloc[entry]) - 1
            port_ret = period_ret_all_funds[selected].mean()
            
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
            'hit_rate': hit_rate
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
        s.name = col
        
        # New strategies
        if strategy_type == 'quality_momentum':
            val = calculate_momentum_quality(s)
        elif strategy_type == 'persistent_alpha':
            val = calculate_rank_persistence(s, hist_data)
        elif strategy_type == 'regime_adaptive':
            val = calculate_volatility_regime_adaptability(s)
        elif strategy_type == 'trend_confirmation':
            val = calculate_trend_following_score(s)
        elif strategy_type == 'best_day_capture' and bench_rets is not None:
            val = calculate_max_gain_capture(rets, bench_rets)
        elif strategy_type == 'low_vol_momentum':
            mom = calculate_flexible_momentum(s, 0.4, 0.4, 0.2, False)
            vol = calculate_volatility(rets)
            if not pd.isna(mom) and not pd.isna(vol) and vol > 0:
                val = mom / vol
        elif strategy_type == 'composite_predictor':
            # For snapshot, use simplified version
            qm = calculate_momentum_quality(s) or 0
            rp = calculate_rank_persistence(s, hist_data) or 0
            ts = calculate_trend_following_score(s) or 0
            val = qm * 0.4 + rp * 0.3 + ts * 0.3
        
        # Existing strategies
        elif strategy_type == 'consistent_alpha':
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
# 9. UI COMPONENTS
# ============================================================================

def display_backtest_results(nav, maps, nifty, strat_key, top_n, target_n, hold, cust_w, mom_cfg, ens_w=None, description=""):
    if description:
        st.info(description)
    
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
    st.markdown("### 🏆 Strategy Leaderboard (Including New High-Hit-Rate Strategies)")
    
    # Include both old and new strategies
    strategies = {
        # NEW STRATEGIES (designed for higher hit rate)
        '🆕 Quality Momentum': ('quality_momentum', {}, {}, None),
        '🆕 Persistent Alpha': ('persistent_alpha', {}, {}, None),
        '🆕 Regime Adaptive': ('regime_adaptive', {}, {}, None),
        '🆕 Trend Confirmation': ('trend_confirmation', {}, {}, None),
        '🆕 Best Day Capture': ('best_day_capture', {}, {}, None),
        '🆕 Low Vol Momentum': ('low_vol_momentum', {}, {}, None),
        '🆕 Composite Predictor': ('composite_predictor', {}, {}, None),
        # EXISTING STRATEGIES
        'Stable Momentum': ('stable_momentum', {}, {}, None),
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
        df_res = pd.DataFrame(results).set_index('Strategy')
        
        # Sort by hit rate to show best hit rate strategies first
        df_res = df_res.sort_values('Hit Rate %', ascending=False)
        
        st.caption(f"Hit Rate Logic: How many of the Top {top_n} selected funds landed in the Top {target_n} actual performers.")
        
        # Highlight new strategies
        st.dataframe(
            df_res.style.format("{:.2f}")
            .background_gradient(subset=['Hit Rate %'], cmap='Greens')
            .background_gradient(subset=['CAGR %'], cmap='Blues'),
            use_container_width=True
        )
        st.plotly_chart(fig, use_container_width=True, key="comparison_chart")


def render_backtest_tab():
    st.header("🚀 Strategy Backtester")
    c1, c2, c3, c4 = st.columns(4)
    cat = c1.selectbox("Category", list(FILE_MAPPING.keys()))
    top_n = c2.number_input("Top N Funds", 1, 20, DEFAULT_TOP_N, help="Funds to Buy")
    target_n = c3.number_input("Target N Funds", 1, 20, DEFAULT_TARGET_N, help="Hit Rate Success Target")
    hold = c4.number_input("Holding Period (Days)", 20, 504, DEFAULT_HOLDING)
    
    with st.spinner("Loading Data..."):
        nav, maps = load_fund_data_raw(cat)
        nifty = load_nifty_data()
    
    if nav is None: st.error("Data not found."); return

    tabs = st.tabs([
        "🏆 Compare All", 
        # NEW STRATEGIES
        "🆕 Quality Mom", "🆕 Persistent α", "🆕 Regime Adapt", 
        "🆕 Trend Confirm", "🆕 Best Day", "🆕 Low Vol Mom", "🆕 Composite",
        # EXISTING
        "⚓ Stable Mom", "🧩 Ensemble", "🧠 Consistent α", "🛡️ Martin", "🚀 Momentum", "⚖️ Sharpe", "Custom"
    ])
    
    with tabs[0]:
        render_comparison_tab(nav, maps, nifty, top_n, target_n, hold)

    # NEW STRATEGIES TABS
    with tabs[1]:
        display_backtest_results(nav, maps, nifty, 'quality_momentum', top_n, target_n, hold, {}, {},
            description="🆕 **Quality Momentum**: Combines momentum with R-squared trend quality. Smooth, consistent uptrends are more likely to continue than volatile ones.")
    
    with tabs[2]:
        display_backtest_results(nav, maps, nifty, 'persistent_alpha', top_n, target_n, hold, {}, {},
            description="🆕 **Persistent Alpha**: Tracks funds that consistently rank in top quartiles across multiple quarters. Skill persists; luck doesn't.")
    
    with tabs[3]:
        display_backtest_results(nav, maps, nifty, 'regime_adaptive', top_n, target_n, hold, {}, {},
            description="🆕 **Regime Adaptive**: Funds that perform well in BOTH high and low volatility environments. Robust across conditions = better future performance.")
    
    with tabs[4]:
        display_backtest_results(nav, maps, nifty, 'trend_confirmation', top_n, target_n, hold, {}, {},
            description="🆕 **Trend Confirmation**: Multi-timeframe trend analysis (50/200 day MAs). Confirmed trends have higher continuation probability.")
    
    with tabs[5]:
        display_backtest_results(nav, maps, nifty, 'best_day_capture', top_n, target_n, hold, {}, {},
            description="🆕 **Best Day Capture**: Funds that capture the biggest up days in the market. Missing best days kills returns.")
    
    with tabs[6]:
        display_backtest_results(nav, maps, nifty, 'low_vol_momentum', top_n, target_n, hold, {}, {},
            description="🆕 **Low Volatility Momentum**: Momentum filtered for lower volatility. Low vol + momentum = more sustainable outperformance.")
    
    with tabs[7]:
        display_backtest_results(nav, maps, nifty, 'composite_predictor', top_n, target_n, hold, {}, {},
            description="🆕 **Composite Predictor**: Ensemble of best predictive signals (Quality Momentum, Rank Persistence, Regime Adaptability, Trend, Sharpe).")

    # EXISTING STRATEGIES TABS
    with tabs[8]:
        display_backtest_results(nav, maps, nifty, 'stable_momentum', top_n, target_n, hold, {}, {},
            description="⚓ **Stable Momentum**: Top funds by Momentum, filtered for Lowest Drawdown.")

    with tabs[9]:
        display_backtest_results(nav, maps, nifty, 'ensemble', top_n, target_n, hold, {}, 
            {'w_3m':0.33, 'w_6m':0.33, 'w_12m':0.33, 'risk_adjust':True}, 
            {'momentum':0.4, 'capture':0.6},
            description="🧩 **Ensemble Strategy**: Weighted combination of Momentum (40%) and Capture Ratio (60%).")

    with tabs[10]:
        display_backtest_results(nav, maps, nifty, 'consistent_alpha', top_n, target_n, hold, {}, {},
            description="🧠 **Consistent Alpha**: Rolling Win Rate (60%) + Information Ratio (40%).")

    with tabs[11]:
        display_backtest_results(nav, maps, nifty, 'martin', top_n, target_n, hold, {}, {},
            description="🛡️ **Martin Ratio**: Risk-adjusted return using Ulcer Index (penalizes drawdown duration).")
        
    with tabs[12]:
        display_backtest_results(nav, maps, nifty, 'momentum', top_n, target_n, hold, {}, 
            {'w_3m':0.33, 'w_6m':0.33, 'w_12m':0.33, 'risk_adjust':True},
            description="🚀 **Momentum**: Weighted 3/6/12 month momentum with volatility adjustment.")

    with tabs[13]:
        display_backtest_results(nav, maps, nifty, 'sharpe', top_n, target_n, hold, {}, {},
            description="⚖️ **Sharpe Ratio**: Classic risk-adjusted return metric.")

    with tabs[14]:
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
    st.title("📈 Advanced Fund Analysis with High-Hit-Rate Strategies")
    
    st.markdown("""
    ### New Strategies Added for Higher Hit Rate:
    
    | Strategy | Key Insight | Why Higher Hit Rate? |
    |----------|-------------|---------------------|
    | **Quality Momentum** | R² of price trend | Smooth trends continue more reliably |
    | **Persistent Alpha** | Quarterly rank consistency | Skill persists; luck doesn't |
    | **Regime Adaptive** | Performance in both high/low vol | Robust funds are more predictable |
    | **Trend Confirmation** | Multi-timeframe trend alignment | Confirmed trends = higher continuation |
    | **Best Day Capture** | Performance on market's best days | Good stock pickers capture upside |
    | **Low Vol Momentum** | Momentum + low volatility | Sustainable outperformance |
    | **Composite Predictor** | Ensemble of predictive signals | Multiple signals = more robust |
    """)
    
    t1, t2 = st.tabs(["📂 Category Explorer", "🚀 Strategy Backtester"])
    with t1: render_explorer_tab()
    with t2: render_backtest_tab()


if __name__ == "__main__":
    main()
