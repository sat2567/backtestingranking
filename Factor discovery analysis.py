import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats
import pickle
import json

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
RESULTS_DIR = "factor_discovery_results"
MAX_DATA_DATE = pd.Timestamp('2025-12-05')

# Create results directory if not exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- File Mapping ---
FILE_MAPPING = {
    "Large Cap": "largecap_merged.xlsx",
    "Mid Cap": "midcap.xlsx",
    "Small Cap": "smallcap.xlsx",
    "Large & Mid Cap": "large_and_midcap_fund.xlsx",
    "Multi Cap": "MULTICAP.xlsx",
    "International": "international_merged.xlsx"
}

# --- Results File Paths ---
def get_results_paths():
    return {
        'factor_importance': os.path.join(RESULTS_DIR, 'factor_importance_all.csv'),
        'factor_comparison': os.path.join(RESULTS_DIR, 'factor_comparison_all.csv'),
        'run_metadata': os.path.join(RESULTS_DIR, 'run_metadata.json'),
        'category_summary': os.path.join(RESULTS_DIR, 'category_summary.csv'),
        'top_factors_by_category': os.path.join(RESULTS_DIR, 'top_factors_by_category.csv'),
        'backtest_results': os.path.join(RESULTS_DIR, 'backtest_results.csv'),
        'period_data': os.path.join(RESULTS_DIR, 'period_data.pkl')
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
# 3. RESULTS LOADING & SAVING
# ============================================================================

def load_cached_results():
    """Load previously saved results if they exist."""
    paths = get_results_paths()
    results = {
        'exists': False,
        'metadata': None,
        'factor_importance': None,
        'factor_comparison': None,
        'category_summary': None,
        'top_factors_by_category': None,
        'backtest_results': None,
        'period_data': None
    }
    
    # Check if metadata exists
    if not os.path.exists(paths['run_metadata']):
        return results
    
    try:
        # Load metadata
        with open(paths['run_metadata'], 'r') as f:
            results['metadata'] = json.load(f)
        
        # Load CSVs
        if os.path.exists(paths['factor_importance']):
            results['factor_importance'] = pd.read_csv(paths['factor_importance'], index_col=[0, 1])
        
        if os.path.exists(paths['factor_comparison']):
            results['factor_comparison'] = pd.read_csv(paths['factor_comparison'])
        
        if os.path.exists(paths['category_summary']):
            results['category_summary'] = pd.read_csv(paths['category_summary'], index_col=0)
        
        if os.path.exists(paths['top_factors_by_category']):
            results['top_factors_by_category'] = pd.read_csv(paths['top_factors_by_category'], index_col=0)
        
        if os.path.exists(paths['backtest_results']):
            results['backtest_results'] = pd.read_csv(paths['backtest_results'], index_col=0)
        
        # Load pickle data
        if os.path.exists(paths['period_data']):
            with open(paths['period_data'], 'rb') as f:
                results['period_data'] = pickle.load(f)
        
        results['exists'] = True
        
    except Exception as e:
        st.warning(f"Error loading cached results: {e}")
        results['exists'] = False
    
    return results


def save_results(factor_importance_all, factor_comparison_all, category_summary, 
                 top_factors_by_category, backtest_results, period_data_all, 
                 holding_period, top_n):
    """Save all results to disk."""
    paths = get_results_paths()
    
    try:
        # Save metadata
        metadata = {
            'last_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'holding_period': holding_period,
            'top_n': top_n,
            'categories_analyzed': list(FILE_MAPPING.keys()),
            'num_factors': len(factor_importance_all.index.get_level_values('factor').unique()) if factor_importance_all is not None else 0
        }
        with open(paths['run_metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save DataFrames
        if factor_importance_all is not None:
            factor_importance_all.to_csv(paths['factor_importance'])
        
        if factor_comparison_all is not None:
            factor_comparison_all.to_csv(paths['factor_comparison'], index=False)
        
        if category_summary is not None:
            category_summary.to_csv(paths['category_summary'])
        
        if top_factors_by_category is not None:
            top_factors_by_category.to_csv(paths['top_factors_by_category'])
        
        if backtest_results is not None:
            backtest_results.to_csv(paths['backtest_results'])
        
        # Save period data (pickle for complex objects)
        if period_data_all is not None:
            with open(paths['period_data'], 'wb') as f:
                pickle.dump(period_data_all, f)
        
        return True
    
    except Exception as e:
        st.error(f"Error saving results: {e}")
        return False


# ============================================================================
# 4. ALL FACTOR CALCULATIONS
# ============================================================================

def calculate_all_factors(series, bench_series=None, nav_df=None):
    """Calculate ALL possible factors for a fund at a given point in time."""
    factors = {}
    
    if len(series) < 126:
        return factors
    
    returns = series.pct_change().dropna()
    
    bench_rets = None
    if bench_series is not None:
        common_idx = returns.index.intersection(bench_series.index)
        if len(common_idx) > 30:
            bench_rets = bench_series.pct_change().dropna()
            bench_rets = bench_rets.loc[bench_rets.index.intersection(returns.index)]
    
    # ==========================================
    # RETURN-BASED FACTORS
    # ==========================================
    
    if len(series) >= 63:
        factors['momentum_3m'] = (series.iloc[-1] / series.iloc[-63]) - 1 if series.iloc[-63] > 0 else np.nan
    
    if len(series) >= 126:
        factors['momentum_6m'] = (series.iloc[-1] / series.iloc[-126]) - 1 if series.iloc[-126] > 0 else np.nan
    
    if len(series) >= 252:
        factors['momentum_12m'] = (series.iloc[-1] / series.iloc[-252]) - 1 if series.iloc[-252] > 0 else np.nan
        factors['momentum_12m_ex_1m'] = (series.iloc[-21] / series.iloc[-252]) - 1 if series.iloc[-252] > 0 else np.nan
    
    # ==========================================
    # RISK-ADJUSTED FACTORS
    # ==========================================
    
    if len(returns) >= 126 and returns.std() > 0:
        excess_ret = returns - DAILY_RISK_FREE_RATE
        factors['sharpe'] = (excess_ret.mean() / returns.std()) * np.sqrt(TRADING_DAYS_YEAR)
    
    if len(returns) >= 126:
        downside = returns[returns < 0]
        if len(downside) > 10 and downside.std() > 0:
            excess_ret = (returns - DAILY_RISK_FREE_RATE).mean()
            factors['sortino'] = (excess_ret / downside.std()) * np.sqrt(TRADING_DAYS_YEAR)
    
    if len(series) >= 252:
        max_dd = calculate_max_dd(series)
        if max_dd is not None and max_dd < 0:
            years = len(series) / TRADING_DAYS_YEAR
            cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1 if years > 0 else 0
            factors['calmar'] = cagr / abs(max_dd)
    
    # ==========================================
    # VOLATILITY FACTORS
    # ==========================================
    
    if len(returns) >= 63:
        factors['volatility'] = returns.std() * np.sqrt(TRADING_DAYS_YEAR)
    
    if len(returns) >= 63:
        downside = returns[returns < 0]
        if len(downside) > 10:
            factors['downside_vol'] = downside.std() * np.sqrt(TRADING_DAYS_YEAR)
    
    if len(returns) >= 126:
        rolling_vol = returns.rolling(21).std()
        factors['vol_of_vol'] = rolling_vol.std() if rolling_vol.std() > 0 else np.nan
    
    # ==========================================
    # DRAWDOWN FACTORS
    # ==========================================
    
    max_dd = calculate_max_dd(series)
    if max_dd is not None:
        factors['max_drawdown'] = max_dd
    
    if len(series) >= 126:
        cum_max = series.expanding(min_periods=1).max()
        drawdowns = (series / cum_max) - 1
        factors['avg_drawdown'] = drawdowns.mean()
        in_drawdown = series < cum_max
        factors['drawdown_duration_pct'] = in_drawdown.sum() / len(series)
    
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
        
        if len(common_idx) > 30:
            cov_matrix = np.cov(f_ret, b_ret)
            if cov_matrix[1, 1] > 0:
                factors['beta'] = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        if 'beta' in factors:
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
        
        if 'up_capture' in factors and 'down_capture' in factors:
            if factors['down_capture'] > 0:
                factors['capture_ratio'] = factors['up_capture'] / factors['down_capture']
        
        factors['batting_avg'] = (f_ret > b_ret).sum() / len(common_idx)
        
        if len(down_market) > 10:
            fund_on_down = f_ret.loc[down_market.index]
            factors['win_rate_down_days'] = (fund_on_down > down_market).sum() / len(down_market)
    
    # ==========================================
    # CONSISTENCY FACTORS
    # ==========================================
    
    if len(returns) >= 126:
        monthly_rets = series.resample('ME').last().pct_change().dropna()
        if len(monthly_rets) > 3:
            factors['pct_positive_months'] = (monthly_rets > 0).sum() / len(monthly_rets)
    
    if len(returns) >= 252:
        rolling_sharpe = returns.rolling(63).apply(
            lambda x: (x.mean() - DAILY_RISK_FREE_RATE) / x.std() * np.sqrt(TRADING_DAYS_YEAR) if x.std() > 0 else 0
        )
        factors['sharpe_consistency'] = 1 - (rolling_sharpe.std() / abs(rolling_sharpe.mean())) if rolling_sharpe.mean() != 0 else 0
    
    if nav_df is not None and len(series) >= 252:
        rank_persist = calculate_rank_persistence(series, nav_df)
        if rank_persist is not None:
            factors['rank_persistence'] = rank_persist
    
    # ==========================================
    # TREND QUALITY FACTORS
    # ==========================================
    
    if len(series) >= 126:
        y = np.log(series.iloc[-126:].values)
        x = np.arange(len(y))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            factors['trend_r_squared'] = r_value ** 2
            factors['trend_slope'] = slope * TRADING_DAYS_YEAR
        except:
            pass
    
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
    
    if len(returns) >= 126:
        factors['skewness'] = returns.skew()
        factors['kurtosis'] = returns.kurtosis()
        factors['var_95'] = returns.quantile(0.05)
        var_threshold = returns.quantile(0.05)
        factors['cvar_95'] = returns[returns <= var_threshold].mean()
    
    # ==========================================
    # REGIME FACTORS
    # ==========================================
    
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
# 5. FACTOR DISCOVERY ENGINE (FOR SINGLE CATEGORY)
# ============================================================================

def run_factor_discovery_single(nav_df, benchmark_series, holding_period=252, top_n=5, 
                                 category_name="Unknown", scheme_map=None, progress_callback=None):
    """Run factor discovery for a single category."""
    
    min_history = 370 + holding_period
    start_date = nav_df.index.min() + pd.Timedelta(days=min_history)
    
    if start_date >= nav_df.index.max():
        return None, None, None
    
    rebal_dates = pd.date_range(
        start=start_date, 
        end=nav_df.index.max() - pd.Timedelta(days=holding_period), 
        freq='Q'
    )
    
    factor_winner_comparison = []
    period_data = []
    
    for idx, analysis_date in enumerate(rebal_dates):
        if progress_callback:
            progress_callback(idx, len(rebal_dates), analysis_date)
        
        try:
            analysis_idx = nav_df.index.get_loc(nav_df.index.asof(analysis_date))
        except:
            continue
        
        entry_idx = analysis_idx + 1
        exit_idx = entry_idx + holding_period
        
        if exit_idx >= len(nav_df):
            continue
        
        forward_returns = (nav_df.iloc[exit_idx] / nav_df.iloc[entry_idx]) - 1
        forward_returns = forward_returns.dropna()
        
        if len(forward_returns) < 10:
            continue
        
        actual_top_n = forward_returns.nlargest(top_n).index.tolist()
        actual_bottom_n = forward_returns.nsmallest(top_n).index.tolist()
        
        hist_data = nav_df.loc[:nav_df.index[analysis_idx]]
        
        bench_series_hist = None
        if benchmark_series is not None:
            bench_series_hist = benchmark_series.loc[:nav_df.index[analysis_idx]]
        
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
        
        factors_df = pd.DataFrame(fund_factors).T
        
        for factor in factors_df.columns:
            factor_values = factors_df[factor].dropna()
            
            if len(factor_values) < 10:
                continue
            
            winner_values = factor_values.loc[factor_values.index.intersection(actual_top_n)]
            loser_values = factor_values.loc[factor_values.index.intersection(actual_bottom_n)]
            all_values = factor_values
            
            if len(winner_values) < 2 or len(loser_values) < 2:
                continue
            
            winner_mean = winner_values.mean()
            loser_mean = loser_values.mean()
            all_mean = all_values.mean()
            all_std = all_values.std()
            
            if all_std > 0:
                effect_size = (winner_mean - all_mean) / all_std
            else:
                effect_size = 0
            
            rest_values = factor_values.loc[~factor_values.index.isin(actual_top_n)]
            if len(rest_values) > 5:
                try:
                    t_stat, p_value = stats.ttest_ind(winner_values, rest_values)
                except:
                    t_stat, p_value = np.nan, np.nan
            else:
                t_stat, p_value = np.nan, np.nan
            
            top_quartile_threshold = factor_values.quantile(0.75)
            winners_in_top_quartile = (winner_values >= top_quartile_threshold).sum() / len(winner_values)
            
            factor_winner_comparison.append({
                'category': category_name,
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
        
        period_data.append({
            'category': category_name,
            'period': analysis_date,
            'holding_period': holding_period,
            'top_n_funds': actual_top_n,
            'top_n_fund_names': [scheme_map.get(f, f) for f in actual_top_n] if scheme_map else actual_top_n,
            'top_n_returns': forward_returns.loc[actual_top_n].tolist()
        })
    
    if not factor_winner_comparison:
        return None, None, None
    
    comparison_df = pd.DataFrame(factor_winner_comparison)
    
    factor_importance = comparison_df.groupby('factor').agg({
        'effect_size': ['mean', 'std'],
        'winners_higher': 'mean',
        'winners_in_top_quartile': 'mean',
        'p_value': lambda x: (x < 0.05).mean(),
        'period': 'count'
    }).round(3)
    
    factor_importance.columns = ['Avg_Effect_Size', 'Effect_Size_Std', 'Pct_Winners_Higher', 
                                  'Avg_Winners_TopQuartile', 'Pct_Significant', 'Num_Periods']
    
    factor_importance['Predictive_Score'] = (
        factor_importance['Pct_Winners_Higher'] * 0.3 +
        factor_importance['Avg_Winners_TopQuartile'] * 0.3 +
        factor_importance['Pct_Significant'] * 0.2 +
        (factor_importance['Avg_Effect_Size'].abs() / factor_importance['Avg_Effect_Size'].abs().max()) * 0.2
    )
    
    factor_importance = factor_importance.sort_values('Predictive_Score', ascending=False)
    factor_importance['category'] = category_name
    
    return factor_importance, comparison_df, period_data


# ============================================================================
# 6. RUN ALL CATEGORIES
# ============================================================================

def run_factor_discovery_all_categories(holding_period=252, top_n=5):
    """Run factor discovery for ALL categories and aggregate results."""
    
    all_factor_importance = []
    all_factor_comparison = []
    all_period_data = []
    category_summaries = []
    backtest_results = []
    
    benchmark = load_nifty_data()
    
    total_categories = len(FILE_MAPPING)
    
    # Create progress containers
    overall_progress = st.progress(0)
    category_status = st.empty()
    period_status = st.empty()
    
    for cat_idx, (category, filename) in enumerate(FILE_MAPPING.items()):
        category_status.markdown(f"### Processing: **{category}** ({cat_idx + 1}/{total_categories})")
        
        nav_df, scheme_map = load_fund_data_raw(category)
        
        if nav_df is None:
            st.warning(f"Could not load data for {category}")
            continue
        
        def progress_callback(period_idx, total_periods, current_date):
            period_status.text(f"  Period {period_idx + 1}/{total_periods}: {current_date.strftime('%Y-%m-%d')}")
        
        factor_importance, comparison_df, period_data = run_factor_discovery_single(
            nav_df, benchmark, holding_period, top_n, category, scheme_map, progress_callback
        )
        
        if factor_importance is not None:
            all_factor_importance.append(factor_importance)
            all_factor_comparison.append(comparison_df)
            all_period_data.extend(period_data)
            
            # Category summary
            top_3_factors = factor_importance.head(3).index.tolist()
            category_summaries.append({
                'category': category,
                'num_funds': len(nav_df.columns),
                'num_periods_analyzed': len(comparison_df['period'].unique()),
                'top_factor_1': top_3_factors[0] if len(top_3_factors) > 0 else None,
                'top_factor_2': top_3_factors[1] if len(top_3_factors) > 1 else None,
                'top_factor_3': top_3_factors[2] if len(top_3_factors) > 2 else None,
                'best_predictive_score': factor_importance['Predictive_Score'].max()
            })
            
            # Run data-driven backtest for this category
            hist, eq, bench = run_data_driven_backtest(
                nav_df, benchmark, factor_importance, holding_period, min(top_n, 3), scheme_map
            )
            
            if hist is not None and not eq.empty:
                start_date = eq.iloc[0]['date']
                end_date = eq.iloc[-1]['date']
                years = (end_date - start_date).days / 365.25
                
                strat_cagr = (eq.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                bench_cagr = (bench.iloc[-1]['value']/100)**(1/years)-1 if years > 0 and not bench.empty else 0
                avg_hit_rate = hist['hit_rate'].mean()
                
                backtest_results.append({
                    'category': category,
                    'strategy_cagr': strat_cagr,
                    'benchmark_cagr': bench_cagr,
                    'outperformance': strat_cagr - bench_cagr,
                    'avg_hit_rate': avg_hit_rate,
                    'num_trades': len(hist)
                })
        
        overall_progress.progress((cat_idx + 1) / total_categories)
    
    overall_progress.empty()
    category_status.empty()
    period_status.empty()
    
    if not all_factor_importance:
        return None, None, None, None, None, None
    
    # Combine all results
    combined_factor_importance = pd.concat(all_factor_importance)
    combined_factor_importance = combined_factor_importance.reset_index()
    combined_factor_importance = combined_factor_importance.set_index(['category', 'factor'])
    
    combined_comparison = pd.concat(all_factor_comparison, ignore_index=True)
    
    category_summary_df = pd.DataFrame(category_summaries).set_index('category')
    
    # Create top factors by category table
    top_factors_by_cat = []
    for cat_fi in all_factor_importance:
        cat_name = cat_fi['category'].iloc[0]
        for rank, (factor, row) in enumerate(cat_fi.head(10).iterrows(), 1):
            top_factors_by_cat.append({
                'category': cat_name,
                'rank': rank,
                'factor': factor,
                'predictive_score': row['Predictive_Score'],
                'pct_winners_higher': row['Pct_Winners_Higher']
            })
    
    top_factors_df = pd.DataFrame(top_factors_by_cat)
    top_factors_pivot = top_factors_df.pivot(index='factor', columns='category', values='rank')
    top_factors_pivot['avg_rank'] = top_factors_pivot.mean(axis=1)
    top_factors_pivot = top_factors_pivot.sort_values('avg_rank')
    
    backtest_df = pd.DataFrame(backtest_results).set_index('category') if backtest_results else None
    
    return (combined_factor_importance, combined_comparison, category_summary_df, 
            top_factors_pivot, backtest_df, all_period_data)


# ============================================================================
# 7. DATA-DRIVEN BACKTEST
# ============================================================================

def run_data_driven_backtest(nav_df, benchmark_series, factor_importance, holding_period=252, top_n=3, scheme_map=None):
    """Uses discovered factor weights to pick funds."""
    
    if factor_importance is None or factor_importance.empty:
        return None, None, None
    
    # Handle multi-index (category, factor) - get just factor level
    if isinstance(factor_importance.index, pd.MultiIndex):
        fi = factor_importance.reset_index(level=0, drop=True)
    else:
        fi = factor_importance
    
    top_factors = fi.head(10).index.tolist()
    
    weights = fi.loc[top_factors, 'Predictive_Score'].to_dict()
    total_weight = sum(weights.values())
    if total_weight == 0:
        return None, None, None
    weights = {k: v/total_weight for k, v in weights.items()}
    
    factor_direction = {}
    for factor in top_factors:
        pct_higher = fi.loc[factor, 'Pct_Winners_Higher']
        factor_direction[factor] = 1 if pct_higher > 0.5 else -1
    
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
        
        bench_series_hist = None
        if benchmark_series is not None:
            bench_series_hist = benchmark_series.loc[:date]
        
        fund_scores = {}
        
        for col in nav_df.columns:
            series = hist_data[col].dropna()
            if len(series) < 126:
                continue
            series.name = col
            
            factors = calculate_all_factors(series, bench_series_hist, hist_data)
            
            if not factors:
                continue
            
            score = 0
            valid_factors = 0
            
            for factor, weight in weights.items():
                if factor in factors and not pd.isna(factors[factor]):
                    normalized_value = factors[factor]
                    score += normalized_value * weight * factor_direction.get(factor, 1)
                    valid_factors += 1
            
            if valid_factors >= 3:
                fund_scores[col] = score
        
        if not fund_scores:
            selected = []
        else:
            score_series = pd.Series(fund_scores)
            selected = score_series.nlargest(top_n).index.tolist()
        
        entry = i + 1
        exit_i = min(i + 1 + holding_period, len(nav_df) - 1)
        
        b_ret = 0.0
        if benchmark_series is not None:
            try:
                b_ret = (benchmark_series.asof(nav_df.index[exit_i]) / benchmark_series.asof(nav_df.index[entry])) - 1
            except:
                pass
        
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
            'selected': selected,
            'selected_names': [scheme_map.get(s, s) for s in selected] if scheme_map else selected,
            'return': port_ret,
            'hit_rate': hit_rate
        })
        
        eq_curve.append({'date': nav_df.index[exit_i], 'value': cap})
        bench_curve.append({'date': nav_df.index[exit_i], 'value': b_cap})
    
    return pd.DataFrame(history), pd.DataFrame(eq_curve), pd.DataFrame(bench_curve)


# ============================================================================
# 8. UI COMPONENTS
# ============================================================================

def display_cached_results(results):
    """Display previously cached results."""
    
    metadata = results['metadata']
    
    st.success(f"✅ Showing results from last run: **{metadata['last_run']}**")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Holding Period", f"{metadata['holding_period']} days")
    col2.metric("Top N Analyzed", metadata['top_n'])
    col3.metric("Factors Analyzed", metadata['num_factors'])
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Cross-Category Analysis",
        "🏆 Category Comparison", 
        "🧬 Factor Rankings",
        "📈 Backtest Results",
        "🔬 Detailed Data"
    ])
    
    with tab1:
        st.subheader("Which Factors Predict Winners Across ALL Categories?")
        
        if results['top_factors_by_category'] is not None:
            st.markdown("""
            **How to read this table:**
            - Each cell shows the RANK of that factor within each category (1 = most predictive)
            - Lower average rank = more consistently predictive across categories
            - Factors at top are the "universal" winner predictors
            """)
            
            display_df = results['top_factors_by_category'].head(20).copy()
            
            # Style the dataframe
            st.dataframe(
                display_df.style.background_gradient(subset=['avg_rank'], cmap='Greens_r')
                .format('{:.1f}', na_rep='-'),
                use_container_width=True,
                height=500
            )
            
            # Top universal factors
            st.subheader("🎯 Top Universal Predictors (Work Across Categories)")
            top_universal = display_df.head(5)
            for idx, (factor, row) in enumerate(top_universal.iterrows(), 1):
                st.markdown(f"**{idx}. {factor}** - Avg Rank: {row['avg_rank']:.1f}")
    
    with tab2:
        st.subheader("Category-by-Category Summary")
        
        if results['category_summary'] is not None:
            st.dataframe(
                results['category_summary'].style.format({
                    'best_predictive_score': '{:.3f}'
                }),
                use_container_width=True
            )
            
            # Visualization
            fig = go.Figure()
            
            for category in results['category_summary'].index:
                row = results['category_summary'].loc[category]
                factors = [row['top_factor_1'], row['top_factor_2'], row['top_factor_3']]
                factors = [f for f in factors if f is not None]
                
                st.markdown(f"### {category}")
                st.markdown(f"**Top 3 Predictive Factors:** {', '.join(factors)}")
                st.markdown("---")
    
    with tab3:
        st.subheader("Factor Importance by Category")
        
        if results['factor_importance'] is not None:
            # Category selector
            categories = results['factor_importance'].index.get_level_values('category').unique().tolist()
            selected_cat = st.selectbox("Select Category", categories)
            
            # Filter for selected category
            cat_fi = results['factor_importance'].xs(selected_cat, level='category')
            cat_fi = cat_fi.sort_values('Predictive_Score', ascending=False)
            
            st.dataframe(
                cat_fi.head(20).style.background_gradient(subset=['Predictive_Score'], cmap='Greens')
                .format({
                    'Pct_Winners_Higher': '{:.1%}',
                    'Avg_Winners_TopQuartile': '{:.1%}',
                    'Pct_Significant': '{:.1%}',
                    'Predictive_Score': '{:.3f}'
                }),
                use_container_width=True
            )
            
            # Chart
            fig = go.Figure()
            top_15 = cat_fi.head(15)
            
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
                title=f'Factor Predictive Power - {selected_cat}',
                xaxis_tickangle=-45,
                yaxis_title='Percentage (%)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Data-Driven Strategy Backtest Results")
        
        if results['backtest_results'] is not None:
            bt = results['backtest_results']
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Strategy CAGR", f"{bt['strategy_cagr'].mean():.2%}")
            col2.metric("Avg Outperformance", f"{bt['outperformance'].mean():.2%}")
            col3.metric("Avg Hit Rate", f"{bt['avg_hit_rate'].mean():.1%}")
            
            # Table
            st.dataframe(
                bt.style.format({
                    'strategy_cagr': '{:.2%}',
                    'benchmark_cagr': '{:.2%}',
                    'outperformance': '{:.2%}',
                    'avg_hit_rate': '{:.1%}'
                }).background_gradient(subset=['avg_hit_rate'], cmap='Greens'),
                use_container_width=True
            )
            
            # Chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Strategy CAGR',
                x=bt.index,
                y=bt['strategy_cagr'] * 100,
                marker_color='green'
            ))
            
            fig.add_trace(go.Bar(
                name='Benchmark CAGR',
                x=bt.index,
                y=bt['benchmark_cagr'] * 100,
                marker_color='gray'
            ))
            
            fig.update_layout(
                barmode='group',
                title='Strategy vs Benchmark by Category',
                yaxis_title='CAGR (%)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Raw Factor Comparison Data")
        
        if results['factor_comparison'] is not None:
            st.markdown("Download the full factor comparison data for your own analysis:")
            
            csv = results['factor_comparison'].to_csv(index=False)
            st.download_button(
                "📥 Download Full Factor Comparison (CSV)",
                csv,
                "factor_comparison_all.csv",
                "text/csv"
            )
            
            st.dataframe(results['factor_comparison'].head(100), use_container_width=True)


def render_main_page():
    st.title("🧬 Factor Discovery: Winner DNA Analysis")
    
    st.markdown("""
    ### What This Does
    
    Discovers **what winning funds had in common** by analyzing historical data across ALL categories.
    
    **Process:**
    1. For each category and time period, identify the ACTUAL top 5 funds
    2. Analyze what factors these winners had BEFORE they won
    3. Find patterns that consistently predict winners
    4. Use discovered factors for future fund selection
    """)
    
    st.divider()
    
    # Check for cached results
    cached_results = load_cached_results()
    
    # Controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col1:
        if cached_results['exists']:
            st.info(f"📁 Last run: {cached_results['metadata']['last_run']}")
    
    with col2:
        holding_period = st.selectbox(
            "Holding Period", 
            [126, 252, 378, 504], 
            index=1,
            format_func=lambda x: f"{x} days"
        )
    
    with col3:
        top_n = st.number_input("Top N", 3, 10, 5)
    
    with col4:
        run_new = st.button("🔄 Run New Discovery (All Categories)", type="primary", use_container_width=True)
    
    st.divider()
    
    # Run new discovery if requested
    if run_new:
        st.warning("⏳ Running factor discovery for ALL categories. This may take several minutes...")
        
        with st.spinner("Processing..."):
            (factor_importance_all, factor_comparison_all, category_summary, 
             top_factors_by_cat, backtest_results, period_data_all) = run_factor_discovery_all_categories(
                holding_period, top_n
            )
        
        if factor_importance_all is not None:
            # Save results
            success = save_results(
                factor_importance_all, factor_comparison_all, category_summary,
                top_factors_by_cat, backtest_results, period_data_all,
                holding_period, top_n
            )
            
            if success:
                st.success("✅ Analysis complete! Results saved to disk.")
                st.rerun()
            else:
                st.error("Failed to save results")
        else:
            st.error("Analysis failed - no data returned")
    
    # Display cached results if available
    elif cached_results['exists']:
        display_cached_results(cached_results)
    
    else:
        st.info("👆 Click 'Run New Discovery' to analyze all categories and discover winning factors.")
        
        st.markdown("""
        ### What will be analyzed:
        
        | Category | Description |
        |----------|-------------|
        | Large Cap | Large cap mutual funds |
        | Mid Cap | Mid cap mutual funds |
        | Small Cap | Small cap mutual funds |
        | Large & Mid Cap | Combined large/mid cap funds |
        | Multi Cap | Multi cap mutual funds |
        | International | International/global funds |
        
        ### Factors that will be tested:
        
        - **Momentum**: 3m, 6m, 12m returns
        - **Risk-Adjusted**: Sharpe, Sortino, Calmar ratios
        - **Volatility**: Total, downside, vol-of-vol
        - **Drawdown**: Max DD, avg DD, recovery speed
        - **Benchmark-Relative**: Alpha, Beta, Information Ratio, Capture ratios
        - **Consistency**: Rank persistence, % positive months
        - **Trend**: R-squared, slope, MA relationships
        - **Higher Moments**: Skewness, kurtosis, VaR, CVaR
        - **Regime**: Performance in high/low volatility environments
        """)


def main():
    render_main_page()


if __name__ == "__main__":
    main()
