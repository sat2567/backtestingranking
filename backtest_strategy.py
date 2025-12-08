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

# Default Parameters
DEFAULT_HOLDING = 126              # Approx 6 months holding (trading days)
DEFAULT_TOP_N = 5
RISK_FREE_RATE = 0.06              # 6% annual
EXECUTION_LAG = 1                  # Days between Analysis and Entry (T+1)
TRADING_DAYS_YEAR = 252            # Standardized trading days in a year

# Data Cleaning Parameters
MAX_DATA_DATE = pd.Timestamp('2025-12-05')    # Maximum date for data
LAST_REBALANCE_DATE = pd.Timestamp('2025-05-01')  # Last allowed rebalance date

# Calculate Daily Risk Free Rate (Compounded)
DAILY_RISK_FREE_RATE = (1 + RISK_FREE_RATE) ** (1/TRADING_DAYS_YEAR) - 1

CATEGORY_MAP = {
    "Large Cap": "largecap",
    "Small Cap": "smallcap",
    "Mid Cap": "midcap",
    "Large & Mid Cap": "large_and_midcap",
    "Multi Cap": "multicap",
    "International": "international",
}

# Data locations
DATA_DIR = "data"
FILE_MAPPING = {
    "largecap": "largecap_merged.xlsx",
    "smallcap": "smallcap.xlsx",
    "midcap": "midcap.xlsx",
    "large_and_midcap": "large_and_midcap_fund.xlsx",
    "multicap": "MULTICAP.xlsx",
    "international": "international_merged.xlsx",
}

# ============================================================================
# 2. HELPER FUNCTIONS (DATA CLEANING & MATH)
# ============================================================================

def clean_weekday_data(df):
    """
    Clean data to only include Monday-Friday:
    1. Keep only weekday data (Monday=0 to Friday=4)
    2. Remove weekend data (Saturday=5, Sunday=6)
    3. Forward fill missing weekday data (up to 5 days)
    4. Filter to max date of Dec 1, 2025
    """
    if df is None or df.empty:
        return df
    
    # Filter to max date first
    df = df[df.index <= MAX_DATA_DATE]
    
    # Remove weekend data (Saturday=5, Sunday=6)
    df = df[df.index.dayofweek < 5]
    
    # Create a complete weekday date range
    if len(df) > 0:
        start_date = df.index.min()
        end_date = min(df.index.max(), MAX_DATA_DATE)
        
        # Generate all dates in range
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Keep only weekdays
        weekdays = all_dates[all_dates.dayofweek < 5]
        
        # Reindex to include all weekdays
        df = df.reindex(weekdays)
        
        # Forward fill missing weekday data (up to 5 trading days)
        df = df.ffill(limit=5)
    
    return df

def calculate_sharpe_ratio(returns):
    """Returns Sharpe Ratio (Higher is Better)"""
    if len(returns) < 10 or returns.std() == 0: return np.nan
    excess_returns = returns - DAILY_RISK_FREE_RATE
    return (excess_returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS_YEAR)

def calculate_sortino_ratio(returns):
    """Returns Sortino Ratio (Higher is Better)"""
    if len(returns) < 10: return np.nan
    downside = returns[returns < 0]
    if len(downside) == 0: return np.nan
    downside_std = downside.std()
    if downside_std == 0: return np.nan
    mean_return = (returns - DAILY_RISK_FREE_RATE).mean()
    return (mean_return / downside_std) * np.sqrt(TRADING_DAYS_YEAR)

def calculate_vol_adj_return(series):
    """Volatility-Adjusted Return (annualized return / annualized volatility)"""
    if series is None or len(series) < 63:
        return np.nan
    cleaned = series.dropna()
    if len(cleaned) < 63:
        return np.nan
    returns = cleaned.pct_change().dropna()
    if returns.empty:
        return np.nan
    total_days = (cleaned.index[-1] - cleaned.index[0]).days
    if total_days <= 0:
        return np.nan
    years = total_days / 365.25
    if years <= 0:
        return np.nan
    start_val, end_val = cleaned.iloc[0], cleaned.iloc[-1]
    if start_val <= 0 or end_val <= 0:
        return np.nan
    annual_return = (end_val / start_val) ** (1 / years) - 1
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS_YEAR)
    if annual_vol == 0 or np.isnan(annual_vol):
        return np.nan
    return annual_return / annual_vol

def calculate_volatility(returns):
    """Returns Annualized Volatility (Lower is Better)"""
    if len(returns) < 10: return np.nan
    return returns.std() * np.sqrt(TRADING_DAYS_YEAR)

def calculate_information_ratio(fund_returns, bench_returns):
    """Returns Information Ratio (Higher is Better)"""
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 10: return np.nan
    
    f_ret = fund_returns.loc[common_idx]
    b_ret = bench_returns.loc[common_idx]
    
    active_return = f_ret - b_ret
    tracking_error = active_return.std() * np.sqrt(TRADING_DAYS_YEAR)
    
    if tracking_error == 0: return np.nan
    return (active_return.mean() * TRADING_DAYS_YEAR) / tracking_error

def calculate_max_dd(series):
    """Returns Maximum Drawdown (Lower is Better, as more negative)"""
    if len(series) < 10 or series.isna().all(): return np.nan
    cum_returns = (1 + series.pct_change().dropna()).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def calculate_calmar_ratio(series, analysis_date):
    """Returns Calmar Ratio (Higher is Better)"""
    if len(series) < 10 or series.isna().all(): return np.nan
    start_date = series.index[0]
    end_val = series.iloc[-1]
    start_val = series.iloc[0]
    years = (analysis_date - start_date).days / 365.25
    if years <= 0 or start_val <= 0: return np.nan
    cagr = (end_val / start_val) ** (1 / years) - 1
    max_dd = calculate_max_dd(series)
    if max_dd >= 0: return np.nan  # No drawdown
    return cagr / abs(max_dd)

def calculate_beta_alpha_treynor(fund_returns, bench_returns):
    """Returns Beta, Alpha (annual), Treynor Ratio (Higher Treynor/Alpha Better)"""
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 30: return np.nan, np.nan, np.nan
    
    f_ret = fund_returns.loc[common_idx]
    b_ret = bench_returns.loc[common_idx]
    
    # Daily excess returns
    excess_fund = f_ret - DAILY_RISK_FREE_RATE
    excess_bench = b_ret - DAILY_RISK_FREE_RATE
    
    # Covariance matrix
    cov_matrix = np.cov(excess_fund, excess_bench)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else np.nan
    
    if np.isnan(beta): return np.nan, np.nan, np.nan
    
    # Annual excess returns
    annual_excess_fund = excess_fund.mean() * TRADING_DAYS_YEAR
    annual_excess_bench = excess_bench.mean() * TRADING_DAYS_YEAR
    
    alpha = annual_excess_fund - beta * annual_excess_bench
    treynor = annual_excess_fund / beta if beta != 0 else np.nan
    
    return beta, alpha, treynor

def calculate_flexible_momentum(series, w_3m, w_6m, w_12m, use_risk_adjust=False):
    """
    Calculates a composite momentum score based on user-defined weights.
    Best Practice: Divide by volatility (Risk Adjusted) if use_risk_adjust is True.
    """
    if len(series) < 70: return np.nan 
    
    price_cur = series.iloc[-1]
    current_date = series.index[-1]
    
    def get_past_price(days_ago):
        target_date = current_date - pd.Timedelta(days=days_ago)
        sub_series = series[series.index <= target_date]
        if sub_series.empty: return np.nan
        return sub_series.iloc[-1]

    # Calculate Returns
    ret_3m = 0.0
    ret_6m = 0.0
    ret_12m = 0.0
    
    if w_3m > 0:
        p_3m = get_past_price(91) # ~3 Months
        if pd.isna(p_3m) or p_3m == 0: return np.nan
        ret_3m = (price_cur / p_3m) - 1

    if w_6m > 0:
        p_6m = get_past_price(182) # ~6 Months
        if pd.isna(p_6m) or p_6m == 0: return np.nan
        ret_6m = (price_cur / p_6m) - 1

    if w_12m > 0:
        p_12m = get_past_price(365) # ~1 Year
        if pd.isna(p_12m) or p_12m == 0: return np.nan
        ret_12m = (price_cur / p_12m) - 1

    # Weighted Score
    raw_score = (ret_3m * w_3m) + (ret_6m * w_6m) + (ret_12m * w_12m)
    
    # Risk Adjustment
    if use_risk_adjust:
        date_1y_ago = current_date - pd.Timedelta(days=365)
        hist_vol_data = series[series.index >= date_1y_ago]
        if len(hist_vol_data) < 20: return np.nan
        
        vol = hist_vol_data.pct_change().dropna().std() * np.sqrt(TRADING_DAYS_YEAR)
        if vol == 0: return np.nan
        return raw_score / vol
        
    return raw_score

# ============================================================================
# 3. DATA LOADING
# ============================================================================

@st.cache_data
def _fetch_fund_data(category_key: str):
    filename = FILE_MAPPING.get(category_key)
    if not filename:
        raise FileNotFoundError(f"No file mapping configured for category: {category_key}")
    
    local_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)
    
    # Read Excel file without header (new wide format)
    df = pd.read_excel(local_path, header=None)
    
    # Extract fund names from row 2 (0-indexed), starting from column 1
    fund_names = df.iloc[2, 1:].tolist()
    
    # Data starts from row 4 (0-indexed)
    data_df = df.iloc[4:, :].copy()
    
    # Remove disclaimer row if present
    if isinstance(data_df.iloc[-1, 0], str) and 'Accord Fintech' in str(data_df.iloc[-1, 0]):
        data_df = data_df.iloc[:-1, :]
    
    # Parse dates
    dates = pd.to_datetime(data_df.iloc[:, 0], errors='coerce')
    nav_wide = pd.DataFrame(index=dates)
    scheme_map = {}
    
    for i, fund_name in enumerate(fund_names):
        if pd.notna(fund_name) and fund_name != '':
            scheme_code = str(abs(hash(fund_name)) % (10 ** 8))
            scheme_map[scheme_code] = fund_name
            nav_values = pd.to_numeric(data_df.iloc[:, i+1], errors='coerce')
            nav_wide[scheme_code] = nav_values.values
    
    nav_wide = nav_wide[nav_wide.index.notna()].sort_index()
    nav_wide = nav_wide[~nav_wide.index.duplicated(keep='last')]
    nav_wide = clean_weekday_data(nav_wide)
    
    return nav_wide, scheme_map


def load_fund_data(category_key: str, show_errors: bool = True):
    try:
        return _fetch_fund_data(category_key)
    except FileNotFoundError as e:
        if show_errors:
            missing_name = os.path.basename(str(e)) if os.path.isabs(str(e)) else str(e)
            st.error(f"File not found for '{category_key}' category: {missing_name}")
            st.info("Ensure the Excel exists under the data folder. For Large Cap run 'python merge_and_download_fixed.py'; for International run 'python merge_international.py'.")
        return None, None
    except Exception as e:
        if show_errors:
            st.error(f"Error loading data for '{category_key}': {e}")
            st.info("Close any open Excel files, rerun the merge scripts, then restart the dashboard.")
        return None, None


@st.cache_data(show_spinner=False)
def get_category_status():
    """Return summary info (fund count & last date) for each category."""
    status_rows = []
    for display_name, key in CATEGORY_MAP.items():
        nav_wide, scheme_map = load_fund_data(key, show_errors=False)
        if nav_wide is not None and not nav_wide.empty:
            last_date = nav_wide.index.max()
            status_rows.append({
                'category': display_name,
                'key': key,
                'last_date': last_date,
                'funds': len(scheme_map),
                'total_days': len(nav_wide),
            })
        else:
            status_rows.append({
                'category': display_name,
                'key': key,
                'last_date': None,
                'funds': len(scheme_map) if scheme_map else 0,
                'total_days': 0,
            })
    return status_rows

@st.cache_data
def load_nifty_data():
    # Load Nifty 100 data from local CSV file
    local_path = os.path.join(DATA_DIR, "nifty100_data.csv")
    
    try:
        if not os.path.exists(local_path):
            st.warning(f"Nifty 100 data not found at: {local_path}")
            st.info("Please ensure the nifty100_data.csv file exists in the data folder.")
            return None
            
        df = pd.read_csv(local_path)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Handle date parsing
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df = df.dropna(subset=['date', 'nav'])
        
        result = df.set_index('date')['nav'].sort_index()
        
        # Apply weekday cleaning: remove weekends, forward fill weekdays, filter to Dec 1, 2025
        result = clean_weekday_data(result.to_frame()).squeeze()
        
        if not result.empty:
            st.sidebar.success(f"📊 Nifty 100: {len(result)} days ({result.index.min().strftime('%Y-%m-%d')} to {result.index.max().strftime('%Y-%m-%d')})")
        
        return result
        
    except Exception as e:
        st.error(f"Error loading Nifty 100 data: {e}")
        st.info("Please check the nifty100_data.csv file format and ensure it contains 'date' and 'nav' columns.")
        return None

# ============================================================================
# 4. BACKTESTING ENGINE
# ============================================================================

def get_lookback_data(nav_wide, analysis_date, strategy_type):
    """Retrieves historical data based on strategy requirements."""
    # Buffer for 12M momentum + holidays
    max_days = 400 
    
    start_date = analysis_date - pd.Timedelta(days=max_days)
    
    hist_data = nav_wide[nav_wide.index >= start_date]
    hist_data = hist_data[hist_data.index < analysis_date]
    return hist_data

def run_backtest(nav_wide, strategy_type, top_n, holding_days, custom_weights=None, momentum_config=None, benchmark_series=None):
    
    # Find start index (require ~1 year history)
    start_date_required = nav_wide.index.min() + pd.Timedelta(days=370)
    try:
        start_idx = nav_wide.index.searchsorted(start_date_required)
    except:
        start_idx = 0
        
    natural_end_idx = len(nav_wide) - holding_days - EXECUTION_LAG
    if start_idx >= natural_end_idx:
        return pd.DataFrame(), pd.DataFrame()

    # Determine maximum index allowed by LAST_REBALANCE_DATE (pad if exact date missing)
    max_idx_allowed = nav_wide.index.searchsorted(LAST_REBALANCE_DATE, side='right') - 1
    max_idx_allowed = min(max_idx_allowed, len(nav_wide) - EXECUTION_LAG - 1)
    if max_idx_allowed < start_idx:
        return pd.DataFrame(), pd.DataFrame()
    
    base_end_idx = min(natural_end_idx, max_idx_allowed)
    if base_end_idx < start_idx:
        return pd.DataFrame(), pd.DataFrame()

    rebalance_indices = list(range(start_idx, base_end_idx + 1, holding_days))
    if max_idx_allowed >= start_idx:
        if not rebalance_indices:
            rebalance_indices = [max_idx_allowed]
        elif rebalance_indices[-1] != max_idx_allowed:
            rebalance_indices.append(max_idx_allowed)
    rebalance_indices = sorted(set(rebalance_indices))
    
    if not rebalance_indices:
        return pd.DataFrame(), pd.DataFrame()
    
    history, equity_curve = [], [{'date': nav_wide.index[rebalance_indices[0]], 'value': 100.0}]
    current_capital = 100.0
    
    for i in rebalance_indices:
        analysis_date = nav_wide.index[i]
        
        hist_data = get_lookback_data(nav_wide, analysis_date, strategy_type)
        scores = {}
        
        # --- A) STANDARD STRATEGIES ---
        if strategy_type != 'custom':
            for col in nav_wide.columns:
                series = hist_data[col].dropna()
                if len(series) < 126: continue
                
                val = np.nan
                if strategy_type == 'momentum':
                    w3 = momentum_config.get('w_3m', 0)
                    w6 = momentum_config.get('w_6m', 0)
                    w12 = momentum_config.get('w_12m', 0)
                    risk_adj = momentum_config.get('risk_adjust', False)
                    val = calculate_flexible_momentum(series, w3, w6, w12, risk_adj)
                elif strategy_type == 'var':
                    val = calculate_vol_adj_return(series)
                else:
                    date_1y_ago = analysis_date - pd.Timedelta(days=365)
                    short_series = series[series.index >= date_1y_ago]
                    rets = short_series.pct_change().dropna()
                    
                    if strategy_type == 'sharpe': val = calculate_sharpe_ratio(rets)
                    elif strategy_type == 'sortino': val = calculate_sortino_ratio(rets)
                
                if not np.isnan(val): scores[col] = val

        # --- B) CUSTOM STRATEGY ---
        else:
            temp_metrics = []
            bench_rets = None
            if benchmark_series is not None:
                bench_slice = get_lookback_data(benchmark_series.to_frame(), analysis_date, 'sharpe')
                bench_rets = bench_slice['nav'].pct_change().dropna()

            for col in nav_wide.columns:
                series = hist_data[col].dropna()
                if len(series) < 126: continue
                
                row = {'id': col}
                
                date_1y_ago = analysis_date - pd.Timedelta(days=365)
                short_series = series[series.index >= date_1y_ago]
                rets = short_series.pct_change().dropna()
                
                if custom_weights.get('sharpe', 0) > 0: row['sharpe'] = calculate_sharpe_ratio(rets)
                if custom_weights.get('sortino', 0) > 0: row['sortino'] = calculate_sortino_ratio(rets)
                if custom_weights.get('volatility', 0) > 0: row['volatility'] = calculate_volatility(rets)
                if custom_weights.get('info_ratio', 0) > 0 and benchmark_series is not None:
                     if bench_rets is not None and not bench_rets.empty:
                         row['info_ratio'] = calculate_information_ratio(rets, bench_rets)
                
                if custom_weights.get('max_dd', 0) > 0: row['max_dd'] = calculate_max_dd(series)
                if custom_weights.get('calmar', 0) > 0: row['calmar'] = calculate_calmar_ratio(series, analysis_date)
                
                if (custom_weights.get('beta', 0) > 0 or custom_weights.get('alpha', 0) > 0 or custom_weights.get('treynor', 0) > 0) and bench_rets is not None and not bench_rets.empty:
                    beta, alpha, treynor = calculate_beta_alpha_treynor(rets, bench_rets)
                    if custom_weights.get('beta', 0) > 0: row['beta'] = beta
                    if custom_weights.get('alpha', 0) > 0: row['alpha'] = alpha
                    if custom_weights.get('treynor', 0) > 0: row['treynor'] = treynor

                if custom_weights.get('momentum', 0) > 0: 
                    w3 = momentum_config.get('w_3m', 0)
                    w6 = momentum_config.get('w_6m', 0)
                    w12 = momentum_config.get('w_12m', 0)
                    risk_adj = momentum_config.get('risk_adjust', False)
                    row['momentum'] = calculate_flexible_momentum(series, w3, w6, w12, risk_adj)

                temp_metrics.append(row)
            
            if temp_metrics:
                metrics_df = pd.DataFrame(temp_metrics).set_index('id')
                final_score_col = pd.Series(0.0, index=metrics_df.index)
                
                for metric in ['sharpe', 'sortino', 'momentum', 'info_ratio', 'calmar', 'alpha', 'treynor']:
                    if metric in metrics_df.columns:
                        final_score_col = final_score_col.add(metrics_df[metric].rank(pct=True) * custom_weights[metric], fill_value=0)
                
                if 'volatility' in metrics_df.columns:
                    final_score_col = final_score_col.add(metrics_df['volatility'].rank(pct=True, ascending=False) * custom_weights['volatility'], fill_value=0)
                
                if 'max_dd' in metrics_df.columns:
                    final_score_col = final_score_col.add(metrics_df['max_dd'].rank(pct=True, ascending=False) * custom_weights['max_dd'], fill_value=0)
                
                if 'beta' in metrics_df.columns:
                    final_score_col = final_score_col.add(metrics_df['beta'].rank(pct=True) * custom_weights['beta'], fill_value=0)

                scores = final_score_col.to_dict()

        if not scores: continue
        selected = [k for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        
        # --- EXECUTION ---
        entry_idx = i + EXECUTION_LAG
        exit_idx = entry_idx + holding_days
        if exit_idx >= len(nav_wide): break
        
        entry_date, exit_date = nav_wide.index[entry_idx], nav_wide.index[exit_idx]
        
        # Calculate Absolute Return (NO TAX)
        period_rets = (nav_wide.iloc[exit_idx] / nav_wide.iloc[entry_idx]) - 1
        
        valid_rets = period_rets[selected].dropna()
        avg_ret = valid_rets.mean() if not valid_rets.empty else 0.0
        
        current_capital *= (1 + avg_ret)
        
        history.append({
            'analysis_date': analysis_date,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'selected_funds': ",".join(map(str, selected)),
            'period_return': avg_ret,
            'cum_value': current_capital
        })
        equity_curve.append({'date': exit_date, 'value': current_capital})
        
    return pd.DataFrame(history), pd.DataFrame(equity_curve)

def generate_snapshot_table(nav_wide, analysis_date, holding_days, strategy_type, names_map, custom_weights=None, momentum_config=None, benchmark_series=None):
    try: idx = nav_wide.index.get_loc(analysis_date)
    except KeyError: return pd.DataFrame()

    entry_idx = idx + EXECUTION_LAG
    exit_idx = entry_idx + holding_days
    has_future = False if exit_idx >= len(nav_wide) else True
    
    hist_data = get_lookback_data(nav_wide, analysis_date, strategy_type) 
    
    bench_rets = None
    if benchmark_series is not None:
        bench_slice = get_lookback_data(benchmark_series.to_frame(), analysis_date, 'sharpe')
        bench_rets = bench_slice['nav'].pct_change().dropna()

    temp_data = []

    for col in nav_wide.columns:
        series = hist_data[col].dropna()
        if len(series) < 126: continue
        
        row = {'id': col, 'name': names_map.get(col, col)}
        
        date_1y_ago = analysis_date - pd.Timedelta(days=365)
        short_series = series[series.index >= date_1y_ago]
        rets = short_series.pct_change().dropna()

        # --- SCORES ---
        if strategy_type == 'sharpe':
            row['Score'] = calculate_sharpe_ratio(rets)
        elif strategy_type == 'sortino':
            row['Score'] = calculate_sortino_ratio(rets)
        elif strategy_type == 'momentum':
            w3, w6, w12 = momentum_config['w_3m'], momentum_config['w_6m'], momentum_config['w_12m']
            row['Score'] = calculate_flexible_momentum(series, w3, w6, w12, momentum_config['risk_adjust'])
        elif strategy_type == 'var':
            row['Score'] = calculate_vol_adj_return(series)
        elif strategy_type == 'custom':
            if custom_weights.get('sharpe',0)>0: row['sharpe'] = calculate_sharpe_ratio(rets)
            if custom_weights.get('sortino',0)>0: row['sortino'] = calculate_sortino_ratio(rets)
            if custom_weights.get('volatility',0)>0: row['volatility'] = calculate_volatility(rets)
            
            if custom_weights.get('momentum',0)>0: 
                 w3, w6, w12 = momentum_config['w_3m'], momentum_config['w_6m'], momentum_config['w_12m']
                 row['momentum'] = calculate_flexible_momentum(series, w3, w6, w12, momentum_config['risk_adjust'])

            if custom_weights.get('info_ratio',0)>0 and bench_rets is not None and not bench_rets.empty: 
                row['info_ratio'] = calculate_information_ratio(rets, bench_rets)
            
            if custom_weights.get('max_dd',0)>0: row['max_dd'] = calculate_max_dd(series)
            if custom_weights.get('calmar',0)>0: row['calmar'] = calculate_calmar_ratio(series, analysis_date)
            
            if (custom_weights.get('beta',0)>0 or custom_weights.get('alpha',0)>0 or custom_weights.get('treynor',0)>0) and bench_rets is not None and not bench_rets.empty:
                beta, alpha, treynor = calculate_beta_alpha_treynor(rets, bench_rets)
                if custom_weights.get('beta',0)>0: row['beta'] = beta
                if custom_weights.get('alpha',0)>0: row['alpha'] = alpha
                if custom_weights.get('treynor',0)>0: row['treynor'] = treynor
        
        # --- FORWARD RETURN (NO TAX) ---
        raw_ret = np.nan
        
        if has_future:
            p_entry = nav_wide[col].iloc[entry_idx]
            p_exit = nav_wide[col].iloc[exit_idx]
            
            if pd.notnull(p_entry) and pd.notnull(p_exit) and p_entry > 0:
                raw_ret = (p_exit / p_entry) - 1

        # Store as Percentage for display (e.g. 5.5 instead of 0.055)
        row['Forward Return %'] = raw_ret * 100 if not np.isnan(raw_ret) else np.nan
        
        temp_data.append(row)

    df = pd.DataFrame(temp_data)
    if df.empty: return df

    # --- RANKING ---
    if strategy_type == 'custom':
        df['Score'] = 0.0
        for metric in ['sharpe', 'sortino', 'momentum', 'info_ratio', 'calmar', 'alpha', 'treynor']:
            if metric in df.columns:
                df['Score'] = df['Score'].add(df[metric].rank(pct=True) * custom_weights[metric], fill_value=0)
        if 'volatility' in df.columns:
             df['Score'] = df['Score'].add(df['volatility'].rank(pct=True, ascending=False) * custom_weights['volatility'], fill_value=0)
        if 'max_dd' in df.columns:
             df['Score'] = df['Score'].add(df['max_dd'].rank(pct=True, ascending=False) * custom_weights['max_dd'], fill_value=0)
        if 'beta' in df.columns:
             df['Score'] = df['Score'].add(df['beta'].rank(pct=True) * custom_weights['beta'], fill_value=0)

    df['Strategy Rank'] = df['Score'].rank(ascending=False, method='min')
    
    if has_future:
        df['Actual Rank'] = df['Forward Return %'].rank(ascending=False, method='min')
    
    return df.sort_values('Strategy Rank')

# ============================================================================
# 5. DASHBOARD UI
# ============================================================================

def main():
    st.title("📊 Fund Analysis: Custom & Standard Strategies")
    
    # Data Cleaning Info
    st.info(f"📅 **Data Range:** Weekdays only (Monday-Friday) till {MAX_DATA_DATE.strftime('%B %d, %Y')} | **Last Rebalance:** {LAST_REBALANCE_DATE.strftime('%B %d, %Y')}")
    status_data = pd.DataFrame(get_category_status())
    
    # --- Sidebar ---
    st.sidebar.header("⚙️ General Settings")
    if not status_data.empty:
        with st.sidebar.expander("📅 Data Availability", expanded=False):
            display_df = status_data[['category', 'last_date', 'funds']].copy()
            display_df['last_date'] = display_df['last_date'].apply(lambda d: d.strftime('%Y-%m-%d') if pd.notna(d) else 'Missing')
            display_df = display_df.rename(columns={'category': 'Category', 'last_date': 'Last Date', 'funds': '# Funds'})
            st.table(display_df)
    cat_key = CATEGORY_MAP[st.sidebar.selectbox("Category", list(CATEGORY_MAP.keys()))]
    
    col_s1, col_s2 = st.sidebar.columns(2)
    top_n = col_s1.number_input("Top N Funds", 1, 20, DEFAULT_TOP_N)
    holding_period = col_s2.number_input("Rebalance (Days)", 20, TRADING_DAYS_YEAR, DEFAULT_HOLDING)
    
    st.sidebar.divider()
    st.sidebar.header("🚀 Momentum Configuration")
    st.sidebar.caption("Weights for Trend Calculation:")
    
    # Momentum Sliders
    mom_c1, mom_c2, mom_c3 = st.sidebar.columns(3)
    w_3m = mom_c1.number_input("3M Weight", 0.0, 10.0, 1.0, 0.1)
    w_6m = mom_c2.number_input("6M Weight", 0.0, 10.0, 1.0, 0.1)
    w_12m = mom_c3.number_input("1Y Weight", 0.0, 10.0, 1.0, 0.1)
    
    risk_adjust_mom = st.sidebar.checkbox("Risk Adjust Momentum?", value=True, help="Divide score by Volatility")
    
    # Normalize weights
    total_w = w_3m + w_6m + w_12m
    momentum_config = {
        'w_3m': w_3m/total_w if total_w > 0 else 0, 
        'w_6m': w_6m/total_w if total_w > 0 else 0, 
        'w_12m': w_12m/total_w if total_w > 0 else 0,
        'risk_adjust': risk_adjust_mom
    }
    
    # --- Load Data ---
    with st.spinner("Loading Data..."):
        nav_data, names_map = load_fund_data(cat_key)
        nifty_data = load_nifty_data()
        
    if nav_data is None:
        st.error("❌ Fund data not found or failed to load."); return

    # --- Display Function ---
    def display_strategy_results(strat_type, c_weights=None):
        hist_df, eq_curve = run_backtest(nav_data, strat_type, top_n, holding_period, c_weights, momentum_config, nifty_data)
        
        if hist_df is None or hist_df.empty:
            st.warning("Insufficient data for backtest."); return

        # Summary
        start_date = eq_curve.iloc[0]['date']
        end_date = eq_curve.iloc[-1]['date']
        final_val = eq_curve.iloc[-1]['value']
        time_period_years = (end_date-start_date).days/365.25
        strat_cagr = (final_val/100)**(1/time_period_years) - 1 if time_period_years > 0 else 0.0
        
        bench_curve = None
        bench_cagr = 0.0
        if nifty_data is not None:
            sub_nifty = nifty_data[(nifty_data.index >= start_date) & (nifty_data.index <= end_date)]
            if not sub_nifty.empty and sub_nifty.iloc[0] > 0:
                bench_cagr = (sub_nifty.iloc[-1]/sub_nifty.iloc[0])**(1/((sub_nifty.index[-1]-sub_nifty.index[0]).days/365.25)) - 1
                bench_curve = (sub_nifty / sub_nifty.iloc[0]) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy CAGR", f"{strat_cagr:.2%}")
        c2.metric("Benchmark CAGR", f"{bench_cagr:.2%}")
        c3.metric("Total Return", f"{final_val - 100:.1f}%")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_curve['date'], y=eq_curve['value'], name='Strategy', line=dict(color='#00CC96')))
        if bench_curve is not None:
            fig.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve.values, name='Benchmark', line=dict(color='#EF553B', dash='dot')))
        fig.update_layout(height=400, title="Equity Curve", hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🔍 Deep Dive Snapshot")
        date_options = hist_df['analysis_date'].dt.strftime('%Y-%m-%d').tolist()
        sel_date_str = st.selectbox("Select Rebalance Date", date_options, key=f"sd_{strat_type}")
        
        if sel_date_str:
            sel_date = pd.to_datetime(sel_date_str)
            df = generate_snapshot_table(nav_data, sel_date, holding_period, strat_type, names_map, c_weights, momentum_config, nifty_data)
            
            if not df.empty:
                # Highlight Top N
                def highlight_top_n(row):
                    if row['Strategy Rank'] <= top_n:
                        return ['background-color: green'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    df.style.format("{:.2f}", subset=['Forward Return %', 'Score'])
                            .apply(highlight_top_n, axis=1),
                    use_container_width=True,
                    column_config={
                        "Forward Return %": st.column_config.NumberColumn(
                            "Forward Return", 
                            help=f"Absolute Return over the next {holding_period} days.",
                            format="%.2f%%" 
                        ),
                    },
                    hide_index=True
                )

    # --- Tabs ---
    tab_mom, tab_sharpe, tab_var, tab_custom = st.tabs(["🚀 Momentum Strategy", "⚖️ Sharpe Ratio", "📉 VAR Strategy", "🛠️ Custom Strategy"])

    with tab_mom:
        st.info(f"Momentum Weights: 3M={momentum_config['w_3m']:.2f}, 6M={momentum_config['w_6m']:.2f}, 1Y={momentum_config['w_12m']:.2f}. Risk Adjust: {momentum_config['risk_adjust']}")
        display_strategy_results('momentum')

    with tab_sharpe:
        display_strategy_results('sharpe')

    with tab_var:
        st.info("Ranks funds by Volatility-Adjusted Return (annualized return divided by annualized volatility over the last year). Higher is better.")
        display_strategy_results('var')

    with tab_custom:
        with st.form("custom_strat_form"):
            col_c1, col_c2, col_c3 = st.columns(3)
            w_sharpe = col_c1.slider("Sharpe Weight", 0, 100, 50, 10)
            w_sortino = col_c2.slider("Sortino Weight", 0, 100, 0, 10)
            w_mom = col_c3.slider("Momentum Weight", 0, 100, 50, 10)
            
            col_c4, col_c5, col_c6 = st.columns(3)
            w_vol = col_c4.slider("Low Volatility Weight", 0, 100, 0, 10)
            w_ir = col_c5.slider("Info Ratio Weight", 0, 100, 0, 10)
            w_maxdd = col_c6.slider("Low MaxDD Weight", 0, 100, 0, 10)
            
            col_c7, col_c8, col_c9 = st.columns(3)
            w_calmar = col_c7.slider("Calmar Ratio Weight", 0, 100, 0, 10)
            w_beta = col_c8.slider("Beta Weight", 0, 100, 0, 10)
            w_alpha = col_c9.slider("Alpha Weight", 0, 100, 0, 10)
            
            col_c10, col_c11 = st.columns(2)
            w_treynor = col_c10.slider("Treynor Ratio Weight", 0, 100, 0, 10)
            
            submit_btn = st.form_submit_button("🚀 Run Custom Strategy")

        if submit_btn:
            weights = {
                'sharpe': w_sharpe/100, 'sortino': w_sortino/100, 'momentum': w_mom/100,
                'volatility': w_vol/100, 'info_ratio': w_ir/100, 'max_dd': w_maxdd/100,
                'calmar': w_calmar/100, 'beta': w_beta/100, 'alpha': w_alpha/100, 'treynor': w_treynor/100
            }
            if sum(weights.values()) == 0: st.error("Select at least one weight > 0")
            else:
                st.session_state['custom_run'] = True
                st.session_state['custom_weights'] = weights
        
        if st.session_state.get('custom_run'):
            display_strategy_results('custom', st.session_state['custom_weights'])

if __name__ == "__main__":
    if 'custom_run' not in st.session_state:
        st.session_state['custom_run'] = False
        st.session_state['custom_weights'] = {}
    main()
