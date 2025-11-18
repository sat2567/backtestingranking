import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1️⃣ DATA PREPARATION
# ============================================================================
print("Loading NAV data...")

# --- CLI args for category/funds file ---
parser = argparse.ArgumentParser(description="Backtest fund selection across categories")
parser.add_argument("--category", default="largecap", choices=[
    "largecap", "smallcap", "midcap", "large_and_midcap", "multicap", "international"
], help="Category key to select default funds CSV and output suffix")
parser.add_argument("--funds-csv", dest="funds_csv", default=None, help="Path to funds NAV CSV (overrides category)")
parser.add_argument("--top-n", dest="top_n", type=int, default=None, help="Override TOP_N")
args, _ = parser.parse_known_args()

category_to_csv = {
    "largecap": "data/largecap_funds.csv",
    "smallcap": "data/smallcap_funds.csv",
    "midcap": "data/midcap_funds.csv",
    "large_and_midcap": "data/large_and_midcap_funds.csv",
    "multicap": "data/multicap_funds.csv",
    "international": "data/international_funds.csv",
}

FUNDS_CSV = args.funds_csv or category_to_csv.get(args.category, "data/largecap_funds.csv")

# infer category key from funds csv if user provided a path
def _infer_key_from_csv(path: str, fallback: str) -> str:
    base = os.path.basename(path).lower()
    for k, v in category_to_csv.items():
        if base == v.lower():
            return k
    # best-effort pattern strip
    if base.endswith("_funds.csv"):
        return base.replace("_funds.csv", "")
    return fallback

cat_key = args.category if args.funds_csv is None else _infer_key_from_csv(args.funds_csv, args.category)

df = pd.read_csv(FUNDS_CSV)

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['nav'] = pd.to_numeric(df['nav'], errors='coerce')

# Remove duplicates (keep last entry if multiple NAVs for same date)
df = df.drop_duplicates(subset=['scheme_code', 'date'], keep='last')
print(f"Records after removing duplicates: {len(df)}")

# Pivot to wide format (rows=dates, columns=scheme_codes, values=NAV)
nav_wide = df.pivot(index='date', columns='scheme_code', values='nav')
nav_wide = nav_wide.sort_index()

# Store scheme names mapping
scheme_names = df[['scheme_code', 'scheme_name']].drop_duplicates().set_index('scheme_code')['scheme_name'].to_dict()

print(f"Data shape: {nav_wide.shape} (dates x funds)")
print(f"Date range: {nav_wide.index.min()} to {nav_wide.index.max()}")

# Handle missing NAVs
# Forward fill up to 5 days
nav_wide = nav_wide.ffill(limit=5)

# Keep all funds (do not drop based on missingness). Metrics and forward returns
# will be computed using available data per fund and per window.
funds_to_keep = nav_wide.columns
print(f"Funds retained (no filtering): {len(funds_to_keep)}")
print(f"Final funds count: {nav_wide.shape[1]}")

# Load benchmark (Nifty 100) and compute daily returns
print("Loading benchmark (Nifty 100) data...")
bm_df = pd.read_csv('data/nifty100_fileter_data.csv')
bm_df['Date'] = pd.to_datetime(bm_df['Date'])
bm_df = bm_df.sort_values('Date')
bm_df = bm_df.set_index('Date')
benchmark = bm_df['Close'].astype(str).str.replace(' ', '').astype(float)
benchmark_returns = benchmark.pct_change().dropna()
print(f"Benchmark date range: {benchmark_returns.index.min()} to {benchmark_returns.index.max()}")

# ============================================================================
# 2️⃣ DEFINE PARAMETERS
# ============================================================================
LOOKBACK = 252          # 1 year in trading days
HOLDING_PERIOD = 126    # 6 months in trading days
TOP_N = int(os.getenv("TOP_N", "15"))
if args.top_n is not None:
    TOP_N = args.top_n
RISK_FREE_RATE = 0.05   # 5% annual

# Weights for composite score
WEIGHT_SHARPE = 0.4
WEIGHT_SORTINO = 0.4
WEIGHT_MDD = -0.2       # Negative because lower MDD is better

print("\n" + "="*70)
print("BACKTEST PARAMETERS")
print("="*70)
print(f"Lookback period: {LOOKBACK} days (~1 year)")
print(f"Holding period: {HOLDING_PERIOD} days (~6 months)")
print(f"Category: {cat_key}")
print(f"Funds file: {FUNDS_CSV}")
print(f"Top N funds: {TOP_N}")
print(f"Risk-free rate: {RISK_FREE_RATE*100}%")
print(f"Composite weights - Sharpe: {WEIGHT_SHARPE}, Sortino: {WEIGHT_SORTINO}, MDD: {WEIGHT_MDD}")

# ============================================================================
# 3️⃣ COMPUTE RISK METRICS FUNCTIONS
# ============================================================================

def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    excess_returns = returns - risk_free_rate/252
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
    return sharpe

def calculate_sortino_ratio(returns, risk_free_rate=0.05):
    """Calculate annualized Sortino ratio"""
    if len(returns) == 0:
        return np.nan
    excess_returns = returns - risk_free_rate/252
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.nan
    sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    return sortino

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    if len(prices) < 2:
        return np.nan
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown.min()

def annual_return(returns):
    if len(returns) == 0:
        return np.nan
    return returns.mean() * 252

def annual_volatility(returns):
    if len(returns) == 0:
        return np.nan
    return returns.std() * np.sqrt(252)

def tracking_error(active_returns):
    if len(active_returns) == 0:
        return np.nan
    return active_returns.std() * np.sqrt(252)

def information_ratio(active_returns):
    te = tracking_error(active_returns)
    if te is None or np.isnan(te) or te == 0 or len(active_returns) == 0:
        return np.nan
    return (active_returns.mean() * np.sqrt(252)) / te

def beta_vs_benchmark(fund_returns, bench_returns):
    aligned = pd.concat([fund_returns, bench_returns], axis=1, join='inner').dropna()
    if aligned.shape[0] < 20 or aligned.iloc[:,1].var() == 0:
        return np.nan
    cov = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])[0,1]
    var_b = aligned.iloc[:,1].var()
    return cov / var_b if var_b != 0 else np.nan

def compute_metrics_for_date(nav_wide, date_idx, lookback=252):
    """Compute risk and benchmark-aware metrics for all funds at a given date"""
    if date_idx < lookback:
        return None
    
    # Get lookback window
    lookback_data = nav_wide.iloc[date_idx-lookback:date_idx]
    window_dates = lookback_data.index
    
    # Benchmark returns for the same window
    bm_window = benchmark_returns.reindex(window_dates).dropna()
    
    metrics = {}
    for fund in nav_wide.columns:
        fund_prices = lookback_data[fund].astype(float).dropna()
        fund_returns = fund_prices.pct_change().dropna()
        # align fund and benchmark (may reduce rows)
        aligned = pd.concat([fund_returns, bm_window], axis=1, join='inner').dropna()
        aligned.columns = ['rf', 'rb']
        r_f = aligned['rf']
        r_b = aligned['rb']
        r_a = r_f - r_b

        sharpe = calculate_sharpe_ratio(fund_returns, RISK_FREE_RATE)
        sortino = calculate_sortino_ratio(fund_returns, RISK_FREE_RATE)
        mdd = calculate_max_drawdown(fund_prices)
        # annual return/vol should use ALL available fund data, not aligned subset
        ann_ret = annual_return(fund_returns)
        ann_vol = annual_volatility(fund_returns)
        # Momentum 6M: annualized return over last 126 days (if available)
        if len(fund_returns) >= 126:
            momentum_6m = annual_return(fund_returns.tail(126))
        else:
            momentum_6m = np.nan
        # IR/TE/Beta only if we have enough aligned observations and variance
        if len(r_f) >= 20 and r_f.std() > 0 and r_b.std() > 0:
            te = tracking_error(r_a)
            ir = information_ratio(r_a)
            beta = beta_vs_benchmark(r_f, r_b)
        else:
            te = np.nan
            ir = np.nan
            beta = np.nan
        
        metrics[fund] = {
            'sharpe': sharpe,
            'sortino': sortino,
            'mdd': mdd,
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'momentum_6m': momentum_6m,
            'te': te,
            'ir': ir,
            'beta': beta
        }
    
    return metrics

# ============================================================================
# 4️⃣ CREATE COMPOSITE SCORE & SELECT FUNDS
# ============================================================================

def normalize_metrics(metrics_dict):
    """Normalize metrics using z-score"""
    df = pd.DataFrame(metrics_dict).T
    
    # Z-score normalization
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0 and not np.isnan(std):
            df[col + '_norm'] = (df[col] - mean) / std
        else:
            df[col + '_norm'] = np.nan
    
    return df

def select_top_funds(metrics_dict, top_n=5):
    """Calculate composite score and select top N funds"""
    df = normalize_metrics(metrics_dict)
    
    # Weighted-available composite score across Sharpe, Sortino, MDD
    w_map = {
        'sharpe_norm': WEIGHT_SHARPE,
        'sortino_norm': WEIGHT_SORTINO,
        'mdd_norm': WEIGHT_MDD  # negative because lower MDD is better
    }
    # Numerator: sum(w*m) across available metrics; Denominator: sum(abs(w)) of available
    numer = np.zeros(len(df))
    denom = np.zeros(len(df))
    for col, w in w_map.items():
        if col in df.columns:
            z = df[col].astype(float)
            mask = z.notna()
            z = z.fillna(0.0)
            numer += w * z.values
            denom += (np.abs(w) * mask.astype(float)).values
    denom = np.where(denom == 0, np.nan, denom)
    df['composite_score'] = numer / denom
    
    # Rank by composite score
    df['composite_rank'] = df['composite_score'].rank(ascending=False)
    
    # Select top N
    top_funds = df.nsmallest(top_n, 'composite_rank').index.tolist()
    
    return top_funds, df

# ============================================================================
# 5️⃣ COMPUTE 6-MONTH FORWARD RETURNS & RANKS
# ============================================================================

def compute_forward_returns(nav_wide, date_idx, holding_period=126):
    """Compute forward returns for all funds"""
    if date_idx + holding_period >= len(nav_wide):
        return None
    
    current_prices = nav_wide.iloc[date_idx]
    future_prices = nav_wide.iloc[date_idx + holding_period]
    
    forward_returns = (future_prices / current_prices - 1).dropna()
    forward_ranks = forward_returns.rank(ascending=False)
    
    return forward_returns, forward_ranks

# ============================================================================
# 6️⃣ EVALUATE SELECTION ACCURACY
# ============================================================================

def evaluate_selection(selected_funds, forward_returns, forward_ranks, top_n=5):
    """Evaluate the accuracy of fund selection"""
    # Get actual top performers
    actual_top_funds = forward_returns.nlargest(top_n).index.tolist()
    
    # Calculate overlap
    overlap_count = len(set(selected_funds) & set(actual_top_funds))
    overlap_accuracy = overlap_count / top_n
    
    # Average rank of selected funds
    selected_ranks = [forward_ranks[fund] for fund in selected_funds if fund in forward_ranks]
    avg_rank = np.mean(selected_ranks) if selected_ranks else np.nan
    
    # Mean future return of selected funds
    selected_returns = [forward_returns[fund] for fund in selected_funds if fund in forward_returns]
    mean_future_return = np.mean(selected_returns) if selected_returns else np.nan
    
    return {
        'selected_funds': selected_funds,
        'actual_top_funds': actual_top_funds,
        'overlap_count': overlap_count,
        'overlap_accuracy': overlap_accuracy,
        'avg_rank_of_selected': avg_rank,
        'mean_future_return': mean_future_return,
        'selected_returns': selected_returns
    }

# ============================================================================
# MAIN BACKTEST LOOP
# ============================================================================

print("\n" + "="*70)
print("RUNNING BACKTEST")
print("="*70)

# Generate rebalance dates (every 6 months after first year)
start_idx = LOOKBACK
end_idx = len(nav_wide) - HOLDING_PERIOD
rebalance_indices = range(start_idx, end_idx, HOLDING_PERIOD)

results = []
detailed_results = []

for idx, date_idx in enumerate(rebalance_indices):
    rebalance_date = nav_wide.index[date_idx]
    print(f"\nRebalance {idx+1}: {rebalance_date.strftime('%Y-%m-%d')}")
    
    # Compute metrics for all funds
    metrics = compute_metrics_for_date(nav_wide, date_idx, LOOKBACK)
    if metrics is None or len(metrics) < TOP_N:
        print("  Insufficient data, skipping...")
        continue
    
    # Select top funds
    selected_funds, metrics_df = select_top_funds(metrics, TOP_N)
    print(f"  Selected {len(selected_funds)} funds")
    
    # Compute forward returns
    forward_result = compute_forward_returns(nav_wide, date_idx, HOLDING_PERIOD)
    if forward_result is None:
        print("  Cannot compute forward returns, skipping...")
        continue
    
    forward_returns, forward_ranks = forward_result
    
    # Evaluate selection
    eval_result = evaluate_selection(selected_funds, forward_returns, forward_ranks, TOP_N)
    
    # Compute benchmark forward return for the same period
    bench_forward_ret = np.nan
    if date_idx + HOLDING_PERIOD < len(benchmark):
        bench_start = benchmark.iloc[date_idx]
        bench_end = benchmark.iloc[date_idx + HOLDING_PERIOD]
        bench_forward_ret = (bench_end / bench_start) - 1
    
    print(f"  Benchmark forward return: {bench_forward_ret*100:.2f}%" if not np.isnan(bench_forward_ret) else "  Benchmark forward return: N/A")
    
    print(f"  Overlap accuracy: {eval_result['overlap_accuracy']*100:.1f}%")
    print(f"  Avg rank of selected: {eval_result['avg_rank_of_selected']:.1f}")
    print(f"  Mean future return: {eval_result['mean_future_return']*100:.2f}%")
    
    # Store results
    result = {
        'rebalance_date': rebalance_date,
        'rebalance_idx': idx + 1,
        'top_n_used': TOP_N,
        'overlap_count': eval_result['overlap_count'],
        'overlap_accuracy': eval_result['overlap_accuracy'],
        'avg_rank_of_selected': eval_result['avg_rank_of_selected'],
        'mean_future_return': eval_result['mean_future_return'],
        'bench_forward_return': bench_forward_ret,
        'selected_funds': ','.join(map(str, selected_funds)),
        'actual_top_funds': ','.join(map(str, eval_result['actual_top_funds']))
    }
    results.append(result)
    
    # Store detailed results for all funds (to enable dynamic rankings in dashboard)
    for fund, met in metrics.items():
        fwd_ret = forward_returns.get(fund, np.nan) if isinstance(forward_returns, pd.Series) else np.nan
        fwd_rank = forward_ranks.get(fund, np.nan) if isinstance(forward_ranks, pd.Series) else np.nan
        comp_score = np.nan
        comp_rank = np.nan
        try:
            if isinstance(metrics_df, pd.DataFrame) and fund in metrics_df.index:
                comp_score = metrics_df.loc[fund, 'composite_score'] if 'composite_score' in metrics_df.columns else np.nan
                comp_rank = metrics_df.loc[fund, 'composite_rank'] if 'composite_rank' in metrics_df.columns else np.nan
        except Exception:
            comp_score = np.nan
            comp_rank = np.nan
        detailed_results.append({
            'rebalance_date': rebalance_date,
            'rebalance_idx': idx + 1,
            'scheme_code': fund,
            'scheme_name': scheme_names.get(fund, 'Unknown'),
            'sharpe': met.get('sharpe', np.nan),
            'sortino': met.get('sortino', np.nan),
            'mdd': met.get('mdd', np.nan),
            'ann_return': met.get('ann_return', np.nan),
            'ann_vol': met.get('ann_vol', np.nan),
            'momentum_6m': met.get('momentum_6m', np.nan),
            'beta': met.get('beta', np.nan),
            'te': met.get('te', np.nan),
            'ir': met.get('ir', np.nan),
            'forward_return': fwd_ret,
            'forward_rank': fwd_rank,
            'composite_score': comp_score,
            'composite_rank': comp_rank,
            'is_in_actual_top': fund in eval_result['actual_top_funds']
        })

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Summary results
summary_df = pd.DataFrame(results)
summary_path = f'output/backtest_results_summary_{cat_key}.csv'
summary_df.to_csv(summary_path, index=False)
print(f"Summary saved: {summary_path} ({len(summary_df)} rebalances)")
if cat_key == 'largecap':
    summary_df.to_csv('output/backtest_results_summary.csv', index=False)
    print("Summary also saved: output/backtest_results_summary.csv")

# Detailed results
detailed_df = pd.DataFrame(detailed_results)
detailed_path = f'output/backtest_results_detailed_{cat_key}.csv'
detailed_df.to_csv(detailed_path, index=False)
print(f"Detailed saved: {detailed_path} ({len(detailed_df)} fund selections)")
if cat_key == 'largecap':
    detailed_df.to_csv('output/backtest_results_detailed.csv', index=False)
    print("Detailed also saved: output/backtest_results_detailed.csv")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("BACKTEST SUMMARY STATISTICS")
print("="*70)

if len(summary_df) > 0:
    print(f"\nTotal rebalances: {len(summary_df)}")
    print(f"Average overlap accuracy: {summary_df['overlap_accuracy'].mean()*100:.2f}%")
    print(f"Average rank of selected funds: {summary_df['avg_rank_of_selected'].mean():.2f}")
    print(f"Average forward return: {summary_df['mean_future_return'].mean()*100:.2f}%")
    print(f"Median forward return: {summary_df['mean_future_return'].median()*100:.2f}%")
    print(f"Best forward return: {summary_df['mean_future_return'].max()*100:.2f}%")
    print(f"Worst forward return: {summary_df['mean_future_return'].min()*100:.2f}%")
    
    print("\nOverlap accuracy distribution:")
    print(summary_df['overlap_accuracy'].value_counts(normalize=True).sort_index() * 100)
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETE!")
    print("="*70)
else:
    print("No results generated. Check data availability.")
