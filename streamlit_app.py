"""
Fund Selection Strategy - Backtest Dashboard
A clean, modular Streamlit dashboard for analyzing fund selection backtest results.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

HOLDING_PERIOD = 126  # trading days (~6 months)

CATEGORY_MAP = {
    "Large Cap": "largecap",
    "Small Cap": "smallcap",
    "Mid Cap": "midcap",
    "Large & Mid Cap": "large_and_midcap",
    "Multi Cap": "multicap",
    "International": "international",
}

# Criteria weights definitions
CRITERIA_WEIGHTS = {
    "Composite (Balanced)": {
        "sharpe": 0.25, "sortino": 0.25, "ir": 0.20, "ann_return": 0.20,
        "mdd": 0.10, "ann_vol": 0.10, "te": 0.10
    },
    "Momentum": {
        "momentum_6m": 0.4, "ann_return": 0.4, "ann_vol": 0.2
    },
    "Sharpe-Focused": {
        "sharpe": 0.40, "sortino": 0.40, "mdd": 0.20
    },
    "Risk-Adjusted": {
        "ir": 0.30, "sortino": 0.40, "mdd": 0.20, "ann_vol": 0.10
    },
    "Consistency": {
        "sortino": 0.40, "ir": 0.30, "mdd": 0.20, "ann_vol": 0.10
    },
    "Sharpe Ratio": {"sharpe": 1.0},
    "Sortino Ratio": {"sortino": 1.0},
    "Information Ratio": {"ir": 1.0},
    "Annual Return": {"ann_return": 1.0},
    "Max Drawdown": {"mdd": 1.0},
    "Annual Volatility": {"ann_vol": 1.0},
    "Beta": {"beta": 1.0},
    "Tracking Error": {"te": 1.0},
}

# Metrics that should be inverted (lower is better)
INVERT_METRICS = {"mdd", "ann_vol", "te", "beta"}

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fund Selection Strategy - Backtest Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def get_category_files(category_key: str):
    """Get file paths for a category, with fallback to default files."""
    summary_file = f"output/backtest_results_summary_{category_key}.csv"
    detailed_file = f"output/backtest_results_detailed_{category_key}.csv"
    
    if not os.path.exists(summary_file):
        summary_file = "output/backtest_results_summary.csv"
    if not os.path.exists(detailed_file):
        detailed_file = "output/backtest_results_detailed.csv"
    
    return summary_file, detailed_file

@st.cache_data
def load_data(category_key: str = "largecap"):
    """Load backtest results for a category."""
    try:
        summary_file, detailed_file = get_category_files(category_key)
        summary = pd.read_csv(summary_file)
        detailed = pd.read_csv(detailed_file)
        summary['rebalance_date'] = pd.to_datetime(summary['rebalance_date'])
        detailed['rebalance_date'] = pd.to_datetime(detailed['rebalance_date'])
        return summary, detailed
    except FileNotFoundError:
        st.error("❌ Backtest results files not found. Please run backtest_strategy.py first.")
        return None, None

@st.cache_data
def load_benchmark_data():
    """Load benchmark (Nifty 100) data."""
    try:
        bm = pd.read_csv('data/nifty100_fileter_data.csv')
        bm['Date'] = pd.to_datetime(bm['Date'])
        bm = bm.sort_values('Date').set_index('Date')
        bm['Close'] = bm['Close'].astype(str).str.replace(' ', '').astype(float)
        return bm
    except Exception:
        return None

# ============================================================================
# SCORING & RANKING FUNCTIONS
# ============================================================================

def compute_z_scores(df: pd.DataFrame, metrics: list, invert_metrics: set) -> pd.DataFrame:
    """Compute direction-aware z-scores for metrics."""
    df = df.copy()
    
    for metric in metrics:
        if metric not in df.columns:
            df[metric] = np.nan
        
        col = df[metric].astype(float)
        
        # Invert metrics where lower is better
        if metric in invert_metrics:
            col = -col
        
        # Compute z-score
        mu, sd = col.mean(), col.std()
        if pd.isna(sd) or sd == 0:
            df[f"{metric}_z"] = pd.Series(np.nan, index=df.index)
        else:
            df[f"{metric}_z"] = (col - mu) / sd
    
    return df

def compute_composite_score(df: pd.DataFrame, weights: dict, invert_metrics: set) -> pd.Series:
    """Compute weighted-available composite score."""
    metrics = list(weights.keys())
    df = compute_z_scores(df, metrics, invert_metrics)
    
    z_cols = [f"{m}_z" for m in metrics]
    Z = df[z_cols].copy()
    
    # Create weight matrix
    w_series = pd.Series(weights)
    W = pd.DataFrame({f"{m}_z": w_series[m] for m in metrics}, index=df.index)
    
    # Weighted-available scoring
    eff_W = W.where(Z.notna(), 0.0)
    denom = eff_W.sum(axis=1)
    score = (Z.fillna(0.0) * eff_W).sum(axis=1) / denom.replace(0, np.nan)
    
    return score

def rank_funds_by_criteria(df: pd.DataFrame, criteria: str, weights_map: dict) -> pd.DataFrame:
    """Rank funds using specified criteria."""
    df = df.copy()
    weights = weights_map.get(criteria, {})
    
    if not weights:
        return df
    
    df['score'] = compute_composite_score(df, weights, INVERT_METRICS)
    df['rank'] = df['score'].rank(ascending=False, method='first')
    
    return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_returns_over_time(summary: pd.DataFrame):
    """Plot 6-month forward returns over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summary['rebalance_date'], summary['mean_future_return'] * 100,
            marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('6-Month Forward Returns Over Time', fontweight='bold')
    ax.set_xlabel('Rebalance Date')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_accuracy_over_time(summary: pd.DataFrame):
    """Plot selection accuracy over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summary['rebalance_date'], summary['overlap_accuracy'] * 100,
            marker='s', linewidth=2, markersize=6, color='#A23B72')
    ax.set_title('Selection Accuracy Over Time', fontweight='bold')
    ax.set_xlabel('Rebalance Date')
    ax.set_ylabel('Overlap Accuracy (%)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_cumulative_returns(summary: pd.DataFrame):
    """Plot cumulative returns."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative_returns = (1 + summary['mean_future_return']).cumprod() - 1
    ax.plot(summary['rebalance_date'], cumulative_returns * 100,
            linewidth=2.5, color='#6A994E')
    ax.fill_between(summary['rebalance_date'], 0, cumulative_returns * 100,
                    alpha=0.3, color='#6A994E')
    ax.set_title('Cumulative Returns', fontweight='bold')
    ax.set_xlabel('Rebalance Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# ============================================================================
# CRITERIA PERFORMANCE EVALUATION
# ============================================================================

def evaluate_criteria_performance(summary: pd.DataFrame, detailed: pd.DataFrame, 
                                 top_n: int, benchmark_relative: bool = False) -> pd.DataFrame:
    """Evaluate performance of different criteria across all rebalances."""
    dates_all = sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique())
    bench_forward_dict = summary.set_index('rebalance_date')['bench_forward_return'].to_dict()
    
    perf_rows = []
    
    for date_str in dates_all:
        date_df = detailed[detailed['rebalance_date'].dt.strftime('%Y-%m-%d') == date_str].copy()
        if date_df.empty:
            continue
        
        if 'forward_rank' not in date_df.columns:
            date_df['forward_rank'] = date_df['forward_return'].rank(ascending=False, method='first')
        
        bench_ret = bench_forward_dict.get(pd.to_datetime(date_str), np.nan) if benchmark_relative else np.nan
        
        for criteria, weights in CRITERIA_WEIGHTS.items():
            ranked_df = rank_funds_by_criteria(date_df, criteria, CRITERIA_WEIGHTS)
            selected = ranked_df.nsmallest(top_n, 'rank')
            
            if selected.empty:
                continue
            
            fr_mean = selected['forward_return'].astype(float).mean()
            avg_frank = selected['forward_rank'].astype(float).mean()
            overlap_share = (selected['forward_rank'].astype(float) <= float(top_n)).mean()
            
            row = {
                'criteria': criteria,
                'date': date_str,
                'mean_forward_return': fr_mean,
                'avg_forward_rank': avg_frank,
                'overlap_share': overlap_share,
            }
            
            if benchmark_relative:
                excess = np.nan if pd.isna(fr_mean) or pd.isna(bench_ret) else (fr_mean - bench_ret)
                row.update({
                    'bench_forward_return': bench_ret,
                    'excess_return': excess,
                    'win_flag': float(excess > 0) if not pd.isna(excess) else np.nan
                })
            else:
                row['win_flag'] = float(fr_mean > 0)
            
            perf_rows.append(row)
    
    return pd.DataFrame(perf_rows)

def aggregate_criteria_performance(perf_df: pd.DataFrame, benchmark_relative: bool = False) -> pd.DataFrame:
    """Aggregate criteria performance metrics."""
    if perf_df.empty:
        return pd.DataFrame()
    
    def _agg(group):
        mfr = group['mean_forward_return'].dropna()
        afr = group['avg_forward_rank'].dropna()
        ovs = group['overlap_share'].dropna()
        wf = group['win_flag'].dropna()
        
        result = {
            'Avg 6M Return (%)': mfr.mean() * 100 if len(mfr) > 0 else np.nan,
            'Win Rate (%)': wf.mean() * 100 if len(wf) > 0 else np.nan,
            'Avg Forward Rank': afr.mean() if len(afr) > 0 else np.nan,
            'Top-N Overlap (%)': ovs.mean() * 100 if len(ovs) > 0 else np.nan,
            'Rebalances': group['date'].nunique(),
        }
        
        if benchmark_relative:
            exr = group['excess_return'].dropna()
            result['Avg Excess Return (%)'] = exr.mean() * 100 if len(exr) > 0 else np.nan
            result['Benchmark-Relative Win Rate (%)'] = wf.mean() * 100 if len(wf) > 0 else np.nan
            
            # Calculate CAGR
            if len(mfr) > 0:
                sorted_group = group.sort_values('date')
                returns = sorted_group['mean_forward_return'].dropna()
                if len(returns) > 0:
                    final_value = (1 + returns).prod()
                    first_date = pd.to_datetime(sorted_group['date'].min())
                    last_date = pd.to_datetime(sorted_group['date'].max())
                    total_years = (last_date - first_date).days / 365.25
                    cagr = (final_value ** (1 / total_years) - 1) if total_years > 0 else np.nan
                    result['CAGR (%)'] = cagr * 100
                else:
                    result['CAGR (%)'] = np.nan
            else:
                result['CAGR (%)'] = np.nan
        
        return pd.Series(result)
    
    agg = perf_df.groupby('criteria', as_index=False).apply(_agg).reset_index(drop=True)
    
    sort_col = 'CAGR (%)' if benchmark_relative else 'Avg 6M Return (%)'
    agg = agg.sort_values(sort_col, ascending=False)
    
    return agg

# ============================================================================
# PAGE VIEWS
# ============================================================================

def show_fund_rankings(summary: pd.DataFrame, detailed: pd.DataFrame):
    """Display fund rankings page."""
    st.header("📋 Fund Rankings & Selection Frequency")
    
    # Controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        criteria = st.selectbox(
            "Selection Criteria",
            options=list(CRITERIA_WEIGHTS.keys()),
            index=0
        )
    with col2:
        dates = sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique())
        chosen_date = st.selectbox("Rebalance Date", options=dates, index=len(dates)-1)
    with col3:
        top_n = st.number_input("Top N", min_value=1, max_value=50, value=5, step=1)
    
    show_all = st.checkbox("Show all funds", value=False, key="rankings_show_all")
    
    # Filter and rank
    date_df = detailed[detailed['rebalance_date'].dt.strftime('%Y-%m-%d') == chosen_date].copy()
    if date_df.empty:
        st.warning("No data available for the selected date.")
        return
    
    ranked_df = rank_funds_by_criteria(date_df, criteria, CRITERIA_WEIGHTS)
    
    # Display results
    if show_all:
        st.subheader("📋 Full Ranking (All Funds, Selected Date)")
        all_metrics = ['sharpe', 'sortino', 'mdd', 'ann_return', 'ann_vol', 'momentum_6m', 'beta', 'te', 'ir']
        for m in all_metrics:
            if m not in ranked_df.columns:
                ranked_df[m] = np.nan
        
        display_cols = ['scheme_code', 'scheme_name', 'score', 'rank', 'forward_return', 'forward_rank'] + all_metrics
        display_df = ranked_df[display_cols].copy().sort_values('rank')
        display_df['forward_return'] = (display_df['forward_return'] * 100).round(2)
        display_df['score'] = display_df['score'].round(3)
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True)
    else:
        st.subheader(f"Top {top_n} Funds — {criteria} — {chosen_date}")
        top_funds = ranked_df.nsmallest(top_n, 'rank')[
            ['scheme_code', 'scheme_name', 'score', 'rank', 'forward_return', 'forward_rank']
        ].copy()
        top_funds['forward_return'] = (top_funds['forward_return'] * 100).round(2)
        top_funds['score'] = top_funds['score'].round(3)
        st.dataframe(top_funds.reset_index(drop=True), use_container_width=True)
    
    # Most frequently selected funds
    st.markdown("---")
    st.subheader("🏆 Most Frequently Selected Funds (Historical)")
    col1, col2 = st.columns([2, 1])
    with col1:
        hist_dates = st.multiselect(
            "Limit to Rebalance Dates (optional)",
            options=sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique()),
            default=[],
            key="freq_dates"
        )
    with col2:
        freq_topn = st.number_input("Top N for frequency", min_value=1, max_value=50, value=5, step=1, key="freq_topn")
    
    eval_dates = sorted(hist_dates) if hist_dates else sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique())
    
    # Compute frequency
    sel_rows = []
    for date_str in eval_dates:
        date_df = detailed[detailed['rebalance_date'].dt.strftime('%Y-%m-%d') == date_str].copy()
        if date_df.empty:
            continue
        ranked_df = rank_funds_by_criteria(date_df, criteria, CRITERIA_WEIGHTS)
        top_funds = ranked_df.nsmallest(freq_topn, 'rank')[['scheme_code', 'scheme_name', 'forward_return']].copy()
        top_funds['date'] = date_str
        sel_rows.append(top_funds)
    
    if sel_rows:
        sel_all = pd.concat(sel_rows, ignore_index=True)
        counts = sel_all.groupby('scheme_code').size().rename('Times Selected').reset_index()
        name_map = sel_all[['scheme_code', 'scheme_name']].drop_duplicates()
        counts = counts.merge(name_map, on='scheme_code', how='left')
        avg_ret = detailed.groupby('scheme_code')['forward_return'].mean() * 100
        counts = counts.merge(avg_ret.rename('Avg 6M Return (%)'), left_on='scheme_code', right_index=True, how='left')
        total_rebalances = len(eval_dates)
        counts['Selection Rate (%)'] = (counts['Times Selected'] / total_rebalances * 100).round(1)
        counts = counts.sort_values(['Times Selected', 'Avg 6M Return (%)'], ascending=[False, False])
        display_cols = ['scheme_code', 'scheme_name', 'Times Selected', 'Selection Rate (%)', 'Avg 6M Return (%)']
        st.dataframe(counts[display_cols].head(10).rename(columns={'scheme_code': 'Scheme Code', 'scheme_name': 'Fund Name'}), 
                     use_container_width=True)
    
    # Criteria performance comparison
    st.markdown("---")
    st.subheader("🧪 Criteria Performance Comparison")
    eval_topn = st.number_input("Eval Top N", min_value=1, max_value=50, value=15, step=1, key="critperf_topn")
    benchmark_relative = st.checkbox("Show benchmark-relative metrics", value=True, key="bench_rel")
    
    perf_df = evaluate_criteria_performance(summary, detailed, eval_topn, benchmark_relative)
    agg_df = aggregate_criteria_performance(perf_df, benchmark_relative)
    
    if not agg_df.empty:
        st.dataframe(agg_df, use_container_width=True)
    else:
        st.info("No performance data available.")

def show_detailed_results(summary: pd.DataFrame, detailed: pd.DataFrame):
    """Display detailed results explorer."""
    st.header("🔍 Detailed Results Explorer")
    
    # Quick selector
    st.subheader("🎛️ Quick Selector by Ranking Criteria")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        criteria = st.selectbox("Selection Criteria", options=list(CRITERIA_WEIGHTS.keys()), 
                                index=0, key="detail_criteria")
    with col2:
        dates = sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique())
        chosen_date = st.selectbox("Rebalance Date", options=dates, index=len(dates)-1, key="detail_date")
    with col3:
        top_n = st.number_input("Top N", min_value=1, max_value=50, value=5, step=1, key="detail_topn")
    
    # Filter and rank
    date_df = detailed[detailed['rebalance_date'].dt.strftime('%Y-%m-%d') == chosen_date].copy()
    if not date_df.empty:
        ranked_df = rank_funds_by_criteria(date_df, criteria, CRITERIA_WEIGHTS)
        top_funds = ranked_df.nsmallest(top_n, 'rank')[
            ['scheme_code', 'scheme_name', 'score', 'rank', 'forward_return', 'forward_rank']
        ].copy()
        top_funds['forward_return'] = (top_funds['forward_return'] * 100).round(2)
        top_funds['score'] = top_funds['score'].round(3)
        st.dataframe(top_funds.reset_index(drop=True), use_container_width=True)
    
    # Filters
    st.markdown("---")
    st.subheader("🔍 Filter Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_filter = st.multiselect(
            "Filter by Rebalance Date:",
            options=sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique()),
            default=[]
        )
    with col2:
        fund_filter = st.multiselect(
            "Filter by Fund:",
            options=sorted(detailed['scheme_name'].unique()),
            default=[]
        )
    with col3:
        show_rank_slider = ('composite_rank' in detailed.columns) and detailed['composite_rank'].notna().any()
        if show_rank_slider:
            rank_filter = st.slider("Filter by Composite Rank:", min_value=1, max_value=50, value=(1, 50))
        else:
            st.caption("Composite rank filter unavailable.")
    
    # Apply filters
    filtered_df = detailed.copy()
    if date_filter:
        filtered_df = filtered_df[filtered_df['rebalance_date'].dt.strftime('%Y-%m-%d').isin(date_filter)]
    if fund_filter:
        filtered_df = filtered_df[filtered_df['scheme_name'].isin(fund_filter)]
    if show_rank_slider:
        filtered_df = filtered_df[
            (filtered_df['composite_rank'] >= rank_filter[0]) &
            (filtered_df['composite_rank'] <= rank_filter[1])
        ]
    
    # Display filtered results
    st.subheader(f"Showing {len(filtered_df)} selections")
    display_cols = [
        'rebalance_date', 'scheme_name', 'sharpe', 'sortino', 'mdd', 'ann_return',
        'ann_vol', 'beta', 'te', 'ir', 'composite_score', 'composite_rank',
        'forward_return', 'forward_rank', 'is_in_actual_top'
    ]
    
    for col in display_cols:
        if col not in filtered_df.columns and col not in ['rebalance_date', 'scheme_name']:
            filtered_df[col] = np.nan
    
    display_df = filtered_df[display_cols].copy()
    display_df['rebalance_date'] = display_df['rebalance_date'].dt.strftime('%Y-%m-%d')
    display_df['sharpe'] = display_df['sharpe'].round(3)
    display_df['sortino'] = display_df['sortino'].round(3)
    display_df['mdd'] = (display_df['mdd'] * 100).round(2)
    display_df['composite_score'] = display_df['composite_score'].round(3)
    display_df['forward_return'] = (display_df['forward_return'] * 100).round(2)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Export
    if st.button("📥 Export Filtered Results to CSV"):
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_backtest_results.csv",
            mime="text/csv"
        )

def show_methodology():
    """Display methodology documentation."""
    st.header("📋 Methodology & Implementation")
    
    st.markdown("""
    ## Overview
    This dashboard evaluates mutual funds across categories using risk/return metrics,
    ranks funds using direction-aware z-scores with user-selected criteria, and measures
    how those selections performed over the next 6 months.
    
    ## Key Metrics
    - **Sharpe Ratio**: Risk-adjusted return measure
    - **Sortino Ratio**: Downside risk-adjusted return
    - **Max Drawdown**: Largest peak-to-trough decline
    - **Information Ratio**: Active return per unit of tracking error
    - **Beta**: Sensitivity to benchmark movements
    
    ## Scoring Method
    1. **Direction-aware z-scores**: Metrics where lower is better (MDD, Volatility, TE, Beta) are inverted
    2. **Weighted-available scoring**: Missing metrics don't penalize funds
    3. **Composite scoring**: Multiple criteria available (Composite, Momentum, Sharpe-Focused, etc.)
    
    ## Evaluation
    - Forward returns measured over 6-month holding periods
    - Comparison with actual top performers
    - Benchmark-relative performance analysis
    """)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    st.title("📊 Fund Selection Strategy - Backtest Dashboard")
    st.markdown("---")
    
    # Category selection
    cat_name = st.sidebar.selectbox("Category", list(CATEGORY_MAP.keys()), index=0)
    cat_key = CATEGORY_MAP[cat_name]
    
    # Load data
    summary, detailed = load_data(cat_key)
    if summary is None or detailed is None:
        return
    
    # Reload button
    if st.sidebar.button("🔄 Reload Data"):
        load_data.clear()
        st.rerun()
    
    # Key metrics overview
    st.subheader("📊 Strategy Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rebalances", len(summary), 
                 f"{summary['overlap_accuracy'].mean()*100:.1f}% Avg Accuracy")
    with col2:
        avg_return = summary['mean_future_return'].mean() * 100
        st.metric("Average 6M Return", f"{avg_return:.1f}%",
                 f"Win Rate: {(summary['mean_future_return'] > 0).mean()*100:.1f}%")
    with col3:
        st.metric("Selection Accuracy", f"{summary['overlap_accuracy'].mean()*100:.1f}%",
                 f"Avg Rank: {summary['avg_rank_of_selected'].mean():.1f}")
    with col4:
        st.metric("Funds Universe", f"{detailed['scheme_code'].nunique()} funds",
                 f"{len(detailed)} selections")
    
    # Quick charts
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_returns_over_time(summary))
    with col2:
        st.pyplot(plot_accuracy_over_time(summary))
    
    st.pyplot(plot_cumulative_returns(summary))
    
    st.markdown("---")
    
    # Navigation
    st.sidebar.header("🔍 Navigation")
    page = st.sidebar.radio("Go to:", [
        "📋 Fund Rankings",
        "🔍 Detailed Results",
        "📋 Methodology"
    ])
    
    # Display selected page
    if page == "📋 Fund Rankings":
        show_fund_rankings(summary, detailed)
    elif page == "🔍 Detailed Results":
        show_detailed_results(summary, detailed)
    elif page == "📋 Methodology":
        show_methodology()

if __name__ == "__main__":
    main()
