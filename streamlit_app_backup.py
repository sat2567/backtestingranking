import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Constants (match backtest)
HOLDING_PERIOD = 126  # trading days (~6 months)

# Set page config
st.set_page_config(
    page_title="Large Cap Fund Selection Backtest Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-left: 0px;
        border-top: 0px;
        border-right: 0px;
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: transparent;
        border-left: 0px;
        border-top: 0px;
        border-right: 0px;
        border-bottom: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

CATEGORY_MAP = {
    "Large Cap": "largecap",
    "Small Cap": "smallcap",
    "Mid Cap": "midcap",
    "Large & Mid Cap": "large_and_midcap",
    "Multi Cap": "multicap",
    "International": "international",
}

def _category_files(key: str):
    s = f"output/backtest_results_summary_{key}.csv"
    d = f"output/backtest_results_detailed_{key}.csv"
    if not os.path.exists(s):
        s = "output/backtest_results_summary.csv"
    if not os.path.exists(d):
        d = "output/backtest_results_detailed.csv"
    return s, d

@st.cache_data
def load_data(category_key: str = "largecap"):
    try:
        s_file, d_file = _category_files(category_key)
        summary = pd.read_csv(s_file)
        detailed = pd.read_csv(d_file)
        summary['rebalance_date'] = pd.to_datetime(summary['rebalance_date'])
        detailed['rebalance_date'] = pd.to_datetime(detailed['rebalance_date'])
        return summary, detailed
    except FileNotFoundError:
        st.error("❌ Backtest results files not found. Please run backtest_strategy.py first.")
        return None, None

# Load benchmark and pre-compute forward 6M returns
@st.cache_data
def load_benchmark_forward(holding_period_days: int = HOLDING_PERIOD):
    try:
        bm = pd.read_csv('data/nifty100_fileter_data.csv')
        bm['Date'] = pd.to_datetime(bm['Date'])
        bm = bm.sort_values('Date').set_index('Date')
        close = bm['Close'].astype(str).str.replace(' ', '').astype(float)
        # Resample to 6-month ends
        resampled = close.resample('6ME').last()
        # Historical 6M returns: (current / previous) - 1
        hist = resampled / resampled.shift(1) - 1
        return hist.dropna()  # Drop first NaN
    except Exception:
        return None

# Main app
def main():
    st.title("📊 Large Cap Fund Selection Strategy - Backtest Dashboard")
    st.markdown("---")

    cat_name = st.sidebar.selectbox("Category", list(CATEGORY_MAP.keys()), index=0)
    cat_key = CATEGORY_MAP[cat_name]

    summary, detailed = load_data(cat_key)
    if summary is None or detailed is None:
        return

    # Provide a way to reload data from disk (clear cache)
    if st.button("🔄 Reload data from CSV"):
        load_data.clear()
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.info("Cache cleared. Press R in Streamlit to reload the app.")

    # Sidebar
    st.sidebar.header("🔍 Navigation")
    page = st.sidebar.radio("Go to:", [
        "📋 Fund Rankings",
        "🔍 Detailed Results",
        "📋 Methodology"
    ])

    if page == "📋 Fund Rankings":
        show_fund_rankings(summary, detailed)
    elif page == "🔍 Detailed Results":
        show_detailed_results(summary, detailed)
    elif page == "📋 Methodology":
        show_methodology()

def show_overview(summary, detailed):
    st.header("📈 Strategy Overview")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Rebalances",
            value=len(summary),
            delta=f"{summary['overlap_accuracy'].mean()*100:.1f}% Avg Accuracy"
        )

    with col2:
        avg_return = summary['mean_future_return'].mean() * 100
        st.metric(
            label="Average 6M Return",
            value=f"{avg_return:.1f}%",
            delta=f"Win Rate: {(summary['mean_future_return'] > 0).mean()*100:.1f}%"
        )

    with col3:
        st.metric(
            label="Selection Accuracy",
            value=f"{summary['overlap_accuracy'].mean()*100:.1f}%",
            delta=f"Avg Rank: {summary['avg_rank_of_selected'].mean():.1f}"
        )

    with col4:
        st.metric(
            label="Funds Universe",
            value=f"{detailed['scheme_code'].nunique()} funds",
            delta=f"{len(detailed)} selections"
        )

    st.markdown("---")

    # Time series plots
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(summary['rebalance_date'], summary['mean_future_return'] * 100,
                marker='o', linewidth=2, markersize=6, color='#2E86AB')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('6-Month Forward Returns Over Time', fontweight='bold')
        ax.set_xlabel('Rebalance Date')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- Criteria performance across all rebalances ---
    st.markdown("---")
    st.subheader("🧪 Criteria Performance (All Rebalances)")
    colp1, _ = st.columns([1,3])
    with colp1:
        eval_topn_rank = st.number_input("Eval Top N", min_value=1, max_value=50, value=15, step=1, key="critperf_topn_rankings")

    weights_perf = {
        "Composite (Balanced)": {"sharpe":0.25, "sortino":0.25, "ir":0.20, "ann_return":0.20, "mdd":0.10, "ann_vol":0.10, "te":0.10},
        "Momentum": {"momentum_6m": 0.4, "ann_return": 0.4, "ann_vol": 0.2},
        "Sharpe-Focused": {"sharpe":0.40, "sortino":0.40, "mdd":0.20},
        "Risk-Adjusted": {"ir":0.30, "sortino":0.40, "mdd":0.20, "ann_vol":0.10},
        "Consistency": {"sortino":0.40, "ir":0.30, "mdd":0.20, "ann_vol":0.10},
        "Sharpe Ratio": {"sharpe": 1.0},
        "Sortino Ratio": {"sortino": 1.0},
        "Information Ratio": {"ir": 1.0},
        "Annual Return": {"ann_return": 1.0},
        "Max Drawdown": {"mdd": 1.0},
        "Annual Volatility": {"ann_vol": 1.0},
        "Beta": {"beta": 1.0},
        "Tracking Error": {"te": 1.0},
    }

    dates_all = sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique())
    perf_rows = []
    for ds in dates_all:
        dd = detailed[detailed['rebalance_date'].dt.strftime('%Y-%m-%d') == ds].copy()
        if dd.empty:
            continue
        if 'forward_rank' not in dd.columns:
            dd['forward_rank'] = dd['forward_return'].rank(ascending=False, method='first')
        for crit, wmap in weights_perf.items():
            need = list(wmap.keys())
            inv = {"mdd", "ann_vol", "te", "beta"}
            for m in need:
                if m not in dd.columns:
                    dd[m] = np.nan
                col = dd[m].astype(float)
                if m in inv:
                    col = -col
                mu, sd = col.mean(), col.std()
                z = pd.Series(np.nan, index=col.index) if pd.isna(sd) or sd == 0 else (col - mu) / sd
                dd[m+"_z"] = z
            z_cols = [m+"_z" for m in need]
            Z = dd[z_cols].copy()
            w_series = pd.Series(wmap)
            W = pd.DataFrame({m+"_z": w_series[m] for m in need}, index=dd.index)
            eff_W = W.where(Z.notna(), 0.0)
            denom = eff_W.sum(axis=1)
            dd['__score'] = (Z.fillna(0.0) * eff_W).sum(axis=1) / denom.replace(0, np.nan)
            dd['__rank'] = dd['__score'].rank(ascending=False, method='first')
            sel = dd.nsmallest(int(eval_topn_rank), '__rank')
            fr = sel['forward_return'].astype(float)
            fr_mean = fr.mean()
            avg_frank = sel['forward_rank'].astype(float).mean()
            overlap_share = (sel['forward_rank'].astype(float) <= float(eval_topn_rank)).mean()
            perf_rows.append({
                'criteria': crit,
                'date': ds,
                'mean_forward_return': fr_mean,
                'avg_forward_rank': avg_frank,
                'overlap_share': overlap_share,
                'win_flag': float(fr_mean > 0)
            })

    perf_df = pd.DataFrame(perf_rows)
    if not perf_df.empty:
        def _agg(g):
            mfr = g['mean_forward_return'].dropna()
            afr = g['avg_forward_rank'].dropna()
            ovs = g['overlap_share'].dropna()
            wf = g['win_flag'].dropna()
            return pd.Series({
                'Avg 6M Return (%)': mfr.mean()*100 if len(mfr)>0 else np.nan,
                'Win Rate (%)': wf.mean()*100 if len(wf)>0 else np.nan,
                'Avg Forward Rank': afr.mean() if len(afr)>0 else np.nan,
                'Top-N Overlap (%)': ovs.mean()*100 if len(ovs)>0 else np.nan,
                'Rebalances': g['date'].nunique(),
                'Avg Forward Return': mfr.mean() if len(mfr)>0 else np.nan,
                'Avg Forward Rank': afr.mean() if len(afr)>0 else np.nan,
                'Avg Overlap Share': ovs.mean() if len(ovs)>0 else np.nan,
                'Avg Win Flag': wf.mean() if len(wf)>0 else np.nan,
            })
        agg = perf_df.groupby('criteria', as_index=False).apply(_agg)
        agg = agg.sort_values('Avg 6M Return (%)', ascending=False)
        st.dataframe(agg.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No performance data available to compare criteria.")

    # Criteria performance across all rebalances (benchmark-relative)
    st.subheader("🧪 Criteria Performance (All Rebalances)")
    colp1, _ = st.columns([1,3])
    with colp1:
        eval_topn = st.number_input("Eval Top N", min_value=1, max_value=50, value=15, step=1, key="critperf_topn")

    weights_perf = {
        "Composite (Balanced)": {"sharpe":0.25, "sortino":0.25, "ir":0.20, "ann_return":0.20, "mdd":0.10, "ann_vol":0.10, "te":0.10},
        "Momentum": {"momentum_6m": 0.4, "ann_return": 0.4, "ann_vol": 0.2},
        "Sharpe-Focused": {"sharpe":0.40, "sortino":0.40, "mdd":0.20},
        "Risk-Adjusted": {"ir":0.30, "sortino":0.40, "mdd":0.20, "ann_vol":0.10},
        "Consistency": {"sortino":0.40, "ir":0.30, "mdd":0.20, "ann_vol":0.10},
        "Sharpe Ratio": {"sharpe": 1.0},
        "Sortino Ratio": {"sortino": 1.0},
        "Information Ratio": {"ir": 1.0},
        "Annual Return": {"ann_return": 1.0},
        "Max Drawdown": {"mdd": 1.0},
        "Annual Volatility": {"ann_vol": 1.0},
        "Beta": {"beta": 1.0},
        "Tracking Error": {"te": 1.0},
    }

    dates_all = sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique())
    bench_forward_dict = summary.set_index('rebalance_date')['bench_forward_return'].to_dict()
    perf_rows = []
    for ds in dates_all:
        dd = detailed[detailed['rebalance_date'].dt.strftime('%Y-%m-%d') == ds].copy()
        if dd.empty:
            continue
        if 'forward_rank' not in dd.columns:
            dd['forward_rank'] = dd['forward_return'].rank(ascending=False, method='first')
        # Benchmark forward return for this date
        bench_ret = np.nan
        ds_ts = pd.to_datetime(ds)
        bench_ret = bench_forward_dict.get(ds_ts, np.nan)
        for crit, wmap in weights_perf.items():
            need = list(wmap.keys())
            inv = {"mdd", "ann_vol", "te", "beta"}
            for m in need:
                if m not in dd.columns:
                    dd[m] = np.nan
                col = dd[m].astype(float)
                if m in inv:
                    col = -col
                mu, sd = col.mean(), col.std()
                z = pd.Series(np.nan, index=col.index) if pd.isna(sd) or sd == 0 else (col - mu) / sd
                dd[m+"_z"] = z
            z_cols = [m+"_z" for m in need]
            Z = dd[z_cols].copy()
            w_series = pd.Series(wmap)
            W = pd.DataFrame({m+"_z": w_series[m] for m in need}, index=dd.index)
            eff_W = W.where(Z.notna(), 0.0)
            denom = eff_W.sum(axis=1)
            dd['__score'] = (Z.fillna(0.0) * eff_W).sum(axis=1) / denom.replace(0, np.nan)
            dd['__rank'] = dd['__score'].rank(ascending=False, method='first')
            sel = dd.nsmallest(int(eval_topn), '__rank')
            fr = sel['forward_return'].astype(float)
            fr_mean = fr.mean()
            avg_frank = sel['forward_rank'].astype(float).mean()
            overlap_share = (sel['forward_rank'].astype(float) <= float(eval_topn)).mean()
            excess = np.nan if pd.isna(fr_mean) or pd.isna(bench_ret) else (fr_mean - bench_ret)
            win_flag = float(excess > 0) if not pd.isna(excess) else np.nan
            perf_rows.append({
                'criteria': crit,
                'date': ds,
                'mean_forward_return': fr_mean,
                'bench_forward_return': bench_ret,
                'excess_return': excess,
                'avg_forward_rank': avg_frank,
                'overlap_share': overlap_share,
                'win_flag': win_flag
            })

    perf_df = pd.DataFrame(perf_rows)
    if not perf_df.empty:
        def _agg(g):
            mfr = g['mean_forward_return'].dropna()
            bfr = g['bench_forward_return'].dropna()
            exr = g['excess_return'].dropna()
            afr = g['avg_forward_rank'].dropna()
            ovs = g['overlap_share'].dropna()
            wf = g['win_flag'].dropna()
            
            # Calculate compounded return for funds
            if len(mfr) > 0:
                sorted_g = g.sort_values('date')
                returns = sorted_g['mean_forward_return'].dropna()
                if len(returns) > 0:
                    final_value = (1 + returns).prod()
                    first_date = pd.to_datetime(sorted_g['date'].min())
                    last_date = pd.to_datetime(sorted_g['date'].max())
                    total_days = (last_date - first_date).days
                    total_years = total_days / 365.25
                    cagr = (final_value ** (1 / total_years) - 1) if total_years > 0 else np.nan
                    avg_6m = cagr * 100
                else:
                    avg_6m = np.nan
            else:
                avg_6m = np.nan
                
            # Calculate compounded return for benchmark (full period CAGR)
            nifty = pd.read_csv('data/nifty100_fileter_data.csv')
            nifty['Date'] = pd.to_datetime(nifty['Date'])
            nifty = nifty.sort_values('Date')
            nifty['Close'] = nifty['Close'].astype(str).str.replace(' ', '').astype(float)
            first_close = nifty['Close'].iloc[0]
            last_close = nifty['Close'].iloc[-1]
            total_days = (nifty['Date'].iloc[-1] - nifty['Date'].iloc[0]).days
            total_years = total_days / 365.25
            bench_cagr = (last_close / first_close) ** (1 / total_years) - 1 if total_years > 0 else np.nan
            bench_avg_6m = bench_cagr * 100
            
            return pd.Series({
                'CAGR (%)': avg_6m,
                'Benchmark Avg 6M Return (%)': bench_avg_6m,
                'Avg Excess Return (%)': exr.mean()*100 if len(exr)>0 else np.nan,
                'Benchmark-Relative Win Rate (%)': wf.mean()*100 if len(wf)>0 else np.nan,
                'Avg Forward Rank': afr.mean() if len(afr)>0 else np.nan,
                'Top-N Overlap (%)': ovs.mean()*100 if len(ovs)>0 else np.nan,
                'Rebalances': g['date'].nunique()
            })
        agg = perf_df.groupby('criteria', as_index=False).apply(_agg)
        agg = agg.sort_values('CAGR (%)', ascending=False)
        st.dataframe(agg.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No performance data available to compare criteria.")

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(summary['rebalance_date'], summary['overlap_accuracy'] * 100,
                marker='s', linewidth=2, markersize=6, color='#A23B72')
        ax.set_title('Selection Accuracy Over Time', fontweight='bold')
        ax.set_xlabel('Rebalance Date')
        ax.set_ylabel('Overlap Accuracy (%)')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Cumulative returns
    st.subheader("📈 Cumulative Returns")
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
    st.pyplot(fig)

def show_performance_analysis(summary, detailed):
    st.header("📊 Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Return Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(summary['mean_future_return'] * 100, bins=15, color='#2E86AB',
                edgecolor='black', alpha=0.7)
        ax.axvline(x=summary['mean_future_return'].mean() * 100,
                   color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {summary["mean_future_return"].mean()*100:.2f}%')
        ax.set_xlabel('6-Month Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)

    with col2:
        st.subheader("Risk Metrics of Selected Funds")
        metrics_df = detailed[['sharpe', 'sortino', 'mdd']].agg(['mean', 'std', 'min', 'max']).round(3)
        st.dataframe(metrics_df.style.background_gradient(axis=1, cmap='RdYlGn'))

    # Correlation analysis
    st.subheader("📈 Correlation Analysis")
    corr_data = detailed[['sharpe', 'sortino', 'mdd', 'composite_score', 'forward_return']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_data.columns)))
    ax.set_yticks(range(len(corr_data.columns)))
    ax.set_xticklabels(corr_data.columns, rotation=45)
    ax.set_yticklabels(corr_data.columns)
    plt.colorbar(cax)

    # Add correlation values
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            text = ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10)

    ax.set_title('Correlation Matrix: Risk Metrics vs Forward Returns', fontweight='bold')
    st.pyplot(fig)

    # Performance by overlap accuracy
    st.subheader("🎯 Returns by Selection Accuracy")
    overlap_returns = summary.groupby('overlap_count')['mean_future_return'].agg(['mean', 'count', 'std']).round(4)
    overlap_returns['mean_pct'] = (overlap_returns['mean'] * 100).round(2)
    overlap_returns['std_pct'] = (overlap_returns['std'] * 100).round(2)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(overlap_returns.index, overlap_returns['mean_pct'],
                  yerr=overlap_returns['std_pct'], capsize=5,
                  color=['#F18F01' if x < 3 else '#C73E1D' if x == 3 else '#6A994E' for x in overlap_returns.index],
                  alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Funds Correctly Selected (out of 5)')
    ax.set_ylabel('Average 6-Month Return (%)')
    ax.set_title('Returns by Selection Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, count in zip(bars, overlap_returns['count']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'n={int(count)}', ha='center', va='bottom')

    st.pyplot(fig)

    # Selection Accuracy tab removed as per request

def show_fund_rankings(summary, detailed):
    st.header("📋 Fund Rankings & Selection Frequency")

    # --- Dynamic selection by criteria ---
    st.subheader("🎛️ Select Funds by Ranking Criteria")

    # controls
    colc1, colc2, colc3 = st.columns([2,2,1])
    with colc1:
        criteria = st.selectbox(
            "Selection Criteria",
            options=[
                "Composite (Balanced)",
                "Momentum",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Max Drawdown",
                "Consistency"
            ],
            index=0
        )
    with colc2:
        dates = sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique())
        chosen_date = st.selectbox("Rebalance Date", options=dates, index=len(dates)-1)
    with colc3:
        top_n = st.number_input("Top N", min_value=1, max_value=50, value=5, step=1)
    show_all = st.checkbox("Show all funds", value=False, key="rankings_show_all")

    # weights per criteria (apply to z-scores; use negative weights where specified)
    weights_map = {
        "Composite (Balanced)": {"sharpe":0.25, "sortino":0.25, "ir":0.20, "ann_return":0.20, "mdd":0.10, "ann_vol":0.10, "te":0.10},
        "Momentum": {"momentum_6m": 0.4, "ann_return": 0.4, "ann_vol": 0.2},
        "Sharpe-Focused": {"sharpe":0.40, "sortino":0.40, "mdd":0.20},
        "Risk-Adjusted": {"ir":0.30, "sortino":0.40, "mdd":0.20, "ann_vol":0.10},
        "Consistency": {"sortino":0.40, "ir":0.30, "mdd":0.20, "ann_vol":0.10},
        "Sharpe Ratio": {"sharpe": 1.0},
        "Sortino Ratio": {"sortino": 1.0},
        "Information Ratio": {"ir": 1.0},
        "Annual Return": {"ann_return": 1.0},
        "Max Drawdown": {"mdd": 1.0},
        "Annual Volatility": {"ann_vol": 1.0},
        "Beta": {"beta": 1.0},
        "Tracking Error": {"te": 1.0},
    }

    # filter data for selected rebalance date
    d = detailed[detailed['rebalance_date'].dt.strftime('%Y-%m-%d') == chosen_date].copy()
    if d.empty:
        st.warning("No data available for the selected date.")
        return

    # compute direction-aware z-scores
    needed = list(weights_map[criteria].keys())
    invert = {"mdd", "ann_vol", "te", "beta"}
    for m in needed:
        if m not in d.columns:
            d[m] = np.nan
        col = d[m].astype(float)
        if m in invert:
            col = -col
        mu, sd = col.mean(), col.std()
        z = pd.Series(np.nan, index=col.index) if pd.isna(sd) or sd == 0 else (col - mu) / sd
        d[m+"_z"] = z

    # score (use available metrics per fund; place NaN scores at bottom)
    z_cols = [m+"_z" for m in needed]
    Z = d[z_cols].copy()
    w_series = pd.Series(weights_map[criteria])
    W = pd.DataFrame({m+"_z": w_series[m] for m in needed}, index=d.index)
    eff_W = W.where(Z.notna(), 0.0)
    denom = eff_W.sum(axis=1)
    d['score'] = (Z.fillna(0.0) * eff_W).sum(axis=1) / denom.replace(0, np.nan)
    d['rank'] = d['score'].rank(ascending=False, method='first')

    top = d.nsmallest(int(top_n), 'rank')[['scheme_code','scheme_name','score','rank','forward_return','forward_rank'] + needed]
    top = top.rename(columns={'forward_return':'6M_forward_return', 'forward_rank':'6M_forward_rank'})
    top['6M_forward_return'] = (top['6M_forward_return'] * 100).round(2)
    top['score'] = top['score'].round(3)
    if not show_all:
        st.markdown(f"**Top {int(top_n)} funds — {criteria} — {chosen_date}**")
        st.dataframe(top.reset_index(drop=True), use_container_width=True)

    # Full ranking table for the chosen date with all metrics (only if Show all)
    if show_all:
        st.subheader("📋 Full Ranking (All Funds, Selected Date)")
        # Ensure metric columns exist (some older CSVs may miss them)
        all_needed = ['sharpe','sortino','mdd','ann_return','ann_vol','momentum_6m','beta','te','ir']
        for m in all_needed:
            if m not in d.columns:
                d[m] = np.nan

        full_cols = ['scheme_code','scheme_name','score','rank','forward_return','forward_rank'] + all_needed
        full = d[full_cols].copy()
        full = full.sort_values('rank')
        full['forward_return'] = (full['forward_return'] * 100).round(2)
        full['score'] = full['score'].round(3)
        st.dataframe(full.reset_index(drop=True), use_container_width=True)

    # Criteria performance across all rebalances (evaluate which criteria wins vs Nifty 100)
    st.markdown("---")
    st.subheader("🧪 Criteria Performance (All Rebalances)")
    colp1, _ = st.columns([1,3])
    with colp1:
        eval_topn_fundrank = st.number_input("Eval Top N", min_value=1, max_value=50, value=15, step=1, key="fr_critperf_topn")

    weights_perf = {
        "Composite (Balanced)": {"sharpe":0.25, "sortino":0.25, "ir":0.20, "ann_return":0.20, "mdd":0.10, "ann_vol":0.10, "te":0.10},
        "Momentum": {"momentum_6m": 0.4, "ann_return": 0.4, "ann_vol": 0.2},
        "Sharpe-Focused": {"sharpe":0.40, "sortino":0.40, "mdd":0.20},
        "Risk-Adjusted": {"ir":0.30, "sortino":0.40, "mdd":0.20, "ann_vol":0.10},
        "Consistency": {"sortino":0.40, "ir":0.30, "mdd":0.20, "ann_vol":0.10},
        "Sharpe Ratio": {"sharpe": 1.0},
        "Sortino Ratio": {"sortino": 1.0},
        "Information Ratio": {"ir": 1.0},
        "Annual Return": {"ann_return": 1.0},
        "Max Drawdown": {"mdd": 1.0},
        "Annual Volatility": {"ann_vol": 1.0},
        "Beta": {"beta": 1.0},
        "Tracking Error": {"te": 1.0},
    }

    dates_all = sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique())
    bench_forward_dict = summary.set_index('rebalance_date')['bench_forward_return'].to_dict()
    perf_rows = []
    for ds in dates_all:
        dd = detailed[detailed['rebalance_date'].dt.strftime('%Y-%m-%d') == ds].copy()
        if dd.empty:
            continue
        if 'forward_rank' not in dd.columns:
            dd['forward_rank'] = dd['forward_return'].rank(ascending=False, method='first')
        # Benchmark forward return for this date
        bench_ret = np.nan
        ds_ts = pd.to_datetime(ds)
        bench_ret = bench_forward_dict.get(ds_ts, np.nan)
        for crit, wmap in weights_perf.items():
            need = list(wmap.keys())
            inv = {"mdd", "ann_vol", "te", "beta"}
            for m in need:
                if m not in dd.columns:
                    dd[m] = np.nan
                col = dd[m].astype(float)
                if m in inv:
                    col = -col
                mu, sd = col.mean(), col.std()
                z = pd.Series(np.nan, index=col.index) if pd.isna(sd) or sd == 0 else (col - mu) / sd
                dd[m+"_z"] = z
            z_cols = [m+"_z" for m in need]
            Z = dd[z_cols].copy()
            w_series = pd.Series(wmap)
            W = pd.DataFrame({m+"_z": w_series[m] for m in need}, index=dd.index)
            eff_W = W.where(Z.notna(), 0.0)
            denom = eff_W.sum(axis=1)
            dd['__score'] = (Z.fillna(0.0) * eff_W).sum(axis=1) / denom.replace(0, np.nan)
            dd['__rank'] = dd['__score'].rank(ascending=False, method='first')
            sel = dd.nsmallest(int(eval_topn_fundrank), '__rank')
            fr = sel['forward_return'].astype(float)
            fr_mean = fr.mean()
            avg_frank = sel['forward_rank'].astype(float).mean()
            overlap_share = (sel['forward_rank'].astype(float) <= float(eval_topn_fundrank)).mean()
            excess = np.nan if pd.isna(fr_mean) or pd.isna(bench_ret) else (fr_mean - bench_ret)
            win_flag = float(excess > 0) if not pd.isna(excess) else np.nan
            perf_rows.append({
                'criteria': crit,
                'date': ds,
                'mean_forward_return': fr_mean,
                'bench_forward_return': bench_ret,
                'excess_return': excess,
                'avg_forward_rank': avg_frank,
                'overlap_share': overlap_share,
                'win_flag': win_flag
            })

    perf_df = pd.DataFrame(perf_rows)
    if not perf_df.empty:
        def _agg(g):
            mfr = g['mean_forward_return'].dropna()
            bfr = g['bench_forward_return'].dropna()
            exr = g['excess_return'].dropna()
            afr = g['avg_forward_rank'].dropna()
            ovs = g['overlap_share'].dropna()
            wf = g['win_flag'].dropna()
            
            # Calculate compounded return for funds
            if len(mfr) > 0:
                sorted_g = g.sort_values('date')
                returns = sorted_g['mean_forward_return'].dropna()
                if len(returns) > 0:
                    final_value = (1 + returns).prod()
                    first_date = pd.to_datetime(sorted_g['date'].min())
                    last_date = pd.to_datetime(sorted_g['date'].max())
                    total_days = (last_date - first_date).days
                    total_years = total_days / 365.25
                    cagr = (final_value ** (1 / total_years) - 1) if total_years > 0 else np.nan
                    avg_6m = cagr * 100
                else:
                    avg_6m = np.nan
            else:
                avg_6m = np.nan
                
            # Calculate compounded return for benchmark (full period CAGR)
            nifty = pd.read_csv('data/nifty100_fileter_data.csv')
            nifty['Date'] = pd.to_datetime(nifty['Date'])
            nifty = nifty.sort_values('Date')
            nifty['Close'] = nifty['Close'].astype(str).str.replace(' ', '').astype(float)
            first_close = nifty['Close'].iloc[0]
            last_close = nifty['Close'].iloc[-1]
            total_days = (nifty['Date'].iloc[-1] - nifty['Date'].iloc[0]).days
            total_years = total_days / 365.25
            bench_cagr = (last_close / first_close) ** (1 / total_years) - 1 if total_years > 0 else np.nan
            bench_avg_6m = bench_cagr * 100
            
            return pd.Series({
                'CAGR (%)': avg_6m,
                'Benchmark Avg 6M Return (%)': bench_avg_6m,
                'Avg Excess Return (%)': exr.mean()*100 if len(exr)>0 else np.nan,
                'Benchmark-Relative Win Rate (%)': wf.mean()*100 if len(wf)>0 else np.nan,
                'Avg Forward Rank': afr.mean() if len(afr)>0 else np.nan,
                'Top-N Overlap (%)': ovs.mean()*100 if len(ovs)>0 else np.nan,
                'Rebalances': g['date'].nunique()
            })
        agg = perf_df.groupby('criteria', as_index=False).apply(_agg)
        agg = agg.sort_values('CAGR (%)', ascending=False)
        st.dataframe(agg.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No performance data available to compare criteria.")

    # --- Frequency and top selections (existing content) ---
    st.markdown("---")
    st.subheader("🏆 Most Frequently Selected Funds (Historical)")
    colf1, colf2 = st.columns([2,1])
    with colf1:
        hist_dates = st.multiselect(
            "Limit to Rebalance Dates (optional)",
            options=sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique()),
            default=[]
        )
    with colf2:
        freq_topn = st.number_input("Top N for frequency", min_value=1, max_value=50, value=5, step=1, key="freq_topn")

    # Determine dates to evaluate
    if hist_dates:
        eval_dates = sorted(hist_dates)
    else:
        eval_dates = sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique())

    # Recompute Top N per date for the CURRENT criteria, then count frequency
    needed_freq = list(weights_map[criteria].keys())
    invert_freq = {"mdd", "ann_vol", "te", "beta"}
    sel_rows = []
    for ds in eval_dates:
        dd = detailed[detailed['rebalance_date'].dt.strftime('%Y-%m-%d') == ds].copy()
        if dd.empty:
            continue
        # ensure needed columns
        for m in needed_freq:
            if m not in dd.columns:
                dd[m] = np.nan
            col = dd[m].astype(float)
            if m in invert_freq:
                col = -col
            mu, sd = col.mean(), col.std()
            z = pd.Series(np.nan, index=col.index) if pd.isna(sd) or sd == 0 else (col - mu) / sd
            dd[m+"_z"] = z
        # weighted-available scoring
        z_cols = [m+"_z" for m in needed_freq]
        Z = dd[z_cols].copy()
        w_series = pd.Series(weights_map[criteria])
        W = pd.DataFrame({m+"_z": w_series[m] for m in needed_freq}, index=dd.index)
        eff_W = W.where(Z.notna(), 0.0)
        denom = eff_W.sum(axis=1)
        dd['__score'] = (Z.fillna(0.0) * eff_W).sum(axis=1) / denom.replace(0, np.nan)
        dd['__rank'] = dd['__score'].rank(ascending=False, method='first')
        top_dd = dd.nsmallest(int(freq_topn), '__rank')[['scheme_code','scheme_name','forward_return']].copy()
        top_dd['ds'] = ds
        sel_rows.append(top_dd)

    if sel_rows:
        sel_all = pd.concat(sel_rows, ignore_index=True)
        counts = sel_all.groupby('scheme_code').size().rename('Times Selected').reset_index()
        name_map = sel_all[['scheme_code','scheme_name']].drop_duplicates()
        counts = counts.merge(name_map, on='scheme_code', how='left')
        avg_ret = detailed.groupby('scheme_code')['forward_return'].apply(
            lambda x: (((1 + x.mean()) ** 2) - 1) / 2
        ).rename('Avg 6M Return')
        counts = counts.merge(avg_ret, on='scheme_code', how='left')
        total_rebalances = max(len(eval_dates), 1)
        fund_counts_df = pd.DataFrame({
            'Scheme Code': counts['scheme_code'],
            'Fund Name': counts['scheme_name'],
            'Times Selected': counts['Times Selected'],
            'Selection Rate (%)': np.round(counts['Times Selected'] / total_rebalances * 100, 1),
            'Avg 6M Return (%)': np.round(counts['Avg 6M Return'] * 100, 2)
        })
        fund_counts_df = fund_counts_df.sort_values(['Times Selected', 'Avg 6M Return (%)'], ascending=[False, False])
        st.dataframe(fund_counts_df.head(10), use_container_width=True)
    else:
        st.info("No data available to compute frequency for the selected dates.")

    st.subheader("💰 Top Performing Individual Selections (Historical)")
    if 'sel_all' in locals() and isinstance(sel_all, pd.DataFrame) and not sel_all.empty:
        # Prepare keys for merge
        sel_keys = sel_all[['ds', 'scheme_code']].copy()
        sel_keys['rebalance_date'] = pd.to_datetime(sel_keys['ds'])
        sel_keys['scheme_code'] = sel_keys['scheme_code'].astype(str)

        detailed_sel_cols = detailed[['rebalance_date', 'scheme_code', 'scheme_name', 'forward_return', 'forward_rank', 'is_in_actual_top',
                      'sharpe', 'sortino', 'mdd']].copy()
        detailed_sel_cols['scheme_code'] = detailed_sel_cols['scheme_code'].astype(str)

        sel_detail = sel_keys.merge(
            detailed_sel_cols,
            on=['rebalance_date', 'scheme_code'],
            how='left'
        )
        top_returns = sel_detail.nlargest(10, 'forward_return').copy()
        top_returns['rebalance_date'] = top_returns['rebalance_date'].dt.strftime('%Y-%m-%d')
        top_returns['forward_return'] = (top_returns['forward_return'] * 100).round(2)
        top_returns['sharpe'] = top_returns['sharpe'].round(3)
        top_returns['sortino'] = top_returns['sortino'].round(3)
        top_returns['mdd'] = (top_returns['mdd'] * 100).round(2)

        column_names = {
            'rebalance_date': 'Date',
            'scheme_name': 'Fund Name',
            'forward_return': '6M Return (%)',
            'forward_rank': 'Rank',
            'is_in_actual_top': 'In Top N',
            'sharpe': 'Sharpe',
            'sortino': 'Sortino',
            'mdd': 'Max DD (%)'
        }
        top_returns = top_returns.rename(columns=column_names)

        st.dataframe(top_returns, use_container_width=True)
    else:
        st.info("No selections available to compute top individual selections. Adjust dates or Top N above.")

    # Fund performance heat map
    st.subheader("🔥 Fund Performance Heat Map")
    pivot_data = detailed.pivot_table(
        values='forward_return',
        index='scheme_name',
        columns=detailed['rebalance_date'].dt.year,
        aggfunc='mean'
    ) * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.matshow(pivot_data, cmap='RdYlGn', aspect='auto')

    # Set ticks
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in pivot_data.index])

    plt.colorbar(cax, label='Average 6M Return (%)')
    ax.set_title('Fund Performance by Year (Heat Map)', fontweight='bold')

    # Add value labels for significant returns
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.iloc[i, j]
            if not np.isnan(val) and abs(val) > 15:  # Only show extreme values
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                       color='black', fontsize=8, fontweight='bold')

    st.pyplot(fig)

def show_detailed_results(summary, detailed):
    st.header("🔍 Detailed Results Explorer")

    # --- Criteria-based selection (optional quick picker) ---
    st.subheader("🎛️ Quick Selector by Ranking Criteria")
    colc1, colc2, colc3 = st.columns([2,2,1])
    with colc1:
        criteria = st.selectbox(
            "Selection Criteria",
            options=[
                "Composite (Balanced)",
                "Momentum",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Max Drawdown",
                "Consistency"
            ],
            index=0,
            key="detail_criteria_sel"
        )
    with colc2:
        dates = sorted(summary['rebalance_date'].dt.strftime('%Y-%m-%d').unique())
        chosen_date = st.selectbox("Rebalance Date", options=dates, index=len(dates)-1, key="detail_date_sel")
    with colc3:
        top_n = st.number_input("Top N", min_value=1, max_value=50, value=5, step=1, key="detail_topn")
    show_all_detail = st.checkbox("Show all funds", value=False, key="details_show_all")

    weights_map = {
        "Composite (Balanced)": {"sharpe":0.25, "sortino":0.25, "ir":0.20, "ann_return":0.20, "mdd":0.10, "ann_vol":0.10, "te":0.10},
        "Momentum": {"momentum_6m": 0.4, "ann_return": 0.4, "ann_vol": 0.2},
        "Sharpe-Focused": {"sharpe":0.40, "sortino":0.40, "mdd":0.20},
        "Risk-Adjusted": {"ir":0.30, "sortino":0.40, "mdd":0.20, "ann_vol":0.10},
        "Consistency": {"sortino":0.40, "ir":0.30, "mdd":0.20, "ann_vol":0.10},
        "Sharpe Ratio": {"sharpe": 1.0},
        "Sortino Ratio": {"sortino": 1.0},
        "Information Ratio": {"ir": 1.0},
        "Annual Return": {"ann_return": 1.0},
        "Max Drawdown": {"mdd": 1.0},
        "Annual Volatility": {"ann_vol": 1.0},
        "Beta": {"beta": 1.0},
        "Tracking Error": {"te": 1.0},
    }

    dcrit = detailed[detailed['rebalance_date'].dt.strftime('%Y-%m-%d') == chosen_date].copy()
    if not dcrit.empty:
        needed = list(weights_map[criteria].keys())
        invert = {"mdd", "ann_vol", "te", "beta"}
        for m in needed:
            if m not in dcrit.columns:
                dcrit[m] = np.nan
            col = dcrit[m].astype(float)
            if m in invert:
                col = -col
            mu, sd = col.mean(), col.std()
            z = pd.Series(np.nan, index=col.index) if pd.isna(sd) or sd == 0 else (col - mu) / sd
            dcrit[m+"_z"] = z
        score = 0
        for m, w in weights_map[criteria].items():
            score = score + w * dcrit[m+"_z"]
        # Replace with weighted-available scoring across all metrics
        z_cols = [m+"_z" for m in needed]
            
        Z = dcrit[z_cols].copy()
        w_series = pd.Series(weights_map[criteria])
        W = pd.DataFrame({m+"_z": w_series[m] for m in needed}, index=dcrit.index)
        eff_W = W.where(Z.notna(), 0.0)
        denom = eff_W.sum(axis=1)
        dcrit['score'] = (Z.fillna(0.0) * eff_W).sum(axis=1) / denom.replace(0, np.nan)
        dcrit['rank'] = dcrit['score'].rank(ascending=False, method='first')
        # Ensure all metric columns exist for display
        all_needed = ['sharpe','sortino','mdd','ann_return','ann_vol','momentum_6m','beta','te','ir']
        for colname in all_needed:
            if colname not in dcrit.columns:
                dcrit[colname] = np.nan
        if show_all_detail:
            st.subheader("📋 Full Ranking (All Funds, Selected Date)")
            full = dcrit[['scheme_code','scheme_name','score','rank','forward_return','forward_rank'] + all_needed].copy()
            full = full.sort_values('rank')
            full['forward_return'] = (full['forward_return'] * 100).round(2)
            full['score'] = full['score'].round(3)
            st.dataframe(full.reset_index(drop=True), use_container_width=True)
        else:
            topcrit = dcrit.nsmallest(int(top_n), 'rank')[[
                'scheme_code','scheme_name','score','rank','forward_return','forward_rank'] + needed
            ].copy()
            topcrit = topcrit.rename(columns={'forward_return':'6M_forward_return','forward_rank':'6M_forward_rank'})
            topcrit['6M_forward_return'] = (topcrit['6M_forward_return'] * 100).round(2)
            topcrit['score'] = topcrit['score'].round(3)
            st.markdown(f"**Top {int(top_n)} funds — {criteria} — {chosen_date}**")
            st.dataframe(topcrit.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No data for the chosen date.")

    st.markdown("---")

    # Filters
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
            rank_filter = st.slider(
                "Filter by Composite Rank:",
                min_value=1,
                max_value=50,
                value=(1, 50)
            )
        else:
            st.caption("Composite rank filter unavailable (showing all funds).")

    # Apply filters
    filtered_detailed = detailed.copy()
    if date_filter:
        filtered_detailed = filtered_detailed[
            filtered_detailed['rebalance_date'].dt.strftime('%Y-%m-%d').isin(date_filter)
        ]
    if fund_filter:
        filtered_detailed = filtered_detailed[filtered_detailed['scheme_name'].isin(fund_filter)]
    if 'show_rank_slider' in locals() and show_rank_slider:
        filtered_detailed = filtered_detailed[
            (filtered_detailed['composite_rank'] >= rank_filter[0]) &
            (filtered_detailed['composite_rank'] <= rank_filter[1])
        ]

    st.subheader(f"Showing {len(filtered_detailed)} selections")

    # Display filtered results
    display_cols = [
        'rebalance_date', 'scheme_name', 'sharpe', 'sortino', 'mdd', 'ann_return', 'ann_vol', 'beta', 'te', 'ir',
        'composite_score', 'composite_rank', 'forward_return', 'forward_rank', 'is_in_actual_top'
    ]

    # Ensure columns exist (in case user hasn't rerun the new backtest yet)
    for col in display_cols:
        if col not in filtered_detailed.columns:
            if col == 'rebalance_date' or col == 'scheme_name':
                continue
            filtered_detailed[col] = np.nan

    display_df = filtered_detailed[display_cols].copy()
    display_df['rebalance_date'] = display_df['rebalance_date'].dt.strftime('%Y-%m-%d')
    display_df['sharpe'] = display_df['sharpe'].round(3)
    display_df['sortino'] = display_df['sortino'].round(3)
    display_df['mdd'] = (display_df['mdd'] * 100).round(2)
    display_df['composite_score'] = display_df['composite_score'].round(3)
    display_df['forward_return'] = (display_df['forward_return'] * 100).round(2)

    column_names = {
        'rebalance_date': 'Date',
        'scheme_name': 'Fund Name',
        'sharpe': 'Sharpe',
        'sortino': 'Sortino',
        'mdd': 'Max DD (%)',
        'composite_score': 'Composite Score',
        'composite_rank': 'Composite Rank',
        'forward_return': '6M Return (%)',
        'forward_rank': 'Forward Rank',
        'is_in_actual_top': 'In Top N?'
    }
    display_df = display_df.rename(columns=column_names)

    st.dataframe(display_df, use_container_width=True)

    # Export option
    if st.button("📥 Export Filtered Results to CSV"):
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_backtest_results.csv",
            mime="text/csv"
        )

def show_methodology():
    st.header("📋 Methodology & Implementation")

    st.markdown("""
    ## Overview
    This dashboard evaluates mutual funds across categories (Large, Small, Mid, Large & Mid, Multi, International). For the chosen category, we compute risk/return metrics, rank funds using direction‑aware z‑scores with user‑selected criteria, and measure how those selections performed over the next 6 months.

    ## Glossary
    - **Lookback window**: Number of past trading days used to compute metrics (252 ≈ 1 year).
    - **Holding period (forward horizon)**: Number of trading days ahead for evaluating results (126 ≈ 6 months).
    - **Rebalance date**: The date when we compute metrics using the lookback window and make a selection for the next holding period.
    - **Top N**: The number of funds selected at each rebalance date based on the chosen criteria (user‑configurable; default 15 in backtests).
    - **Forward return**: Performance over the next 6 months from the rebalance date: NAV[t+H]/NAV[t] − 1.
    - **Forward rank**: Rank of each fund by forward return (1 = best) within the category at that date.
    - **Criteria**: A weighted combination of metrics used to score funds (Composite, Momentum, Sharpe‑Focused, Risk‑Adjusted, Consistency).

    ## Data Preparation
    - Input: category‑specific NAV CSVs with columns `scheme_code`, `scheme_name`, `date` (DD‑MM‑YYYY), `nav`.
    - De‑duplication: keep the last NAV per `scheme_code` + `date`.
    - Sort by date; forward‑fill gaps up to 5 trading days.
    - No hard exclusion for missingness: metrics are computed per window using available data; undefined metrics remain `NaN`.

    ## Metrics (per fund, per rebalance)
    Let daily return be `r_t = P_t / P_{t-1} − 1`. Risk‑free daily rate `rf_d = 0.05 / 252`.

    - **Sharpe (annualized)**: `(mean(r_t − rf_d) / std(r_t)) × sqrt(252)`
    - **Sortino (annualized)**: `(mean(r_t − rf_d) / std(r_t where r_t < 0)) × sqrt(252)`
    - **Max Drawdown**: `min((P_t − cummax(P_t)) / cummax(P_t))` over the lookback
    - **Annual Return**: `mean(r_t) × 252`
    - **Annual Volatility**: `std(r_t) × sqrt(252)`
    - **Tracking Error (annualized)**: `std((r_t − r_b,t)) × sqrt(252)`
    - **Information Ratio (annualized)**: `(mean(r_t − r_b,t) × sqrt(252)) / TE`
    - **Beta vs benchmark**: `cov(r_t, r_b,t) / var(r_b,t)`
    - Safeguards: IR/TE/Beta require ≥ 20 aligned observations with non‑zero variance; otherwise `NaN`.

    ## Benchmark
    - Current benchmark: Nifty 100 (Close). Used for TE/IR/Beta and benchmark‑relative win rate.
    - Per‑category benchmarks (e.g., smallcap/midcap indexes) can be integrated similarly.

    ## Scoring & Ranking
    1. **Direction‑aware z‑scores** (cross‑sectional per date):
       - For metrics where lower is better (MDD, Annual Volatility, TE, Beta), we invert the values before z‑scoring.
       - If a metric’s cross‑section has zero variance or insufficient data, its z‑score is `NaN`.
    2. **Weighted‑available scoring**:
       - `score_i = sum_j w_j × z_{i,j}  /  sum_j |w_j| × 1(z_{i,j} is not NaN)`
       - Missing metrics do not penalize a fund; if all metrics are missing, score is `NaN` and the fund ranks last.
    3. **Criteria (example weights)**:
       - Composite (Balanced): Sharpe 0.25, Sortino 0.25, IR 0.20, Ann Return 0.20, MDD 0.10, Ann Vol 0.10, TE 0.10
       - Momentum: Ann Return 0.40, Sharpe 0.30, Beta 0.20, MDD 0.10
       - Sharpe‑Focused: Sharpe 0.40, Sortino 0.40, MDD 0.20
       - Risk‑Adjusted: IR 0.30, Sortino 0.40, MDD 0.20, Ann Vol 0.10
       - Consistency: Sortino 0.40, IR 0.30, MDD 0.20, Ann Vol 0.10

    ### Criteria formulas (using z‑scores and weighted‑available denominator)
    Let `Z_sharpe = z(Sharpe)`, `Z_sortino = z(Sortino)`, `Z_ir = z(IR)`, `Z_ret = z(Annual Return)`,
    `Z_mdd = z(−MDD)`, `Z_vol = z(−Annual Volatility)`, `Z_te = z(−TE)`, `Z_beta = z(−Beta)`.
    Denominator `D_i = sum of |w| for metrics where Z is available for fund i`.

    - Composite (Balanced):
      `score_i = (0.25·Z_sharpe + 0.25·Z_sortino + 0.20·Z_ir + 0.20·Z_ret + 0.10·Z_mdd + 0.10·Z_vol + 0.10·Z_te) / D_i`

    - Momentum:
      `score_i = (0.40·Z_ret + 0.30·Z_sharpe + 0.20·Z_beta + 0.10·Z_mdd) / D_i`

    - Sharpe‑Focused:
      `score_i = (0.40·Z_sharpe + 0.40·Z_sortino + 0.20·Z_mdd) / D_i`

    - Risk‑Adjusted:
      `score_i = (0.30·Z_ir + 0.40·Z_sortino + 0.20·Z_mdd + 0.10·Z_vol) / D_i`

    - Consistency:
      `score_i = (0.40·Z_sortino + 0.30·Z_ir + 0.20·Z_mdd + 0.10·Z_vol) / D_i`

    ## Selection & Evaluation
    - Selection: At each rebalance date, pick Top N by the chosen criteria.
    - Forward evaluation:
      - **Avg 6M return of selections**
      - **Avg forward rank** of selections
      - **Top‑N overlap share** with actual top performers by forward return
      - **Benchmark‑relative win rate**: share of rebalances where selections’ average 6M return > benchmark’s 6M forward return

    ## Exploratory Views
    - **Most Frequently Selected Funds**: recomputed historically from per‑date Top N for the current criteria and user‑chosen Top N; optional date filter.
    - **Top Individual Selections**: best historical 6M returns among selected funds.
    - **Heat Map**: average 6M forward return by fund × year.

    ## Files & Naming
    - Inputs: `{category}_funds.csv` (e.g., `largecap_funds.csv`, `smallcap_funds.csv`, ...)
    - Outputs per category: `backtest_results_summary_{category}.csv`, `backtest_results_detailed_{category}.csv`
    - The dashboard loads the files for the selected category from the sidebar.

    ## Limitations
    - Metrics depend on historical data availability; short histories produce `NaN` for some metrics.
    - Z‑scores are relative to the peer group and date; they do not convey absolute attractiveness.
    - Single benchmark is used across categories unless replaced with category‑specific benchmarks.
    - No transaction costs, taxes, or liquidity constraints are modeled.
    """)

if __name__ == "__main__":
    main()
