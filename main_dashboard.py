"""
Improved Fund Selection Strategy
================================
Based on hit rate analysis, this implements:
1. Elimination approach (avoid worst) instead of ranking
2. Multiple independent filters
3. Regime-aware selection
4. Better hit rate measurement

Run: streamlit run improved_strategy.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Improved Fund Selection Strategy",
    page_icon="🎯",
    layout="wide"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

RISK_FREE_RATE = 0.06
TRADING_DAYS_YEAR = 252
DAILY_RISK_FREE_RATE = (1 + RISK_FREE_RATE) ** (1/TRADING_DAYS_YEAR) - 1
DATA_DIR = "data"
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
# DATA LOADING
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
# FACTOR CALCULATIONS
# ============================================================================

def calculate_factors_for_fund(series, bench_series=None):
    """Calculate key factors for a single fund."""
    factors = {}
    
    if len(series) < 200:
        return factors
    
    returns = series.pct_change().dropna()
    
    # 1. Momentum factors
    if len(series) >= 63:
        factors['momentum_3m'] = (series.iloc[-1] / series.iloc[-63]) - 1
    if len(series) >= 126:
        factors['momentum_6m'] = (series.iloc[-1] / series.iloc[-126]) - 1
    if len(series) >= 252:
        factors['momentum_12m'] = (series.iloc[-1] / series.iloc[-252]) - 1
    
    # 2. Volatility
    if len(returns) >= 126:
        factors['volatility'] = returns.std() * np.sqrt(252)
        downside = returns[returns < 0]
        if len(downside) > 20:
            factors['downside_vol'] = downside.std() * np.sqrt(252)
    
    # 3. Max Drawdown
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.expanding().max()
    drawdown = (cum_ret / peak) - 1
    factors['max_drawdown'] = drawdown.min()
    
    # 4. Sharpe & Sortino
    if len(returns) >= 126 and returns.std() > 0:
        excess = returns - DAILY_RISK_FREE_RATE
        factors['sharpe'] = (excess.mean() / returns.std()) * np.sqrt(252)
        
        downside = returns[returns < 0]
        if len(downside) > 20 and downside.std() > 0:
            factors['sortino'] = (excess.mean() / downside.std()) * np.sqrt(252)
    
    # 5. Consistency (% positive months)
    monthly = series.resample('ME').last().pct_change().dropna()
    if len(monthly) >= 6:
        factors['pct_positive_months'] = (monthly > 0).mean()
    
    # 6. Moving averages
    if len(series) >= 200:
        ma50 = series.rolling(50).mean().iloc[-1]
        ma200 = series.rolling(200).mean().iloc[-1]
        current = series.iloc[-1]
        factors['price_vs_ma50'] = (current / ma50) - 1
        factors['price_vs_ma200'] = (current / ma200) - 1
        factors['ma50_vs_ma200'] = (ma50 / ma200) - 1
    
    # 7. Benchmark relative (if available)
    if bench_series is not None:
        common_idx = returns.index.intersection(bench_series.index)
        if len(common_idx) > 60:
            f_ret = returns.loc[common_idx]
            b_ret = bench_series.pct_change().dropna().loc[common_idx]
            
            # Batting average
            factors['batting_avg'] = (f_ret > b_ret).mean()
            
            # Information ratio
            active = f_ret - b_ret
            te = active.std() * np.sqrt(252)
            if te > 0:
                factors['information_ratio'] = (active.mean() * 252) / te
            
            # Up/Down capture
            up_days = b_ret > 0
            down_days = b_ret < 0
            if up_days.sum() > 20 and b_ret[up_days].mean() != 0:
                factors['up_capture'] = f_ret[up_days].mean() / b_ret[up_days].mean()
            if down_days.sum() > 20 and b_ret[down_days].mean() != 0:
                factors['down_capture'] = f_ret[down_days].mean() / b_ret[down_days].mean()
    
    return factors

# ============================================================================
# ELIMINATION STRATEGY
# ============================================================================

def strategy_elimination(nav_df, benchmark_series, date, top_n=3, scheme_map=None):
    """
    Elimination-based fund selection.
    
    Instead of ranking by a score, we ELIMINATE funds that fail criteria.
    This is more robust than trying to pick the "best".
    
    Process:
    1. Start with all funds
    2. Eliminate bottom 25% by max drawdown (avoid highest risk)
    3. Eliminate bottom 25% by volatility (avoid most volatile)
    4. From remaining, eliminate any below-median momentum
    5. Pick top N by Sharpe ratio
    """
    hist_data = nav_df.loc[:date]
    
    # Calculate factors for all funds
    fund_factors = {}
    for col in nav_df.columns:
        series = hist_data[col].dropna()
        if len(series) >= 200:
            factors = calculate_factors_for_fund(series, benchmark_series.loc[:date] if benchmark_series is not None else None)
            if factors:
                fund_factors[col] = factors
    
    if len(fund_factors) < 10:
        return []
    
    df = pd.DataFrame(fund_factors).T
    
    # Step 1: Eliminate worst drawdowns (bottom 25%)
    if 'max_drawdown' in df.columns:
        dd_threshold = df['max_drawdown'].quantile(0.25)  # More negative = worse
        survivors = df[df['max_drawdown'] >= dd_threshold].index.tolist()
        df = df.loc[survivors]
    
    if len(df) < top_n * 2:
        return df.index.tolist()[:top_n]
    
    # Step 2: Eliminate highest volatility (bottom 25% of remaining)
    if 'volatility' in df.columns:
        vol_threshold = df['volatility'].quantile(0.75)  # Higher = worse
        survivors = df[df['volatility'] <= vol_threshold].index.tolist()
        df = df.loc[survivors]
    
    if len(df) < top_n * 2:
        return df.index.tolist()[:top_n]
    
    # Step 3: Require above-median momentum
    if 'momentum_6m' in df.columns:
        mom_threshold = df['momentum_6m'].median()
        survivors = df[df['momentum_6m'] >= mom_threshold].index.tolist()
        df = df.loc[survivors]
    
    if len(df) < top_n:
        return df.index.tolist()[:top_n]
    
    # Step 4: From survivors, pick top N by Sharpe
    if 'sharpe' in df.columns:
        selected = df.nlargest(top_n, 'sharpe').index.tolist()
    else:
        selected = df.index.tolist()[:top_n]
    
    return selected


def strategy_regime_adaptive(nav_df, benchmark_series, date, top_n=3, scheme_map=None):
    """
    Regime-aware selection.
    
    1. Detect market regime (bull/bear/sideways)
    2. Apply different selection criteria based on regime
    """
    hist_data = nav_df.loc[:date]
    
    # Detect regime using benchmark
    if benchmark_series is not None:
        bench_hist = benchmark_series.loc[:date]
        if len(bench_hist) >= 200:
            current = bench_hist.iloc[-1]
            ma50 = bench_hist.rolling(50).mean().iloc[-1]
            ma200 = bench_hist.rolling(200).mean().iloc[-1]
            
            # Regime detection
            if current > ma200 and ma50 > ma200:
                regime = 'bull'
            elif current < ma200 and ma50 < ma200:
                regime = 'bear'
            else:
                regime = 'transition'
        else:
            regime = 'unknown'
    else:
        regime = 'unknown'
    
    # Calculate factors
    fund_factors = {}
    for col in nav_df.columns:
        series = hist_data[col].dropna()
        if len(series) >= 200:
            factors = calculate_factors_for_fund(series, benchmark_series.loc[:date] if benchmark_series is not None else None)
            if factors:
                fund_factors[col] = factors
    
    if len(fund_factors) < 10:
        return [], regime
    
    df = pd.DataFrame(fund_factors).T
    
    # Apply regime-specific strategy
    if regime == 'bull':
        # In bull market: Focus on momentum and upside capture
        # Eliminate low momentum
        if 'momentum_6m' in df.columns:
            threshold = df['momentum_6m'].quantile(0.5)
            df = df[df['momentum_6m'] >= threshold]
        
        # Pick by momentum
        if 'momentum_3m' in df.columns:
            selected = df.nlargest(top_n, 'momentum_3m').index.tolist()
        else:
            selected = df.index.tolist()[:top_n]
    
    elif regime == 'bear':
        # In bear market: Focus on low volatility and downside protection
        # Eliminate high volatility
        if 'volatility' in df.columns:
            threshold = df['volatility'].quantile(0.5)
            df = df[df['volatility'] <= threshold]
        
        # Eliminate bad drawdowns
        if 'max_drawdown' in df.columns:
            threshold = df['max_drawdown'].quantile(0.5)
            df = df[df['max_drawdown'] >= threshold]
        
        # Pick by Sortino or Sharpe
        if 'sortino' in df.columns:
            selected = df.nlargest(top_n, 'sortino').index.tolist()
        elif 'sharpe' in df.columns:
            selected = df.nlargest(top_n, 'sharpe').index.tolist()
        else:
            selected = df.index.tolist()[:top_n]
    
    else:  # transition or unknown
        # Use balanced approach - elimination strategy
        return strategy_elimination(nav_df, benchmark_series, date, top_n, scheme_map), regime
    
    return selected, regime


def strategy_avoid_worst(nav_df, benchmark_series, date, top_n=3, scheme_map=None):
    """
    "Avoid the Worst" strategy.
    
    Instead of picking "best", we identify and AVOID the worst funds.
    Then from the "safe" pool, we pick by simple momentum.
    
    This works because:
    - Bad funds are easier to identify than good funds
    - Avoiding disasters is more reliable than finding winners
    """
    hist_data = nav_df.loc[:date]
    
    # Calculate factors
    fund_factors = {}
    for col in nav_df.columns:
        series = hist_data[col].dropna()
        if len(series) >= 200:
            factors = calculate_factors_for_fund(series, benchmark_series.loc[:date] if benchmark_series is not None else None)
            if factors:
                fund_factors[col] = factors
    
    if len(fund_factors) < 10:
        return []
    
    df = pd.DataFrame(fund_factors).T
    n_funds = len(df)
    
    # AVOID funds with:
    avoid_flags = pd.Series(False, index=df.index)
    
    # 1. Worst 20% by drawdown
    if 'max_drawdown' in df.columns:
        worst_dd = df['max_drawdown'].nsmallest(int(n_funds * 0.2)).index
        avoid_flags.loc[worst_dd] = True
    
    # 2. Worst 20% by volatility  
    if 'volatility' in df.columns:
        worst_vol = df['volatility'].nlargest(int(n_funds * 0.2)).index
        avoid_flags.loc[worst_vol] = True
    
    # 3. Negative momentum (if significant)
    if 'momentum_6m' in df.columns:
        negative_mom = df[df['momentum_6m'] < -0.05].index
        avoid_flags.loc[negative_mom] = True
    
    # 4. Worst 20% by Sharpe (if available)
    if 'sharpe' in df.columns:
        worst_sharpe = df['sharpe'].nsmallest(int(n_funds * 0.2)).index
        avoid_flags.loc[worst_sharpe] = True
    
    # Keep the "safe" funds
    safe_funds = df[~avoid_flags]
    
    if len(safe_funds) < top_n:
        # If too few safe funds, relax criteria
        safe_funds = df.nlargest(top_n * 2, 'momentum_6m' if 'momentum_6m' in df.columns else df.columns[0])
    
    # From safe funds, pick top N by momentum
    if 'momentum_6m' in safe_funds.columns:
        selected = safe_funds.nlargest(top_n, 'momentum_6m').index.tolist()
    else:
        selected = safe_funds.index.tolist()[:top_n]
    
    return selected


def strategy_consistency_first(nav_df, benchmark_series, date, top_n=3, scheme_map=None):
    """
    Consistency-First Strategy.
    
    Pick funds that have been CONSISTENTLY good, not just recently good.
    
    Requires:
    - Top 50% performance in at least 3 of last 4 quarters
    - Then rank by recent momentum
    """
    hist_data = nav_df.loc[:date]
    
    # Check quarterly consistency
    consistent_funds = []
    
    for col in nav_df.columns:
        series = hist_data[col].dropna()
        if len(series) < 300:  # Need ~1 year of history
            continue
        
        # Calculate quarterly returns for this fund vs all funds
        quarters_in_top_half = 0
        
        for q in range(4):  # Last 4 quarters
            q_end = date - pd.Timedelta(days=q * 91)
            q_start = q_end - pd.Timedelta(days=91)
            
            try:
                # Get returns for all funds in this quarter
                idx_start = hist_data.index.asof(q_start)
                idx_end = hist_data.index.asof(q_end)
                
                if pd.isna(idx_start) or pd.isna(idx_end):
                    continue
                
                all_returns = (hist_data.loc[idx_end] / hist_data.loc[idx_start]) - 1
                all_returns = all_returns.dropna()
                
                if col not in all_returns.index:
                    continue
                
                fund_return = all_returns[col]
                median_return = all_returns.median()
                
                if fund_return >= median_return:
                    quarters_in_top_half += 1
                    
            except:
                continue
        
        # Require at least 3 out of 4 quarters in top half
        if quarters_in_top_half >= 3:
            consistent_funds.append(col)
    
    if len(consistent_funds) < top_n:
        # Fall back to elimination strategy
        return strategy_elimination(nav_df, benchmark_series, date, top_n, scheme_map)
    
    # From consistent funds, calculate recent factors
    fund_factors = {}
    for col in consistent_funds:
        series = hist_data[col].dropna()
        factors = calculate_factors_for_fund(series, benchmark_series.loc[:date] if benchmark_series is not None else None)
        if factors:
            fund_factors[col] = factors
    
    if not fund_factors:
        return []
    
    df = pd.DataFrame(fund_factors).T
    
    # Pick by recent momentum
    if 'momentum_3m' in df.columns:
        selected = df.nlargest(top_n, 'momentum_3m').index.tolist()
    else:
        selected = df.index.tolist()[:top_n]
    
    return selected


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest(nav_df, benchmark_series, strategy_func, holding_period=252, top_n=3, scheme_map=None):
    """Run backtest for any strategy function."""
    
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
        
        # Get strategy selection
        result = strategy_func(nav_df, benchmark_series, date, top_n, scheme_map)
        
        # Handle regime-aware strategies that return tuple
        if isinstance(result, tuple):
            selected, regime = result
        else:
            selected = result
            regime = None
        
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
        in_top_half = 0.0
        avoided_bottom = 0.0
        
        if selected:
            period_ret_all = (nav_df.iloc[exit_i] / nav_df.iloc[entry]) - 1
            period_ret_all = period_ret_all.dropna()
            port_ret = period_ret_all[selected].mean()
            
            # Standard hit rate (in top N)
            actual_top = period_ret_all.nlargest(top_n).index.tolist()
            matches = len(set(selected).intersection(set(actual_top)))
            hit_rate = matches / top_n
            
            # NEW METRICS:
            # 1. How many of our picks were in top 50%?
            median_ret = period_ret_all.median()
            in_top_half = sum(1 for s in selected if period_ret_all.get(s, -999) >= median_ret) / top_n
            
            # 2. How many did we AVOID from bottom 25%?
            bottom_25 = period_ret_all.nsmallest(int(len(period_ret_all) * 0.25)).index
            avoided_bottom = sum(1 for s in selected if s not in bottom_25) / top_n
        
        cap *= (1 + (port_ret if not pd.isna(port_ret) else 0))
        b_cap *= (1 + b_ret)
        
        history.append({
            'date': date,
            'selected': [scheme_map.get(s, s)[:30] for s in selected] if scheme_map else selected,
            'return': port_ret,
            'hit_rate': hit_rate,
            'in_top_half': in_top_half,
            'avoided_bottom': avoided_bottom,
            'regime': regime
        })
        
        eq_curve.append({'date': nav_df.index[exit_i], 'value': cap})
        bench_curve.append({'date': nav_df.index[exit_i], 'value': b_cap})
    
    return pd.DataFrame(history), pd.DataFrame(eq_curve), pd.DataFrame(bench_curve)


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.title("🎯 Improved Fund Selection Strategies")
    
    st.markdown("""
    ### Why Standard Approaches Fail
    
    The factor discovery analysis revealed that even the BEST factors only have ~40-45% of winners 
    in the top quartile. This means traditional "rank and pick" approaches have fundamental limits.
    
    ### Alternative Approaches
    
    This dashboard tests strategies that work DIFFERENTLY:
    
    1. **🛡️ Elimination Strategy**: Avoid the worst instead of picking the best
    2. **📊 Regime Adaptive**: Different strategies for bull/bear/sideways markets
    3. **🚫 Avoid Worst**: Explicitly identify and avoid problem funds
    4. **📈 Consistency First**: Require sustained performance, not just recent
    
    ### Better Success Metrics
    
    Instead of just "hit rate", we measure:
    - **Hit Rate**: Did we pick actual top N? (hardest metric)
    - **Top Half Rate**: Were our picks in top 50%? (more realistic)
    - **Avoided Bottom**: Did we avoid bottom 25%? (risk management)
    """)
    
    st.divider()
    
    # Controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        category = st.selectbox("Category", list(FILE_MAPPING.keys()))
    with col2:
        holding_period = st.selectbox("Holding Period", [126, 252, 378, 504], index=1,
                                       format_func=lambda x: f"{x} days")
    with col3:
        top_n = st.number_input("Funds to Select", 2, 10, 3)
    with col4:
        run_bt = st.button("🚀 Run All Strategies", type="primary", use_container_width=True)
    
    if run_bt:
        with st.spinner("Loading data..."):
            nav_df, scheme_map = load_fund_data_raw(category)
            benchmark = load_nifty_data()
        
        if nav_df is None:
            st.error("Could not load data")
            return
        
        st.info(f"Loaded {len(nav_df.columns)} funds")
        
        # Define strategies
        strategies = {
            '🛡️ Elimination': strategy_elimination,
            '📊 Regime Adaptive': strategy_regime_adaptive,
            '🚫 Avoid Worst': strategy_avoid_worst,
            '📈 Consistency First': strategy_consistency_first
        }
        
        results = []
        fig = go.Figure()
        
        progress = st.progress(0)
        
        for idx, (name, func) in enumerate(strategies.items()):
            with st.spinner(f"Testing {name}..."):
                history, eq, bench = run_backtest(
                    nav_df, benchmark, func, holding_period, top_n, scheme_map
                )
            
            if history is not None and not eq.empty:
                start_date = eq.iloc[0]['date']
                end_date = eq.iloc[-1]['date']
                years = (end_date - start_date).days / 365.25
                
                strat_cagr = (eq.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                bench_cagr = (bench.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                
                avg_hit_rate = history['hit_rate'].mean()
                avg_top_half = history['in_top_half'].mean()
                avg_avoided = history['avoided_bottom'].mean()
                
                results.append({
                    'Strategy': name,
                    'CAGR %': strat_cagr * 100,
                    'vs Benchmark': (strat_cagr - bench_cagr) * 100,
                    'Hit Rate %': avg_hit_rate * 100,
                    'Top Half %': avg_top_half * 100,
                    'Avoided Bottom %': avg_avoided * 100
                })
                
                fig.add_trace(go.Scatter(
                    x=eq['date'], y=eq['value'],
                    name=name, mode='lines'
                ))
            
            progress.progress((idx + 1) / len(strategies))
        
        # Add benchmark
        if not bench.empty:
            fig.add_trace(go.Scatter(
                x=bench['date'], y=bench['value'],
                name='Benchmark', line=dict(color='gray', dash='dot', width=2)
            ))
        
        progress.empty()
        
        # Show results
        st.subheader("📊 Strategy Comparison")
        
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.set_index('Strategy')
            
            # Highlight the metrics
            st.dataframe(
                results_df.style.format({
                    'CAGR %': '{:.2f}',
                    'vs Benchmark': '{:.2f}',
                    'Hit Rate %': '{:.1f}',
                    'Top Half %': '{:.1f}',
                    'Avoided Bottom %': '{:.1f}'
                }).background_gradient(subset=['Top Half %', 'Avoided Bottom %'], cmap='Greens'),
                use_container_width=True
            )
            
            st.markdown("""
            **Interpreting Results:**
            - **Hit Rate**: % of times our picks were in actual top N (expect 20-40%)
            - **Top Half**: % of picks that were in top 50% (expect 50-70%)
            - **Avoided Bottom**: % of picks NOT in bottom 25% (expect 70-85%)
            """)
        
        # Chart
        fig.update_layout(
            title=f'{category} - Strategy Comparison',
            yaxis_title='Value (100 = Start)',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results by strategy
        st.divider()
        st.subheader("📋 Detailed Results by Strategy")
        
        tabs = st.tabs(list(strategies.keys()))
        
        for tab, (name, func) in zip(tabs, strategies.items()):
            with tab:
                history, eq, bench = run_backtest(
                    nav_df, benchmark, func, holding_period, top_n, scheme_map
                )
                
                if history is not None:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg Hit Rate", f"{history['hit_rate'].mean():.1%}")
                    col2.metric("Avg Top Half", f"{history['in_top_half'].mean():.1%}")
                    col3.metric("Avoided Bottom", f"{history['avoided_bottom'].mean():.1%}")
                    
                    # Show trade history
                    display_df = history[['date', 'selected', 'return', 'hit_rate', 'in_top_half', 'avoided_bottom']].copy()
                    display_df['selected'] = display_df['selected'].apply(lambda x: ', '.join(x) if x else '')
                    
                    st.dataframe(
                        display_df.style.format({
                            'return': '{:.2%}',
                            'hit_rate': '{:.0%}',
                            'in_top_half': '{:.0%}',
                            'avoided_bottom': '{:.0%}'
                        }),
                        use_container_width=True,
                        height=300
                    )

if __name__ == "__main__":
    main()
