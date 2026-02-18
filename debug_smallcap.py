"""
ML Fund Pro v5 - Dual Accuracy Edition
======================================
NEW FEATURES:
  - "Top 5 Precision" Metric: How many of our picks were actual Top 5 funds?
  - "Win Rate" Metric: How many picks beat the median?
  - Enhanced Reporting: Side-by-side accuracy comparison.

Run: streamlit run ml_fund_pro_v5.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from datetime import timedelta

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dual-Accuracy ML Pro", page_icon="🎯", layout="wide")

# ============================================================================
# 1. CONSTANTS & DATA LOADING
# ============================================================================

DATA_DIR = "data"
FILE_MAPPING = {"Large Cap": "largecap_merged.xlsx"}

@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, FILE_MAPPING["Large Cap"])
    if not os.path.exists(path): return None, None, None
    try:
        df = pd.read_excel(path, header=None)
        fund_names = df.iloc[2, 1:].tolist()
        data_df = df.iloc[4:, :].copy()
        if isinstance(data_df.iloc[-1, 0], str) and 'Accord' in str(data_df.iloc[-1, 0]):
            data_df = data_df.iloc[:-1, :]
        dates = pd.to_datetime(data_df.iloc[:, 0], errors='coerce')
        nav_wide = pd.DataFrame(index=dates)
        scheme_map = {}
        for i, name in enumerate(fund_names):
            if pd.notna(name) and str(name).strip() and 'idcw' not in str(name).lower():
                code = f"fund_{i}"
                scheme_map[code] = name
                nav_wide[code] = pd.to_numeric(data_df.iloc[:, i+1], errors='coerce').values
        
        # Load Benchmark
        b_path = os.path.join(DATA_DIR, "nifty100_data.csv")
        bench = None
        if os.path.exists(b_path):
            b_df = pd.read_csv(b_path)
            b_df.columns = [c.lower().strip() for c in b_df.columns]
            b_df['date'] = pd.to_datetime(b_df['date'])
            b_df = b_df.set_index('date').sort_index()
            bench = b_df['nav'] if 'nav' in b_df.columns else b_df.iloc[:,0]
            bench = bench.reindex(nav_wide.index).ffill()
            
        return nav_wide.sort_index().ffill(), scheme_map, bench
    except Exception as e: st.error(f"Error: {e}"); return None, None, None

# ============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# ============================================================================

def calculate_technical_features(s, benchmark_series):
    # 1. Volatility Adjusted Momentum
    ret_6m = s.pct_change(126).iloc[-1]
    vol_6m = s.pct_change().tail(126).std() * np.sqrt(252)
    vol_adj_mom = ret_6m / vol_6m if vol_6m > 0 else 0
    
    # 2. Beta (6-month rolling)
    f_ret = s.pct_change().tail(126).dropna()
    b_ret = benchmark_series.pct_change().tail(126).dropna()
    common = f_ret.index.intersection(b_ret.index)
    
    if len(common) > 60:
        cov = np.cov(f_ret.loc[common], b_ret.loc[common])[0, 1]
        var = np.var(b_ret.loc[common])
        beta = cov / var if var != 0 else 1.0
    else:
        beta = 1.0
        
    return vol_adj_mom, beta

def prepare_enhanced_dataset(nav_df, benchmark, holding_period=63):
    features = []
    
    # Market Regime (1 = Bull, 0 = Bear)
    bench_ma200 = benchmark.rolling(200).mean()
    market_regime = (benchmark > bench_ma200).astype(int)
    
    step_size = 21 # Monthly steps
    valid_dates = nav_df.index[252:-holding_period:step_size]
    
    with st.spinner("🧠 Engineering 'Context-Aware' Features..."):
        for dt in valid_dates:
            hist_nav = nav_df.loc[:dt].tail(253)
            curr_regime = market_regime.loc[dt]
            
            # Peer Stats for Relative Strength
            period_returns = hist_nav.pct_change(126).iloc[-1]
            median_ret = period_returns.median()
            
            future_end = nav_df.index[min(nav_df.index.get_loc(dt) + holding_period, len(nav_df)-1)]
            
            for fund_id in nav_df.columns:
                s = hist_nav[fund_id].dropna()
                if len(s) < 252: continue
                
                try:
                    curr_price = s.iloc[-1]
                    fut_price = nav_df.loc[future_end, fund_id]
                    
                    # FEATURES
                    abs_ret_6m = period_returns[fund_id]
                    rel_strength = abs_ret_6m - median_ret
                    vol_adj_mom, beta = calculate_technical_features(s, benchmark.loc[:dt])
                    
                    rets = s.pct_change().tail(126)
                    neg_rets = rets[rets < 0]
                    downside_dev = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 else 0.01
                    sortino_proxy = abs_ret_6m / downside_dev
                    
                    # TARGET (Future Return)
                    future_ret = (fut_price / curr_price) - 1
                    
                    features.append({
                        'date': dt,
                        'fund_id': fund_id,
                        'f_regime': curr_regime,
                        'f_rel_strength': rel_strength,
                        'f_vol_mom': vol_adj_mom,
                        'f_beta': beta,
                        'f_sortino': sortino_proxy,
                        'target_ret': future_ret
                    })
                except: continue
                
    df = pd.DataFrame(features)
    
    # Target Class: Did it beat the median? (Broad definition of "Winner")
    df['target_class'] = df.groupby('date')['target_ret'].transform(
        lambda x: x > x.median()
    ).astype(int)
    
    return df.dropna()

# ============================================================================
# 3. ENSEMBLE MODEL TRAINING WITH DUAL ACCURACY
# ============================================================================

def train_ensemble_model(ml_df, train_window_yrs, threshold=0.60, top_n=5):
    """
    Trains model and calculates BOTH Broad Accuracy and Exact Top-N Precision.
    """
    dates = sorted(ml_df['date'].unique())
    start_idx = int(train_window_yrs * 12)
    results = []
    
    # ENSEMBLE
    rf = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
    
    feature_cols = [c for c in ml_df.columns if c.startswith('f_')]
    
    prog = st.progress(0)
    
    for i in range(start_idx, len(dates)):
        test_date = dates[i]
        
        # Expanding Window Train
        train = ml_df[ml_df['date'] < test_date]
        test = ml_df[ml_df['date'] == test_date]
        
        if len(train) < 500 or len(test) == 0: continue
            
        X_train, y_train = train[feature_cols], train['target_class']
        X_test, y_test = test[feature_cols], test['target_class']
        
        ensemble.fit(X_train, y_train)
        
        # Predictions
        probs = ensemble.predict_proba(X_test)[:, 1]
        
        test_res = pd.DataFrame({
            'fund_id': test['fund_id'].values,
            'prob': probs,
            'actual_ret': test['target_ret'].values,
            'is_winner': y_test.values # Did it beat median?
        })
        
        # --- SELECTION LOGIC ---
        high_conf_picks = test_res[test_res['prob'] > threshold]
        
        if high_conf_picks.empty:
            avg_ret = test['target_ret'].mean()
            broad_accuracy = np.nan
            top_n_precision = np.nan
            n_picks = 0
        else:
            # Select Top N from high confidence ones
            my_top_picks = high_conf_picks.nlargest(top_n, 'prob')
            
            # Metric 1: Broad Accuracy (Did we pick funds that beat median?)
            broad_accuracy = my_top_picks['is_winner'].mean()
            
            # Metric 2: Top N Precision (Did we pick actual Top N funds?)
            # Find the ACTUAL Top N funds for this period based on return
            actual_top_funds = test_res.nlargest(top_n, 'actual_ret')['fund_id'].tolist()
            my_picked_funds = my_top_picks['fund_id'].tolist()
            
            # Count intersection
            correct_picks = len(set(my_picked_funds).intersection(set(actual_top_funds)))
            top_n_precision = correct_picks / top_n
            
            avg_ret = my_top_picks['actual_ret'].mean()
            n_picks = len(my_top_picks)
            
        results.append({
            'date': test_date,
            'strategy_ret': avg_ret,
            'benchmark_ret': test['target_ret'].mean(),
            'broad_accuracy': broad_accuracy,
            'top_n_precision': top_n_precision,
            'trades_taken': n_picks
        })
        prog.progress((i - start_idx) / (len(dates) - start_idx))
        
    return pd.DataFrame(results)

# ============================================================================
# 4. DASHBOARD
# ============================================================================

def main():
    st.markdown("## 🎯 Dual-Accuracy ML Strategy")
    st.markdown("""
    This dashboard calculates two types of accuracy:
    1.  **Win Rate (Broad):** % of picks that beat the median fund.
    2.  **Top 5 Precision (Strict):** % of picks that landed in the exact Top 5 of the universe.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: prob_thresh = st.slider("Confidence Threshold", 0.5, 0.8, 0.60, 0.05)
    with col2: hold_period = st.selectbox("Holding Period", [21, 63, 126], index=1, format_func=lambda x: f"{x} Days")
    with col3: top_n = st.number_input("Top N Funds", 1, 10, 5)
    
    nav, _, bench = load_data()
    if nav is None: return
    
    if st.button("Run Advanced Backtest"):
        ml_df = prepare_enhanced_dataset(nav, bench, hold_period)
        res = train_ensemble_model(ml_df, train_window_yrs=3, threshold=prob_thresh, top_n=top_n)
        
        if res.empty:
            st.warning("No trades generated.")
            return
            
        res['cum_strat'] = (1 + res['strategy_ret']).cumprod()
        res['cum_bench'] = (1 + res['benchmark_ret']).cumprod()
        
        trades_only = res.dropna(subset=['broad_accuracy'])
        
        final_ret = (res['cum_strat'].iloc[-1] - 1) * 100
        bench_ret = (res['cum_bench'].iloc[-1] - 1) * 100
        
        # Dual Accuracy Metrics
        avg_broad_acc = trades_only['broad_accuracy'].mean() * 100
        avg_strict_acc = trades_only['top_n_precision'].mean() * 100
        trade_count = len(trades_only)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategy Return", f"{final_ret:+.1f}%", delta=f"{final_ret - bench_ret:+.1f}%")
        m2.metric("Win Rate (Beat Median)", f"{avg_broad_acc:.1f}%", help="Did we pick better-than-average funds?")
        m3.metric(f"Top {top_n} Precision", f"{avg_strict_acc:.1f}%", help=f"Did we pick the exact Top {top_n} funds?")
        m4.metric("Trades Taken", f"{trade_count}")
        
        # Visualization
        tab1, tab2 = st.tabs(["📈 Performance & Accuracy", "📋 Detailed Logs"])
        
        with tab1:
            st.markdown("### Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['date'], y=res['cum_strat'], name='Smart ML', line=dict(color='#00C853', width=3)))
            fig.add_trace(go.Scatter(x=res['date'], y=res['cum_bench'], name='Benchmark', line=dict(color='gray', dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Accuracy Over Time")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=trades_only['date'], y=trades_only['broad_accuracy'], name='Win Rate (Broad)', marker_color='#64B5F6'))
            fig2.add_trace(go.Bar(x=trades_only['date'], y=trades_only['top_n_precision'], name=f'Top {top_n} Precision', marker_color='#1976D2'))
            fig2.update_layout(title="Accuracy Comparison per Period", barmode='group', yaxis_title="Accuracy (0-1)")
            st.plotly_chart(fig2, use_container_width=True)
             

        with tab2:
            st.dataframe(trades_only[['date', 'strategy_ret', 'broad_accuracy', 'top_n_precision', 'trades_taken']].style.format({
                'strategy_ret': '{:.2%}', 
                'broad_accuracy': '{:.0%}', 
                'top_n_precision': '{:.0%}'
            }))

if __name__ == "__main__":
    main()
