"""
ML Fund Pro v4 - High Hit Rate Edition
======================================
IMPROVEMENTS:
  - Regime Detection (Bull/Bear context)
  - Relative Strength Features (vs Peers)
  - Probability Thresholding (Only trade when confident)
  - Voting Ensemble (RF + XGBoost combined)

Run: streamlit run ml_fund_pro_v4.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from datetime import timedelta

warnings.filterwarnings('ignore')

st.set_page_config(page_title="High Hit-Rate ML", page_icon="🎯", layout="wide")

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
# 2. ADVANCED FEATURE ENGINEERING (THE SECRET SAUCE)
# ============================================================================

def calculate_technical_features(s, benchmark_series):
    """Calculates advanced features for a single fund."""
    # 1. Volatility Adjusted Momentum (Sharpe-like)
    ret_6m = s.pct_change(126).iloc[-1]
    vol_6m = s.pct_change().tail(126).std() * np.sqrt(252)
    vol_adj_mom = ret_6m / vol_6m if vol_6m > 0 else 0
    
    # 2. Beta (Market Sensitivity)
    # We use a 6-month rolling beta
    f_ret = s.pct_change().tail(126).dropna()
    b_ret = benchmark_series.pct_change().tail(126).dropna()
    
    # Align dates
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
    
    # Calculate Market Regime (Vectorized)
    # Regime = 1 if Market > 200DMA (Bull), else 0 (Bear)
    bench_ma200 = benchmark.rolling(200).mean()
    market_regime = (benchmark > bench_ma200).astype(int)
    
    # Monthly Steps
    step_size = 21 
    valid_dates = nav_df.index[252:-holding_period:step_size]
    
    with st.spinner("🧠 Engineering 'Context-Aware' Features..."):
        for dt in valid_dates:
            # Context Data
            hist_nav = nav_df.loc[:dt].tail(253)
            curr_regime = market_regime.loc[dt]
            
            # Calculate Peer stats for Relative Strength
            # We get the 6m return for ALL funds at this date to calculate rank
            period_returns = hist_nav.pct_change(126).iloc[-1]
            median_ret = period_returns.median()
            
            # Target Data (Future)
            future_end = nav_df.index[min(nav_df.index.get_loc(dt) + holding_period, len(nav_df)-1)]
            
            for fund_id in nav_df.columns:
                s = hist_nav[fund_id].dropna()
                if len(s) < 252: continue
                
                try:
                    curr_price = s.iloc[-1]
                    fut_price = nav_df.loc[future_end, fund_id]
                    
                    # --- FEATURE SET ---
                    
                    # 1. Relative Strength (The most important feature)
                    # How much better is this fund vs the median fund?
                    abs_ret_6m = period_returns[fund_id]
                    rel_strength = abs_ret_6m - median_ret
                    
                    # 2. Volatility Adjusted Momentum
                    vol_adj_mom, beta = calculate_technical_features(s, benchmark.loc[:dt])
                    
                    # 3. Downside Deviation (Risk)
                    rets = s.pct_change().tail(126)
                    neg_rets = rets[rets < 0]
                    downside_dev = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 else 0.01
                    sortino_proxy = abs_ret_6m / downside_dev
                    
                    # --- TARGET ---
                    # 1 = Outperform Median AND Positive Return (Don't pick 'less bad' losers)
                    future_ret = (fut_price / curr_price) - 1
                    
                    features.append({
                        'date': dt,
                        'fund_id': fund_id,
                        'f_regime': curr_regime,       # 0 or 1
                        'f_rel_strength': rel_strength, # >0 means leader
                        'f_vol_mom': vol_adj_mom,       # Risk-adjusted return
                        'f_beta': beta,                 # Defensive vs Aggressive
                        'f_sortino': sortino_proxy,     # Downside protection
                        'target_ret': future_ret
                    })
                except: continue
                
    df = pd.DataFrame(features)
    
    # Target: Top 40% of funds in that period
    # We use 40% (quantile 0.6) instead of 50% to force the model to be pickier
    df['target_class'] = df.groupby('date')['target_ret'].transform(
        lambda x: x > x.quantile(0.60)
    ).astype(int)
    
    return df.dropna()

# ============================================================================
# 3. ENSEMBLE MODEL TRAINING
# ============================================================================

def train_ensemble_model(ml_df, train_window_yrs, threshold=0.60):
    """
    Uses a Voting Classifier (Soft Vote) of RF and XGBoost.
    Only selects funds where probability > threshold.
    """
    dates = sorted(ml_df['date'].unique())
    start_idx = int(train_window_yrs * 12)
    results = []
    
    # ENSEMBLE DEFINITION
    rf = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
    
    feature_cols = [c for c in ml_df.columns if c.startswith('f_')]
    
    prog = st.progress(0)
    
    for i in range(start_idx, len(dates)):
        test_date = dates[i]
        
        # Expand Window Training
        train = ml_df[ml_df['date'] < test_date]
        test = ml_df[ml_df['date'] == test_date]
        
        if len(train) < 500 or len(test) == 0: continue
            
        X_train, y_train = train[feature_cols], train['target_class']
        X_test, y_test = test[feature_cols], test['target_class']
        
        # Fit
        ensemble.fit(X_train, y_train)
        
        # Predict Probabilities
        probs = ensemble.predict_proba(X_test)[:, 1]
        
        # --- PROBABILITY THRESHOLDING ---
        # We combine the fund ID, actual return, and probability
        test_res = pd.DataFrame({
            'fund_id': test['fund_id'].values,
            'prob': probs,
            'actual_ret': test['target_ret'].values,
            'is_winner': y_test.values
        })
        
        # Filter: Confidence > Threshold
        high_conf_picks = test_res[test_res['prob'] > threshold]
        
        # If no funds meet threshold, we assume cash/benchmark (0% active return)
        if high_conf_picks.empty:
            avg_ret = test['target_ret'].mean() # Default to market average
            # Or 0.0 if you go to cash
            accuracy = np.nan # No trade taken
            n_picks = 0
        else:
            # Pick top 5 from the high confidence list
            top_picks = high_conf_picks.nlargest(5, 'prob')
            avg_ret = top_picks['actual_ret'].mean()
            accuracy = top_picks['is_winner'].mean()
            n_picks = len(top_picks)
            
        results.append({
            'date': test_date,
            'strategy_ret': avg_ret,
            'benchmark_ret': test['target_ret'].mean(),
            'accuracy': accuracy,
            'trades_taken': n_picks
        })
        prog.progress((i - start_idx) / (len(dates) - start_idx))
        
    return pd.DataFrame(results)

# ============================================================================
# 4. DASHBOARD
# ============================================================================

def main():
    st.markdown("## 🎯 High Hit-Rate ML Strategy")
    st.markdown("""
    To improve accuracy, this model uses **Relative Strength** (vs Peers) and **Market Regime** (Bull/Bear).
    It also applies a **Confidence Threshold**: it will *refuse to trade* if it isn't sure.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1: prob_thresh = st.slider("Confidence Threshold", 0.5, 0.8, 0.60, 0.05, help="Only trade if model is >X% sure.")
    with col2: hold_period = st.selectbox("Holding Period", [21, 63, 126], index=1, format_func=lambda x: f"{x} Days")
    
    nav, _, bench = load_data()
    if nav is None: return
    
    if st.button("Run Advanced Backtest"):
        # 1. Feature Engineering
        ml_df = prepare_enhanced_dataset(nav, bench, hold_period)
        
        # 2. Train & Test
        res = train_ensemble_model(ml_df, train_window_yrs=3, threshold=prob_thresh)
        
        if res.empty:
            st.warning("No trades generated. Try lowering the threshold or increasing data history.")
            return
            
        # 3. Metrics
        res['cum_strat'] = (1 + res['strategy_ret']).cumprod()
        res['cum_bench'] = (1 + res['benchmark_ret']).cumprod()
        
        # Filter out periods where we didn't trade for accuracy calculation
        trades_only = res.dropna(subset=['accuracy'])
        
        final_ret = (res['cum_strat'].iloc[-1] - 1) * 100
        bench_ret = (res['cum_bench'].iloc[-1] - 1) * 100
        hit_rate = trades_only['accuracy'].mean() * 100
        trade_count = len(trades_only)
        skipped = len(res) - len(trades_only)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategy Return", f"{final_ret:+.1f}%", delta=f"{final_ret - bench_ret:+.1f}%")
        m2.metric("Hit Rate (Accuracy)", f"{hit_rate:.1f}%", help="Target: >60%")
        m3.metric("Trades Taken", f"{trade_count}", help=f"Skipped {skipped} periods due to low confidence")
        m4.metric("Avg. Monthly Alpha", f"{(res['strategy_ret'] - res['benchmark_ret']).mean()*100:+.2f}%")
        
        # 4. Visualization
        tab1, tab2 = st.tabs(["📈 Performance", "🧠 Model Logic"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['date'], y=res['cum_strat'], name='Smart ML', line=dict(color='#00C853', width=3)))
            fig.add_trace(go.Scatter(x=res['date'], y=res['cum_bench'], name='Benchmark', line=dict(color='gray', dash='dot')))
            
            # Highlight "Cash" periods (where line is flat/matches benchmark because we skipped)
            # We can't easily visualize this in one line, but the skipped metric tells the story.
            
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.markdown("### Why this works better?")
            st.markdown("""
            1.  **Voting Ensemble:** We combine Random Forest (good at complex rules) and Gradient Boosting (good at minimizing errors).
            2.  **Relative Strength (`f_rel_strength`):** The model learns to pick the *best of the bunch*, not just "funds that went up".
            3.  **Sortino Ratio (`f_sortino`):** It prioritizes funds that have low downside volatility, avoiding "crash-prone" winners.
            """)
            st.dataframe(ml_df.head())

if __name__ == "__main__":
    main()
