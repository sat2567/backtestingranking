import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="DeepQuant: Robust Batch Analyzer",
    page_icon="🛡️",
    layout="wide"
)

# Constants
DATA_DIR = "data"
HISTORY_FILE = "batch_history.csv"

FILE_MAPPING = {
    "Large Cap": "largecap_merged.xlsx",
    "Mid Cap": "midcap.xlsx",
    "Small Cap": "smallcap.xlsx",
    "Large & Mid": "large_and_midcap_fund.xlsx",
    "Multi Cap": "MULTICAP.xlsx",
    "International": "international_merged.xlsx"
}

# ============================================================================
# 2. ROBUST MATH FUNCTIONS (THE FIX)
# ============================================================================

def safe_std(series):
    """Calculates std dev, returns None if 0 to avoid division by zero."""
    val = series.std()
    if val < 1e-9 or pd.isna(val): return None
    return val

def calculate_sharpe_ratio(returns):
    if len(returns) < 10: return np.nan
    std = safe_std(returns)
    if std is None: return np.nan # Fix for Infinity
    
    # Simple Annualized Sharpe (Assuming 0 risk free for ranking relative strength)
    return (returns.mean() / std) * np.sqrt(252)

def calculate_flexible_momentum(series, w_3m=0.33, w_6m=0.33, w_12m=0.33, risk_adjust=False):
    if len(series) < 200: return np.nan
    try:
        p_now = series.iloc[-1]
        # Safe lookback function
        def get_ret(days):
            idx = series.index.get_indexer([series.index[-1] - pd.Timedelta(days=days)], method='nearest')[0]
            if idx < 0 or idx >= len(series): return 0
            p_past = series.iloc[idx]
            return (p_now / p_past) - 1 if p_past > 0 else 0

        r3 = get_ret(63)  # ~3 months
        r6 = get_ret(126) # ~6 months
        r12 = get_ret(252)# ~12 months
        
        score = (r3 * w_3m) + (r6 * w_6m) + (r12 * w_12m)
        
        if risk_adjust:
            vol = safe_std(series.pct_change().dropna())
            if vol is None: return np.nan # Fix for Infinity
            score = score / (vol * np.sqrt(252))
            
        return score
    except:
        return np.nan

def calculate_residual_momentum(series, benchmark_series):
    """
    Robust Residual Momentum.
    Returns np.nan if calculation is mathematically unstable (Infinity/Zero).
    """
    # 1. Align Data
    s_ret = series.pct_change().dropna()
    b_ret = benchmark_series.pct_change().dropna()
    
    aligned = pd.concat([s_ret, b_ret], axis=1, join='inner').dropna()
    
    # 2. Sanity Checks
    if len(aligned) < 60: return np.nan
    
    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]
    
    # Check for flat lines (Zero Variance)
    if safe_std(x) is None or safe_std(y) is None: return np.nan
    
    # 3. Regression
    try:
        slope, intercept, _, _, _ = linregress(x, y)
        if not np.isfinite(slope) or not np.isfinite(intercept): return np.nan
        
        # 4. Residuals
        expected = (x * slope) + intercept
        residuals = y - expected
        
        # 5. Score (Info Ratio of Residuals)
        res_mean = residuals.mean()
        res_std = safe_std(residuals)
        
        if res_std is None: return np.nan # Fix for Zero/Inf
        
        score = res_mean / res_std
        
        # Final check
        if not np.isfinite(score): return np.nan
        if abs(score) < 1e-9: return np.nan # Treat nearly zero as noise
        
        return score
        
    except:
        return np.nan

def get_market_regime(benchmark_series, current_date):
    """Returns 'bull' or 'bear' based on 200 DMA."""
    try:
        # Get data up to today
        subset = benchmark_series[benchmark_series.index <= current_date]
        if len(subset) < 200: return 'neutral'
        
        curr_price = subset.iloc[-1]
        dma_200 = subset.iloc[-200:].mean()
        
        return 'bull' if curr_price > dma_200 else 'bear'
    except:
        return 'neutral'

# ============================================================================
# 3. DATA LOADING
# ============================================================================

def universal_date_parser(series):
    try:
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().sum() > len(series) * 0.5:
            return pd.to_datetime(numeric, unit='D', origin='1899-12-30')
    except: pass
    return pd.to_datetime(series, errors='coerce', dayfirst=True)

@st.cache_data
def load_data_batch(category_key):
    filename = FILE_MAPPING.get(category_key)
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path): return None, None

    try: df = pd.read_csv(path, header=None, low_memory=False)
    except: 
        try: df = pd.read_excel(path, header=None)
        except: return None, None

    header_idx = -1
    for i in range(min(50, len(df))):
        s = " ".join(df.iloc[i].astype(str).values).lower()
        if "nav date" in s or "adjusted nav" in s:
            header_idx = i
            break
    if header_idx == -1: return None, None

    names = df.iloc[header_idx-1, 1:].tolist()
    data = df.iloc[header_idx+1:, :].copy()
    
    dates = universal_date_parser(data.iloc[:, 0])
    nav = pd.DataFrame(index=dates)
    
    for i, name in enumerate(names):
        if pd.isna(name) or "Unnamed" in str(name): continue
        vals = pd.to_numeric(data.iloc[:, i+1], errors='coerce')
        cid = str(abs(hash(str(name))) % (10**8))
        nav[cid] = vals.values

    nav = nav[nav.index.notna()]
    nav.index = nav.index.normalize()
    nav = nav.sort_index()
    nav = nav[~nav.index.duplicated()]
    nav = nav.ffill(limit=5)
    nav = nav.dropna(how='all')
    
    # Returns (keep NaNs for ragged data handling)
    returns = np.log(nav / nav.shift(1))
    
    return nav, returns

@st.cache_data
def load_benchmark():
    path = os.path.join(DATA_DIR, "nifty100_data.csv")
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        if 'date' not in df.columns: df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        if 'close' in df.columns: return df['close']
        elif 'adj close' in df.columns: return df['adj close']
        else: return df.iloc[:, -1]
    except: return None

# ============================================================================
# 4. STRATEGY ENGINE
# ============================================================================

def run_single_backtest(nav, returns, benchmark, hold_days, top_n, target_n, strategy_name):
    # Strategy Config
    TRAIN_WINDOW = 365 * 2
    
    if len(returns) < 300: return None
    start_idx = 252 
    
    rebal_idx = range(start_idx, len(returns) - hold_days, hold_days)
    if not rebal_idx: return None

    hits = []
    capital = 100.0
    
    for i in rebal_idx:
        curr = returns.index[i]
        nxt = returns.index[i+hold_days]
        
        # Get historical slice
        t_start = curr - pd.Timedelta(days=TRAIN_WINDOW)
        if t_start < returns.index.min(): t_start = returns.index.min()
        
        hist_returns = returns.loc[t_start:curr]
        hist_nav = nav.loc[t_start:curr]
        
        scores = {}
        
        # --- STRATEGY LOGIC ---
        
        if strategy_name == 'Residual Momentum':
            if benchmark is not None:
                bench_slice = benchmark.loc[t_start:curr]
                for col in returns.columns:
                    s = hist_nav[col].dropna() # Use NAV for momentum calc
                    if len(s) > 126:
                        val = calculate_residual_momentum(s, bench_slice)
                        if pd.notna(val) and np.isfinite(val): scores[col] = val
                        
        elif strategy_name == 'Regime Switch':
            if benchmark is not None:
                regime = get_market_regime(benchmark, curr)
                for col in returns.columns:
                    s = hist_nav[col].dropna()
                    if len(s) > 126:
                        if regime == 'bull':
                            # Bull: Risk-Adjusted Momentum
                            val = calculate_flexible_momentum(s, risk_adjust=True)
                        else:
                            # Bear: Sharpe Ratio (Safety)
                            # Use returns for Sharpe
                            s_ret = hist_returns[col].dropna()
                            val = calculate_sharpe_ratio(s_ret)
                            
                        if pd.notna(val) and np.isfinite(val): scores[col] = val

        elif strategy_name == 'Stable Momentum':
             for col in returns.columns:
                 s = hist_nav[col].dropna()
                 if len(s) > 252:
                     # 1. Calculate Momentum
                     mom = calculate_flexible_momentum(s, risk_adjust=False)
                     if pd.notna(mom) and np.isfinite(mom): scores[col] = mom
             
             # Filter Top 2N by Momentum -> Select Top N by Low Drawdown
             if scores:
                 top_pool = sorted(scores, key=scores.get, reverse=True)[:top_n*2]
                 dd_scores = {}
                 for col in top_pool:
                     s = hist_nav[col].dropna()
                     dd = calculate_max_dd(s) # Imported from previous steps logic
                     # Simple Max DD calc if missing
                     if pd.notna(dd): dd_scores[col] = dd
                     else: 
                         comp = (1 + s.pct_change().fillna(0)).cumprod()
                         dd = ((comp / comp.expanding().max()) - 1).min()
                         dd_scores[col] = dd
                 
                 # Re-rank by Drawdown (Higher (closer to 0) is better)
                 scores = dd_scores

        elif strategy_name == 'Consistent Alpha':
             # Use Sharpe as proxy for Alpha consistency in this lightweight batch version
             for col in returns.columns:
                 s_ret = hist_returns[col].dropna()
                 if len(s_ret) > 126:
                     val = calculate_sharpe_ratio(s_ret)
                     if pd.notna(val) and np.isfinite(val): scores[col] = val

        # --- SELECTION ---
        selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
        
        # --- VALIDATION ---
        try:
            p0 = nav.loc[curr]
            p1 = nav.loc[nxt]
            period_rets = (p1 / p0) - 1
            period_rets = period_rets.dropna()
            
            if period_rets.empty: continue
            
            winners = period_rets.nlargest(target_n).index.tolist()
            n_hits = len([f for f in selected if f in winners])
            hit_rate = n_hits / top_n if top_n > 0 else 0
            hits.append(hit_rate)
            
            valid_sel = [f for f in selected if f in period_rets.index]
            r = period_rets[valid_sel].mean() if valid_sel else 0.0
            capital *= (1 + r)
        except:
            pass
        
    if not hits: return None
    
    avg_hit = np.mean(hits)
    total_ret = (capital / 100) - 1
    
    start_d = returns.index[rebal_idx[0]]
    end_d = returns.index[rebal_idx[-1] + hold_days]
    years = (end_d - start_d).days / 365.25
    cagr = (capital / 100) ** (1/years) - 1 if years > 0 else 0
    
    b_cagr = 0.0
    if benchmark is not None:
        try:
            b0 = benchmark.asof(start_d)
            b1 = benchmark.asof(end_d)
            b_ret = (b1/b0) - 1
            b_cagr = (1 + b_ret) ** (1/years) - 1
        except: pass
        
    return {
        "Avg Hit Rate": avg_hit,
        "CAGR": cagr,
        "Alpha": cagr - b_cagr,
        "Total Return": total_ret,
        "Trades": len(hits)
    }

# ============================================================================
# 5. UI & EXECUTION
# ============================================================================

def main():
    st.title("🛡️ DeepQuant: Robust Batch Strategy Lab")
    
    with st.sidebar:
        cat = st.selectbox("Category", list(FILE_MAPPING.keys()))
        st.divider()
        
        if os.path.exists(HISTORY_FILE):
            if st.button("🗑️ Clear History"):
                os.remove(HISTORY_FILE)
                st.rerun()

    # Load Data
    nav, returns = load_data_batch(cat)
    bench = load_benchmark()
    
    if nav is None: 
        st.error("Data Load Error.")
        return

    tab1, tab2 = st.tabs(["🚀 Run Batch", "📜 History"])

    with tab1:
        st.info(f"Loaded {len(nav.columns)} funds. Range: {nav.index.min().date()} - {nav.index.max().date()}")
        
        # CONFIGURATION
        scenarios = [
            {"top": 3, "target": 5, "label": "Top 3 in 5"},
            {"top": 5, "target": 10, "label": "Top 5 in 10"}
        ]
        
        # Include 504 days as requested
        periods = [
            {"days": 126, "label": "6 Months"},
            {"days": 252, "label": "1 Year"},
            {"days": 504, "label": "2 Years"} 
        ]
        
        strategies = ["Residual Momentum", "Regime Switch", "Stable Momentum", "Consistent Alpha"]
        
        if st.button("🚀 Run Analysis"):
            results = []
            progress = st.progress(0)
            status = st.empty()
            
            total_steps = len(scenarios) * len(periods) * len(strategies)
            step = 0
            
            run_id = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            for strat in strategies:
                for p in periods:
                    for s in scenarios:
                        step += 1
                        progress.progress(step / total_steps)
                        status.text(f"Testing {strat} | {p['label']} | {s['label']}")
                        
                        try:
                            res = run_single_backtest(
                                nav, returns, bench, 
                                p['days'], s['top'], s['target'], strat
                            )
                            
                            entry = {
                                "Run ID": run_id,
                                "Category": cat,
                                "Strategy": strat,
                                "Timeframe": p['label'],
                                "Target": s['label']
                            }
                            
                            if res:
                                entry.update({
                                    "Hit Rate": res['Avg Hit Rate'],
                                    "CAGR": res['CAGR'],
                                    "Alpha": res['Alpha'],
                                    "Trades": res['Trades']
                                })
                            else:
                                entry.update({"Hit Rate": 0, "CAGR": 0, "Alpha": 0, "Trades": 0})
                                
                            results.append(entry)
                            
                        except Exception as e:
                            # print(e) # Debug
                            pass
            
            progress.empty()
            status.empty()
            
            if results:
                new_df = pd.DataFrame(results)
                
                # Save
                if os.path.exists(HISTORY_FILE):
                    pd.concat([pd.read_csv(HISTORY_FILE), new_df]).to_csv(HISTORY_FILE, index=False)
                else:
                    new_df.to_csv(HISTORY_FILE, index=False)
                
                st.success("Batch Run Complete")
                st.dataframe(new_df.style.format({
                    'Hit Rate': '{:.1%}', 'CAGR': '{:.1%}', 'Alpha': '{:.1%}'
                }), use_container_width=True)
                
                st.rerun()

    with tab2:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            st.dataframe(df.style.format({
                'Hit Rate': '{:.1%}', 'CAGR': '{:.1%}', 'Alpha': '{:.1%}'
            }), use_container_width=True)
            
            # Heatmap
            if not df.empty:
                st.subheader("Hit Rate Heatmap")
                try:
                    pivot = df.pivot_table(index="Strategy", columns="Timeframe", values="Hit Rate", aggfunc="mean")
                    st.dataframe(pivot.style.format("{:.1%}"), use_container_width=True)
                except: pass
        else:
            st.info("No history found.")

if __name__ == "__main__":
    main()
