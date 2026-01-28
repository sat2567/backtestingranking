import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
from datetime import datetime

# Filter warnings for cleaner logs
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="Advanced Fund Analysis",
    page_icon="📈",
    layout="wide"
)

# Financial Constants
RISK_FREE_RATE = 0.06
TRADING_DAYS_YEAR = 252
DAILY_RF = (1 + RISK_FREE_RATE) ** (1/TRADING_DAYS_YEAR) - 1
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
# 2. CORE ANALYTICAL ENGINE
# ============================================================================

class FundMetrics:
    """Namespace for risk and return calculations."""
    
    @staticmethod
    def sharpe_ratio(returns):
        if len(returns) < 10 or returns.std() == 0: return np.nan
        return ((returns - DAILY_RF).mean() / returns.std()) * np.sqrt(TRADING_DAYS_YEAR)

    @staticmethod
    def martin_ratio(series):
        """Calculates (CAGR - RF) / Ulcer Index."""
        if len(series) < 30: return np.nan
        cum_max = series.expanding().max()
        drawdowns = (series / cum_max) - 1
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
        
        years = (series.index[-1] - series.index[0]).days / 365.25
        cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
        return (cagr - RISK_FREE_RATE) / ulcer_index if ulcer_index != 0 else np.nan

    @staticmethod
    def max_drawdown(series):
        if series.empty: return np.nan
        comp_ret = (1 + series.pct_change().fillna(0)).cumprod()
        peak = comp_ret.expanding().max()
        return ((comp_ret / peak) - 1).min()

    @staticmethod
    def momentum_score(series, weights=(0.33, 0.33, 0.33), risk_adj=False):
        """Weighted momentum: 3m, 6m, and 12m returns."""
        if len(series) < 252: return np.nan
        curr = series.iloc[-1]
        
        # Helper to find price at specific intervals
        def get_ret(days):
            target = series.index[-1] - pd.Timedelta(days=days)
            past = series[series.index <= target]
            return (curr / past.iloc[-1]) - 1 if not past.empty else 0

        r3, r6, r12 = get_ret(91), get_ret(182), get_ret(365)
        raw = (r3 * weights[0]) + (r6 * weights[1]) + (r12 * weights[2])
        
        if risk_adj:
            vol = series.pct_change().std() * np.sqrt(TRADING_DAYS_YEAR)
            return raw / vol if vol > 0 else np.nan
        return raw

# ============================================================================
# 3. DATA LOADERS
# ============================================================================

@st.cache_data
def load_and_clean_data(category_key):
    """Loads fund data and ensures business-day frequency."""
    filename = FILE_MAPPING.get(category_key)
    path = os.path.join(DATA_DIR, filename) if filename else None
    
    if not path or not os.path.exists(path):
        return None, None

    try:
        df_raw = pd.read_excel(path, header=None)
        names = df_raw.iloc[2, 1:].tolist()
        data = df_raw.iloc[4:, :].copy()
        
        # Date and NAV cleaning
        dates = pd.to_datetime(data.iloc[:, 0], errors='coerce')
        nav_df = pd.DataFrame(index=dates)
        name_map = {}

        for i, name in enumerate(names):
            if pd.isna(name): continue
            code = str(abs(hash(name)) % (10**8))
            name_map[code] = name
            nav_df[code] = pd.to_numeric(data.iloc[:, i+1], errors='coerce')

        nav_df = nav_df[~nav_df.index.duplicated(keep='last')].sort_index()
        nav_df = nav_df[nav_df.index <= MAX_DATA_DATE]
        
        # Business day reindexing
        all_days = pd.date_range(nav_df.index.min(), nav_df.index.max(), freq='B')
        nav_df = nav_df.reindex(all_days).ffill(limit=5)
        
        return nav_df, name_map
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None, None

# ============================================================================
# 4. BACKTESTING ENGINE
# ============================================================================

def execute_backtest(nav, strategy_type, top_n, hold_days):
    """
    Simulates portfolio rebalancing based on selected strategy.
    """
    start_idx = 260 # Wait for 1 year of data
    rebal_points = range(start_idx, len(nav) - hold_days, hold_days)
    
    results = []
    equity_path = [100.0]
    dates = [nav.index[start_idx]]

    for i in rebal_points:
        curr_date = nav.index[i]
        lookback = nav.iloc[i-252 : i] # 1 year window
        
        # 1. Score Funds
        scores = {}
        for fund in nav.columns:
            s = lookback[fund].dropna()
            if len(s) < 126: continue
            
            if strategy_type == 'Momentum':
                scores[fund] = FundMetrics.momentum_score(s, risk_adj=True)
            elif strategy_type == 'Sharpe':
                scores[fund] = FundMetrics.sharpe_ratio(s.pct_change())
            elif strategy_type == 'Martin':
                scores[fund] = FundMetrics.martin_ratio(s)
        
        # 2. Select Top N
        selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
        
        # 3. Calculate Period Return
        if selected:
            ret = (nav[selected].iloc[i + hold_days] / nav[selected].iloc[i+1]) - 1
            period_ret = ret.mean()
        else:
            period_ret = 0
            
        new_val = equity_path[-1] * (1 + period_ret)
        equity_path.append(new_val)
        dates.append(nav.index[i + hold_days])
        
    return pd.DataFrame({'Date': dates, 'Value': equity_path})

# ============================================================================
# 5. UI COMPONENTS
# ============================================================================

def render_sidebar():
    with st.sidebar:
        st.title("Settings")
        cat = st.selectbox("Fund Category", list(FILE_MAPPING.keys()))
        hold = st.slider("Holding Period (Days)", 20, 252, 126)
        top_n = st.number_input("Selection Size (Top N)", 1, 10, 3)
        return cat, hold, top_n

def main():
    cat, hold, top_n = render_sidebar()
    nav, names = load_and_clean_data(cat)
    
    if nav is None:
        st.warning("Please ensure Excel files are in the 'data/' folder.")
        return

    tab1, tab2 = st.tabs(["📊 Explorer", "🚀 Backtester"])

    with tab1:
        st.subheader(f"Raw Fund Performance: {cat}")
        # Show recent returns table
        recent_ret = (nav.iloc[-1] / nav.iloc[-252] - 1).sort_values(ascending=False)
        display_df = pd.DataFrame({
            "Fund": [names.get(c) for c in recent_ret.index],
            "1Y Return": recent_ret.values * 100
        })
        st.dataframe(display_df.style.format({"1Y Return": "{:.2f}%"}), height=400)

    with tab2:
        st.subheader("Strategy Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with st.spinner("Calculating strategies..."):
            res_mom = execute_backtest(nav, 'Momentum', top_n, hold)
            res_sha = execute_backtest(nav, 'Sharpe', top_n, hold)
            res_mar = execute_backtest(nav, 'Martin', top_n, hold)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_mom['Date'], y=res_mom['Value'], name="Momentum Strategy"))
        fig.add_trace(go.Scatter(x=res_sha['Date'], y=res_sha['Value'], name="Sharpe Strategy"))
        fig.add_trace(go.Scatter(x=res_mar['Date'], y=res_mar['Value'], name="Martin Strategy"))
        
        fig.update_layout(title="Growth of 100", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

        # Performance Summary Table
        summary = []
        for name, df in [("Momentum", res_mom), ("Sharpe", res_sha), ("Martin", res_mar)]:
            total_ret = (df['Value'].iloc[-1] / 100) - 1
            summary.append({"Strategy": name, "Total Return %": total_ret * 100})
        
        st.table(pd.DataFrame(summary).set_index("Strategy"))

if __name__ == "__main__":
    main()
