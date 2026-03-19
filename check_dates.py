"""
Fund Analysis Dashboard — Rolling Returns Focus
================================================
Simple dashboard showing:
  - Rolling 1Y / 3Y / 5Y: Mean, Median, Min returns per fund
  - Fund vs Nifty 100 rolling return comparison chart
  - Category selector, fund selector
Run: streamlit run backtest_strategy.py
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
 
warnings.filterwarnings('ignore')
 
# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Fund Rolling Returns",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 100%; }
    h1 { color: #1E3A5F; font-weight: 700; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 16px 18px;
        border-left: 4px solid #2196F3;
        margin: 6px 0;
    }
    .metric-card .label { font-size: 0.78rem; color: #666; margin-bottom: 4px; }
    .metric-card .value { font-size: 1.4rem; font-weight: 700; color: #1E3A5F; }
    .metric-card .sub   { font-size: 0.75rem; color: #888; margin-top: 2px; }
    .warn-card {
        background: #fff8e1;
        border-left: 4px solid #ff9800;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #7c5800;
    }
</style>
""", unsafe_allow_html=True)
 
# ============================================================================
# CONSTANTS
# ============================================================================
DATA_DIR      = "data"
MAX_DATA_DATE = pd.Timestamp('2025-12-05')
TRADING_DAYS  = 260          # trading days per year
 
W1  = 260                    # 1Y window
W3  = 780                    # 3Y window
W5  = 1300                   # 5Y window
 
FILE_MAPPING = {
    "Large Cap":       "largecap_merged.xlsx",
    "Mid Cap":         "midcap.xlsx",
    "Small Cap":       "smallcap.xlsx",
    "Large & Mid Cap": "large_and_midcap_fund.xlsx",
    "Multi Cap":       "MULTICAP.xlsx",
    "Flexi Cap":       "flexicap.xlsx",
    "International":   "international_merged.xlsx"
}
 
# ============================================================================
# DATA LOADING
# ============================================================================
def is_regular_growth_fund(name):
    n = str(name).lower()
    exclude = ['idcw','dividend','div ','div.','div)','direct','dir ','dir)',
               'bonus','institutional','segregated','payout','reinvestment',
               'monthly','quarterly','annual','option','opt']
    return not any(kw in n for kw in exclude)
 
def clean_weekday_data(df):
    if df is None or df.empty: return df
    df = df[df.index <= MAX_DATA_DATE]
    df = df[df.index.dayofweek < 5]
    if len(df) > 0:
        bdays = pd.bdate_range(start=df.index.min(),
                               end=min(df.index.max(), MAX_DATA_DATE))
        df = df.reindex(bdays).ffill(limit=5)
    return df
 
@st.cache_data
def load_fund_data(category_key):
    filename = FILE_MAPPING.get(category_key)
    if not filename: return None, None
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        st.error(f"❌ File not found: {os.path.abspath(path)}")
        return None, None
    try:
        df = pd.read_excel(path, header=None)
        fund_names = df.iloc[2, 1:].tolist()
        data_df = df.iloc[4:].copy()
        # strip trailing non-date rows
        while len(data_df) > 0:
            if pd.isna(pd.to_datetime(data_df.iloc[-1, 0], errors='coerce')):
                data_df = data_df.iloc[:-1]
            else:
                break
        dates = pd.to_datetime(data_df.iloc[:, 0], errors='coerce')
        mask = dates.notna()
        data_df = data_df[mask.values]
        dates   = dates[mask]
        nav = pd.DataFrame(index=dates)
        scheme_map = {}
        for i, name in enumerate(fund_names):
            if pd.notna(name) and str(name).strip():
                if not is_regular_growth_fund(name): continue
                code = str(abs(hash(name)) % (10**8))
                scheme_map[code] = name
                nav[code] = pd.to_numeric(data_df.iloc[:, i+1], errors='coerce').values
        nav = nav.sort_index()
        nav = nav[~nav.index.duplicated(keep='last')]
        return clean_weekday_data(nav), scheme_map
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None
 
@st.cache_data
def load_nifty_data():
    path = os.path.join(DATA_DIR, "nifty100_data.csv")
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['nav']  = pd.to_numeric(df['nav'], errors='coerce')
        df = df.dropna(subset=['date','nav']).set_index('date').sort_index()
        return clean_weekday_data(df).squeeze()
    except:
        return None
 
# ============================================================================
# ROLLING RETURN CALCULATION
# ============================================================================
def rolling_returns(series, window):
    """Returns Series of annualised rolling CAGR using 260 trading-day basis."""
    exp = TRADING_DAYS / window
    if len(series) <= window:
        return pd.Series(dtype=float)
    rolled = ((series / series.shift(window)) ** exp - 1).dropna()
    return rolled
 
def rolling_stats(series, window):
    """Returns (mean, median, min) of rolling CAGR. All as % values."""
    r = rolling_returns(series, window)
    if len(r) == 0:
        return np.nan, np.nan, np.nan
    return r.mean() * 100, r.median() * 100, r.min() * 100
 
# ============================================================================
# BUILD METRICS TABLE
# ============================================================================
@st.cache_data
def build_metrics_table(nav_pkl, scheme_map):
    """Returns DataFrame with rolling stats for all funds."""
    rows = []
    for col in nav_pkl.columns:
        s = nav_pkl[col].dropna()
        if len(s) < W1 + 10:
            continue
        name  = scheme_map.get(col, col)
        yrs   = (s.index.max() - s.index.min()).days / 365.25
        since = s.index.min().strftime('%b %Y')
 
        row = {
            'Fund Name': name,
            'Since':     since,
            'Years':     round(yrs, 1),
        }
 
        # 1Y
        m, med, mn = rolling_stats(s, W1)
        row['1Y Mean %']   = m
        row['1Y Median %'] = med
        row['1Y Min %']    = mn
 
        # 3Y
        if len(s) >= W3 + 10:
            m, med, mn = rolling_stats(s, W3)
            row['3Y Mean %']   = m
            row['3Y Median %'] = med
            row['3Y Min %']    = mn
        else:
            row['3Y Mean %'] = row['3Y Median %'] = row['3Y Min %'] = np.nan
 
        # 5Y
        if len(s) >= W5 + 10:
            m, med, mn = rolling_stats(s, W5)
            row['5Y Mean %']   = m
            row['5Y Median %'] = med
            row['5Y Min %']    = mn
        else:
            row['5Y Mean %'] = row['5Y Median %'] = row['5Y Min %'] = np.nan
 
        rows.append(row)
 
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('3Y Mean %', ascending=False, na_position='last')
    return df
 
# ============================================================================
# COMPARISON CHART: fund rolling returns vs Nifty 100
# ============================================================================
def comparison_chart(fund_series, nifty_series, fund_name, window, period_label):
    """Plot rolling return time series for fund vs Nifty 100."""
    f_roll = rolling_returns(fund_series, window)
    if nifty_series is not None:
        b_roll = rolling_returns(nifty_series, window)
        common = f_roll.index.intersection(b_roll.index)
    else:
        common = f_roll.index
 
    if len(common) == 0:
        return None
 
    fig = go.Figure()
 
    # Fund line
    fig.add_trace(go.Scatter(
        x=f_roll.loc[common].index,
        y=(f_roll.loc[common] * 100).round(2),
        name=fund_name[:40],
        line=dict(color='#2196F3', width=2),
        hovertemplate='%{x|%b %Y}: %{y:.2f}%<extra></extra>'
    ))
 
    # Benchmark line
    if nifty_series is not None and len(common) > 0:
        fig.add_trace(go.Scatter(
            x=common,
            y=(b_roll.loc[common] * 100).round(2),
            name='Nifty 100',
            line=dict(color='#FF7043', width=2, dash='dot'),
            hovertemplate='Nifty %{x|%b %Y}: %{y:.2f}%<extra></extra>'
        ))
 
    # Zero line
    fig.add_hline(y=0, line=dict(color='grey', width=1, dash='dash'))
 
    fig.update_layout(
        title=f'{period_label} Rolling Return (Annualised CAGR) — {fund_name[:45]}',
        yaxis_title='Rolling Return %',
        xaxis_title='',
        hovermode='x unified',
        height=380,
        legend=dict(orientation='h', y=1.06, x=0.5, xanchor='center'),
        margin=dict(t=60, b=40, l=50, r=20)
    )
    return fig
 
# ============================================================================
# SUMMARY METRIC CARDS (for selected fund)
# ============================================================================
def render_fund_cards(fund_row, nifty_stats):
    """Show mean/median/min cards for 1Y, 3Y, 5Y side by side."""
 
    def card(label, val, sub='', color='#2196F3'):
        val_str = f"{val:.2f}%" if not np.isnan(val) else "N/A"
        color_str = color if not np.isnan(val) else '#aaa'
        return f"""<div class="metric-card" style="border-left-color:{color_str}">
            <div class="label">{label}</div>
            <div class="value" style="color:{color_str}">{val_str}</div>
            <div class="sub">{sub}</div>
        </div>"""
 
    def get(row, key):
        v = row.get(key, np.nan)
        return v if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan
 
    periods = [
        ('1Y', '#1976D2'),
        ('3Y', '#388E3C'),
        ('5Y', '#7B1FA2'),
    ]
    stat_keys = ['Mean %', 'Median %', 'Min %']
    stat_labels = ['Mean', 'Median', 'Worst (Min)']
 
    # Nifty reference row
    nifty_ref = {
        '1Y': nifty_stats.get('1Y', {}),
        '3Y': nifty_stats.get('3Y', {}),
        '5Y': nifty_stats.get('5Y', {}),
    }
 
    for period, color in periods:
        cols = st.columns([1, 1, 1, 1])
        with cols
