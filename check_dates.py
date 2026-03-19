"""
Fund Analysis Dashboard — Rolling Returns Focus
================================================
Simple dashboard showing:
  - Rolling 1Y / 3Y / 5Y: Mean, Median, Min returns per fund
  - Fund vs Nifty 100 rolling return comparison chart
  - Category selector, fund selector
Run: streamlit run check_dates.py
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
    "Small Cap":
