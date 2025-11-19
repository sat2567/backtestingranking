"""
Basic Dashboard - Detailed Fund Selection Only
Shows selected funds by Sharpe and Sortino ratios for a selected date.
This version reads data from local CSV files in the `data/` folder (no upload required).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

LOOKBACK = 252          # 1 year in trading days
HOLDING_PERIOD = 126    # 6 months in trading days (approx)
RISK_FREE_RATE = 0.05   # 5% annual
DEFAULT_TOP_N = 5

CATEGORY_MAP = {
    "Large Cap": "largecap",
    "Small Cap": "smallcap",
    "Mid Cap": "midcap",
    "Large & Mid Cap": "large_and_midcap",
    "Multi Cap": "multicap",
    "International": "international",
}

DATA_FOLDER = "data"
FUNDS_CSV_MAP = {
    "largecap": "largecap_funds.csv",
    "smallcap": "smallcap_funds.csv",
    "midcap": "midcap_funds.csv",
    "large_and_midcap": "large_and_midcap_funds.csv",
    "multicap": "multicap_funds.csv",
    "international": "international_funds.csv",
}
NIFTY_CSV = "nifty100_fileter_data.csv"

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Basic Fund Selection Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """Calculate annualized Sharpe ratio."""
    returns = returns.dropna()
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    excess_returns = returns - risk_free_rate / 252
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
    return sharpe


def calculate_sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """Calculate annualized Sortino ratio."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.nan
    sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    return sortino

# ============================================================================
# DATA LOADING FUNCTIONS (LOCAL FILEs)
# ============================================================================

@st.cache_data
def load_fund_data(category_key: str):
    """Load and prepare fu
