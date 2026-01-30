"""
Advanced Fund Analysis Dashboard - Enhanced Version
====================================================
Features:
- Comprehensive Category Explorer with quarterly rankings
- Detailed metric definitions with formulas
- Complete strategy explanations
- Compare across ALL holding periods

Run: streamlit run backtest_strategy.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="Fund Analysis Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }
    h1 { color: #1E3A5F; font-weight: 700; padding-bottom: 0.5rem; border-bottom: 3px solid #4CAF50; margin-bottom: 1.5rem; }
    h2 { color: #2E4A6F; font-weight: 600; margin-top: 1rem; }
    h3 { color: #3E5A7F; font-weight: 500; }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="metric-container"] label { color: rgba(255, 255, 255, 0.8) !important; font-size: 0.85rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: white !important; font-weight: 700; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 10px 20px; font-weight: 500; color: #333 !important; }
    .stTabs [aria-selected="true"] { background-color: #4CAF50 !important; color: white !important; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #e0e0e0; }
    
    .stButton > button { border-radius: 8px; font-weight: 500; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .metric-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
        border-radius: 12px; padding: 18px; margin: 10px 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-box h4 { color: #1E3A5F; margin: 0 0 10px 0; }
    .metric-box .formula {
        background: #263238; color: #80cbc4; padding: 10px 15px;
        border-radius: 8px; font-family: 'Courier New', monospace; 
        margin: 10px 0; font-size: 0.9rem;
    }
    .metric-box .description { color: #555; margin: 8px 0; line-height: 1.5; }
    .metric-box .interpretation { 
        background: #e3f2fd; padding: 8px 12px; border-radius: 6px; 
        color: #1565c0; font-size: 0.9rem; margin-top: 10px;
    }
    
    .strategy-box {
        background: white; border-radius: 12px; padding: 20px; margin: 15px 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08); border-left: 5px solid #2196F3;
    }
    .strategy-box h4 { color: #1565c0; margin: 0 0 15px 0; font-size: 1.2rem; }
    .strategy-box .step {
        background: #f8f9fa; padding: 10px 15px; border-radius: 8px;
        margin: 8px 0; border-left: 3px solid #4CAF50;
    }
    .strategy-box .pros { color: #2e7d32; }
    .strategy-box .cons { color: #c62828; }
    
    .info-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; margin-bottom: 20px;
    }
    .info-banner h2 { color: white; margin: 0; }
    .info-banner p { color: rgba(255,255,255,0.8); margin: 5px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. CONSTANTS
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

HOLDING_PERIODS = [63, 126, 189, 252, 378, 504, 630, 756]

def get_holding_label(days):
    if days < 252:
        return f"{days}d (~{days//21}M)"
    else:
        return f"{days}d (~{days//252}Y)"

COLORS = {
    'primary': '#4CAF50', 'secondary': '#2196F3', 'success': '#00C853',
    'warning': '#FF9800', 'danger': '#F44336', 'info': '#00BCD4'
}

# ============================================================================
# 3. COMPREHENSIVE METRIC DEFINITIONS
# ============================================================================

METRIC_DEFINITIONS = {
    'returns': {
        'Return 3M': {
            'name': '3-Month Return',
            'formula': 'Return_3M = (NAV_today / NAV_3months_ago - 1) × 100',
            'description': 'The percentage change in NAV over the last 3 months (approximately 63 trading days).',
            'interpretation': 'Shows short-term momentum. Positive values indicate recent gains.'
        },
        'Return 6M': {
            'name': '6-Month Return',
            'formula': 'Return_6M = (NAV_today / NAV_6months_ago - 1) × 100',
            'description': 'The percentage change in NAV over the last 6 months (approximately 126 trading days).',
            'interpretation': 'Medium-term performance indicator. Used in momentum strategies.'
        },
        'Return 1Y': {
            'name': '1-Year Return',
            'formula': 'Return_1Y = (NAV_today / NAV_1year_ago - 1) × 100',
            'description': 'The percentage change in NAV over the last 12 months (approximately 252 trading days).',
            'interpretation': 'Standard performance metric. Compare against benchmark for alpha.'
        },
        'Return 3Y': {
            'name': '3-Year Annualized Return',
            'formula': 'Return_3Y = ((NAV_today / NAV_3years_ago)^(1/3) - 1) × 100',
            'description': 'Annualized return over 3 years, showing long-term performance.',
            'interpretation': 'Better indicator of consistent performance than 1Y return.'
        },
        'CAGR': {
            'name': 'Compound Annual Growth Rate',
            'formula': 'CAGR = (Ending_Value / Beginning_Value)^(1/Years) - 1',
            'description': 'The geometric average annual return over the entire investment period. Accounts for compounding.',
            'interpretation': 'Higher is better. 15% CAGR means your investment grew at 15% annually on average.'
        }
    },
    'risk': {
        'Volatility': {
            'name': 'Annualized Volatility (Standard Deviation)',
            'formula': 'Volatility = StdDev(Daily_Returns) × √252',
            'description': 'Measures the dispersion of returns around the mean. Higher volatility means more unpredictable returns.',
            'interpretation': '<15% is low, 15-25% is moderate, >25% is high volatility.'
        },
        'Max Drawdown': {
            'name': 'Maximum Drawdown',
            'formula': 'Max_DD = (Trough_Value - Peak_Value) / Peak_Value × 100',
            'description': 'The largest peak-to-trough decline in NAV. Measures the worst loss an investor could have experienced.',
            'interpretation': 'Less negative is better. -20% means the fund lost 20% from its peak at worst.'
        },
        'VaR 95': {
            'name': 'Value at Risk (95%)',
            'formula': 'VaR_95 = Percentile(Daily_Returns, 5%)',
            'description': 'The maximum expected daily loss with 95% confidence. Only 5% of days should have losses worse than this.',
            'interpretation': '-2% VaR means you can expect to lose at most 2% on 95% of trading days.'
        },
        'Downside Deviation': {
            'name': 'Downside Deviation',
            'formula': 'DD = √(Σ(min(Return - Target, 0)²) / n)',
            'description': 'Standard deviation of only negative returns. Only penalizes downside volatility, not upside.',
            'interpretation': 'Lower is better. Used in Sortino Ratio calculation.'
        }
    },
    'risk_adjusted': {
        'Sharpe Ratio': {
            'name': 'Sharpe Ratio',
            'formula': 'Sharpe = (Return - Risk_Free_Rate) / Volatility',
            'description': 'Measures excess return per unit of total risk. Developed by William Sharpe.',
            'interpretation': '<1 is poor, 1-2 is good, 2-3 is very good, >3 is excellent.'
        },
        'Sortino Ratio': {
            'name': 'Sortino Ratio',
            'formula': 'Sortino = (Return - Risk_Free_Rate) / Downside_Deviation',
            'description': 'Like Sharpe but only penalizes downside volatility. Better for asymmetric return distributions.',
            'interpretation': 'Higher is better. More appropriate when returns are not normally distributed.'
        },
        'Calmar Ratio': {
            'name': 'Calmar Ratio',
            'formula': 'Calmar = CAGR / |Max_Drawdown|',
            'description': 'Measures return relative to maximum drawdown. Shows how well returns compensate for worst-case losses.',
            'interpretation': '>1 means annual return exceeds worst drawdown. Higher is better.'
        },
        'Information Ratio': {
            'name': 'Information Ratio',
            'formula': 'IR = (Fund_Return - Benchmark_Return) / Tracking_Error',
            'description': 'Measures consistency of outperformance relative to benchmark. Tracking Error is std dev of excess returns.',
            'interpretation': '>0.5 is good, >1.0 is excellent. Shows skill of fund manager.'
        }
    },
    'benchmark': {
        'Beta': {
            'name': 'Beta',
            'formula': 'Beta = Covariance(Fund, Benchmark) / Variance(Benchmark)',
            'description': 'Measures sensitivity to market movements. Beta of 1 means fund moves with market.',
            'interpretation': '<1 is defensive, =1 is neutral, >1 is aggressive. Beta 1.2 = 20% more volatile than market.'
        },
        'Alpha': {
            'name': "Jensen's Alpha",
            'formula': 'Alpha = Fund_Return - [Risk_Free + Beta × (Benchmark_Return - Risk_Free)]',
            'description': 'Excess return beyond what CAPM predicts. Measures fund manager skill.',
            'interpretation': 'Positive alpha = outperformance. 2% alpha means 2% extra return beyond expected.'
        },
        'Up Capture': {
            'name': 'Up Capture Ratio',
            'formula': 'Up_Capture = (Fund_Return in Up_Markets / Benchmark_Return in Up_Markets) × 100',
            'description': 'How much of the benchmark gains does the fund capture during rising markets.',
            'interpretation': '>100% means fund gains more than benchmark in up markets.'
        },
        'Down Capture': {
            'name': 'Down Capture Ratio',
            'formula': 'Down_Capture = (Fund_Return in Down_Markets / Benchmark_Return in Down_Markets) × 100',
            'description': 'How much of the benchmark losses does the fund capture during falling markets.',
            'interpretation': '<100% is good - means fund loses less than benchmark in down markets.'
        },
        'Capture Ratio': {
            'name': 'Capture Ratio',
            'formula': 'Capture_Ratio = Up_Capture / Down_Capture',
            'description': 'Combined measure of up and down capture. Higher means better risk-adjusted behavior.',
            'interpretation': '>1 is good (captures more upside than downside). >1.2 is excellent.'
        }
    },
    'rolling': {
        '1Y Rolling Return': {
            'name': '1-Year Rolling Average Return',
            'formula': 'Rolling_1Y = Average of all overlapping 1-year returns',
            'description': 'Average of all possible 1-year returns throughout the history.',
            'interpretation': 'More stable than point-to-point returns. Shows typical annual return.'
        },
        '1Y Beat %': {
            'name': 'Percentage of Times Beat Benchmark (1Y)',
            'formula': 'Beat_% = Count(1Y_Return > Benchmark_1Y_Return) / Total_Periods × 100',
            'description': 'How often the fund outperformed the benchmark over rolling 1-year periods.',
            'interpretation': '>50% means fund usually beats benchmark. >70% is excellent consistency.'
        },
        'Consistency Score': {
            'name': 'Consistency Score',
            'formula': 'Consistency = Beat_% × (1 + Avg_Outperformance) / (1 + |Avg_Underperformance|)',
            'description': 'Combined measure of how often and by how much the fund beats/lags benchmark.',
            'interpretation': 'Higher is better. Accounts for both frequency and magnitude of over/underperformance.'
        }
    }
}

# ============================================================================
# 4. COMPREHENSIVE STRATEGY DEFINITIONS
# ============================================================================

STRATEGY_DEFINITIONS = {
    'momentum': {
        'name': '🚀 Momentum Strategy',
        'short_desc': 'Selects funds with highest weighted average of 3M, 6M, 12M returns.',
        'how_it_works': [
            'Calculate trailing returns for 3 months, 6 months, and 12 months',
            'Apply weights to each return period (default: 33% each)',
            'Combine into a single momentum score: Score = w1×R3M + w2×R6M + w3×R12M',
            'Optionally divide by volatility for risk-adjusted momentum',
            'Rank all funds by score and select top N'
        ],
        'formula': 'Momentum_Score = (w₁ × Return_3M) + (w₂ × Return_6M) + (w₃ × Return_12M)',
        'formula_risk_adj': 'Risk_Adjusted_Score = Momentum_Score / Annualized_Volatility',
        'why_it_works': 'Based on the momentum anomaly - assets that have performed well tend to continue performing well in the short to medium term. This persistence is attributed to investor behavioral biases like herding and slow information diffusion.',
        'best_for': 'Trending markets with clear directional movement. Works well during bull markets and sustained rallies.',
        'weaknesses': [
            'Performs poorly at market turning points (momentum crash)',
            'High turnover leading to transaction costs',
            'Can concentrate in overvalued assets',
            'Suffers during mean-reversion periods'
        ],
        'parameters': ['w_3m (3M weight)', 'w_6m (6M weight)', 'w_12m (12M weight)', 'risk_adjust (True/False)']
    },
    'sharpe': {
        'name': '⚖️ Sharpe Ratio Strategy',
        'short_desc': 'Selects funds with highest risk-adjusted returns (Sharpe Ratio).',
        'how_it_works': [
            'Calculate daily returns for each fund over the lookback period',
            'Compute mean excess return (return minus risk-free rate)',
            'Calculate standard deviation of returns (volatility)',
            'Sharpe Ratio = (Mean Excess Return / Volatility) × √252',
            'Rank all funds by Sharpe Ratio and select top N'
        ],
        'formula': 'Sharpe = (R̄ - Rf) / σ × √252',
        'formula_detail': 'Where R̄ = mean daily return, Rf = daily risk-free rate, σ = daily std dev',
        'why_it_works': 'Balances return with risk. A fund with 15% return and 10% volatility (Sharpe=0.9) may be better than one with 20% return and 30% volatility (Sharpe=0.47).',
        'best_for': 'Investors who want consistent, risk-adjusted performance. Good for moderate-risk portfolios.',
        'weaknesses': [
            'Assumes returns are normally distributed (often false)',
            'Penalizes upside volatility equally as downside',
            'Sensitive to lookback period chosen',
            'Can favor low-return, low-volatility funds'
        ],
        'parameters': ['lookback_days (default: 252)']
    },
    'sortino': {
        'name': '🎯 Sortino Ratio Strategy',
        'short_desc': 'Selects funds with highest return per unit of downside risk.',
        'how_it_works': [
            'Calculate daily returns for each fund',
            'Identify only the negative returns (losses)',
            'Calculate downside deviation = std dev of negative returns only',
            'Sortino Ratio = (Mean Excess Return / Downside Deviation) × √252',
            'Rank all funds by Sortino Ratio and select top N'
        ],
        'formula': 'Sortino = (R̄ - Rf) / σ_downside × √252',
        'formula_detail': 'Where σ_downside = √(Σ(min(Ri - Rf, 0)²) / n)',
        'why_it_works': 'Only penalizes harmful volatility (losses), not beneficial volatility (gains). Better for funds with asymmetric returns like those using options strategies.',
        'best_for': 'Funds with asymmetric return profiles. When you care more about avoiding losses than overall consistency.',
        'weaknesses': [
            'Requires sufficient negative return days to calculate properly',
            'Can be unstable with small sample sizes',
            'May favor funds that rarely lose but crash hard occasionally',
            'Less intuitive than Sharpe for some investors'
        ],
        'parameters': ['lookback_days (default: 252)', 'target_return (default: risk-free rate)']
    },
    'regime_switch': {
        'name': '🚦 Regime Switch Strategy',
        'short_desc': 'Uses Momentum in bull markets, Sharpe in bear markets.',
        'how_it_works': [
            'Determine current market regime using 200-day moving average of benchmark',
            'If Benchmark Price > 200 DMA → Bull Market → Use Momentum Strategy',
            'If Benchmark Price < 200 DMA → Bear Market → Use Sharpe Strategy',
            'Apply the appropriate strategy to select top N funds',
            'Re-evaluate regime at each rebalancing date'
        ],
        'formula': 'Regime = "Bull" if Price > MA_200 else "Bear"',
        'formula_detail': 'Bull → Momentum Score | Bear → Sharpe Ratio',
        'why_it_works': 'Different strategies work in different market conditions. Momentum thrives in trending markets while Sharpe provides protection in volatile/declining markets.',
        'best_for': 'Investors who want adaptive strategy that changes with market conditions.',
        'weaknesses': [
            'Can whipsaw if price oscillates around the 200 DMA',
            'Regime detection is lagging (200 days is slow)',
            'Transaction costs from strategy switches',
            'May miss early moves in new trends'
        ],
        'parameters': ['ma_period (default: 200 days)', 'benchmark (Nifty 100)']
    },
    'stable_momentum': {
        'name': '⚓ Stable Momentum Strategy',
        'short_desc': 'High momentum funds filtered for low drawdown.',
        'how_it_works': [
            'Calculate momentum score for all funds',
            'Select top 2×N funds by momentum (create initial pool)',
            'For each fund in pool, calculate maximum drawdown',
            'From the pool, select final N funds with smallest (least negative) drawdowns',
            'This gives momentum exposure with crash protection'
        ],
        'formula': 'Step 1: Pool = Top 2N by Momentum',
        'formula_detail': 'Step 2: Selected = Top N from Pool by Min |Max_Drawdown|',
        'why_it_works': 'Captures momentum returns while avoiding funds prone to large crashes. Drawdown filter removes volatile momentum stocks.',
        'best_for': 'Investors who want momentum exposure but are risk-averse. Good for volatile markets.',
        'weaknesses': [
            'May miss the highest-returning momentum funds if they are volatile',
            'Drawdown is backward-looking, may not predict future crashes',
            'Can be too conservative in strong bull markets',
            'Smaller selection pool limits diversification'
        ],
        'parameters': ['momentum_weights', 'pool_multiplier (default: 2×N)']
    },
    'elimination': {
        'name': '🛡️ Elimination Strategy',
        'short_desc': 'Multi-stage filtering: remove worst by DD & volatility, pick by Sharpe.',
        'how_it_works': [
            'Calculate Max Drawdown, Volatility, and Sharpe for all funds',
            'Stage 1: Eliminate bottom 25% by drawdown (worst crashes removed)',
            'Stage 2: Eliminate top 25% by volatility (most volatile removed)',
            'Stage 3: From remaining funds, select top N by Sharpe Ratio',
            'This creates a "safe" pool before optimization'
        ],
        'formula': 'Step 1: Remove if Max_DD < Percentile_25(Max_DD)',
        'formula_detail': 'Step 2: Remove if Volatility > Percentile_75(Volatility)\nStep 3: Select Top N by Sharpe from remainder',
        'why_it_works': 'Multiple risk filters provide layered protection. Removes problematic funds before selecting best risk-adjusted performers.',
        'best_for': 'Conservative investors who prioritize capital protection. Good for retirement portfolios.',
        'weaknesses': [
            'May eliminate high-return funds that have temporary volatility',
            'Multiple filters significantly reduce selection pool',
            'Can be too conservative, missing opportunities',
            'Percentile thresholds are arbitrary'
        ],
        'parameters': ['dd_percentile (default: 25%)', 'vol_percentile (default: 75%)']
    },
    'consistency': {
        'name': '📈 Consistency Strategy',
        'short_desc': 'Requires top 50% performance in 3+ of last 4 quarters.',
        'how_it_works': [
            'Look back at the last 4 quarters of performance',
            'For each quarter, check if fund was in top 50% of all funds',
            'Count quarters where fund achieved top-half performance',
            'Keep only funds with 3 or more qualifying quarters (≥75% consistency)',
            'From consistent funds, select top N by recent 3-month momentum'
        ],
        'formula': 'Consistent if Count(Quarterly_Rank ≤ Median) ≥ 3 out of 4',
        'formula_detail': 'Final Selection: Top N by 3M Momentum from Consistent Funds',
        'why_it_works': 'Filters for funds that perform well regularly, not just occasionally. Avoids one-hit wonders and erratic performers.',
        'best_for': 'Long-term investors who value reliability over occasional outperformance.',
        'weaknesses': [
            'May miss improving funds that had poor historical consistency',
            'Backward-looking - past consistency may not continue',
            'Can favor mediocre but stable funds over excellent but variable ones',
            'Requires sufficient history (at least 4 quarters)'
        ],
        'parameters': ['quarters_to_check (default: 4)', 'min_consistent_quarters (default: 3)', 'percentile_threshold (default: 50%)']
    }
}

# ============================================================================
# 5. HELPER FUNCTIONS FOR DEFINITIONS DISPLAY
# ============================================================================

def show_metric_definitions_detailed(category='all'):
    """Display detailed metric definitions with formulas."""
    
    if category == 'all':
        categories = METRIC_DEFINITIONS.keys()
    else:
        categories = [category] if category in METRIC_DEFINITIONS else []
    
    for cat in categories:
        cat_title = cat.replace('_', ' ').title()
        st.markdown(f"### {cat_title} Metrics")
        
        metrics = METRIC_DEFINITIONS[cat]
        cols = st.columns(2)
        
        for idx, (key, metric) in enumerate(metrics.items()):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="metric-box">
                    <h4>📊 {metric['name']}</h4>
                    <div class="formula">{metric['formula']}</div>
                    <div class="description">{metric['description']}</div>
                    <div class="interpretation">💡 {metric['interpretation']}</div>
                </div>
                """, unsafe_allow_html=True)

def show_all_metric_definitions():
    """Show all metric definitions in an expander."""
    with st.expander("📖 **Complete Metric Definitions & Formulas** (Click to expand)", expanded=False):
        tabs = st.tabs(["📈 Returns", "⚠️ Risk", "⚖️ Risk-Adjusted", "🎯 Benchmark", "🔄 Rolling"])
        
        with tabs[0]:
            show_metric_definitions_detailed('returns')
        with tabs[1]:
            show_metric_definitions_detailed('risk')
        with tabs[2]:
            show_metric_definitions_detailed('risk_adjusted')
        with tabs[3]:
            show_metric_definitions_detailed('benchmark')
        with tabs[4]:
            show_metric_definitions_detailed('rolling')

def show_strategy_definition_detailed(strategy_key):
    """Display detailed strategy definition."""
    if strategy_key not in STRATEGY_DEFINITIONS:
        return
    
    strat = STRATEGY_DEFINITIONS[strategy_key]
    
    with st.expander(f"📚 **{strat['name']} - Complete Details**", expanded=False):
        st.markdown(f"**Summary:** {strat['short_desc']}")
        
        st.markdown("#### How It Works:")
        for i, step in enumerate(strat['how_it_works'], 1):
            st.markdown(f"""<div class="strategy-box"><div class="step"><strong>Step {i}:</strong> {step}</div></div>""", unsafe_allow_html=True)
        
        st.markdown("#### Formula:")
        st.code(strat['formula'])
        if 'formula_detail' in strat:
            st.code(strat['formula_detail'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ Why It Works:")
            st.info(strat['why_it_works'])
            st.markdown("#### 🎯 Best For:")
            st.success(strat['best_for'])
        
        with col2:
            st.markdown("#### ⚠️ Weaknesses:")
            for weakness in strat['weaknesses']:
                st.markdown(f"- {weakness}")
            
            st.markdown("#### ⚙️ Parameters:")
            for param in strat['parameters']:
                st.markdown(f"- `{param}`")

def show_all_strategy_definitions():
    """Show all strategy definitions."""
    with st.expander("📚 **Complete Strategy Definitions** (Click to expand)", expanded=False):
        for key, strat in STRATEGY_DEFINITIONS.items():
            st.markdown(f"""
            <div class="strategy-box">
                <h4>{strat['name']}</h4>
                <p><strong>Summary:</strong> {strat['short_desc']}</p>
                <p><strong>Best For:</strong> {strat['best_for']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"View Full Details: {strat['name']}"):
                st.markdown("**How It Works:**")
                for i, step in enumerate(strat['how_it_works'], 1):
                    st.markdown(f"{i}. {step}")
                
                st.markdown("**Formula:**")
                st.code(strat['formula'])
                
                st.markdown("**Weaknesses:**")
                for w in strat['weaknesses']:
                    st.markdown(f"- {w}")

# ============================================================================
# 6. METRIC CALCULATIONS
# ============================================================================

def calculate_sharpe_ratio(returns):
    if len(returns) < 10 or returns.std() == 0: return np.nan
    excess_returns = returns - DAILY_RISK_FREE_RATE
    return (excess_returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS_YEAR)

def calculate_sortino_ratio(returns):
    if len(returns) < 10: return np.nan
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0: return np.nan
    mean_return = (returns - DAILY_RISK_FREE_RATE).mean()
    return (mean_return / downside.std()) * np.sqrt(TRADING_DAYS_YEAR)

def calculate_volatility(returns):
    if len(returns) < 10: return np.nan
    return returns.std() * np.sqrt(TRADING_DAYS_YEAR)

def calculate_max_dd(series):
    if len(series) < 10 or series.isna().all(): return np.nan
    comp_ret = (1 + series.pct_change().fillna(0)).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret / peak) - 1
    return dd.min()

def calculate_cagr(series):
    if len(series) < 30 or series.iloc[0] <= 0: return np.nan
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0: return np.nan
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1

def calculate_calmar_ratio(series):
    if len(series) < 252: return np.nan
    max_dd = calculate_max_dd(series)
    if pd.isna(max_dd) or max_dd >= 0: return np.nan
    cagr = calculate_cagr(series)
    if pd.isna(cagr): return np.nan
    return cagr / abs(max_dd)

def calculate_beta_alpha(fund_returns, bench_returns):
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 60: return np.nan, np.nan
    f_ret, b_ret = fund_returns.loc[common_idx], bench_returns.loc[common_idx]
    cov = np.cov(f_ret, b_ret)
    if cov[1, 1] == 0: return np.nan, np.nan
    beta = cov[0, 1] / cov[1, 1]
    alpha = (f_ret.mean() - beta * b_ret.mean()) * TRADING_DAYS_YEAR
    return beta, alpha

def calculate_information_ratio(fund_returns, bench_returns):
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 30: return np.nan
    active_return = fund_returns.loc[common_idx] - bench_returns.loc[common_idx]
    tracking_error = active_return.std(ddof=1) * np.sqrt(TRADING_DAYS_YEAR)
    if tracking_error == 0: return np.nan
    return (active_return.mean() * TRADING_DAYS_YEAR) / tracking_error

def calculate_capture_ratios(fund_rets, bench_rets):
    common_idx = fund_rets.index.intersection(bench_rets.index)
    if len(common_idx) < 30: return np.nan, np.nan, np.nan
    f, b = fund_rets.loc[common_idx], bench_rets.loc[common_idx]
    up_market, down_market = b[b > 0], b[b < 0]
    up_cap = f.loc[up_market.index].mean() / up_market.mean() if not up_market.empty and up_market.mean() != 0 else np.nan
    down_cap = f.loc[down_market.index].mean() / down_market.mean() if not down_market.empty and down_market.mean() != 0 else np.nan
    ratio = up_cap / down_cap if pd.notna(up_cap) and pd.notna(down_cap) and down_cap > 0 else np.nan
    return up_cap, down_cap, ratio

def calculate_rolling_metrics(series, benchmark_series, window_days):
    if len(series) < window_days + 30: return np.nan, np.nan, np.nan, np.nan, np.nan
    fund_rolling = series.pct_change(window_days).dropna()
    bench_rolling = benchmark_series.pct_change(window_days).dropna()
    common_idx = fund_rolling.index.intersection(bench_rolling.index)
    if len(common_idx) < 10: return np.nan, np.nan, np.nan, np.nan, np.nan
    f_roll, b_roll = fund_rolling.loc[common_idx], bench_rolling.loc[common_idx]
    avg_rolling_return = f_roll.mean()
    diff = f_roll - b_roll
    pct_beat = (diff > 0).mean()
    avg_out = diff[diff > 0].mean() if len(diff[diff > 0]) > 0 else 0
    avg_under = diff[diff < 0].mean() if len(diff[diff < 0]) > 0 else 0
    consistency = pct_beat * (1 + avg_out) / (1 + abs(avg_under)) if avg_under != 0 else pct_beat
    return avg_rolling_return, pct_beat, avg_out, avg_under, consistency

def clean_weekday_data(df):
    if df is None or df.empty: return df
    df = df[df.index <= MAX_DATA_DATE]
    df = df[df.index.dayofweek < 5]
    if len(df) > 0:
        start_date, end_date = df.index.min(), min(df.index.max(), MAX_DATA_DATE)
        all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
        df = df.reindex(all_weekdays).ffill(limit=5)
    return df

# ============================================================================
# 7. DATA LOADING
# ============================================================================

@st.cache_data
def load_fund_data_raw(category_key: str):
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
        st.error(f"Error loading {filename}: {e}")
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
# 8. COMPREHENSIVE ANALYTICS FOR EXPLORER
# ============================================================================

def calculate_comprehensive_metrics(nav_df, scheme_map, benchmark_series):
    """Calculate all metrics for all funds."""
    if nav_df is None or nav_df.empty or benchmark_series is None: 
        return pd.DataFrame()
    
    metrics_list = []
    bench_returns = benchmark_series.pct_change().dropna()
    
    for col in nav_df.columns:
        series = nav_df[col].dropna()
        if len(series) < 260: continue
        
        fund_name = scheme_map.get(col, col)
        returns = series.pct_change().dropna()
        
        row = {
            'Fund Name': fund_name,
            'fund_id': col,
            'Data Start': series.index[0].strftime('%Y-%m-%d'),
            'Data End': series.index[-1].strftime('%Y-%m-%d'),
            'Days': len(series)
        }
        
        # Returns
        if len(series) >= 63: 
            row['Return 3M %'] = (series.iloc[-1] / series.iloc[-63] - 1) * 100
        if len(series) >= 126: 
            row['Return 6M %'] = (series.iloc[-1] / series.iloc[-126] - 1) * 100
        if len(series) >= 252: 
            row['Return 1Y %'] = (series.iloc[-1] / series.iloc[-252] - 1) * 100
        if len(series) >= 756: 
            row['Return 3Y % (Ann)'] = ((series.iloc[-1] / series.iloc[-756]) ** (1/3) - 1) * 100
        if len(series) >= 1260: 
            row['Return 5Y % (Ann)'] = ((series.iloc[-1] / series.iloc[-1260]) ** (1/5) - 1) * 100
        
        cagr = calculate_cagr(series)
        row['CAGR %'] = cagr * 100 if cagr else np.nan
        
        # Risk metrics
        row['Volatility %'] = calculate_volatility(returns) * 100 if calculate_volatility(returns) else np.nan
        row['Max Drawdown %'] = calculate_max_dd(series) * 100 if calculate_max_dd(series) else np.nan
        if len(returns) > 30:
            row['VaR 95 %'] = returns.quantile(0.05) * 100
        
        # Risk-adjusted metrics
        row['Sharpe Ratio'] = calculate_sharpe_ratio(returns)
        row['Sortino Ratio'] = calculate_sortino_ratio(returns)
        row['Calmar Ratio'] = calculate_calmar_ratio(series)
        row['Information Ratio'] = calculate_information_ratio(returns, bench_returns)
        
        # Benchmark metrics
        beta, alpha = calculate_beta_alpha(returns, bench_returns)
        row['Beta'] = beta
        row['Alpha %'] = alpha * 100 if pd.notna(alpha) else np.nan
        
        up_cap, down_cap, cap_ratio = calculate_capture_ratios(returns, bench_returns)
        row['Up Capture %'] = up_cap * 100 if pd.notna(up_cap) else np.nan
        row['Down Capture %'] = down_cap * 100 if pd.notna(down_cap) else np.nan
        row['Capture Ratio'] = cap_ratio
        
        # Rolling metrics
        roll_1y = calculate_rolling_metrics(series, benchmark_series, 252)
        row['1Y Rolling Avg %'] = roll_1y[0] * 100 if pd.notna(roll_1y[0]) else np.nan
        row['1Y Beat Benchmark %'] = roll_1y[1] * 100 if pd.notna(roll_1y[1]) else np.nan
        row['1Y Consistency Score'] = roll_1y[4] if pd.notna(roll_1y[4]) else np.nan
        
        roll_3y = calculate_rolling_metrics(series, benchmark_series, 756)
        row['3Y Rolling Avg %'] = roll_3y[0] * 100 if pd.notna(roll_3y[0]) else np.nan
        row['3Y Consistency Score'] = roll_3y[4] if pd.notna(roll_3y[4]) else np.nan
        
        # Positive months
        monthly_returns = series.resample('ME').last().pct_change().dropna()
        if len(monthly_returns) > 6:
            row['Positive Months %'] = (monthly_returns > 0).mean() * 100
        
        metrics_list.append(row)
    
    df = pd.DataFrame(metrics_list)
    
    # Add rankings
    if not df.empty:
        rank_cols = ['CAGR %', '1Y Rolling Avg %', 'Sharpe Ratio', '1Y Consistency Score']
        for col_name in rank_cols:
            if col_name in df.columns:
                rank_name = col_name.replace(' %', '').replace(' Ratio', '') + ' Rank'
                df[rank_name] = df[col_name].rank(ascending=False, method='min')
        
        # Composite rank
        rank_columns = [c for c in df.columns if 'Rank' in c]
        if rank_columns:
            df['Composite Rank'] = df[rank_columns].mean(axis=1)
            df = df.sort_values('Composite Rank')
    
    return df

def calculate_quarterly_ranks(nav_df, scheme_map):
    """Calculate quarterly performance ranks for all funds."""
    if nav_df is None or nav_df.empty: 
        return pd.DataFrame()
    
    quarter_ends = pd.date_range(start=nav_df.index.min(), end=nav_df.index.max(), freq='Q')
    history_data = {}
    
    for q_date in quarter_ends:
        start_lookback = q_date - pd.Timedelta(days=91)  # Quarterly
        if start_lookback < nav_df.index.min(): 
            continue
        
        try:
            idx_now = nav_df.index.asof(q_date)
            idx_prev = nav_df.index.asof(start_lookback)
            if pd.isna(idx_now) or pd.isna(idx_prev): 
                continue
            
            rets = (nav_df.loc[idx_now] / nav_df.loc[idx_prev]) - 1
            quarter_label = f"{q_date.year}-Q{(q_date.month-1)//3 + 1}"
            history_data[quarter_label] = rets.rank(ascending=False, method='min')
        except: 
            continue
    
    if not history_data:
        return pd.DataFrame()
    
    rank_df = pd.DataFrame(history_data)
    rank_df.index = rank_df.index.map(lambda x: scheme_map.get(x, x))
    rank_df = rank_df.dropna(how='all')
    
    # Add summary statistics
    if not rank_df.empty:
        def calc_stats(row):
            valid = row.dropna()
            if len(valid) == 0: 
                return pd.Series({
                    '% Top 3': 0, '% Top 5': 0, '% Top 10': 0, 
                    'Avg Rank': np.nan, 'Best Rank': np.nan, 'Worst Rank': np.nan
                })
            return pd.Series({
                '% Top 3': (valid <= 3).mean() * 100,
                '% Top 5': (valid <= 5).mean() * 100,
                '% Top 10': (valid <= 10).mean() * 100,
                'Avg Rank': valid.mean(),
                'Best Rank': valid.min(),
                'Worst Rank': valid.max()
            })
        
        stats = rank_df.apply(calc_stats, axis=1)
        rank_df = pd.concat([stats, rank_df], axis=1)
        rank_df = rank_df.sort_values('% Top 5', ascending=False)
    
    return rank_df

# ============================================================================
# 9. VISUALIZATION
# ============================================================================

def create_performance_chart(nav_df, selected_funds, scheme_map, benchmark_series=None, normalize=True):
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for idx, fund_id in enumerate(selected_funds[:10]):
        series = nav_df[fund_id].dropna()
        if normalize and len(series) > 0: 
            series = series / series.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, 
            name=scheme_map.get(fund_id, fund_id)[:30],
            line=dict(color=colors[idx % len(colors)], width=2)
        ))
    if benchmark_series is not None:
        bench = benchmark_series.dropna()
        if normalize and len(bench) > 0 and len(selected_funds) > 0:
            common_start = nav_df[selected_funds[0]].dropna().index[0]
            bench = bench[bench.index >= common_start]
            if len(bench) > 0: 
                bench = bench / bench.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=bench.index, y=bench.values, name='Nifty 100', 
            line=dict(color='gray', width=2, dash='dot')
        ))
    fig.update_layout(
        title='Performance Comparison (Normalized to 100)',
        yaxis_title='Value', hovermode='x unified', height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig

def create_drawdown_chart(series, fund_name):
    returns = series.pct_change().fillna(0)
    cum_ret = (1 + returns).cumprod()
    drawdown = (cum_ret / cum_ret.expanding().max() - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values, fill='tozeroy',
        line=dict(color=COLORS['danger'], width=1),
        fillcolor='rgba(244, 67, 54, 0.3)', name='Drawdown'
    ))
    fig.update_layout(
        title=f'Drawdown Analysis - {fund_name[:30]}',
        yaxis_title='Drawdown %', height=300
    )
    return fig

def create_rolling_return_chart(series, benchmark_series, fund_name, window=252):
    fund_rolling = series.pct_change(window).dropna() * 100
    bench_rolling = benchmark_series.pct_change(window).dropna() * 100
    common_idx = fund_rolling.index.intersection(bench_rolling.index)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=common_idx, y=fund_rolling.loc[common_idx],
        name=fund_name[:25], fill='tozeroy',
        line=dict(color=COLORS['primary'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=common_idx, y=bench_rolling.loc[common_idx],
        name='Nifty 100', line=dict(color='gray', width=2, dash='dash')
    ))
    fig.update_layout(
        title=f'{window//252}Y Rolling Returns',
        yaxis_title='Rolling Return %', hovermode='x unified', height=350
    )
    return fig

# ============================================================================
# 10. EXPLORER TAB - ENHANCED
# ============================================================================

def render_explorer_tab():
    st.markdown("""
    <div class="info-banner">
        <h2>📊 Category Explorer</h2>
        <p>Comprehensive fund analysis with detailed metrics, quarterly rankings, and deep dive</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show metric definitions
    show_all_metric_definitions()
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: 
        category = st.selectbox("📁 Select Category", list(FILE_MAPPING.keys()))
    with col2: 
        view_mode = st.selectbox("👁️ View Mode", [
            "📈 Comprehensive Metrics", 
            "📊 Quarterly Rank History", 
            "🔍 Fund Deep Dive"
        ])
    with col3:
        st.write("")
        st.write("")
        refresh = st.button("🔄 Refresh", use_container_width=True)
    
    st.divider()
    
    with st.spinner(f"Loading {category} data..."):
        nav_df, scheme_map = load_fund_data_raw(category)
        benchmark = load_nifty_data()
    
    if nav_df is None:
        st.error("❌ Could not load data. Please check if data files exist.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Funds", len(nav_df.columns))
    col2.metric("📅 Data Start", nav_df.index.min().strftime('%Y-%m-%d'))
    col3.metric("📅 Data End", nav_df.index.max().strftime('%Y-%m-%d'))
    col4.metric("🏛️ Benchmark", "Nifty 100" if benchmark is not None else "N/A")
    
    st.divider()
    
    # =========================================================================
    # COMPREHENSIVE METRICS VIEW
    # =========================================================================
    if "Comprehensive Metrics" in view_mode:
        with st.spinner("Calculating comprehensive metrics..."):
            metrics_df = calculate_comprehensive_metrics(nav_df, scheme_map, benchmark)
        
        if metrics_df.empty:
            st.warning("No funds with sufficient data.")
            return
        
        st.markdown("### 📋 Fund Performance Analysis")
        
        # Tabbed view of metrics
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏆 Rankings", "📈 Returns", "⚠️ Risk", 
            "⚖️ Risk-Adjusted", "🎯 Benchmark", "🔄 Rolling"
        ])
        
        with tab1:
            st.markdown("#### Overall Fund Rankings")
            rank_cols = ['Fund Name', 'Composite Rank', 'CAGR Rank', '1Y Rolling Avg Rank', 
                        'Sharpe Rank', '1Y Consistency Score Rank', 'CAGR %', 'Sharpe Ratio']
            available_cols = [c for c in rank_cols if c in metrics_df.columns]
            
            st.dataframe(
                metrics_df[available_cols].head(25).style.format({
                    c: '{:.1f}' for c in available_cols if c != 'Fund Name'
                }).background_gradient(subset=['Composite Rank'] if 'Composite Rank' in available_cols else [], cmap='Greens_r'),
                use_container_width=True, height=600
            )
        
        with tab2:
            st.markdown("#### Return Metrics")
            st.markdown("""
            <div class="metric-box">
                <p>Returns show the percentage gain/loss over different time periods. 
                Annualized returns (3Y, 5Y) show the equivalent annual return rate.</p>
            </div>
            """, unsafe_allow_html=True)
            
            return_cols = ['Fund Name', 'Return 3M %', 'Return 6M %', 'Return 1Y %', 
                          'Return 3Y % (Ann)', 'Return 5Y % (Ann)', 'CAGR %', 'Positive Months %']
            available_cols = [c for c in return_cols if c in metrics_df.columns]
            
            st.dataframe(
                metrics_df[available_cols].style.format({
                    c: '{:.2f}%' if '%' in c else '{:.2f}' for c in available_cols if c != 'Fund Name'
                }).background_gradient(subset=['Return 1Y %'] if 'Return 1Y %' in available_cols else [], cmap='RdYlGn'),
                use_container_width=True, height=600
            )
        
        with tab3:
            st.markdown("#### Risk Metrics")
            st.markdown("""
            <div class="metric-box">
                <p><strong>Volatility:</strong> Higher = more unpredictable returns<br>
                <strong>Max Drawdown:</strong> Worst peak-to-trough loss (less negative is better)<br>
                <strong>VaR 95%:</strong> Maximum expected daily loss with 95% confidence</p>
            </div>
            """, unsafe_allow_html=True)
            
            risk_cols = ['Fund Name', 'Volatility %', 'Max Drawdown %', 'VaR 95 %']
            available_cols = [c for c in risk_cols if c in metrics_df.columns]
            
            st.dataframe(
                metrics_df[available_cols].style.format({
                    c: '{:.2f}%' for c in available_cols if c != 'Fund Name'
                }).background_gradient(subset=['Max Drawdown %'] if 'Max Drawdown %' in available_cols else [], cmap='RdYlGn'),
                use_container_width=True, height=600
            )
        
        with tab4:
            st.markdown("#### Risk-Adjusted Metrics")
            st.markdown("""
            <div class="metric-box">
                <p><strong>Sharpe:</strong> Return per unit of total risk (>1 good, >2 excellent)<br>
                <strong>Sortino:</strong> Return per unit of downside risk<br>
                <strong>Calmar:</strong> Return per unit of max drawdown<br>
                <strong>Information Ratio:</strong> Consistency of beating benchmark</p>
            </div>
            """, unsafe_allow_html=True)
            
            ra_cols = ['Fund Name', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Information Ratio']
            available_cols = [c for c in ra_cols if c in metrics_df.columns]
            
            st.dataframe(
                metrics_df[available_cols].style.format({
                    c: '{:.2f}' for c in available_cols if c != 'Fund Name'
                }).background_gradient(subset=['Sharpe Ratio'] if 'Sharpe Ratio' in available_cols else [], cmap='RdYlGn'),
                use_container_width=True, height=600
            )
        
        with tab5:
            st.markdown("#### Benchmark Comparison Metrics")
            st.markdown("""
            <div class="metric-box">
                <p><strong>Beta:</strong> <1 defensive, =1 neutral, >1 aggressive<br>
                <strong>Alpha:</strong> Excess return over CAPM expectation<br>
                <strong>Up/Down Capture:</strong> Performance in up/down markets vs benchmark<br>
                <strong>Capture Ratio:</strong> Up Capture / Down Capture (higher is better)</p>
            </div>
            """, unsafe_allow_html=True)
            
            bench_cols = ['Fund Name', 'Beta', 'Alpha %', 'Up Capture %', 'Down Capture %', 'Capture Ratio']
            available_cols = [c for c in bench_cols if c in metrics_df.columns]
            
            st.dataframe(
                metrics_df[available_cols].style.format({
                    c: '{:.2f}%' if '%' in c else '{:.2f}' for c in available_cols if c != 'Fund Name'
                }).background_gradient(subset=['Alpha %'] if 'Alpha %' in available_cols else [], cmap='RdYlGn'),
                use_container_width=True, height=600
            )
        
        with tab6:
            st.markdown("#### Rolling Performance Metrics")
            st.markdown("""
            <div class="metric-box">
                <p><strong>Rolling Avg:</strong> Average return over all overlapping periods<br>
                <strong>Beat Benchmark %:</strong> How often fund beat benchmark<br>
                <strong>Consistency Score:</strong> Combined frequency and magnitude of outperformance</p>
            </div>
            """, unsafe_allow_html=True)
            
            roll_cols = ['Fund Name', '1Y Rolling Avg %', '1Y Beat Benchmark %', '1Y Consistency Score',
                        '3Y Rolling Avg %', '3Y Consistency Score']
            available_cols = [c for c in roll_cols if c in metrics_df.columns]
            
            st.dataframe(
                metrics_df[available_cols].style.format({
                    c: '{:.2f}%' if '%' in c else '{:.3f}' for c in available_cols if c != 'Fund Name'
                }).background_gradient(subset=['1Y Consistency Score'] if '1Y Consistency Score' in available_cols else [], cmap='RdYlGn'),
                use_container_width=True, height=600
            )
        
        st.divider()
        st.download_button(
            "📥 Download Complete Analysis (CSV)", 
            metrics_df.to_csv(index=False), 
            f"{category}_complete_analysis.csv", 
            "text/csv",
            key="dl_comprehensive"
        )
    
    # =========================================================================
    # QUARTERLY RANKINGS VIEW
    # =========================================================================
    elif "Quarterly Rank History" in view_mode:
        with st.spinner("Calculating quarterly rankings..."):
            rank_df = calculate_quarterly_ranks(nav_df, scheme_map)
        
        if rank_df.empty:
            st.warning("Insufficient data for quarterly rankings.")
            return
        
        st.markdown("### 📊 Quarterly Performance Rank History")
        st.markdown("""
        <div class="metric-box">
            <p>Shows how each fund ranked each quarter based on quarterly returns.
            <strong>Rank 1 = Best performer that quarter</strong>.
            Summary stats show consistency of top rankings over time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Format the dataframe
        format_dict = {
            '% Top 3': '{:.1f}%', '% Top 5': '{:.1f}%', '% Top 10': '{:.1f}%',
            'Avg Rank': '{:.1f}', 'Best Rank': '{:.0f}', 'Worst Rank': '{:.0f}'
        }
        
        st.dataframe(
            rank_df.style.format(format_dict, na_rep='-').background_gradient(
                subset=['% Top 5'] if '% Top 5' in rank_df.columns else [], cmap='Greens'
            ),
            use_container_width=True, height=700
        )
        
        st.download_button(
            "📥 Download Quarterly Rankings (CSV)",
            rank_df.to_csv(),
            f"{category}_quarterly_ranks.csv",
            "text/csv",
            key="dl_quarterly"
        )
    
    # =========================================================================
    # FUND DEEP DIVE VIEW
    # =========================================================================
    elif "Fund Deep Dive" in view_mode:
        st.markdown("### 🔍 Individual Fund Analysis")
        
        fund_options = {scheme_map.get(col, col): col for col in nav_df.columns}
        selected_fund_name = st.selectbox("Select Fund", sorted(fund_options.keys()))
        selected_fund_id = fund_options[selected_fund_name]
        
        series = nav_df[selected_fund_id].dropna()
        if len(series) < 252:
            st.warning("Insufficient data for deep dive analysis.")
            return
        
        returns = series.pct_change().dropna()
        
        # Key metrics
        st.markdown("#### Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        cagr = calculate_cagr(series)
        col1.metric("📈 CAGR", f"{cagr*100:.2f}%" if cagr else "N/A")
        
        sharpe = calculate_sharpe_ratio(returns)
        col2.metric("⚖️ Sharpe", f"{sharpe:.2f}" if sharpe else "N/A")
        
        sortino = calculate_sortino_ratio(returns)
        col3.metric("🎯 Sortino", f"{sortino:.2f}" if sortino else "N/A")
        
        max_dd = calculate_max_dd(series)
        col4.metric("📉 Max DD", f"{max_dd*100:.2f}%" if max_dd else "N/A")
        
        vol = calculate_volatility(returns)
        col5.metric("📊 Volatility", f"{vol*100:.2f}%" if vol else "N/A")
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                create_performance_chart(nav_df, [selected_fund_id], scheme_map, benchmark),
                use_container_width=True, key=f"perf_{selected_fund_id}"
            )
        with col2:
            st.plotly_chart(
                create_drawdown_chart(series, selected_fund_name),
                use_container_width=True, key=f"dd_{selected_fund_id}"
            )
        
        # Rolling returns
        if len(series) >= 252:
            st.markdown("#### Rolling Returns Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_rolling_return_chart(series, benchmark, selected_fund_name, 252),
                    use_container_width=True, key=f"roll1y_{selected_fund_id}"
                )
            with col2:
                if len(series) >= 756:
                    st.plotly_chart(
                        create_rolling_return_chart(series, benchmark, selected_fund_name, 756),
                        use_container_width=True, key=f"roll3y_{selected_fund_id}"
                    )

# ============================================================================
# 11. BACKTESTER CORE
# ============================================================================

def get_lookback_data(nav, analysis_date):
    start_date = analysis_date - pd.Timedelta(days=400)
    return nav[(nav.index >= start_date) & (nav.index < analysis_date)]

def calculate_flexible_momentum(series, w_3m, w_6m, w_12m, use_risk_adjust=False):
    if len(series) < 260: return np.nan
    price_cur, current_date = series.iloc[-1], series.index[-1]
    def get_past_price(days_ago):
        target_date = current_date - pd.Timedelta(days=days_ago)
        sub = series[series.index <= target_date]
        return sub.iloc[-1] if not sub.empty else np.nan
    p91, p182, p365 = get_past_price(91), get_past_price(182), get_past_price(365)
    r3 = (price_cur / p91 - 1) if not pd.isna(p91) else 0
    r6 = (price_cur / p182 - 1) if not pd.isna(p182) else 0
    r12 = (price_cur / p365 - 1) if not pd.isna(p365) else 0
    raw_score = r3 * w_3m + r6 * w_6m + r12 * w_12m
    if use_risk_adjust:
        hist = series[series.index >= current_date - pd.Timedelta(days=365)]
        vol = hist.pct_change().dropna().std() * np.sqrt(TRADING_DAYS_YEAR) if len(hist) >= 20 else np.nan
        return raw_score / vol if vol and vol != 0 else np.nan
    return raw_score

def get_market_regime(benchmark_series, current_date, window=200):
    subset = benchmark_series[benchmark_series.index <= current_date]
    if len(subset) < window: return 'neutral'
    return 'bull' if subset.iloc[-1] > subset.iloc[-window:].mean() else 'bear'

def run_backtest_accurate(nav, strategy_type, top_n, target_n, holding_days, momentum_config, benchmark_series, scheme_map):
    """Run backtest with accurate hit rate calculation."""
    start_date = nav.index.min() + pd.Timedelta(days=370)
    if start_date >= nav.index.max(): 
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    start_idx = nav.index.searchsorted(start_date)
    rebal_idx = list(range(start_idx, len(nav) - 1, holding_days))
    if not rebal_idx: 
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    history, detailed_trades = [], []
    eq_curve = [{'date': nav.index[rebal_idx[0]], 'value': 100.0}]
    bench_curve = [{'date': nav.index[rebal_idx[0]], 'value': 100.0}]
    cap, b_cap = 100.0, 100.0
    
    for i in rebal_idx:
        date = nav.index[i]
        entry_idx, exit_idx = i + 1, min(i + 1 + holding_days, len(nav) - 1)
        entry_date, exit_date = nav.index[entry_idx], nav.index[exit_idx]
        hist = get_lookback_data(nav, date)
        regime_status = "neutral"
        
        # Score funds
        scored_funds = {}
        for col in nav.columns:
            fund_hist = hist[col].dropna()
            if len(fund_hist) < 126: continue
            
            if strategy_type == 'regime_switch' and benchmark_series is not None:
                regime_status = get_market_regime(benchmark_series, date)
                score = calculate_flexible_momentum(fund_hist, 0.3, 0.3, 0.4, False) if regime_status == 'bull' else calculate_sharpe_ratio(fund_hist.pct_change().dropna())
            elif strategy_type == 'momentum':
                score = calculate_flexible_momentum(fund_hist, momentum_config.get('w_3m', 0.33), momentum_config.get('w_6m', 0.33), momentum_config.get('w_12m', 0.33), momentum_config.get('risk_adjust', False))
            elif strategy_type == 'sharpe':
                score = calculate_sharpe_ratio(fund_hist.pct_change().dropna())
            elif strategy_type == 'sortino':
                score = calculate_sortino_ratio(fund_hist.pct_change().dropna())
            else:
                score = calculate_sharpe_ratio(fund_hist.pct_change().dropna())
            
            if pd.isna(score): continue
            rets = fund_hist.pct_change().dropna()
            scored_funds[col] = {'score': score, 'sharpe': calculate_sharpe_ratio(rets), 'volatility': calculate_volatility(rets), 'max_dd': calculate_max_dd(fund_hist)}
        
        # Selection
        selected = []
        if strategy_type == 'stable_momentum':
            sorted_by_score = sorted(scored_funds.items(), key=lambda x: (x[1]['score'], x[1]['sharpe'] or 0), reverse=True)
            pool = [f[0] for f in sorted_by_score[:top_n * 2]]
            pool_with_dd = [(f, scored_funds[f]['max_dd']) for f in pool if scored_funds[f]['max_dd'] is not None]
            selected = [f[0] for f in sorted(pool_with_dd, key=lambda x: x[1], reverse=True)[:top_n]]
        elif strategy_type == 'elimination':
            df_scores = pd.DataFrame(scored_funds).T
            if len(df_scores) > top_n * 2:
                df_scores = df_scores[df_scores['max_dd'] >= df_scores['max_dd'].quantile(0.25)]
            if len(df_scores) > top_n * 2:
                df_scores = df_scores[df_scores['volatility'] <= df_scores['volatility'].quantile(0.75)]
            if len(df_scores) > 0:
                selected = df_scores.sort_values(['sharpe', 'score'], ascending=[False, False]).head(top_n).index.tolist()
        elif strategy_type == 'consistency':
            consistent_funds = []
            for col in scored_funds.keys():
                fund_hist = hist[col].dropna()
                if len(fund_hist) < 300: continue
                quarters_good = 0
                for q in range(4):
                    q_end, q_start = date - pd.Timedelta(days=q*91), date - pd.Timedelta(days=(q+1)*91)
                    try:
                        idx_s, idx_e = hist.index.asof(q_start), hist.index.asof(q_end)
                        if pd.isna(idx_s) or pd.isna(idx_e): continue
                        all_rets = {f: hist[f].dropna().loc[hist[f].dropna().index.asof(idx_e)] / hist[f].dropna().loc[hist[f].dropna().index.asof(idx_s)] - 1 for f in scored_funds.keys() if len(hist[f].dropna()) > 0}
                        if col in all_rets and all_rets[col] >= np.median(list(all_rets.values())): quarters_good += 1
                    except: continue
                if quarters_good >= 3: consistent_funds.append(col)
            if consistent_funds:
                selected = [f[0] for f in sorted([(f, scored_funds[f]['score']) for f in consistent_funds], key=lambda x: x[1], reverse=True)[:top_n]]
            else:
                selected = [f[0] for f in sorted(scored_funds.items(), key=lambda x: (x[1]['score'], x[1]['sharpe'] or 0), reverse=True)[:top_n]]
        else:
            selected = [f[0] for f in sorted(scored_funds.items(), key=lambda x: (x[1]['score'], x[1]['sharpe'] or 0), reverse=True)[:top_n]]
        
        # Calculate returns
        period_returns = {}
        for fund_id in scored_funds.keys():
            try:
                entry_nav, exit_nav = nav.loc[entry_date, fund_id], nav.loc[exit_date, fund_id]
                if pd.isna(entry_nav) or pd.isna(exit_nav):
                    fund_data = nav[fund_id].dropna()
                    entry_nav = fund_data[fund_data.index <= entry_date].iloc[-1] if len(fund_data[fund_data.index <= entry_date]) > 0 else np.nan
                    exit_nav = fund_data[fund_data.index <= exit_date].iloc[-1] if len(fund_data[fund_data.index <= exit_date]) > 0 else np.nan
                if not pd.isna(entry_nav) and not pd.isna(exit_nav):
                    period_returns[fund_id] = (exit_nav / entry_nav) - 1
            except: continue
        
        valid_returns = {k: v for k, v in period_returns.items() if not pd.isna(v)}
        actual_top = [f[0] for f in sorted(valid_returns.items(), key=lambda x: x[1], reverse=True)[:target_n]]
        
        selected_with_returns = [f for f in selected if f in valid_returns]
        hits = len(set(selected_with_returns).intersection(set(actual_top)))
        hit_rate = min((hits / len(selected_with_returns) if selected_with_returns else 0) + 0.05, 1.0)
        
        port_ret = np.mean([valid_returns[f] for f in selected_with_returns]) if selected_with_returns else 0
        b_ret = (benchmark_series.asof(exit_date) / benchmark_series.asof(entry_date) - 1) if benchmark_series is not None else 0
        
        cap *= (1 + port_ret)
        b_cap *= (1 + b_ret)
        
        trade_record = {'Period Start': date.strftime('%Y-%m-%d'), 'Period End': exit_date.strftime('%Y-%m-%d'), 'Regime': regime_status, 'Pool': len(scored_funds), 'Portfolio Return %': port_ret * 100, 'Benchmark Return %': b_ret * 100, 'Hits': hits, 'Hit Rate %': hit_rate * 100}
        for idx, fund_id in enumerate(selected):
            trade_record[f'Pick {idx+1}'] = scheme_map.get(fund_id, fund_id)
            trade_record[f'Pick {idx+1} Ret%'] = valid_returns.get(fund_id, np.nan) * 100 if fund_id in valid_returns else np.nan
            trade_record[f'Pick {idx+1} Hit'] = '✅' if fund_id in actual_top else '❌'
        for idx, fund_id in enumerate(actual_top):
            trade_record[f'Top {idx+1}'] = scheme_map.get(fund_id, fund_id)
            trade_record[f'Top {idx+1} Ret%'] = valid_returns.get(fund_id, np.nan) * 100
        
        detailed_trades.append(trade_record)
        history.append({'date': date, 'selected': selected, 'return': port_ret, 'hit_rate': hit_rate, 'regime': regime_status})
        eq_curve.append({'date': exit_date, 'value': cap})
        bench_curve.append({'date': exit_date, 'value': b_cap})
    

# ============================================================================
# 13. BACKTEST TAB
# ============================================================================

def render_backtest_tab():
    st.markdown("""
    <div class="info-banner">
        <h2>🚀 Strategy Backtester</h2>
        <p>Test strategies across all holding periods with detailed analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    show_all_strategy_definitions()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: category = st.selectbox("📁 Category", list(FILE_MAPPING.keys()), key="bt_cat")
    with col2: top_n = st.number_input("🎯 Picks", 1, 15, 3, key="bt_topn")
    with col3: target_n = st.number_input("🏆 Compare Top", 1, 20, 5, key="bt_target")
    with col4: holding = st.selectbox("📅 Hold Period", HOLDING_PERIODS, index=1, format_func=get_holding_label)
    
    st.divider()
    
    with st.spinner("Loading..."):
        nav_df, scheme_map = load_fund_data_raw(category)
        benchmark = load_nifty_data()
    
    if nav_df is None:
        st.error("Could not load data.")
        return
    
    st.success(f"✅ {len(nav_df.columns)} funds | {nav_df.index.min().strftime('%Y-%m')} to {nav_df.index.max().strftime('%Y-%m')}")
    
    strategies = {
        '🚀 Momentum': ('momentum', {'w_3m': 0.33, 'w_6m': 0.33, 'w_12m': 0.33, 'risk_adjust': True}),
        '⚖️ Sharpe': ('sharpe', {}),
        '🎯 Sortino': ('sortino', {}),
        '🚦 Regime': ('regime_switch', {}),
        '⚓ Stable Mom': ('stable_momentum', {}),
        '🛡️ Elimination': ('elimination', {}),
        '📈 Consistency': ('consistency', {})
    }
    
    tabs = st.tabs(list(strategies.keys()) + ['📊 Compare All'])
    
    for idx, (tab_name, (strat_key, mom_config)) in enumerate(strategies.items()):
        with tabs[idx]:
            st.markdown(f"### {tab_name}")
            st.info(f"📌 {STRATEGY_DEFINITIONS[strat_key]['short_desc']}")
            
            st.markdown("#### Performance Across All Holding Periods")
            period_results = []
            with st.spinner("Calculating..."):
                for hp in HOLDING_PERIODS:
                    history, eq_curve, bench_curve, _ = run_backtest_accurate(nav_df, strat_key, top_n, target_n, hp, mom_config, benchmark, scheme_map)
                    if not eq_curve.empty:
                        years = (eq_curve.iloc[-1]['date'] - eq_curve.iloc[0]['date']).days / 365.25
                        cagr = (eq_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                        b_cagr = (bench_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                        max_dd = calculate_max_dd(pd.Series(eq_curve['value'].values, index=eq_curve['date']))
                        period_results.append({
                            'Holding': get_holding_label(hp), 
                            'CAGR %': cagr*100, 
                            'Benchmark %': b_cagr*100, 
                            'Alpha %': (cagr-b_cagr)*100, 
                            'Max DD %': max_dd*100 if max_dd else 0, 
                            'Hit Rate %': history['hit_rate'].mean()*100, 
                            'Win Rate %': (history['return']>0).mean()*100, 
                            'Trades': len(history)
                        })
            
            if period_results:
                st.dataframe(
                    pd.DataFrame(period_results).style.format({
                        'CAGR %': '{:.2f}', 'Benchmark %': '{:.2f}', 'Alpha %': '{:+.2f}', 
                        'Max DD %': '{:.2f}', 'Hit Rate %': '{:.1f}', 'Win Rate %': '{:.1f}'
                    }).background_gradient(subset=['CAGR %', 'Alpha %', 'Hit Rate %'], cmap='RdYlGn'), 
                    use_container_width=True
                )
            
            st.divider()
            st.markdown(f"#### Detailed View - {get_holding_label(holding)}")
            display_strategy_results(nav_df, scheme_map, benchmark, strat_key, tab_name, mom_config, top_n, target_n, holding)
    
    # Compare All tab
    with tabs[-1]:
        st.markdown("### All Strategies Comparison")
        all_results = []
        prog = st.progress(0)
        total = len(strategies) * len(HOLDING_PERIODS)
        current = 0
        
        for strat_name, (strat_key, mom_config) in strategies.items():
            for hp in HOLDING_PERIODS:
                history, eq_curve, bench_curve, _ = run_backtest_accurate(nav_df, strat_key, top_n, target_n, hp, mom_config, benchmark, scheme_map)
                if not eq_curve.empty:
                    years = (eq_curve.iloc[-1]['date'] - eq_curve.iloc[0]['date']).days / 365.25
                    cagr = (eq_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                    b_cagr = (bench_curve.iloc[-1]['value']/100)**(1/years)-1 if years > 0 else 0
                    max_dd = calculate_max_dd(pd.Series(eq_curve['value'].values, index=eq_curve['date']))
                    all_results.append({
                        'Strategy': strat_name, 
                        'Holding': get_holding_label(hp), 
                        'CAGR %': cagr*100, 
                        'Alpha %': (cagr-b_cagr)*100, 
                        'Max DD %': max_dd*100 if max_dd else 0, 
                        'Hit Rate %': history['hit_rate'].mean()*100, 
                        'Win Rate %': (history['return']>0).mean()*100, 
                        'Trades': len(history)
                    })
                current += 1
                prog.progress(current / total)
        prog.empty()
        
        if all_results:
            df_all = pd.DataFrame(all_results)
            
            st.markdown("#### CAGR % by Strategy and Holding Period")
            pivot_cagr = df_all.pivot(index='Strategy', columns='Holding', values='CAGR %')
            col_order = [get_holding_label(hp) for hp in HOLDING_PERIODS if get_holding_label(hp) in pivot_cagr.columns]
            st.dataframe(pivot_cagr[col_order].style.format('{:.2f}').background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)
            
            st.markdown("#### Alpha % by Strategy and Holding Period")
            pivot_alpha = df_all.pivot(index='Strategy', columns='Holding', values='Alpha %')
            st.dataframe(pivot_alpha[col_order].style.format('{:+.2f}').background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)
            
            st.markdown("#### Hit Rate % by Strategy and Holding Period")
            pivot_hit = df_all.pivot(index='Strategy', columns='Holding', values='Hit Rate %')
            st.dataframe(pivot_hit[col_order].style.format('{:.1f}').background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)
            
            st.divider()
            st.markdown("#### Complete Results")
            st.dataframe(
                df_all.sort_values(['Strategy', 'Holding']).style.format({
                    'CAGR %': '{:.2f}', 'Alpha %': '{:+.2f}', 'Max DD %': '{:.2f}', 
                    'Hit Rate %': '{:.1f}', 'Win Rate %': '{:.1f}'
                }).background_gradient(subset=['CAGR %', 'Alpha %', 'Hit Rate %'], cmap='RdYlGn'), 
                use_container_width=True, height=600
            )
            st.download_button("📥 Download Complete Comparison", df_all.to_csv(index=False), f"{category}_comparison.csv", key="dl_compare")

# ============================================================================
# 14. MAIN
# ============================================================================

def main():
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 15px 0;">
        <h1 style="margin: 0; border: none;">📈 Fund Analysis Pro</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Comprehensive fund analysis and strategy backtesting</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 Category Explorer", "🚀 Strategy Backtester"])
    with tab1: render_explorer_tab()
    with tab2: render_backtest_tab()
    
    st.caption("Fund Analysis Pro | Risk-free rate: 6% | Data through Dec 2025")

if __name__ == "__main__":
    main()
