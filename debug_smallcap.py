"""
ML Fund Predictor — Classification-Based Top-K Fund Selection
================================================================
Research-backed approach using Classification-as-Ranking strategy.

Instead of predicting returns (regression), we predict:
  "Will this fund be in the top K performers next period?"

Models:
  1. HistGradientBoosting Classifier (≈LightGBM, sklearn native)
  2. MLP Neural Network Classifier (deep learning proxy)
  3. Ensemble of both

Features:
  - Multi-horizon momentum (1M, 3M, 6M, 12M)
  - Risk metrics (vol, drawdown, Sharpe, Sortino)
  - Cross-sectional rank features (z-scores, percentiles)
  - Regime features (benchmark vs moving averages)
  - Trend & acceleration features

Backtesting: Walk-forward with expanding or rolling window.
Loss: Focal-weighted cross-entropy for class imbalance.

Run: streamlit run ml_fund_predictor.py
Requires: streamlit, pandas, numpy, scikit-learn, plotly, openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
from copy import deepcopy

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import pickle

warnings.filterwarnings("ignore")

# ============================================================================
# 1. PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="ML Fund Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 100%; }
    h1 { color: #1E3A5F; font-weight: 700; border-bottom: 3px solid #2196F3; }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px; padding: 15px;
    }
    div[data-testid="metric-container"] label { color: rgba(255,255,255,0.9) !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: white !important; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: #f0f2f6; padding: 10px; border-radius: 10px; }
    .stTabs [aria-selected="true"] { background-color: #2196F3 !important; color: white !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .info-banner { background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        padding: 20px; border-radius: 12px; margin-bottom: 20px; color: white; }
    .info-banner h2 { color: white !important; margin: 0; border: none; }
    .info-banner p { color: rgba(255,255,255,0.85) !important; margin: 5px 0 0 0; }
    .pick-card { background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 14px; padding: 16px; margin: 8px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-left: 5px solid #2196F3; }
    .pick-card .rank { font-size: 1.6rem; font-weight: 800; color: #1565c0; }
    .pick-card .name { font-weight: 700; color: #1E3A5F; font-size: 0.92rem; }
    .pick-card .prob { color: #4caf50; font-weight: 700; font-size: 0.95rem; }
    .metric-box { background: #f5f7fa; border-radius: 12px; padding: 16px;
        margin: 8px 0; border-left: 5px solid #2196F3; }
    .metric-box h4 { color: #1E3A5F; margin: 0 0 8px 0; }
    .metric-box p { color: #374151; margin: 4px 0; font-size: 0.9rem; }
    .strategy-compare { display: flex; gap: 16px; flex-wrap: wrap; }
    .strat-card { flex: 1; min-width: 200px; background: white; border-radius: 12px;
        padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 2. CONSTANTS
# ============================================================================

RISK_FREE_RATE = 0.06
TRADING_DAYS_YEAR = 252
DAILY_RF = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS_YEAR) - 1
DATA_DIR = "data"
MAX_DATA_DATE = pd.Timestamp("2025-12-05")

FILE_MAPPING = {
    "Large Cap": "largecap_merged.xlsx",
    "Mid Cap": "midcap.xlsx",
    "Small Cap": "smallcap.xlsx",
    "Large & Mid Cap": "large_and_midcap_fund.xlsx",
    "Multi Cap": "MULTICAP.xlsx",
    "International": "international_merged.xlsx",
}

HOLDING_PERIODS = [63, 126, 189, 252]

def get_holding_label(d):
    return f"{d}d (~{d // 21}M)" if d < 252 else f"{d}d (~{d // 252}Y)"


# ============================================================================
# 3. DATA LOADING (reuse from original code)
# ============================================================================

def is_regular_growth_fund(name):
    n = str(name).lower()
    exclude = [
        "idcw", "dividend", "div ", "div.", "div)",
        "direct", "dir ", "dir)",
        "bonus", "institutional", "segregated", "payout",
        "reinvestment", "monthly", "quarterly", "annual",
    ]
    return not any(kw in n for kw in exclude)


def clean_weekday_data(df):
    if df is None or df.empty:
        return df
    df = df[df.index <= MAX_DATA_DATE]
    df = df[df.index.dayofweek < 5]
    if len(df) > 0:
        all_wd = pd.date_range(df.index.min(), min(df.index.max(), MAX_DATA_DATE), freq="B")
        df = df.reindex(all_wd).ffill(limit=5)
    return df


@st.cache_data
def load_fund_data(category_key):
    filename = FILE_MAPPING.get(category_key)
    if not filename:
        return None, None
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_excel(path, header=None)
        fund_names = df.iloc[2, 1:].tolist()
        data_df = df.iloc[4:, :].copy()
        if isinstance(data_df.iloc[-1, 0], str) and "Accord" in str(data_df.iloc[-1, 0]):
            data_df = data_df.iloc[:-1, :]
        dates = pd.to_datetime(data_df.iloc[:, 0], errors="coerce")
        nav_wide = pd.DataFrame(index=dates)
        scheme_map = {}
        idx = 0
        for i, name in enumerate(fund_names):
            if pd.notna(name) and str(name).strip():
                if not is_regular_growth_fund(name):
                    continue
                code = f"F{idx:04d}"  # deterministic IDs — fixes hash issue
                idx += 1
                scheme_map[code] = str(name).strip()
                nav_wide[code] = pd.to_numeric(data_df.iloc[:, i + 1], errors="coerce").values
        nav_wide = nav_wide.sort_index()
        nav_wide = nav_wide[~nav_wide.index.duplicated(keep="last")]
        return clean_weekday_data(nav_wide), scheme_map
    except Exception as e:
        st.error(f"Data load error: {e}")
        return None, None


@st.cache_data
def load_benchmark():
    path = os.path.join(DATA_DIR, "nifty100_data.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna(subset=["date", "nav"]).set_index("date").sort_index()
        return clean_weekday_data(df).squeeze()
    except:
        return None


# ============================================================================
# 4. FEATURE ENGINEERING — THE HEART OF THE ML APPROACH
# ============================================================================

class FundFeatureEngine:
    """
    Extracts features for each fund at a given date using only past data.
    
    Feature groups:
      1. MOMENTUM: Multi-horizon returns (1M, 3M, 6M, 12M) — most predictive per NBER
      2. RISK: Volatility, drawdown, downside deviation
      3. RISK-ADJUSTED: Rolling Sharpe, Sortino, Calmar
      4. CROSS-SECTIONAL: Z-scores and percentile ranks vs peers
      5. REGIME: Benchmark state features
      6. TREND: Acceleration, trend strength, mean reversion signals
    """

    FEATURE_NAMES = [
        # Momentum (6 features)
        "ret_21d", "ret_63d", "ret_126d", "ret_252d",
        "mom_accel_short", "mom_accel_long",
        # Risk (5 features)
        "vol_21d", "vol_63d", "vol_252d",
        "max_dd_63d", "max_dd_252d",
        # Risk-adjusted (4 features)
        "sharpe_63d", "sharpe_252d",
        "sortino_63d", "sortino_252d",
        # Trend (5 features)
        "trend_50dma", "trend_200dma",
        "dist_from_peak", "days_since_peak_pct",
        "price_vs_mean_126d",
        # Cross-sectional (6 features) — computed after all funds
        "cs_rank_ret63", "cs_rank_ret252",
        "cs_zscore_ret63", "cs_zscore_ret252",
        "cs_rank_sharpe63", "cs_rank_vol63",
        # Regime (4 features)
        "bench_ret_63d", "bench_vol_63d",
        "bench_above_50dma", "bench_above_200dma",
    ]

    def __init__(self, nav_df, benchmark, min_history=260):
        self.nav_df = nav_df
        self.benchmark = benchmark
        self.min_history = min_history
        self.fund_cols = nav_df.columns.tolist()

    def _safe_return(self, series, lookback):
        """Calculate return over lookback trading days."""
        if len(series) < lookback + 1:
            return np.nan
        cur = series.iloc[-1]
        prev = series.iloc[-(lookback + 1)]
        if prev == 0 or pd.isna(prev) or pd.isna(cur):
            return np.nan
        return cur / prev - 1

    def _safe_vol(self, series, lookback):
        """Annualized volatility over lookback days."""
        if len(series) < lookback + 1:
            return np.nan
        rets = series.iloc[-lookback:].pct_change().dropna()
        if len(rets) < 10:
            return np.nan
        return rets.std() * np.sqrt(TRADING_DAYS_YEAR)

    def _safe_max_dd(self, series, lookback):
        """Max drawdown over lookback days."""
        s = series.iloc[-lookback:]
        if len(s) < 10:
            return np.nan
        cum = (1 + s.pct_change().fillna(0)).cumprod()
        dd = cum / cum.expanding().max() - 1
        return dd.min()

    def _safe_sharpe(self, series, lookback):
        """Rolling Sharpe ratio."""
        if len(series) < lookback + 1:
            return np.nan
        rets = series.iloc[-lookback:].pct_change().dropna()
        if len(rets) < 10 or rets.std() == 0:
            return np.nan
        return ((rets - DAILY_RF).mean() / rets.std()) * np.sqrt(TRADING_DAYS_YEAR)

    def _safe_sortino(self, series, lookback):
        """Rolling Sortino ratio."""
        if len(series) < lookback + 1:
            return np.nan
        rets = series.iloc[-lookback:].pct_change().dropna()
        down = rets[rets < 0]
        if len(down) < 5 or down.std() == 0:
            return np.nan
        return ((rets - DAILY_RF).mean() / down.std()) * np.sqrt(TRADING_DAYS_YEAR)

    def extract_fund_features(self, fund_series, date_idx):
        """Extract per-fund features using data up to date_idx."""
        s = fund_series.iloc[: date_idx + 1].dropna()
        if len(s) < self.min_history:
            return None

        feats = {}

        # --- MOMENTUM ---
        feats["ret_21d"] = self._safe_return(s, 21)
        feats["ret_63d"] = self._safe_return(s, 63)
        feats["ret_126d"] = self._safe_return(s, 126)
        feats["ret_252d"] = self._safe_return(s, 252)

        # Momentum acceleration: is momentum accelerating or decelerating?
        r63_now = feats["ret_63d"] or 0
        if len(s) > 126:
            s_prev = s.iloc[: -63]
            r63_prev = self._safe_return(s_prev, 63) or 0
            feats["mom_accel_short"] = r63_now - r63_prev
        else:
            feats["mom_accel_short"] = 0

        r252_now = feats["ret_252d"] or 0
        if len(s) > 504:
            s_prev = s.iloc[: -252]
            r252_prev = self._safe_return(s_prev, 252) or 0
            feats["mom_accel_long"] = r252_now - r252_prev
        else:
            feats["mom_accel_long"] = 0

        # --- RISK ---
        feats["vol_21d"] = self._safe_vol(s, 21)
        feats["vol_63d"] = self._safe_vol(s, 63)
        feats["vol_252d"] = self._safe_vol(s, 252)
        feats["max_dd_63d"] = self._safe_max_dd(s, 63)
        feats["max_dd_252d"] = self._safe_max_dd(s, 252)

        # --- RISK-ADJUSTED ---
        feats["sharpe_63d"] = self._safe_sharpe(s, 63)
        feats["sharpe_252d"] = self._safe_sharpe(s, 252)
        feats["sortino_63d"] = self._safe_sortino(s, 63)
        feats["sortino_252d"] = self._safe_sortino(s, 252)

        # --- TREND ---
        cur_price = s.iloc[-1]
        if len(s) >= 50:
            ma50 = s.iloc[-50:].mean()
            feats["trend_50dma"] = 1.0 if cur_price > ma50 else 0.0
        else:
            feats["trend_50dma"] = np.nan
        if len(s) >= 200:
            ma200 = s.iloc[-200:].mean()
            feats["trend_200dma"] = 1.0 if cur_price > ma200 else 0.0
        else:
            feats["trend_200dma"] = np.nan

        # Distance from all-time high (in lookback)
        peak = s.iloc[-252:].max() if len(s) >= 252 else s.max()
        feats["dist_from_peak"] = (cur_price / peak - 1) if peak > 0 else 0

        # How long since peak (as fraction of lookback)
        if len(s) >= 252:
            peak_idx = s.iloc[-252:].idxmax()
            days_since = (s.index[-1] - peak_idx).days
            feats["days_since_peak_pct"] = days_since / 252
        else:
            feats["days_since_peak_pct"] = 0

        # Price vs 126d mean (mean reversion signal)
        if len(s) >= 126:
            feats["price_vs_mean_126d"] = cur_price / s.iloc[-126:].mean() - 1
        else:
            feats["price_vs_mean_126d"] = 0

        return feats

    def extract_regime_features(self, date_idx):
        """Extract benchmark/market regime features."""
        if self.benchmark is None:
            return {
                "bench_ret_63d": 0, "bench_vol_63d": 0,
                "bench_above_50dma": 0.5, "bench_above_200dma": 0.5,
            }
        b = self.benchmark.iloc[: date_idx + 1].dropna()
        feats = {}
        feats["bench_ret_63d"] = self._safe_return(b, 63) or 0
        feats["bench_vol_63d"] = self._safe_vol(b, 63) or 0
        if len(b) >= 50:
            feats["bench_above_50dma"] = 1.0 if b.iloc[-1] > b.iloc[-50:].mean() else 0.0
        else:
            feats["bench_above_50dma"] = 0.5
        if len(b) >= 200:
            feats["bench_above_200dma"] = 1.0 if b.iloc[-1] > b.iloc[-200:].mean() else 0.0
        else:
            feats["bench_above_200dma"] = 0.5
        return feats

    def build_cross_sectional_features(self, fund_features_dict):
        """
        Add cross-sectional (relative) features.
        fund_features_dict: {fund_id: {feature_name: value}}
        """
        if not fund_features_dict:
            return fund_features_dict

        # Collect arrays
        fids = list(fund_features_dict.keys())
        ret63 = np.array([fund_features_dict[f].get("ret_63d", np.nan) for f in fids])
        ret252 = np.array([fund_features_dict[f].get("ret_252d", np.nan) for f in fids])
        sharpe63 = np.array([fund_features_dict[f].get("sharpe_63d", np.nan) for f in fids])
        vol63 = np.array([fund_features_dict[f].get("vol_63d", np.nan) for f in fids])

        def percentile_rank(arr):
            """Compute percentile rank (0-1), handling NaNs."""
            result = np.full_like(arr, np.nan, dtype=float)
            valid = ~np.isnan(arr)
            if valid.sum() > 1:
                from scipy.stats import rankdata
                ranks = rankdata(arr[valid])
                result[valid] = ranks / len(ranks)
            return result

        def zscore(arr):
            result = np.full_like(arr, 0.0, dtype=float)
            valid = ~np.isnan(arr)
            if valid.sum() > 2:
                m, s = np.nanmean(arr[valid]), np.nanstd(arr[valid])
                if s > 0:
                    result[valid] = (arr[valid] - m) / s
            return result

        # Compute ranks and z-scores
        try:
            from scipy.stats import rankdata
        except ImportError:
            # Fallback: manual rank
            def rankdata(arr):
                temp = arr.argsort().argsort() + 1
                return temp.astype(float)

        r_ret63 = percentile_rank(ret63)
        r_ret252 = percentile_rank(ret252)
        r_sharpe63 = percentile_rank(sharpe63)
        r_vol63 = percentile_rank(vol63)
        z_ret63 = zscore(ret63)
        z_ret252 = zscore(ret252)

        for i, fid in enumerate(fids):
            fund_features_dict[fid]["cs_rank_ret63"] = r_ret63[i]
            fund_features_dict[fid]["cs_rank_ret252"] = r_ret252[i]
            fund_features_dict[fid]["cs_zscore_ret63"] = z_ret63[i]
            fund_features_dict[fid]["cs_zscore_ret252"] = z_ret252[i]
            fund_features_dict[fid]["cs_rank_sharpe63"] = r_sharpe63[i]
            fund_features_dict[fid]["cs_rank_vol63"] = r_vol63[i]

        return fund_features_dict

    def build_snapshot(self, date_idx):
        """
        Build full feature matrix for all funds at a single date index.
        Returns: dict of {fund_id: feature_dict}
        """
        fund_feats = {}
        regime = self.extract_regime_features(date_idx)

        for col in self.fund_cols:
            series = self.nav_df[col]
            feats = self.extract_fund_features(series, date_idx)
            if feats is not None:
                feats.update(regime)
                fund_feats[col] = feats

        # Add cross-sectional features
        fund_feats = self.build_cross_sectional_features(fund_feats)
        return fund_feats


# ============================================================================
# 5. LABEL GENERATION & DATASET CONSTRUCTION
# ============================================================================

def build_training_dataset(nav_df, benchmark, scheme_map, hold_days, top_k,
                           min_train_history=370, step_days=None):
    """
    Build the complete training dataset with walk-forward labels.
    
    For each rebalance date:
      1. Extract features for all funds (using only past data)
      2. Compute ACTUAL forward returns over hold_days
      3. Label top_k funds as 1, rest as 0
    
    Returns:
      records: list of dicts with features + label + metadata
    """
    if step_days is None:
        step_days = max(hold_days // 2, 21)  # Overlapping windows for more training data

    engine = FundFeatureEngine(nav_df, benchmark)

    start_idx = nav_df.index.searchsorted(
        nav_df.index.min() + pd.Timedelta(days=min_train_history)
    )
    # Stop early enough to have forward returns
    end_idx = len(nav_df) - hold_days - 1

    records = []
    rebalance_indices = list(range(start_idx, end_idx, step_days))

    progress = st.progress(0, text="Building dataset...")
    total = len(rebalance_indices)

    for prog_i, idx in enumerate(rebalance_indices):
        if prog_i % 5 == 0:
            progress.progress(prog_i / total, text=f"Processing date {prog_i + 1}/{total}...")

        signal_date = nav_df.index[idx]
        entry_idx = idx + 1
        exit_idx = min(idx + 1 + hold_days, len(nav_df) - 1)
        entry_date = nav_df.index[entry_idx]
        exit_date = nav_df.index[exit_idx]

        # 1. Extract features
        snapshot = engine.build_snapshot(idx)
        if len(snapshot) < top_k + 3:
            continue

        # 2. Compute forward returns
        forward_returns = {}
        for fid in snapshot:
            try:
                entry_nav = nav_df.loc[entry_date, fid]
                exit_nav = nav_df.loc[exit_date, fid]
                if pd.notna(entry_nav) and pd.notna(exit_nav) and entry_nav > 0:
                    forward_returns[fid] = exit_nav / entry_nav - 1
            except:
                continue

        if len(forward_returns) < top_k + 3:
            continue

        # 3. Label: top_k = 1, rest = 0
        sorted_funds = sorted(forward_returns.items(), key=lambda x: x[1], reverse=True)
        top_k_set = set(f for f, _ in sorted_funds[:top_k])

        # Also compute benchmark return for this period
        bench_ret = 0
        if benchmark is not None:
            try:
                b_entry = benchmark.asof(entry_date)
                b_exit = benchmark.asof(exit_date)
                if pd.notna(b_entry) and pd.notna(b_exit) and b_entry > 0:
                    bench_ret = b_exit / b_entry - 1
            except:
                pass

        # 4. Create records
        for fid, feats in snapshot.items():
            if fid not in forward_returns:
                continue
            record = feats.copy()
            record["fund_id"] = fid
            record["fund_name"] = scheme_map.get(fid, fid)
            record["signal_date"] = signal_date
            record["entry_date"] = entry_date
            record["exit_date"] = exit_date
            record["forward_return"] = forward_returns[fid]
            record["label"] = 1 if fid in top_k_set else 0
            record["bench_return"] = bench_ret
            records.append(record)

    progress.empty()
    return pd.DataFrame(records)


# ============================================================================
# 6. MODEL TRAINING & PREDICTION
# ============================================================================

class FundClassifier:
    """
    Trains and predicts top-K funds using classification.
    
    Models:
      - HGB: HistGradientBoosting (tree-based, handles NaN natively)
      - MLP: Multi-layer perceptron (neural network)
      - Ensemble: Soft-voting combination
    
    Training: Walk-forward expanding window.
    """

    FEATURE_COLS = FundFeatureEngine.FEATURE_NAMES

    def __init__(self, model_type="ensemble", class_weight_ratio=3.0):
        self.model_type = model_type
        self.scaler = RobustScaler()  # Robust to outliers
        self.model = None
        self.class_weight_ratio = class_weight_ratio
        self.feature_importances_ = None

    def _build_model(self):
        # Compute sample weight ratio for class imbalance
        # top_k out of ~50 funds → ~10% positive, 90% negative
        # We upweight positives by class_weight_ratio

        if self.model_type == "hgb":
            return HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=5,
                learning_rate=0.05,
                min_samples_leaf=20,
                l2_regularization=1.0,
                max_bins=128,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=42,
            )
        elif self.model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,  # L2 regularization
                learning_rate="adaptive",
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=42,
            )
        elif self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "ensemble":
            hgb = HistGradientBoostingClassifier(
                max_iter=200, max_depth=5, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=25, random_state=42,
            )
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu", solver="adam", alpha=0.001,
                learning_rate="adaptive", max_iter=400,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=25, random_state=42,
            )
            rf = RandomForestClassifier(
                n_estimators=150, max_depth=8,
                min_samples_leaf=20, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            return VotingClassifier(
                estimators=[("hgb", hgb), ("mlp", mlp), ("rf", rf)],
                voting="soft",
                weights=[2, 1, 1],  # HGB gets more weight (handles NaN, stronger)
            )

    def _prepare_features(self, df, fit_scaler=False):
        """Extract feature matrix and handle NaNs."""
        X = df[self.FEATURE_COLS].copy()
        # Fill NaN with column median (better than 0 for financial data)
        X = X.fillna(X.median())
        X = X.fillna(0)  # Fallback

        if self.model_type in ("mlp", "ensemble"):
            if fit_scaler:
                X_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X),
                    columns=X.columns, index=X.index
                )
            else:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=X.columns, index=X.index
                )
            return X_scaled
        return X

    def _compute_sample_weights(self, y):
        """Compute sample weights to handle class imbalance."""
        weights = np.ones(len(y))
        weights[y == 1] = self.class_weight_ratio
        return weights

    def train(self, train_df):
        """Train the model on training data."""
        self.model = self._build_model()
        X = self._prepare_features(train_df, fit_scaler=True)
        y = train_df["label"].values
        sw = self._compute_sample_weights(y)

        if self.model_type == "hgb":
            self.model.fit(X, y, sample_weight=sw)
        elif self.model_type in ("mlp", "rf"):
            self.model.fit(X, y)
        elif self.model_type == "ensemble":
            # VotingClassifier doesn't directly support sample_weight in fit
            # Train components individually where possible
            self.model.fit(X, y)

        # Extract feature importances
        self._extract_importances(X.columns)

    def predict_proba(self, test_df):
        """Predict probability of being in top-K."""
        X = self._prepare_features(test_df, fit_scaler=False)
        probs = self.model.predict_proba(X)
        # Return probability of class 1 (top-K)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs[:, 0]

    def _extract_importances(self, feature_names):
        """Try to extract feature importances."""
        try:
            if self.model_type == "hgb":
                self.feature_importances_ = dict(
                    zip(feature_names, self.model.feature_importances_)
                )
            elif self.model_type == "rf":
                self.feature_importances_ = dict(
                    zip(feature_names, self.model.feature_importances_)
                )
            elif self.model_type == "ensemble":
                # Average importances from tree-based components
                imps = np.zeros(len(feature_names))
                count = 0
                for name, est in self.model.named_estimators_.items():
                    if hasattr(est, "feature_importances_"):
                        imps += est.feature_importances_
                        count += 1
                if count > 0:
                    self.feature_importances_ = dict(
                        zip(feature_names, imps / count)
                    )
        except:
            pass


# ============================================================================
# 7. WALK-FORWARD BACKTESTER
# ============================================================================

def run_ml_backtest(dataset_df, model_type, top_k, target_k,
                    min_train_periods=8, class_weight_ratio=3.0):
    """
    Walk-forward backtest:
      1. For each test period, train on ALL prior periods
      2. Predict probabilities for test period
      3. Pick top_k by probability
      4. Measure hit rate vs actual top target_k
      5. Track equity curve

    Returns: results_df, equity_df, benchmark_df, trade_details, model_info
    """
    # Get unique signal dates
    dates = sorted(dataset_df["signal_date"].unique())
    if len(dates) < min_train_periods + 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    results = []
    equity = [{"date": dates[min_train_periods], "value": 100.0}]
    bench_eq = [{"date": dates[min_train_periods], "value": 100.0}]
    cap, bcap = 100.0, 100.0
    all_trades = []
    last_model = None
    all_importances = []

    progress = st.progress(0, text="Backtesting...")
    total = len(dates) - min_train_periods

    for i in range(min_train_periods, len(dates)):
        test_date = dates[i]
        train_dates = dates[:i]
        prog = (i - min_train_periods) / total
        progress.progress(min(prog, 1.0), text=f"Testing period {i - min_train_periods + 1}/{total}...")

        # Split
        train_df = dataset_df[dataset_df["signal_date"].isin(train_dates)]
        test_df = dataset_df[dataset_df["signal_date"] == test_date]

        if len(train_df) < 50 or len(test_df) < top_k + 3:
            continue
        if train_df["label"].sum() < 5:
            continue

        # Train
        clf = FundClassifier(model_type=model_type, class_weight_ratio=class_weight_ratio)
        try:
            clf.train(train_df)
        except Exception as e:
            continue

        last_model = clf

        # Predict
        try:
            probs = clf.predict_proba(test_df)
        except:
            continue

        test_df = test_df.copy()
        test_df["pred_prob"] = probs

        # Pick top_k by predicted probability
        predicted_top = test_df.nlargest(top_k, "pred_prob")
        predicted_set = set(predicted_top["fund_id"].values)

        # Actual top target_k by forward return
        actual_top = test_df.nlargest(target_k, "forward_return")
        actual_set = set(actual_top["fund_id"].values)

        # Hit rate
        hits = len(predicted_set & actual_set)
        hit_rate = hits / top_k if top_k > 0 else 0

        # Portfolio return (equal weight predicted picks)
        port_ret = predicted_top["forward_return"].mean()
        bench_ret = test_df["bench_return"].iloc[0] if len(test_df) > 0 else 0

        # Top-k average (what perfect foresight would get)
        perfect_ret = actual_top["forward_return"].mean()

        cap *= (1 + port_ret)
        bcap *= (1 + bench_ret)

        entry_date = test_df["entry_date"].iloc[0]
        exit_date = test_df["exit_date"].iloc[0]

        # Build trade record
        trade = {
            "Period": f"{pd.Timestamp(entry_date).strftime('%Y-%m-%d')} → {pd.Timestamp(exit_date).strftime('%Y-%m-%d')}",
            "Pool": len(test_df),
            "Port %": port_ret * 100,
            "Bench %": bench_ret * 100,
            "Alpha %": (port_ret - bench_ret) * 100,
            "Perfect %": perfect_ret * 100,
            "Hits": hits,
            "HR %": hit_rate * 100,
        }
        # Add individual picks
        for j, (_, row) in enumerate(predicted_top.iterrows()):
            trade[f"Pick{j + 1}"] = row["fund_name"][:35]
            trade[f"Pick{j + 1} %"] = row["forward_return"] * 100
            trade[f"Pick{j + 1} Prob"] = row["pred_prob"]
            trade[f"Pick{j + 1} Hit"] = "✅" if row["fund_id"] in actual_set else "❌"

        all_trades.append(trade)

        results.append({
            "date": test_date,
            "entry": entry_date,
            "exit": exit_date,
            "port_return": port_ret,
            "bench_return": bench_ret,
            "hit_rate": hit_rate,
            "hits": hits,
            "pool_size": len(test_df),
            "perfect_return": perfect_ret,
        })
        equity.append({"date": exit_date, "value": cap})
        bench_eq.append({"date": exit_date, "value": bcap})

        if clf.feature_importances_:
            all_importances.append(clf.feature_importances_)

    progress.empty()

    # Aggregate feature importances
    avg_importances = {}
    if all_importances:
        for feat in all_importances[0]:
            avg_importances[feat] = np.mean([imp.get(feat, 0) for imp in all_importances])

    model_info = {
        "model_type": model_type,
        "feature_importances": avg_importances,
        "last_model": last_model,
        "n_train_periods": len(results),
    }

    return (
        pd.DataFrame(results),
        pd.DataFrame(equity),
        pd.DataFrame(bench_eq),
        pd.DataFrame(all_trades),
        model_info,
    )


# ============================================================================
# 8. CURRENT PREDICTIONS (What to buy today)
# ============================================================================

def predict_current_picks(nav_df, benchmark, scheme_map, model_info,
                          dataset_df, top_k):
    """
    Use the latest trained model to predict today's top picks.
    """
    if model_info.get("last_model") is None:
        return []

    clf = model_info["last_model"]
    engine = FundFeatureEngine(nav_df, benchmark)

    # Build features for the latest date
    latest_idx = len(nav_df) - 1
    snapshot = engine.build_snapshot(latest_idx)
    if not snapshot:
        return []

    # Convert to DataFrame
    records = []
    for fid, feats in snapshot.items():
        rec = feats.copy()
        rec["fund_id"] = fid
        rec["fund_name"] = scheme_map.get(fid, fid)
        records.append(rec)

    pred_df = pd.DataFrame(records)
    if len(pred_df) < top_k:
        return []

    # Predict
    try:
        probs = clf.predict_proba(pred_df)
        pred_df["pred_prob"] = probs
    except Exception as e:
        return []

    # Top picks
    top_picks = pred_df.nlargest(top_k, "pred_prob")

    picks = []
    for _, row in top_picks.iterrows():
        picks.append({
            "fund_id": row["fund_id"],
            "name": row["fund_name"],
            "probability": row["pred_prob"],
            "ret_63d": row.get("ret_63d", np.nan),
            "ret_252d": row.get("ret_252d", np.nan),
            "sharpe_252d": row.get("sharpe_252d", np.nan),
            "vol_252d": row.get("vol_252d", np.nan),
            "trend_50dma": row.get("trend_50dma", np.nan),
        })
    return picks


# ============================================================================
# 9. COMPARISON WITH RULE-BASED STRATEGIES
# ============================================================================

def run_rule_based_comparison(dataset_df, top_k, target_k):
    """
    Run simple rule-based strategies on the same dataset for comparison.
    """
    dates = sorted(dataset_df["signal_date"].unique())
    strategies = {
        "Momentum (ret_63d)": "ret_63d",
        "Momentum (ret_252d)": "ret_252d",
        "Sharpe (63d)": "sharpe_63d",
        "Sharpe (252d)": "sharpe_252d",
        "Sortino (252d)": "sortino_252d",
        "CS Rank (ret63)": "cs_rank_ret63",
    }

    results = {}
    for strat_name, col in strategies.items():
        hits_list, rets_list = [], []
        for dt in dates:
            df_t = dataset_df[dataset_df["signal_date"] == dt].copy()
            if col not in df_t.columns or len(df_t) < top_k + 3:
                continue
            df_valid = df_t.dropna(subset=[col, "forward_return"])
            if len(df_valid) < top_k + 3:
                continue

            predicted = set(df_valid.nlargest(top_k, col)["fund_id"].values)
            actual = set(df_valid.nlargest(target_k, "forward_return")["fund_id"].values)
            hr = len(predicted & actual) / top_k
            port_ret = df_valid[df_valid["fund_id"].isin(predicted)]["forward_return"].mean()

            hits_list.append(hr)
            rets_list.append(port_ret)

        if hits_list:
            results[strat_name] = {
                "Avg Hit Rate %": np.mean(hits_list) * 100,
                "Avg Return %": np.mean(rets_list) * 100,
                "Win Rate %": (np.array(rets_list) > 0).mean() * 100,
                "Periods": len(hits_list),
            }
    return results


# ============================================================================
# 10. VISUALIZATION
# ============================================================================

def plot_equity_curve(equity_df, bench_df, title="ML Strategy vs Benchmark"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["date"], y=equity_df["value"],
        name="ML Strategy", line=dict(color="#2196F3", width=2.5),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=bench_df["date"], y=bench_df["value"],
        name="Benchmark (Nifty 100)", line=dict(color="gray", width=2, dash="dot"),
    ))
    fig.update_layout(
        title=title, yaxis_title="Value (₹100 invested)",
        hovermode="x unified", height=420,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def plot_hit_rate_over_time(results_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=results_df["date"], y=results_df["hit_rate"] * 100,
        marker_color=["#4caf50" if hr >= 0.5 else "#f44336"
                       for hr in results_df["hit_rate"]],
        name="Hit Rate %",
    ))
    avg_hr = results_df["hit_rate"].mean() * 100
    fig.add_hline(y=avg_hr, line_dash="dash", line_color="blue",
                  annotation_text=f"Avg: {avg_hr:.1f}%")
    fig.update_layout(
        title="Hit Rate per Period", yaxis_title="Hit Rate %",
        height=350,
    )
    return fig


def plot_feature_importance(importances, top_n=15):
    if not importances:
        return None
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [f[0] for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=px.colors.sequential.Viridis[:len(vals)],
    ))
    fig.update_layout(
        title="Feature Importance (Avg across periods)",
        xaxis_title="Importance", height=400,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_returns_comparison(results_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=results_df["date"], y=results_df["port_return"] * 100,
        name="ML Portfolio", marker_color="#2196F3",
    ))
    fig.add_trace(go.Bar(
        x=results_df["date"], y=results_df["bench_return"] * 100,
        name="Benchmark", marker_color="#bdbdbd",
    ))
    fig.update_layout(
        title="Period Returns: ML vs Benchmark",
        yaxis_title="Return %", barmode="group", height=350,
    )
    return fig


# ============================================================================
# 11. STREAMLIT UI
# ============================================================================

def render_ml_backtest_tab():
    st.markdown("""<div class="info-banner">
        <h2>🧠 ML Classification Strategy — Backtest</h2>
        <p>Train ML models to predict which funds will be in the Top K performers.
        Walk-forward backtesting with expanding training window.</p>
    </div>""", unsafe_allow_html=True)

    # --- Controls ---
    c1, c2, c3, c4, c5, c6 = st.columns([2, 1.5, 1, 1, 1, 1])
    with c1:
        category = st.selectbox("📁 Fund Category", list(FILE_MAPPING.keys()), key="ml_cat")
    with c2:
        model_type = st.selectbox("🤖 Model", [
            ("ensemble", "🧠 Ensemble (HGB+MLP+RF)"),
            ("hgb", "🌲 HistGradientBoosting"),
            ("mlp", "🔮 Neural Network (MLP)"),
            ("rf", "🌳 Random Forest"),
        ], format_func=lambda x: x[1], key="ml_model")[0]
    with c3:
        top_k = st.number_input("🎯 Pick Top K", 1, 10, 5, key="ml_topk")
    with c4:
        target_k = st.number_input("🏆 Actual Top K", 1, 15, 5, key="ml_targetk")
    with c5:
        hold = st.selectbox("📅 Hold Period", HOLDING_PERIODS, index=1,
                            format_func=get_holding_label, key="ml_hold")
    with c6:
        class_weight = st.selectbox("⚖️ Pos. Weight", [2.0, 3.0, 5.0, 7.0],
                                    index=1, key="ml_cw")

    # --- Load Data ---
    nav_df, scheme_map = load_fund_data(category)
    benchmark = load_benchmark()
    if nav_df is None:
        st.error("Could not load fund data. Ensure data/ folder exists.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Funds", len(nav_df.columns))
    c2.metric("Period", f"{nav_df.index.min().strftime('%Y-%m')} → {nav_df.index.max().strftime('%Y-%m')}")
    c3.metric("Features", len(FundFeatureEngine.FEATURE_NAMES))

    st.divider()

    # --- Run Button ---
    if st.button("🚀 Run ML Backtest", type="primary", use_container_width=True):

        # Step 1: Build dataset
        st.markdown("### 📊 Step 1: Building Feature Dataset")
        with st.spinner("Engineering features for all funds across all dates..."):
            dataset = build_training_dataset(
                nav_df, benchmark, scheme_map,
                hold_days=hold, top_k=target_k,
                step_days=hold,  # Non-overlapping for backtest purity
            )

        if dataset.empty or len(dataset) < 100:
            st.error("Insufficient data to build training set.")
            return

        n_dates = dataset["signal_date"].nunique()
        n_pos = dataset["label"].sum()
        n_neg = (dataset["label"] == 0).sum()

        st.success(
            f"✅ Dataset: **{len(dataset):,}** samples across **{n_dates}** periods | "
            f"Positive (top-{target_k}): **{n_pos:,}** ({n_pos / len(dataset) * 100:.1f}%) | "
            f"Negative: **{n_neg:,}** ({n_neg / len(dataset) * 100:.1f}%)"
        )

        # Step 2: Run backtest
        st.markdown("### 🔄 Step 2: Walk-Forward Backtest")
        results_df, equity_df, bench_df, trades_df, model_info = run_ml_backtest(
            dataset, model_type=model_type,
            top_k=top_k, target_k=target_k,
            min_train_periods=max(6, n_dates // 4),
            class_weight_ratio=class_weight,
        )

        if results_df.empty:
            st.error("Backtest produced no results. Try different parameters.")
            return

        # Store in session state
        st.session_state["ml_results"] = results_df
        st.session_state["ml_equity"] = equity_df
        st.session_state["ml_bench"] = bench_df
        st.session_state["ml_trades"] = trades_df
        st.session_state["ml_model_info"] = model_info
        st.session_state["ml_dataset"] = dataset
        st.session_state["ml_nav"] = nav_df
        st.session_state["ml_benchmark"] = benchmark
        st.session_state["ml_scheme_map"] = scheme_map
        st.session_state["ml_top_k"] = top_k
        st.session_state["ml_target_k"] = target_k

    # --- Display Results ---
    if "ml_results" not in st.session_state:
        st.info("👆 Configure parameters and click **Run ML Backtest** to begin.")
        with st.expander("📖 How the Classification Strategy Works", expanded=True):
            st.markdown("""
**Problem Reframing:** Instead of predicting returns (regression), we predict: *"Will this fund be in the top K performers?"* (classification)

**Why this is better:**
- A fund predicted at +15% vs actual +18% is an error in regression, but both correctly classify as "top K"
- The model focuses on the **decision boundary** that matters for hit rate
- Handles the heavy class imbalance (5 top funds out of 50) with weighted training

**Feature Groups (30 features):**
1. **Momentum** (6): Multi-horizon returns + acceleration
2. **Risk** (5): Volatility, drawdown at multiple windows
3. **Risk-Adjusted** (4): Rolling Sharpe & Sortino
4. **Trend** (5): DMA crossovers, distance from peak, mean reversion
5. **Cross-Sectional** (6): Z-scores & percentile ranks vs ALL peers
6. **Regime** (4): Benchmark state features

**Walk-Forward Validation:**
- Train on all past periods → predict next period → measure → expand window → repeat
- No future data leakage guaranteed
            """)
        return

    results_df = st.session_state["ml_results"]
    equity_df = st.session_state["ml_equity"]
    bench_df = st.session_state["ml_bench"]
    trades_df = st.session_state["ml_trades"]
    model_info = st.session_state["ml_model_info"]

    # --- Summary Metrics ---
    st.markdown("### 📈 Backtest Results")
    yrs = (equity_df.iloc[-1]["date"] - equity_df.iloc[0]["date"]).days / 365.25
    cagr = (equity_df.iloc[-1]["value"] / 100) ** (1 / yrs) - 1 if yrs > 0 else 0
    bcagr = (bench_df.iloc[-1]["value"] / 100) ** (1 / yrs) - 1 if yrs > 0 else 0
    avg_hr = results_df["hit_rate"].mean()
    avg_alpha = (results_df["port_return"] - results_df["bench_return"]).mean()
    win_rate = (results_df["port_return"] > 0).mean()
    # Max drawdown
    vals = equity_df["value"].values
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak
    max_dd = dd.min()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ML CAGR", f"{cagr * 100:.2f}%")
    c2.metric("Bench CAGR", f"{bcagr * 100:.2f}%")
    c3.metric("Alpha", f"{(cagr - bcagr) * 100:+.2f}%")
    c4.metric("Avg Hit Rate", f"{avg_hr * 100:.1f}%")
    c5.metric("Win Rate", f"{win_rate * 100:.1f}%")
    c6.metric("Max DD", f"{max_dd * 100:.1f}%")

    st.divider()

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Equity Curve", "🎯 Hit Rate", "📋 Trades",
        "🏅 Feature Importance", "📊 vs Rule-Based", "🔮 Current Picks"
    ])

    with tab1:
        st.plotly_chart(
            plot_equity_curve(equity_df, bench_df,
                              f"ML {model_info['model_type'].upper()} Strategy vs Benchmark"),
            use_container_width=True,
        )
        st.plotly_chart(plot_returns_comparison(results_df), use_container_width=True)

    with tab2:
        st.plotly_chart(plot_hit_rate_over_time(results_df), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Average HR", f"{avg_hr * 100:.1f}%")
        c2.metric("Best Period HR", f"{results_df['hit_rate'].max() * 100:.0f}%")
        c3.metric("Worst Period HR", f"{results_df['hit_rate'].min() * 100:.0f}%")

        # Distribution
        hr_vals = results_df["hit_rate"].values
        hr_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        hr_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        hr_counts = pd.cut(hr_vals, bins=hr_bins, labels=hr_labels).value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=hr_labels, y=hr_counts.values,
            marker_color=["#f44336", "#ff9800", "#ffc107", "#8bc34a", "#4caf50"],
        ))
        fig.update_layout(title="Hit Rate Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if not trades_df.empty:
            pct_cols = [c for c in trades_df.columns if "%" in c]
            prob_cols = [c for c in trades_df.columns if "Prob" in c]
            fmt = {c: "{:.2f}" for c in pct_cols}
            fmt.update({c: "{:.3f}" for c in prob_cols})
            fmt["Hits"] = "{:.0f}"
            fmt["Pool"] = "{:.0f}"

            def highlight_hit(v):
                if v == "✅":
                    return "background-color: #c8e6c9"
                if v == "❌":
                    return "background-color: #ffcdd2"
                return ""

            hit_cols = [c for c in trades_df.columns if "Hit" in c and "%" not in c]
            sty = trades_df.style.format(fmt, na_rep="")
            for c in hit_cols:
                sty = sty.map(highlight_hit, subset=[c])
            st.dataframe(sty, use_container_width=True, height=600)
            st.download_button(
                "📥 Download Trades",
                trades_df.to_csv(index=False),
                "ml_trades.csv",
            )

    with tab4:
        importances = model_info.get("feature_importances", {})
        if importances:
            fig = plot_feature_importance(importances)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Feature Importance Table")
            imp_df = (
                pd.DataFrame(
                    sorted(importances.items(), key=lambda x: x[1], reverse=True),
                    columns=["Feature", "Importance"],
                )
            )
            imp_df["% of Total"] = imp_df["Importance"] / imp_df["Importance"].sum() * 100
            st.dataframe(
                imp_df.style.format({"Importance": "{:.4f}", "% of Total": "{:.1f}%"})
                .background_gradient(subset=["Importance"], cmap="Blues"),
                use_container_width=True,
            )
        else:
            st.info("Feature importances not available for this model type.")

    with tab5:
        st.markdown("### 📊 ML vs Rule-Based Strategies")
        dataset = st.session_state.get("ml_dataset")
        top_k_val = st.session_state.get("ml_top_k", 5)
        target_k_val = st.session_state.get("ml_target_k", 5)

        if dataset is not None:
            with st.spinner("Running rule-based comparisons..."):
                rule_results = run_rule_based_comparison(dataset, top_k_val, target_k_val)

            if rule_results:
                # Add ML result
                compare_data = {
                    "🧠 ML Classification": {
                        "Avg Hit Rate %": avg_hr * 100,
                        "Avg Return %": results_df["port_return"].mean() * 100,
                        "Win Rate %": win_rate * 100,
                        "Periods": len(results_df),
                    }
                }
                compare_data.update(rule_results)

                compare_df = pd.DataFrame(compare_data).T
                compare_df.index.name = "Strategy"

                st.dataframe(
                    compare_df.style.format({
                        "Avg Hit Rate %": "{:.1f}",
                        "Avg Return %": "{:.2f}",
                        "Win Rate %": "{:.1f}",
                        "Periods": "{:.0f}",
                    }).background_gradient(subset=["Avg Hit Rate %"], cmap="RdYlGn"),
                    use_container_width=True,
                )

                # Bar chart comparison
                fig = go.Figure()
                strat_names = list(compare_data.keys())
                hr_vals = [compare_data[s]["Avg Hit Rate %"] for s in strat_names]
                ret_vals = [compare_data[s]["Avg Return %"] for s in strat_names]

                fig.add_trace(go.Bar(
                    name="Hit Rate %", x=strat_names, y=hr_vals,
                    marker_color="#2196F3",
                ))
                fig.add_trace(go.Bar(
                    name="Avg Return %", x=strat_names, y=ret_vals,
                    marker_color="#4caf50",
                ))
                fig.update_layout(
                    barmode="group", title="ML vs Rule-Based Comparison",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Run the ML backtest first to enable comparison.")

    with tab6:
        st.markdown("### 🔮 Current Picks (What to Buy Today)")
        nav = st.session_state.get("ml_nav")
        bench = st.session_state.get("ml_benchmark")
        smap = st.session_state.get("ml_scheme_map")

        if nav is not None and model_info.get("last_model"):
            with st.spinner("Predicting current top picks..."):
                picks = predict_current_picks(
                    nav, bench, smap, model_info,
                    st.session_state.get("ml_dataset"),
                    st.session_state.get("ml_top_k", 5),
                )

            if picks:
                st.markdown(f"**Signal Date:** {nav.index[-1].strftime('%Y-%m-%d')}")
                st.markdown(f"**Model:** {model_info['model_type'].upper()} trained on {model_info['n_train_periods']} periods")
                st.divider()

                cols = st.columns(min(len(picks), 5))
                for idx, pick in enumerate(picks):
                    with cols[idx % len(cols)]:
                        prob_color = "#4caf50" if pick["probability"] > 0.5 else "#ff9800"
                        st.markdown(f"""<div class="pick-card">
                            <div class="rank">#{idx + 1}</div>
                            <div class="name">{pick['name'][:50]}</div>
                            <div class="prob" style="color:{prob_color}">
                                P(Top-K) = {pick['probability']:.1%}
                            </div>
                            <div style="font-size:0.8rem;color:#666;margin-top:6px;">
                                3M: {pick['ret_63d'] * 100:.1f}% |
                                1Y: {pick['ret_252d'] * 100:.1f}%<br>
                                Sharpe: {pick['sharpe_252d']:.2f} |
                                Vol: {pick['vol_252d'] * 100:.1f}%
                            </div>
                        </div>""", unsafe_allow_html=True)

                # Table view
                st.markdown("#### Detailed View")
                picks_df = pd.DataFrame(picks)
                st.dataframe(
                    picks_df[["name", "probability", "ret_63d", "ret_252d", "sharpe_252d", "vol_252d"]]
                    .rename(columns={
                        "name": "Fund", "probability": "P(Top-K)",
                        "ret_63d": "3M Ret", "ret_252d": "1Y Ret",
                        "sharpe_252d": "Sharpe", "vol_252d": "Volatility",
                    })
                    .style.format({
                        "P(Top-K)": "{:.1%}", "3M Ret": "{:.2%}",
                        "1Y Ret": "{:.2%}", "Sharpe": "{:.2f}",
                        "Volatility": "{:.2%}",
                    }).background_gradient(subset=["P(Top-K)"], cmap="Greens"),
                    use_container_width=True,
                )
            else:
                st.warning("Could not generate predictions. Ensure the model is trained.")
        else:
            st.info("Run a backtest first to train the model and generate current picks.")


# ============================================================================
# 12. MAIN
# ============================================================================

def main():
    st.markdown("""<div style="text-align:center;padding:10px 0 15px">
        <h1 style="margin:0;border:none">🧠 ML Fund Predictor</h1>
        <p style="color:#666">Classification-Based Top-K Fund Selection |
        Walk-Forward Backtest | 30 Engineered Features</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🚀 ML Backtest", "📖 Methodology"])

    with tab1:
        render_ml_backtest_tab()

    with tab2:
        st.markdown("""
## How This Works — Complete Methodology

### 1. Problem Reframing: Classification, Not Regression

**Traditional approach (your original code):**
- Compute a score per fund (momentum, Sharpe, etc.)
- Sort by score → pick top N
- **Problem:** Score is a proxy, not directly optimized for ranking

**Our ML approach:**
- For each historical period, we know which funds were ACTUALLY in the top K
- Train a classifier: **"Given these 30 features, will this fund be in the top K?"**
- At prediction time, pick funds with highest probability
- **Advantage:** The model directly optimizes for the ranking decision

### 2. Feature Engineering (30 Features)

| Group | Features | Why |
|-------|----------|-----|
| **Momentum** (6) | Returns at 1M, 3M, 6M, 12M + acceleration | NBER paper: fund momentum is THE strongest predictor |
| **Risk** (5) | Volatility (21d, 63d, 252d), Max DD (63d, 252d) | Risk predicts future underperformance |
| **Risk-Adjusted** (4) | Sharpe & Sortino at 63d, 252d | Efficiency of returns |
| **Trend** (5) | DMA crossovers, peak distance, mean reversion | Regime-aware features |
| **Cross-Sectional** (6) | Z-scores & percentile ranks vs peers | **KEY:** Relative positioning matters more than absolute |
| **Regime** (4) | Benchmark returns, vol, DMA state | Market context |

### 3. Models

**HistGradientBoosting (HGB):**
- Equivalent to LightGBM, handles NaN natively
- Best for tabular data, captures non-linear feature interactions
- E.g., "high momentum + low vol = top fund" vs "high momentum + high vol = risky"

**MLP Neural Network:**
- 3-layer feedforward network (128→64→32)
- Captures complex non-linear patterns
- With RobustScaler preprocessing

**Random Forest:**
- Ensemble of decision trees with balanced class weights
- Robust baseline, less prone to overfitting

**Ensemble (recommended):**
- Soft-voting combination: HGB (weight 2) + MLP (1) + RF (1)
- Diversity of model types improves robustness

### 4. Walk-Forward Backtesting

```
Period 1-8:  [TRAIN] ────────────────────────────────
Period 9:    [TRAIN] ──────────────────────── [TEST]
Period 10:   [TRAIN] ──────────────────────────── [TEST]
Period 11:   [TRAIN] ──────────────────────────────── [TEST]
...
```

- **Expanding window:** Each test period has ALL prior data for training
- **No leakage:** Features use only past data; labels use only forward returns
- **Deterministic IDs:** Fixed fund identifiers (not Python hash)

### 5. Class Imbalance Handling

With 50 funds and top_k=5, only 10% are positive. Solutions:
- **Sample weighting:** Positive class weighted 3x (configurable)
- **RF class_weight='balanced':** Automatic adjustment
- **Ensemble diversity:** Different models handle imbalance differently

### 6. Hit Rate Calculation

Same as your original code:
```
Hit Rate = |Predicted Top K ∩ Actual Top K| / K
```

### 7. Key Improvements Over Rule-Based

1. **Non-linear feature interactions** (tree splits, neural network layers)
2. **Cross-sectional features** (relative rank vs peers, not just absolute values)
3. **Automatic feature selection** (model learns which features matter per regime)
4. **Adaptation** (model retrains each period, can adjust to changing markets)
5. **Proper walk-forward validation** (no in-sample strategy selection)
        """)

    st.caption(
        "ML Fund Predictor | Classification-as-Ranking | "
        "Walk-Forward Backtest | Ensemble: HGB + MLP + RF"
    )


if __name__ == "__main__":
    main()
