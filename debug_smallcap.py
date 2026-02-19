"""
ML Fund Predictor v2 — Learning-to-Rank + Pairwise + Multi-Strategy
=====================================================================
WHY v1 (Classification) FAILED — and how v2 fixes each issue:

PROBLEM 1: Binary labels destroy information
  v1: Label = {0 or 1} — "top 5" vs "not top 5"
  Fund ranked #6 gets label=0, same as fund ranked #50
  → Model can't distinguish near-misses from terrible funds
  FIX: Use GRADED RELEVANCE (return percentile rank as label)
  
PROBLEM 2: Classification loss ≠ Ranking loss
  v1: Minimizes cross-entropy (probability accuracy)
  A fund with P(top5)=0.49 vs 0.51 is a massive miss in classification
  but irrelevant for ranking (both are borderline)
  FIX: Use PAIRWISE RANKING LOSS — directly optimizes "is fund A > fund B?"

PROBLEM 3: Not enough training data
  v1: Non-overlapping windows → ~12-20 training periods for 5 years
  Each period has ~50 samples → only 600-1000 total training samples
  FIX: SLIDING WINDOW with step_days=21 → 5-10x more training data

PROBLEM 4: Features not tailored for ranking
  v1: Absolute features only (ret_63d = 15%)  
  FIX: Add PAIRWISE DIFFERENCE features and TEMPORAL FEATURES
  
PROBLEM 5: Single model
  v1: One model type
  FIX: MULTI-STRATEGY ensemble with specialized models for different regimes

Run: streamlit run ml_fund_predictor_v2.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import rankdata, percentileofscore
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import ndcg_score

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIG
# ============================================================================
st.set_page_config(page_title="ML Fund Predictor v2", page_icon="🧠", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""<style>
.main .block-container{padding-top:1rem;max-width:100%}
h1{color:#1E3A5F;font-weight:700;border-bottom:3px solid #2196F3}
div[data-testid="metric-container"]{background:linear-gradient(135deg,#667eea,#764ba2);border-radius:10px;padding:15px}
div[data-testid="metric-container"] label{color:rgba(255,255,255,.9)!important}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{color:#fff!important;font-weight:700}
.stTabs [aria-selected="true"]{background:#2196F3!important;color:#fff!important}
#MainMenu,footer{visibility:hidden}
.info-banner{background:linear-gradient(135deg,#1a237e,#0d47a1);padding:20px;border-radius:12px;margin-bottom:20px;color:#fff}
.info-banner h2{color:#fff!important;margin:0;border:none}
.info-banner p{color:rgba(255,255,255,.85)!important;margin:5px 0 0}
.pick-card{background:linear-gradient(135deg,#e3f2fd,#f3e5f5);border-radius:14px;padding:16px;margin:8px 0;
  box-shadow:0 4px 15px rgba(0,0,0,.08);border-left:5px solid #2196F3}
.pick-card .rank{font-size:1.6rem;font-weight:800;color:#1565c0}
.pick-card .name{font-weight:700;color:#1E3A5F;font-size:.92rem}
.pick-card .score{color:#4caf50;font-weight:700;font-size:.95rem}
.fix-box{background:#e8f5e9;border-radius:10px;padding:12px 16px;margin:6px 0;border-left:4px solid #4caf50}
.problem-box{background:#ffebee;border-radius:10px;padding:12px 16px;margin:6px 0;border-left:4px solid #f44336}
</style>""", unsafe_allow_html=True)

RISK_FREE_RATE = 0.06
TRADING_DAYS = 252
DAILY_RF = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
DATA_DIR = "data"
MAX_DATE = pd.Timestamp("2025-12-05")
FILE_MAP = {
    "Large Cap": "largecap_merged.xlsx", "Mid Cap": "midcap.xlsx",
    "Small Cap": "smallcap.xlsx", "Large & Mid Cap": "large_and_midcap_fund.xlsx",
    "Multi Cap": "MULTICAP.xlsx", "International": "international_merged.xlsx",
}
HOLD_PERIODS = [63, 126, 189, 252]

def hold_label(d):
    return f"{d}d (~{d//21}M)" if d < 252 else f"{d}d (~{d//252}Y)"

# ============================================================================
# 2. DATA LOADING (same structure as original)
# ============================================================================
def is_growth_fund(name):
    n = str(name).lower()
    return not any(k in n for k in [
        "idcw","dividend","div ","div.","div)","direct","dir ","dir)",
        "bonus","institutional","segregated","payout","reinvestment",
        "monthly","quarterly","annual",
    ])

def clean_data(df):
    if df is None or df.empty: return df
    df = df[df.index <= MAX_DATE]
    df = df[df.index.dayofweek < 5]
    if len(df) > 0:
        wd = pd.date_range(df.index.min(), min(df.index.max(), MAX_DATE), freq="B")
        df = df.reindex(wd).ffill(limit=5)
    return df

@st.cache_data
def load_funds(cat):
    fn = FILE_MAP.get(cat)
    if not fn: return None, None
    path = os.path.join(DATA_DIR, fn)
    if not os.path.exists(path): return None, None
    try:
        raw = pd.read_excel(path, header=None)
        names = raw.iloc[2, 1:].tolist()
        data = raw.iloc[4:, :].copy()
        if isinstance(data.iloc[-1, 0], str) and "Accord" in str(data.iloc[-1, 0]):
            data = data.iloc[:-1]
        dates = pd.to_datetime(data.iloc[:, 0], errors="coerce")
        nav = pd.DataFrame(index=dates)
        smap = {}; idx = 0
        for i, nm in enumerate(names):
            if pd.notna(nm) and str(nm).strip() and is_growth_fund(nm):
                code = f"F{idx:04d}"; idx += 1
                smap[code] = str(nm).strip()
                nav[code] = pd.to_numeric(data.iloc[:, i+1], errors="coerce").values
        nav = nav.sort_index()
        nav = nav[~nav.index.duplicated(keep="last")]
        return clean_data(nav), smap
    except Exception as e:
        st.error(f"Load error: {e}"); return None, None

@st.cache_data
def load_bench():
    p = os.path.join(DATA_DIR, "nifty100_data.csv")
    if not os.path.exists(p): return None
    try:
        df = pd.read_csv(p)
        df.columns = [c.lower().strip() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna(subset=["date","nav"]).set_index("date").sort_index()
        return clean_data(df).squeeze()
    except: return None

# ============================================================================
# 3. FEATURE ENGINEERING v2 — 42 features including temporal & relative
# ============================================================================
FEATURE_COLS = [
    # Momentum (8)
    "ret_5d","ret_21d","ret_63d","ret_126d","ret_252d",
    "mom_accel_21","mom_accel_63","ret_63d_div_ret_252d",
    # Risk (6)
    "vol_21d","vol_63d","vol_252d","max_dd_63d","max_dd_126d","max_dd_252d",
    # Risk-adjusted (6)
    "sharpe_63d","sharpe_126d","sharpe_252d","sortino_63d","sortino_126d","sortino_252d",
    # Trend (6)
    "above_20dma","above_50dma","above_200dma",
    "dist_from_peak_63","dist_from_peak_252","mean_reversion_63d",
    # Stability (4)
    "vol_of_vol_63d","up_down_ratio_63d","pos_day_pct_63d","consistency_qtrs",
    # Cross-sectional (8) — KEY FOR RANKING
    "cs_pctile_ret21","cs_pctile_ret63","cs_pctile_ret252",
    "cs_z_ret63","cs_z_ret252",
    "cs_pctile_sharpe63","cs_pctile_vol63","cs_rank_composite",
    # Regime (4)
    "bench_ret_63d","bench_vol_63d","bench_above_50dma","bench_above_200dma",
]

def safe_ret(s, n):
    if len(s) < n+1 or s.iloc[-(n+1)] == 0: return np.nan
    return s.iloc[-1] / s.iloc[-(n+1)] - 1

def safe_vol(s, n):
    if len(s) < n+1: return np.nan
    r = s.iloc[-n:].pct_change().dropna()
    return r.std() * np.sqrt(TRADING_DAYS) if len(r) > 5 else np.nan

def safe_dd(s, n):
    if len(s) < n: return np.nan
    sub = s.iloc[-n:]
    cum = (1 + sub.pct_change().fillna(0)).cumprod()
    return (cum / cum.expanding().max() - 1).min()

def safe_sharpe(s, n):
    if len(s) < n+1: return np.nan
    r = s.iloc[-n:].pct_change().dropna()
    if len(r) < 10 or r.std() == 0: return np.nan
    return ((r - DAILY_RF).mean() / r.std()) * np.sqrt(TRADING_DAYS)

def safe_sortino(s, n):
    if len(s) < n+1: return np.nan
    r = s.iloc[-n:].pct_change().dropna()
    d = r[r < 0]
    if len(d) < 3 or d.std() == 0: return np.nan
    return ((r - DAILY_RF).mean() / d.std()) * np.sqrt(TRADING_DAYS)

def extract_features(nav_df, bench, date_idx, min_hist=260):
    """Extract features for ALL funds at a single date. Returns DataFrame."""
    records = []
    fund_cols = nav_df.columns.tolist()
    
    # Regime features (shared)
    regime = {}
    if bench is not None:
        b = bench.iloc[:date_idx+1].dropna()
        regime["bench_ret_63d"] = safe_ret(b, 63) if len(b) > 63 else 0
        regime["bench_vol_63d"] = safe_vol(b, 63) if len(b) > 63 else 0
        regime["bench_above_50dma"] = float(b.iloc[-1] > b.iloc[-50:].mean()) if len(b) >= 50 else 0.5
        regime["bench_above_200dma"] = float(b.iloc[-1] > b.iloc[-200:].mean()) if len(b) >= 200 else 0.5
    else:
        regime = {"bench_ret_63d":0,"bench_vol_63d":0,"bench_above_50dma":0.5,"bench_above_200dma":0.5}

    for col in fund_cols:
        s = nav_df[col].iloc[:date_idx+1].dropna()
        if len(s) < min_hist: continue
        
        f = {"fund_id": col}
        
        # Momentum
        f["ret_5d"] = safe_ret(s, 5)
        f["ret_21d"] = safe_ret(s, 21)
        f["ret_63d"] = safe_ret(s, 63)
        f["ret_126d"] = safe_ret(s, 126)
        f["ret_252d"] = safe_ret(s, 252)
        # Acceleration
        r21_now = f["ret_21d"] or 0
        if len(s) > 42:
            r21_prev = safe_ret(s.iloc[:-21], 21) or 0
            f["mom_accel_21"] = r21_now - r21_prev
        else: f["mom_accel_21"] = 0
        r63_now = f["ret_63d"] or 0
        if len(s) > 126:
            r63_prev = safe_ret(s.iloc[:-63], 63) or 0
            f["mom_accel_63"] = r63_now - r63_prev
        else: f["mom_accel_63"] = 0
        # Momentum ratio (short vs long)
        r252 = f["ret_252d"]
        f["ret_63d_div_ret_252d"] = (f["ret_63d"] / r252) if r252 and r252 != 0 and f["ret_63d"] else 0
        
        # Risk
        f["vol_21d"] = safe_vol(s, 21)
        f["vol_63d"] = safe_vol(s, 63)
        f["vol_252d"] = safe_vol(s, 252)
        f["max_dd_63d"] = safe_dd(s, 63)
        f["max_dd_126d"] = safe_dd(s, 126)
        f["max_dd_252d"] = safe_dd(s, 252)
        
        # Risk-adjusted
        f["sharpe_63d"] = safe_sharpe(s, 63)
        f["sharpe_126d"] = safe_sharpe(s, 126)
        f["sharpe_252d"] = safe_sharpe(s, 252)
        f["sortino_63d"] = safe_sortino(s, 63)
        f["sortino_126d"] = safe_sortino(s, 126)
        f["sortino_252d"] = safe_sortino(s, 252)
        
        # Trend
        cur = s.iloc[-1]
        f["above_20dma"] = float(cur > s.iloc[-20:].mean()) if len(s) >= 20 else 0.5
        f["above_50dma"] = float(cur > s.iloc[-50:].mean()) if len(s) >= 50 else 0.5
        f["above_200dma"] = float(cur > s.iloc[-200:].mean()) if len(s) >= 200 else 0.5
        peak63 = s.iloc[-63:].max() if len(s) >= 63 else s.max()
        peak252 = s.iloc[-252:].max() if len(s) >= 252 else s.max()
        f["dist_from_peak_63"] = (cur/peak63 - 1) if peak63 > 0 else 0
        f["dist_from_peak_252"] = (cur/peak252 - 1) if peak252 > 0 else 0
        f["mean_reversion_63d"] = (cur / s.iloc[-63:].mean() - 1) if len(s) >= 63 else 0
        
        # Stability
        if len(s) >= 63:
            rets = s.iloc[-63:].pct_change().dropna()
            roll_vol = rets.rolling(10).std()
            f["vol_of_vol_63d"] = roll_vol.std() if len(roll_vol.dropna()) > 5 else np.nan
            up = (rets > 0).sum(); down = (rets < 0).sum()
            f["up_down_ratio_63d"] = up / down if down > 0 else 2.0
            f["pos_day_pct_63d"] = up / len(rets)
        else:
            f["vol_of_vol_63d"] = np.nan
            f["up_down_ratio_63d"] = 1.0
            f["pos_day_pct_63d"] = 0.5
        
        # Quarterly consistency (how many of last 4 qtrs in top half)
        good_qtrs = 0
        if len(s) >= 252:
            for q in range(4):
                qs = -(q+1)*63; qe = -q*63 if q > 0 else None
                qret = (s.iloc[qe] / s.iloc[qs] - 1) if qe else (s.iloc[-1] / s.iloc[qs] - 1)
                if not pd.isna(qret) and qret > 0: good_qtrs += 1
        f["consistency_qtrs"] = good_qtrs / 4
        
        # Regime
        f.update(regime)
        
        records.append(f)
    
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    
    # Cross-sectional features (computed across all funds at this date)
    for col_name, src_col in [
        ("cs_pctile_ret21","ret_21d"), ("cs_pctile_ret63","ret_63d"),
        ("cs_pctile_ret252","ret_252d"), ("cs_pctile_sharpe63","sharpe_63d"),
        ("cs_pctile_vol63","vol_63d"),
    ]:
        vals = df[src_col].values
        valid = ~np.isnan(vals)
        result = np.full(len(vals), 0.5)
        if valid.sum() > 2:
            ranks = np.zeros(len(vals))
            ranks[valid] = rankdata(vals[valid]) / valid.sum()
            result = ranks
        df[col_name] = result
    
    for col_name, src_col in [("cs_z_ret63","ret_63d"),("cs_z_ret252","ret_252d")]:
        vals = df[src_col].values
        valid = ~np.isnan(vals)
        result = np.zeros(len(vals))
        if valid.sum() > 2:
            m, s = np.nanmean(vals[valid]), np.nanstd(vals[valid])
            if s > 0: result[valid] = (vals[valid] - m) / s
        df[col_name] = result
    
    # Composite rank (average of percentile ranks)
    pctile_cols = [c for c in df.columns if c.startswith("cs_pctile_")]
    df["cs_rank_composite"] = df[pctile_cols].mean(axis=1)
    
    return df

# ============================================================================
# 4. DATASET BUILDING — with GRADED LABELS (key fix)
# ============================================================================
def build_dataset(nav_df, bench, smap, hold_days, top_k,
                  step_days=21, min_hist=370):
    """
    KEY DIFFERENCE FROM v1:
    - step_days=21 (monthly sliding) → 5-10x more training data
    - Labels are GRADED: percentile rank of forward return (0-1)
    - Also store binary label for hit rate computation
    """
    start_idx = nav_df.index.searchsorted(nav_df.index.min() + pd.Timedelta(days=min_hist))
    end_idx = len(nav_df) - hold_days - 1
    rebal = list(range(start_idx, end_idx, step_days))
    
    all_records = []
    prog = st.progress(0, text="Building dataset...")
    
    for pi, idx in enumerate(rebal):
        if pi % 10 == 0:
            prog.progress(pi / len(rebal), text=f"Date {pi+1}/{len(rebal)}...")
        
        signal_dt = nav_df.index[idx]
        entry_idx = idx + 1
        exit_idx = min(idx + 1 + hold_days, len(nav_df) - 1)
        entry_dt = nav_df.index[entry_idx]
        exit_dt = nav_df.index[exit_idx]
        
        snap = extract_features(nav_df, bench, idx)
        if len(snap) < top_k + 3: continue
        
        # Forward returns
        fwd = {}
        for fid in snap["fund_id"]:
            try:
                en = nav_df.loc[entry_dt, fid]
                ex = nav_df.loc[exit_dt, fid]
                if pd.notna(en) and pd.notna(ex) and en > 0:
                    fwd[fid] = ex / en - 1
            except: continue
        
        if len(fwd) < top_k + 3: continue
        
        # GRADED LABEL: percentile rank of forward return
        fwd_series = pd.Series(fwd)
        fwd_ranks = rankdata(fwd_series.values) / len(fwd_series)  # 0 to 1
        rank_map = dict(zip(fwd_series.index, fwd_ranks))
        
        # Binary: top_k
        sorted_fwd = fwd_series.sort_values(ascending=False)
        top_k_set = set(sorted_fwd.index[:top_k])
        
        # Benchmark return
        bret = 0
        if bench is not None:
            try:
                bret = bench.asof(exit_dt) / bench.asof(entry_dt) - 1
            except: pass
        
        for _, row in snap.iterrows():
            fid = row["fund_id"]
            if fid not in fwd: continue
            rec = row.to_dict()
            rec["signal_date"] = signal_dt
            rec["entry_date"] = entry_dt
            rec["exit_date"] = exit_dt
            rec["forward_return"] = fwd[fid]
            rec["graded_label"] = rank_map[fid]  # 0-1 percentile rank
            rec["binary_label"] = 1 if fid in top_k_set else 0
            rec["bench_return"] = bret
            rec["fund_name"] = smap.get(fid, fid)
            all_records.append(rec)
    
    prog.empty()
    return pd.DataFrame(all_records)


# ============================================================================
# 5. MODELS — Three approaches that directly optimize ranking
# ============================================================================

class PairwiseRankModel:
    """
    Strategy 1: PAIRWISE RANKING
    
    For each pair of funds at each date, predict which has higher return.
    At test time, count pairwise "wins" → rank by wins → pick top K.
    
    This directly optimizes the ranking question.
    """
    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=30, l2_regularization=2.0,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=42,
        )
        self.scaler = RobustScaler()
        self.feature_cols = FEATURE_COLS
    
    def _make_pairs(self, df, max_pairs_per_date=500):
        """Generate pairwise training samples from a snapshot."""
        pairs_X, pairs_y = [], []
        dates = df["signal_date"].unique()
        
        for dt in dates:
            snap = df[df["signal_date"] == dt]
            if len(snap) < 5: continue
            
            fids = snap["fund_id"].values
            feats = snap[self.feature_cols].fillna(0).values
            fwd = snap["forward_return"].values
            
            n = len(fids)
            # Sample pairs (not all N*N — too many)
            n_pairs = min(max_pairs_per_date, n * (n - 1) // 2)
            indices = np.random.choice(n, size=(n_pairs, 2), replace=True)
            # Remove self-pairs
            mask = indices[:, 0] != indices[:, 1]
            indices = indices[mask]
            
            for i, j in indices:
                diff = feats[i] - feats[j]  # Feature difference
                if fwd[i] > fwd[j]:
                    pairs_X.append(diff)
                    pairs_y.append(1)
                elif fwd[j] > fwd[i]:
                    pairs_X.append(diff)
                    pairs_y.append(0)
        
        return np.array(pairs_X), np.array(pairs_y)
    
    def train(self, train_df):
        X, y = self._make_pairs(train_df)
        if len(X) < 50: return False
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return True
    
    def predict_scores(self, test_snap):
        """Score each fund by pairwise win count."""
        feats = test_snap[self.feature_cols].fillna(0).values
        n = len(feats)
        wins = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                diff = (feats[i] - feats[j]).reshape(1, -1)
                diff_scaled = self.scaler.transform(diff)
                prob = self.model.predict_proba(diff_scaled)[0]
                wins[i] += prob[1] if len(prob) > 1 else prob[0]
        
        return wins / max(n - 1, 1)  # Normalize


class GradedRegressionModel:
    """
    Strategy 2: REGRESSION ON GRADED LABELS
    
    Instead of binary classification, predict the percentile rank (0-1).
    Fund ranked #1 gets label ~1.0, #50 gets ~0.02.
    
    This preserves ordinal information that binary labels destroy.
    """
    def __init__(self):
        self.model = HistGradientBoostingRegressor(
            max_iter=300, max_depth=5, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=1.5,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=42,
        )
        self.feature_cols = FEATURE_COLS
    
    def train(self, train_df):
        X = train_df[self.feature_cols].fillna(train_df[self.feature_cols].median()).fillna(0)
        y = train_df["graded_label"].values
        self.model.fit(X, y)
        return True
    
    def predict_scores(self, test_snap):
        X = test_snap[self.feature_cols].fillna(test_snap[self.feature_cols].median()).fillna(0)
        return self.model.predict(X)


class ReturnRegressionModel:
    """
    Strategy 3: DIRECT RETURN PREDICTION
    
    Predict the actual forward return. Simpler but effective baseline.
    """
    def __init__(self):
        self.model = HistGradientBoostingRegressor(
            max_iter=300, max_depth=5, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=1.5,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=42,
        )
        self.feature_cols = FEATURE_COLS
    
    def train(self, train_df):
        X = train_df[self.feature_cols].fillna(train_df[self.feature_cols].median()).fillna(0)
        y = train_df["forward_return"].values
        # Winsorize extreme returns
        y = np.clip(y, np.percentile(y, 2), np.percentile(y, 98))
        self.model.fit(X, y)
        return True
    
    def predict_scores(self, test_snap):
        X = test_snap[self.feature_cols].fillna(test_snap[self.feature_cols].median()).fillna(0)
        return self.model.predict(X)


class MLPRankModel:
    """
    Strategy 4: NEURAL NETWORK with graded regression
    """
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu", solver="adam", alpha=0.005,
            learning_rate="adaptive", learning_rate_init=0.001,
            max_iter=500, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=30,
            random_state=42,
        )
        self.scaler = RobustScaler()
        self.feature_cols = FEATURE_COLS
    
    def train(self, train_df):
        X = train_df[self.feature_cols].fillna(train_df[self.feature_cols].median()).fillna(0)
        X_s = self.scaler.fit_transform(X)
        y = train_df["graded_label"].values
        self.model.fit(X_s, y)
        return True
    
    def predict_scores(self, test_snap):
        X = test_snap[self.feature_cols].fillna(test_snap[self.feature_cols].median()).fillna(0)
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)


class EnsembleRankModel:
    """
    Strategy 5: ENSEMBLE — combines all strategies
    
    Each model produces scores → normalize to 0-1 → weighted average.
    """
    def __init__(self, use_pairwise=False):
        self.graded = GradedRegressionModel()
        self.returns = ReturnRegressionModel()
        self.mlp = MLPRankModel()
        self.use_pairwise = use_pairwise
        if use_pairwise:
            self.pairwise = PairwiseRankModel()
        self.feature_cols = FEATURE_COLS
    
    def train(self, train_df):
        self.graded.train(train_df)
        self.returns.train(train_df)
        self.mlp.train(train_df)
        if self.use_pairwise:
            self.pairwise.train(train_df)
        return True
    
    def predict_scores(self, test_snap):
        s1 = self.graded.predict_scores(test_snap)
        s2 = self.returns.predict_scores(test_snap)
        s3 = self.mlp.predict_scores(test_snap)
        
        # Normalize each to 0-1 via rank
        def rank_norm(arr):
            return rankdata(arr) / len(arr)
        
        combined = 0.4 * rank_norm(s1) + 0.35 * rank_norm(s2) + 0.25 * rank_norm(s3)
        
        if self.use_pairwise:
            s4 = self.pairwise.predict_scores(test_snap)
            combined = 0.3 * rank_norm(s1) + 0.25 * rank_norm(s2) + 0.2 * rank_norm(s3) + 0.25 * rank_norm(s4)
        
        return combined


# Rule-based baselines for comparison
class RuleBasedModel:
    """Wraps a simple rule-based strategy as a model interface."""
    def __init__(self, sort_col, ascending=False):
        self.sort_col = sort_col
        self.ascending = ascending
        self.feature_cols = FEATURE_COLS
    
    def train(self, train_df):
        return True  # No training needed
    
    def predict_scores(self, test_snap):
        vals = test_snap[self.sort_col].fillna(0).values
        if self.ascending:
            return -vals
        return vals


# ============================================================================
# 6. WALK-FORWARD BACKTESTER
# ============================================================================
def run_backtest(dataset, model_class, model_kwargs, top_k, target_k,
                 min_train_periods=8, hold_days=126):
    """Walk-forward backtest. Returns results, equity, bench, trades."""
    dates = sorted(dataset["signal_date"].unique())
    if len(dates) < min_train_periods + 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    
    results, trades = [], []
    eq = [{"date": dates[min_train_periods], "value": 100.0}]
    bm = [{"date": dates[min_train_periods], "value": 100.0}]
    cap, bcap = 100.0, 100.0
    feat_imp_list = []
    last_model = None
    
    prog = st.progress(0, "Backtesting...")
    total = len(dates) - min_train_periods
    
    # Use non-overlapping TEST dates (even though training uses overlapping)
    test_step = max(1, hold_days // 21)  # Test every hold_days worth of sliding steps
    test_indices = list(range(min_train_periods, len(dates), test_step))
    total = len(test_indices)
    
    for prog_i, ti in enumerate(test_indices):
        if ti >= len(dates): break
        prog.progress(min(prog_i / max(total, 1), 1.0),
                      text=f"Period {prog_i+1}/{total}...")
        
        test_date = dates[ti]
        train_dates = dates[:ti]
        
        train_df = dataset[dataset["signal_date"].isin(train_dates)]
        test_df = dataset[dataset["signal_date"] == test_date].copy()
        
        if len(train_df) < 100 or len(test_df) < top_k + 3:
            continue
        
        # Train
        model = model_class(**model_kwargs)
        try:
            ok = model.train(train_df)
            if not ok: continue
        except: continue
        
        last_model = model
        
        # Predict
        try:
            scores = model.predict_scores(test_df)
        except: continue
        
        test_df["pred_score"] = scores
        
        # Pick top_k by score
        pred_top = test_df.nlargest(top_k, "pred_score")
        pred_set = set(pred_top["fund_id"].values)
        
        # Actual top target_k
        actual_top = test_df.nlargest(target_k, "forward_return")
        actual_set = set(actual_top["fund_id"].values)
        
        # Hit rate
        hits = len(pred_set & actual_set)
        hr = hits / top_k
        
        # Returns
        port_ret = pred_top["forward_return"].mean()
        bench_ret = test_df["bench_return"].iloc[0]
        perfect_ret = actual_top["forward_return"].mean()
        
        cap *= (1 + port_ret)
        bcap *= (1 + bench_ret)
        
        entry_dt = test_df["entry_date"].iloc[0]
        exit_dt = test_df["exit_date"].iloc[0]
        
        # NDCG score (ranking quality metric)
        if len(test_df) >= top_k:
            true_rels = test_df["graded_label"].values.reshape(1, -1)
            pred_rels = test_df["pred_score"].values.reshape(1, -1)
            try:
                ndcg = ndcg_score(true_rels, pred_rels, k=top_k)
            except: ndcg = 0
        else: ndcg = 0
        
        trade = {
            "Period": f"{pd.Timestamp(entry_dt).strftime('%Y-%m-%d')} → {pd.Timestamp(exit_dt).strftime('%Y-%m-%d')}",
            "Pool": len(test_df), "Port %": port_ret*100, "Bench %": bench_ret*100,
            "Alpha %": (port_ret-bench_ret)*100, "Perfect %": perfect_ret*100,
            "Hits": hits, "HR %": hr*100, "NDCG@K": ndcg,
        }
        for j, (_, row) in enumerate(pred_top.iterrows()):
            trade[f"Pick{j+1}"] = row["fund_name"][:35]
            trade[f"Pick{j+1}%"] = row["forward_return"]*100
            trade[f"Pick{j+1}Hit"] = "✅" if row["fund_id"] in actual_set else "❌"
        
        trades.append(trade)
        results.append({
            "date": test_date, "entry": entry_dt, "exit": exit_dt,
            "port_return": port_ret, "bench_return": bench_ret,
            "hit_rate": hr, "hits": hits, "ndcg": ndcg,
            "pool": len(test_df), "perfect_return": perfect_ret,
        })
        eq.append({"date": exit_dt, "value": cap})
        bm.append({"date": exit_dt, "value": bcap})
        
        # Feature importance
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            feat_imp_list.append(dict(zip(FEATURE_COLS, model.model.feature_importances_)))
        elif hasattr(model, 'graded') and hasattr(model.graded.model, 'feature_importances_'):
            feat_imp_list.append(dict(zip(FEATURE_COLS, model.graded.model.feature_importances_)))
    
    prog.empty()
    
    avg_imp = {}
    if feat_imp_list:
        for f in feat_imp_list[0]:
            avg_imp[f] = np.mean([d.get(f, 0) for d in feat_imp_list])
    
    return (pd.DataFrame(results), pd.DataFrame(eq), pd.DataFrame(bm),
            pd.DataFrame(trades), {"importances": avg_imp, "last_model": last_model})


# ============================================================================
# 7. VISUALIZATION
# ============================================================================
def plot_equity(eq, bm, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["date"], y=eq["value"], name="ML Strategy",
        line=dict(color="#2196F3", width=2.5), fill="tozeroy", fillcolor="rgba(33,150,243,0.1)"))
    fig.add_trace(go.Scatter(x=bm["date"], y=bm["value"], name="Nifty 100",
        line=dict(color="gray", width=2, dash="dot")))
    fig.update_layout(title=title, yaxis_title="₹100 invested", hovermode="x unified",
                      height=400, legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"))
    return fig

def plot_hit_rate(res):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=res["date"], y=res["hit_rate"]*100,
        marker_color=["#4caf50" if h >= 0.4 else "#ff9800" if h >= 0.2 else "#f44336"
                       for h in res["hit_rate"]]))
    avg = res["hit_rate"].mean()*100
    fig.add_hline(y=avg, line_dash="dash", line_color="blue",
                  annotation_text=f"Avg: {avg:.1f}%")
    fig.update_layout(title="Hit Rate per Period", yaxis_title="HR %", height=350)
    return fig

def plot_importance(imp, top_n=20):
    if not imp: return None
    s = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    fig = go.Figure(go.Bar(x=[v for _,v in s], y=[n for n,_ in s], orientation="h",
        marker_color=px.colors.sequential.Viridis[:len(s)]))
    fig.update_layout(title="Feature Importance", height=450, yaxis=dict(autorange="reversed"))
    return fig

# ============================================================================
# 8. STREAMLIT UI
# ============================================================================
def render_main():
    st.markdown("""<div class="info-banner">
        <h2>🧠 ML Fund Predictor v2 — Learning to Rank</h2>
        <p>Pairwise Ranking + Graded Labels + 42 Features + Sliding Window Training</p>
    </div>""", unsafe_allow_html=True)

    # Controls
    c1,c2,c3,c4,c5 = st.columns([2,2,1,1,1])
    with c1: cat = st.selectbox("📁 Category", list(FILE_MAP.keys()))
    with c2:
        strat = st.selectbox("🤖 Strategy", [
            ("ensemble", "🧠 Ensemble (Graded+Return+MLP)"),
            ("graded", "📊 Graded Regression (percentile rank)"),
            ("return_reg", "📈 Return Regression (predict return)"),
            ("mlp", "🔮 MLP Neural Network"),
            ("pairwise", "🔗 Pairwise Ranking"),
        ], format_func=lambda x: x[1])[0]
    with c3: top_k = st.number_input("🎯 Pick K", 1, 10, 5)
    with c4: target_k = st.number_input("🏆 vs Top K", 1, 15, top_k)
    with c5: hold = st.selectbox("📅 Hold", HOLD_PERIODS, index=1, format_func=hold_label)

    adv = st.expander("⚙️ Advanced Settings")
    with adv:
        c1,c2,c3 = st.columns(3)
        with c1: step = st.number_input("Slide step (days)", 7, 63, 21, help="Training window slide. Lower = more training data.")
        with c2: min_periods = st.number_input("Min train periods", 4, 20, 8)
        with c3: run_baselines = st.checkbox("Compare vs rule-based", True)

    nav, smap = load_funds(cat)
    bench = load_bench()
    if nav is None: st.error("No data found."); return

    c1,c2,c3 = st.columns(3)
    c1.metric("Funds", len(nav.columns))
    c2.metric("Period", f"{nav.index.min().strftime('%Y-%m')} → {nav.index.max().strftime('%Y-%m')}")
    c3.metric("Features", len(FEATURE_COLS))
    st.divider()

    if st.button("🚀 Run ML Backtest v2", type="primary", use_container_width=True):
        # Build dataset
        st.markdown("### 📊 Step 1: Build Dataset (sliding window)")
        ds = build_dataset(nav, bench, smap, hold, target_k, step_days=step)
        if ds.empty or len(ds) < 200:
            st.error("Insufficient data."); return
        
        nd = ds["signal_date"].nunique()
        st.success(f"✅ **{len(ds):,}** samples · **{nd}** dates · "
                   f"Pos: {ds['binary_label'].sum():,} ({ds['binary_label'].mean()*100:.1f}%)")

        # Map strategy to model
        model_map = {
            "ensemble": (EnsembleRankModel, {}),
            "graded": (GradedRegressionModel, {}),
            "return_reg": (ReturnRegressionModel, {}),
            "mlp": (MLPRankModel, {}),
            "pairwise": (PairwiseRankModel, {}),
        }
        mcls, mkw = model_map[strat]

        st.markdown("### 🔄 Step 2: Walk-Forward Backtest")
        res, eq, bm, trd, info = run_backtest(
            ds, mcls, mkw, top_k, target_k,
            min_train_periods=min_periods, hold_days=hold)

        if res.empty: st.error("No results."); return

        # Store
        st.session_state["v2_res"] = res
        st.session_state["v2_eq"] = eq
        st.session_state["v2_bm"] = bm
        st.session_state["v2_trd"] = trd
        st.session_state["v2_info"] = info
        st.session_state["v2_ds"] = ds
        st.session_state["v2_nav"] = nav
        st.session_state["v2_bench"] = bench
        st.session_state["v2_smap"] = smap
        st.session_state["v2_strat"] = strat
        st.session_state["v2_topk"] = top_k
        st.session_state["v2_targetk"] = target_k
        
        # Run baselines
        if run_baselines:
            bl_results = {}
            baselines = {
                "Momentum 63d": ("ret_63d", False),
                "Momentum 252d": ("ret_252d", False),
                "Sharpe 252d": ("sharpe_252d", False),
                "CS Rank Composite": ("cs_rank_composite", False),
            }
            for bl_name, (col, asc) in baselines.items():
                bl_cls = RuleBasedModel
                bl_kw = {"sort_col": col, "ascending": asc}
                bl_res, _, _, _, _ = run_backtest(
                    ds, bl_cls, bl_kw, top_k, target_k,
                    min_train_periods=min_periods, hold_days=hold)
                if not bl_res.empty:
                    bl_results[bl_name] = {
                        "Hit Rate %": bl_res["hit_rate"].mean()*100,
                        "Avg Return %": bl_res["port_return"].mean()*100,
                        "NDCG@K": bl_res["ndcg"].mean(),
                    }
            st.session_state["v2_baselines"] = bl_results

    # Display results
    if "v2_res" not in st.session_state:
        st.info("👆 Click **Run ML Backtest v2** to begin.")
        _show_methodology()
        return

    res = st.session_state["v2_res"]
    eq = st.session_state["v2_eq"]
    bm = st.session_state["v2_bm"]
    trd = st.session_state["v2_trd"]
    info = st.session_state["v2_info"]

    # Metrics
    yrs = (eq.iloc[-1]["date"] - eq.iloc[0]["date"]).days / 365.25
    cagr = (eq.iloc[-1]["value"]/100)**(1/yrs)-1 if yrs > 0 else 0
    bcagr = (bm.iloc[-1]["value"]/100)**(1/yrs)-1 if yrs > 0 else 0
    avg_hr = res["hit_rate"].mean()
    avg_ndcg = res["ndcg"].mean()
    win_rate = (res["port_return"] > 0).mean()
    vals = eq["value"].values
    mdd = ((vals - np.maximum.accumulate(vals)) / np.maximum.accumulate(vals)).min()

    st.markdown("### 📈 Results")
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("ML CAGR", f"{cagr*100:.2f}%")
    c2.metric("Bench CAGR", f"{bcagr*100:.2f}%")
    c3.metric("Alpha", f"{(cagr-bcagr)*100:+.2f}%")
    c4.metric("Avg Hit Rate", f"{avg_hr*100:.1f}%")
    c5.metric("NDCG@K", f"{avg_ndcg:.3f}")
    c6.metric("Win Rate", f"{win_rate*100:.1f}%")
    c7.metric("Max DD", f"{mdd*100:.1f}%")

    tabs = st.tabs(["📈 Equity", "🎯 Hit Rate", "📋 Trades", "🏅 Features",
                     "📊 vs Baselines", "🔮 Today's Picks"])

    with tabs[0]:
        sname = st.session_state.get("v2_strat","").upper()
        st.plotly_chart(plot_equity(eq, bm, f"{sname} vs Benchmark"), use_container_width=True)

    with tabs[1]:
        st.plotly_chart(plot_hit_rate(res), use_container_width=True)
        # Distribution
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ["0-20%","20-40%","40-60%","60-80%","80-100%"]
        counts = pd.cut(res["hit_rate"], bins=bins, labels=labels).value_counts().sort_index()
        fig = go.Figure(go.Bar(x=labels, y=counts.values,
            marker_color=["#f44336","#ff9800","#ffc107","#8bc34a","#4caf50"]))
        fig.update_layout(title="HR Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        if not trd.empty:
            pct = [c for c in trd.columns if "%" in c]
            fmt = {c: "{:.2f}" for c in pct}
            fmt["Hits"] = "{:.0f}"; fmt["Pool"] = "{:.0f}"; fmt["NDCG@K"] = "{:.3f}"
            hit_cols = [c for c in trd.columns if "Hit" in c and "%" not in c]
            sty = trd.style.format(fmt, na_rep="")
            for c in hit_cols:
                sty = sty.map(lambda v: "background-color:#c8e6c9" if v=="✅"
                              else "background-color:#ffcdd2" if v=="❌" else "", subset=[c])
            st.dataframe(sty, use_container_width=True, height=600)
            st.download_button("📥 Download", trd.to_csv(index=False), "ml_v2_trades.csv")

    with tabs[3]:
        imp = info.get("importances", {})
        if imp:
            fig = plot_importance(imp)
            if fig: st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        bl = st.session_state.get("v2_baselines", {})
        if bl:
            comp = {"🧠 ML " + st.session_state.get("v2_strat","").upper(): {
                "Hit Rate %": avg_hr*100,
                "Avg Return %": res["port_return"].mean()*100,
                "NDCG@K": avg_ndcg,
            }}
            comp.update(bl)
            cdf = pd.DataFrame(comp).T
            cdf.index.name = "Strategy"
            st.dataframe(cdf.style.format({
                "Hit Rate %": "{:.1f}", "Avg Return %": "{:.2f}", "NDCG@K": "{:.3f}"
            }).background_gradient(subset=["Hit Rate %"], cmap="RdYlGn"), use_container_width=True)
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Hit Rate %", x=list(comp.keys()),
                y=[comp[s]["Hit Rate %"] for s in comp], marker_color="#2196F3"))
            fig.add_trace(go.Bar(name="NDCG@K", x=list(comp.keys()),
                y=[comp[s]["NDCG@K"]*100 for s in comp], marker_color="#4caf50"))
            fig.update_layout(barmode="group", title="ML vs Rule-Based", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enable 'Compare vs rule-based' and rerun.")

    with tabs[5]:
        nav = st.session_state.get("v2_nav")
        bench = st.session_state.get("v2_bench")
        smap = st.session_state.get("v2_smap")
        lm = info.get("last_model")
        
        if nav is not None and lm is not None:
            with st.spinner("Generating today's predictions..."):
                latest_snap = extract_features(nav, bench, len(nav)-1)
                if len(latest_snap) > 0:
                    try:
                        scores = lm.predict_scores(latest_snap)
                        latest_snap["pred_score"] = scores
                        picks = latest_snap.nlargest(st.session_state.get("v2_topk", 5), "pred_score")
                        
                        st.markdown(f"**Date:** {nav.index[-1].strftime('%Y-%m-%d')} | **Funds scored:** {len(latest_snap)}")
                        st.divider()
                        
                        cols = st.columns(min(len(picks), 5))
                        for idx, (_, row) in enumerate(picks.iterrows()):
                            with cols[idx % len(cols)]:
                                nm = smap.get(row["fund_id"], row["fund_id"])
                                r63 = row.get("ret_63d", 0) or 0
                                r252 = row.get("ret_252d", 0) or 0
                                sh = row.get("sharpe_252d", 0) or 0
                                st.markdown(f"""<div class="pick-card">
                                    <div class="rank">#{idx+1}</div>
                                    <div class="name">{nm[:50]}</div>
                                    <div class="score">Score: {row['pred_score']:.4f}</div>
                                    <div style="font-size:.8rem;color:#666;margin-top:4px">
                                        3M: {r63*100:.1f}% · 1Y: {r252*100:.1f}% · Sharpe: {sh:.2f}
                                    </div>
                                </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
        else:
            st.info("Run backtest first to train model.")


def _show_methodology():
    with st.expander("📖 Why v2 is Different — 5 Key Fixes", expanded=True):
        st.markdown("""
<div class="problem-box"><b>❌ Problem 1:</b> Binary labels (top-5 = 1, rest = 0) destroy information. 
Fund #6 gets same label as fund #50.</div>
<div class="fix-box"><b>✅ Fix:</b> <b>Graded labels</b> — use percentile rank (0.0 to 1.0). 
Fund #1 gets 1.0, #6 gets 0.88, #50 gets 0.02.</div>

<div class="problem-box"><b>❌ Problem 2:</b> Classification loss (cross-entropy) optimizes probability accuracy, 
not ranking quality. Small probability errors cause big ranking errors.</div>
<div class="fix-box"><b>✅ Fix:</b> <b>Pairwise ranking</b> — directly asks "is fund A better than fund B?" 
Plus regression on percentile rank which preserves ordinal structure.</div>

<div class="problem-box"><b>❌ Problem 3:</b> Non-overlapping windows → only ~15 training periods → ~750 samples.
Way too little for ML to learn patterns.</div>
<div class="fix-box"><b>✅ Fix:</b> <b>Sliding window</b> with step=21 days → 5-10x more training data (3000-8000 samples).
Test dates are still non-overlapping to avoid double-counting.</div>

<div class="problem-box"><b>❌ Problem 4:</b> Missing key features: no short-term momentum (5d), 
no stability metrics, no momentum ratios.</div>
<div class="fix-box"><b>✅ Fix:</b> <b>42 features</b> including 5d momentum, vol-of-vol, up/down ratio, 
quarterly consistency, momentum acceleration, and short/long momentum ratio.</div>

<div class="problem-box"><b>❌ Problem 5:</b> Single model can't capture all market regimes.</div>
<div class="fix-box"><b>✅ Fix:</b> <b>5 model strategies</b> including Pairwise Ranking, Graded Regression, 
Return Prediction, MLP, and Rank-Normalized Ensemble.</div>

### Evaluation Metric: NDCG@K
We also report **NDCG@K** (Normalized Discounted Cumulative Gain), the standard 
ranking quality metric. This measures not just *if* we got the right funds, 
but *how close* our ranking is to the ideal ranking. NDCG=1.0 means perfect ranking.
        """, unsafe_allow_html=True)


def main():
    st.markdown("""<div style="text-align:center;padding:10px 0 15px">
        <h1 style="margin:0;border:none">🧠 ML Fund Predictor v2</h1>
        <p style="color:#666">Learning-to-Rank · Pairwise · Graded Labels · 42 Features · Sliding Window</p>
    </div>""", unsafe_allow_html=True)
    render_main()
    st.caption("ML Fund Predictor v2 | Fixes: Graded labels, Pairwise ranking, "
               "Sliding window, 42 features, Ensemble | NDCG@K evaluation")


if __name__ == "__main__":
    main()
