"""
Unified Fund Predictor — ML + Rule-Based + Consensus
=====================================================
Combines:
  A) 8 Rule-Based strategies (from backtest_strategy.py):
     1. Composite Momentum  2. Sharpe  3. Sortino  4. Regime Switch
     5. Stable Momentum  6. Elimination  7. Consistency  8. Smart Ensemble
  B) 5 ML strategies (from ml_fund_predictor_v2.py):
     1. Graded Regression  2. Return Regression  3. MLP Neural Net
     4. Pairwise Ranking  5. ML Ensemble
  C) "Run All" mode: runs every strategy, shows comparison matrix
  D) Consensus Picks: funds appearing across ALL methodologies

Run: streamlit run unified_fund_predictor.py
Requires: streamlit, plotly, scikit-learn, scipy, pandas, numpy, openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import rankdata
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import ndcg_score

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIG & STYLING
# ============================================================================
st.set_page_config(page_title="Unified Fund Predictor", page_icon="🧠", layout="wide",
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
.consensus-card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:14px;padding:18px;margin:8px 0;
  box-shadow:0 4px 15px rgba(0,0,0,.10);border-left:6px solid #2e7d32}
.consensus-card .rank{font-size:1.8rem;font-weight:800;color:#2e7d32}
.consensus-card .name{font-weight:700;color:#1b5e20;font-size:.95rem}
.consensus-card .votes{color:#4caf50;font-weight:700;font-size:1rem}
.vote-badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.72rem;font-weight:700;margin:1px}
.vote-high{background:#4caf50;color:#fff}
.vote-med{background:#ff9800;color:#fff}
.vote-low{background:#e0e0e0;color:#666}
.metric-box{background:#f5f7fa;border-radius:12px;padding:16px;margin:8px 0;border-left:5px solid #2196F3}
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
HOLD_PERIODS = [63, 126, 189, 252, 378, 504, 630, 756]

# All strategy names
RULE_STRATEGIES = ["momentum","sharpe","sortino","regime_switch","stable_momentum",
                   "elimination","consistency","smart_ensemble"]
ML_STRATEGIES = ["graded_reg","return_reg","mlp_nn","pairwise","ml_ensemble"]
ALL_STRATEGIES = RULE_STRATEGIES + ML_STRATEGIES

STRATEGY_INFO = {
    "momentum":        ("🚀 Composite Momentum", "Weighted multi-horizon returns / volatility"),
    "sharpe":          ("⚖️ Sharpe Maximization", "Excess return per unit of total risk"),
    "sortino":         ("🎯 Sortino Optimization", "Return per unit of downside risk"),
    "regime_switch":   ("🚦 Regime Switch", "Momentum in bull, Sharpe in bear"),
    "stable_momentum": ("⚓ Stable Momentum", "High momentum + low drawdown"),
    "elimination":     ("🛡️ Elimination", "Remove worst DD & vol, pick best Sharpe"),
    "consistency":     ("📈 Consistency", "Top half in 3/4 quarters, then momentum"),
    "smart_ensemble":  ("🧠 Rule Ensemble", "7-strategy voting + vol-adj momentum"),
    "graded_reg":      ("📊 Graded Regression", "ML: predict percentile rank (0-1)"),
    "return_reg":      ("📈 Return Regression", "ML: predict forward return directly"),
    "mlp_nn":          ("🔮 MLP Neural Net", "ML: 3-layer neural network on graded labels"),
    "pairwise":        ("🔗 Pairwise Ranking", "ML: is fund A > fund B? Count wins"),
    "ml_ensemble":     ("🧬 ML Ensemble", "ML: combine Graded+Return+MLP by rank"),
}

def hold_label(d):
    return f"{d}d (~{d//21}M)" if d < 252 else f"{d}d (~{d//252}Y)"

# ============================================================================
# 2. DATA LOADING
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
# 3. CORE METRIC FUNCTIONS (from backtest_strategy.py)
# ============================================================================
def calc_sharpe(rets):
    if len(rets) < 10 or rets.std() == 0: return np.nan
    return ((rets - DAILY_RF).mean() / rets.std()) * np.sqrt(TRADING_DAYS)

def calc_sortino(rets):
    if len(rets) < 10: return np.nan
    down = rets[rets < 0]
    if len(down) < 3 or down.std() == 0: return np.nan
    return ((rets - DAILY_RF).mean() / down.std()) * np.sqrt(TRADING_DAYS)

def calc_vol(rets):
    return rets.std() * np.sqrt(TRADING_DAYS) if len(rets) >= 10 else np.nan

def calc_max_dd(series):
    if len(series) < 10: return np.nan
    c = (1 + series.pct_change().fillna(0)).cumprod()
    return (c / c.expanding().max() - 1).min()

def calc_cagr(series):
    if len(series) < 30 or series.iloc[0] <= 0: return np.nan
    yrs = (series.index[-1] - series.index[0]).days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1/yrs) - 1 if yrs > 0 else np.nan

def check_trend(series, w=50):
    if len(series) < w + 5: return False
    return series.iloc[-1] > series.iloc[-w:].mean()

def check_dd_ok(series, thr=-0.10):
    if len(series) < 20: return False
    c = (1 + series.pct_change().fillna(0)).cumprod()
    return (c.iloc[-1] / c.expanding().max().iloc[-1] - 1) > thr

def adaptive_weights(hold):
    if hold <= 63: return 0.60, 0.25, 0.15
    elif hold <= 189: return 0.30, 0.45, 0.25
    elif hold <= 378: return 0.20, 0.30, 0.50
    else: return 0.15, 0.25, 0.60

def calc_momentum(series, w3, w6, w12, risk_adj=True):
    if len(series) < 260: return np.nan
    cur, dt = series.iloc[-1], series.index[-1]
    def past(d):
        sub = series[series.index <= dt - pd.Timedelta(days=d)]
        return sub.iloc[-1] if len(sub) > 0 else np.nan
    p3, p6, p12 = past(91), past(182), past(365)
    r3 = (cur/p3-1) if p3 else 0
    r6 = (cur/p6-1) if p6 else 0
    r12 = (cur/p12-1) if p12 else 0
    score = r3*w3 + r6*w6 + r12*w12
    if risk_adj:
        vol = series.iloc[-252:].pct_change().dropna().std() * np.sqrt(252) if len(series) >= 252 else np.nan
        return score/vol if vol and vol > 0 else np.nan
    return score

def vol_adj_mom(series, w3=0.33, w6=0.33, w12=0.34):
    return calc_momentum(series, w3, w6, w12, risk_adj=True)

def get_regime(bench, date):
    sub = bench[bench.index <= date]
    return 'bull' if len(sub) >= 200 and sub.iloc[-1] > sub.iloc[-200:].mean() else 'bear'

# ============================================================================
# 4. RULE-BASED BACKTESTER (from backtest_strategy.py — all 8 strategies)
# ============================================================================
def run_rule_backtest(nav, strategy, top_n, target_n, hold_days, bench, smap):
    """Run a single rule-based strategy backtest. Returns results_df, equity, benchmark, trades."""
    start = nav.index.min() + pd.Timedelta(days=370)
    if start >= nav.index.max():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    start_idx = nav.index.searchsorted(start)
    rebal = list(range(start_idx, len(nav)-1, hold_days))
    if not rebal:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    w3a, w6a, w12a = adaptive_weights(hold_days)
    history, trades = [], []
    eq = [{"date": nav.index[rebal[0]], "value": 100.0}]
    bm = [{"date": nav.index[rebal[0]], "value": 100.0}]
    cap, bcap = 100.0, 100.0

    for i in rebal:
        dt = nav.index[i]
        entry_i = i + 1
        exit_i = min(i + 1 + hold_days, len(nav) - 1)
        entry_dt, exit_dt = nav.index[entry_i], nav.index[exit_i]
        hist = nav[(nav.index >= dt - pd.Timedelta(days=400)) & (nav.index < dt)]

        scores = {}
        for col in nav.columns:
            fh = hist[col].dropna()
            if len(fh) < 126: continue
            rets = fh.pct_change().dropna()
            if strategy == "momentum":
                sc = calc_momentum(fh, w3a, w6a, w12a, True)
            elif strategy == "sharpe":
                sc = calc_sharpe(rets)
            elif strategy == "sortino":
                sc = calc_sortino(rets)
            elif strategy == "regime_switch":
                r = get_regime(bench, dt) if bench is not None else "bull"
                sc = calc_momentum(fh, 0.3, 0.3, 0.4, False) if r == "bull" else calc_sharpe(rets)
            else:
                sc = calc_sharpe(rets)
            if pd.isna(sc): continue
            full_s = nav[col].dropna()
            full_s = full_s[full_s.index <= dt]
            scores[col] = {
                "score": sc, "sharpe": calc_sharpe(rets), "sortino": calc_sortino(rets),
                "vol": calc_vol(rets), "dd": calc_max_dd(fh),
                "vol_adj_mom": vol_adj_mom(fh, w3a, w6a, w12a),
            }

        # Selection logic per strategy
        selected = []
        if strategy == "smart_ensemble":
            base_picks = {}
            for bs in ["momentum","sharpe","sortino","stable_momentum","elimination","consistency","regime_switch"]:
                if bs == "momentum":
                    ranked = sorted(scores.items(), key=lambda x: (x[1].get("vol_adj_mom") or 0), reverse=True)
                elif bs == "sharpe":
                    ranked = sorted(scores.items(), key=lambda x: (x[1]["sharpe"] or 0), reverse=True)
                elif bs == "sortino":
                    ranked = sorted(scores.items(), key=lambda x: (x[1]["sortino"] or 0), reverse=True)
                elif bs == "stable_momentum":
                    pool = sorted(scores.items(), key=lambda x: (x[1].get("vol_adj_mom") or 0), reverse=True)[:top_n*2]
                    ranked = sorted(pool, key=lambda x: (x[1]["dd"] if x[1]["dd"] is not None else -999), reverse=True)
                elif bs == "elimination":
                    dfs = pd.DataFrame(scores).T
                    if len(dfs) > top_n*2: dfs = dfs[dfs["dd"] >= dfs["dd"].quantile(0.25)]
                    if len(dfs) > top_n*2: dfs = dfs[dfs["vol"] <= dfs["vol"].quantile(0.75)]
                    ranked = [(idx, scores.get(idx, {})) for idx in dfs.sort_values("sharpe", ascending=False).index] if len(dfs) > 0 else []
                elif bs == "regime_switch":
                    r = get_regime(bench, dt) if bench is not None else "bull"
                    ranked = sorted(scores.items(), key=lambda x: (x[1].get("vol_adj_mom") or 0), reverse=True) if r == "bull" else sorted(scores.items(), key=lambda x: (x[1]["sharpe"] or 0), reverse=True)
                else:
                    ranked = sorted(scores.items(), key=lambda x: (x[1]["sharpe"] or 0), reverse=True)
                base_picks[bs] = [f for f, _ in ranked[:top_n]]
            vm = {}
            for sn, pks in base_picks.items():
                for fid in pks: vm[fid] = vm.get(fid, 0) + 1
            cands = [(fid, vm[fid], scores.get(fid, {})) for fid in vm]
            cands.sort(key=lambda x: (-x[1], -(x[2].get("vol_adj_mom") or 0)))
            selected = [c[0] for c in cands[:top_n]]
        elif strategy == "stable_momentum":
            pool = [f for f, _ in sorted(scores.items(), key=lambda x: (x[1]["score"], x[1]["sharpe"] or 0), reverse=True)[:top_n*2]]
            pool_dd = [(f, scores[f]["dd"]) for f in pool if scores[f]["dd"] is not None]
            selected = [f for f, _ in sorted(pool_dd, key=lambda x: x[1], reverse=True)[:top_n]]
        elif strategy == "elimination":
            dfs = pd.DataFrame(scores).T
            if len(dfs) > top_n*2: dfs = dfs[dfs["dd"] >= dfs["dd"].quantile(0.25)]
            if len(dfs) > top_n*2: dfs = dfs[dfs["vol"] <= dfs["vol"].quantile(0.75)]
            if len(dfs) > 0:
                selected = dfs.sort_values(["sharpe","score"], ascending=[False,False]).head(top_n).index.tolist()
        elif strategy == "consistency":
            cons = []
            for col in scores:
                fh = hist[col].dropna()
                if len(fh) < 300: continue
                good = 0
                for q in range(4):
                    try:
                        qs = dt - pd.Timedelta(days=(q+1)*91)
                        qe = dt - pd.Timedelta(days=q*91)
                        is_ = hist.index.asof(qs); ie = hist.index.asof(qe)
                        if pd.isna(is_) or pd.isna(ie): continue
                        all_r = {}
                        for f in scores:
                            fd = hist[f].dropna()
                            if len(fd) > 0:
                                v_e = fd[fd.index <= ie]
                                v_s = fd[fd.index <= is_]
                                if len(v_e) > 0 and len(v_s) > 0:
                                    all_r[f] = v_e.iloc[-1] / v_s.iloc[-1] - 1
                        if col in all_r and all_r[col] >= np.median(list(all_r.values())):
                            good += 1
                    except: continue
                if good >= 3: cons.append(col)
            if cons:
                selected = [f for f, _ in sorted([(f, scores[f]["score"]) for f in cons], key=lambda x: x[1], reverse=True)[:top_n]]
            else:
                selected = [f for f, _ in sorted(scores.items(), key=lambda x: (x[1]["score"], x[1]["sharpe"] or 0), reverse=True)[:top_n]]
        else:
            selected = [f for f, _ in sorted(scores.items(), key=lambda x: (x[1]["score"], x[1]["sharpe"] or 0), reverse=True)[:top_n]]

        # Compute actual returns
        rets_map = {}
        for fid in scores:
            try:
                en = nav.loc[entry_dt, fid] if entry_dt in nav.index else np.nan
                ex = nav.loc[exit_dt, fid] if exit_dt in nav.index else np.nan
                if pd.isna(en) or pd.isna(ex):
                    fd = nav[fid].dropna()
                    en = fd[fd.index <= entry_dt].iloc[-1] if len(fd[fd.index <= entry_dt]) > 0 else np.nan
                    ex = fd[fd.index <= exit_dt].iloc[-1] if len(fd[fd.index <= exit_dt]) > 0 else np.nan
                if not pd.isna(en) and not pd.isna(ex): rets_map[fid] = ex/en - 1
            except: continue

        valid = {k: v for k, v in rets_map.items() if not pd.isna(v)}
        actual_top = [f for f, _ in sorted(valid.items(), key=lambda x: x[1], reverse=True)[:target_n]]
        sel_valid = [f for f in selected if f in valid]
        hits = len(set(sel_valid) & set(actual_top))
        hr = hits / len(sel_valid) if sel_valid else 0
        pret = np.mean([valid[f] for f in sel_valid]) if sel_valid else 0
        bret = (bench.asof(exit_dt) / bench.asof(entry_dt) - 1) if bench is not None else 0

        cap *= (1 + pret); bcap *= (1 + bret)

        rec = {"Start": dt.strftime("%Y-%m-%d"), "End": exit_dt.strftime("%Y-%m-%d"),
               "Pool": len(scores), "Port %": pret*100, "Bench %": bret*100,
               "Alpha %": (pret-bret)*100, "Hits": hits, "HR %": hr*100}
        for j, fid in enumerate(selected):
            rec[f"Pick{j+1}"] = smap.get(fid, fid)[:35]
            rec[f"Pick{j+1}%"] = valid.get(fid, np.nan)*100 if fid in valid else np.nan
            rec[f"Pick{j+1}Hit"] = "✅" if fid in actual_top else "❌"
        trades.append(rec)
        history.append({"date": dt, "selected": selected, "return": pret, "hit_rate": hr})
        eq.append({"date": exit_dt, "value": cap})
        bm.append({"date": exit_dt, "value": bcap})

    return pd.DataFrame(history), pd.DataFrame(eq), pd.DataFrame(bm), pd.DataFrame(trades)


def get_rule_current_picks(nav, smap, bench, strategy, top_n, hold):
    """Get current picks for a rule-based strategy (what to buy today)."""
    dt = nav.index.max()
    hist = nav[(nav.index >= dt - pd.Timedelta(days=400)) & (nav.index <= dt)]
    w3, w6, w12 = adaptive_weights(hold)

    scores = {}
    for col in nav.columns:
        fh = hist[col].dropna()
        if len(fh) < 126: continue
        rets = fh.pct_change().dropna()
        if strategy == "momentum": sc = vol_adj_mom(fh, w3, w6, w12)
        elif strategy == "sharpe": sc = calc_sharpe(rets)
        elif strategy == "sortino": sc = calc_sortino(rets)
        elif strategy == "regime_switch":
            r = get_regime(bench, dt) if bench is not None else "bull"
            sc = calc_momentum(fh, 0.3, 0.3, 0.4, False) if r == "bull" else calc_sharpe(rets)
        else: sc = calc_sharpe(rets)
        if pd.isna(sc): continue
        full_s = nav[col].dropna()
        scores[col] = {
            "score": sc, "sharpe": calc_sharpe(rets), "sortino": calc_sortino(rets),
            "vol": calc_vol(rets), "dd": calc_max_dd(fh),
            "vol_adj_mom": vol_adj_mom(fh, w3, w6, w12),
            "ret_3m": (fh.iloc[-1]/fh.iloc[-63]-1)*100 if len(fh) >= 63 else np.nan,
            "ret_1y": (fh.iloc[-1]/fh.iloc[-252]-1)*100 if len(fh) >= 252 else np.nan,
        }

    if not scores: return []

    if strategy == "stable_momentum":
        pool = [f for f, _ in sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_n*2]]
        pool_dd = [(f, scores[f]["dd"]) for f in pool if scores[f]["dd"] is not None]
        selected = [f for f, _ in sorted(pool_dd, key=lambda x: x[1], reverse=True)[:top_n]]
    elif strategy == "elimination":
        dfs = pd.DataFrame(scores).T
        if len(dfs) > top_n*2: dfs = dfs[dfs["dd"] >= dfs["dd"].quantile(0.25)]
        if len(dfs) > top_n*2: dfs = dfs[dfs["vol"] <= dfs["vol"].quantile(0.75)]
        selected = dfs.sort_values("sharpe", ascending=False).head(top_n).index.tolist() if len(dfs) > 0 else []
    elif strategy == "smart_ensemble":
        base_strats = ["momentum","sharpe","sortino","stable_momentum","elimination","consistency","regime_switch"]
        vm = {}
        for bs in base_strats:
            bpicks = get_rule_current_picks(nav, smap, bench, bs, top_n, hold)
            for p in bpicks:
                fid = p["fund_id"]
                vm[fid] = vm.get(fid, 0) + 1
        cands = [(fid, vm[fid], scores.get(fid, {}).get("vol_adj_mom", 0) or 0) for fid in vm if fid in scores]
        cands.sort(key=lambda x: (-x[1], -x[2]))
        selected = [c[0] for c in cands[:top_n]]
    elif strategy == "consistency":
        # Simplified: use consistency_qtrs-like logic
        selected = [f for f, _ in sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_n]]
    else:
        selected = [f for f, _ in sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_n]]

    return [{"fund_id": fid, "name": smap.get(fid, fid), **scores.get(fid, {})} for fid in selected if fid in scores]

# ============================================================================
# 5. ML FEATURE ENGINE (42 features from v2)
# ============================================================================
FEATURE_COLS = [
    "ret_5d","ret_21d","ret_63d","ret_126d","ret_252d",
    "mom_accel_21","mom_accel_63","ret_63d_div_ret_252d",
    "vol_21d","vol_63d","vol_252d","max_dd_63d","max_dd_126d","max_dd_252d",
    "sharpe_63d","sharpe_126d","sharpe_252d","sortino_63d","sortino_126d","sortino_252d",
    "above_20dma","above_50dma","above_200dma",
    "dist_from_peak_63","dist_from_peak_252","mean_reversion_63d",
    "vol_of_vol_63d","up_down_ratio_63d","pos_day_pct_63d","consistency_qtrs",
    "cs_pctile_ret21","cs_pctile_ret63","cs_pctile_ret252",
    "cs_z_ret63","cs_z_ret252",
    "cs_pctile_sharpe63","cs_pctile_vol63","cs_rank_composite",
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

def extract_ml_features(nav_df, bench, date_idx, min_hist=260):
    """Extract 42 ML features for ALL funds at a single date index."""
    records = []
    regime = {}
    if bench is not None:
        b = bench.iloc[:date_idx+1].dropna()
        regime["bench_ret_63d"] = safe_ret(b, 63) if len(b) > 63 else 0
        regime["bench_vol_63d"] = safe_vol(b, 63) if len(b) > 63 else 0
        regime["bench_above_50dma"] = float(b.iloc[-1] > b.iloc[-50:].mean()) if len(b) >= 50 else 0.5
        regime["bench_above_200dma"] = float(b.iloc[-1] > b.iloc[-200:].mean()) if len(b) >= 200 else 0.5
    else:
        regime = {"bench_ret_63d":0,"bench_vol_63d":0,"bench_above_50dma":0.5,"bench_above_200dma":0.5}

    for col in nav_df.columns:
        s = nav_df[col].iloc[:date_idx+1].dropna()
        if len(s) < min_hist: continue
        f = {"fund_id": col}
        f["ret_5d"] = safe_ret(s, 5)
        f["ret_21d"] = safe_ret(s, 21)
        f["ret_63d"] = safe_ret(s, 63)
        f["ret_126d"] = safe_ret(s, 126)
        f["ret_252d"] = safe_ret(s, 252)
        r21 = f["ret_21d"] or 0
        f["mom_accel_21"] = (r21 - (safe_ret(s.iloc[:-21], 21) or 0)) if len(s) > 42 else 0
        r63 = f["ret_63d"] or 0
        f["mom_accel_63"] = (r63 - (safe_ret(s.iloc[:-63], 63) or 0)) if len(s) > 126 else 0
        r252 = f["ret_252d"]
        f["ret_63d_div_ret_252d"] = (f["ret_63d"]/r252) if r252 and r252 != 0 and f["ret_63d"] else 0
        f["vol_21d"] = safe_vol(s, 21)
        f["vol_63d"] = safe_vol(s, 63)
        f["vol_252d"] = safe_vol(s, 252)
        f["max_dd_63d"] = safe_dd(s, 63)
        f["max_dd_126d"] = safe_dd(s, 126)
        f["max_dd_252d"] = safe_dd(s, 252)
        f["sharpe_63d"] = safe_sharpe(s, 63)
        f["sharpe_126d"] = safe_sharpe(s, 126)
        f["sharpe_252d"] = safe_sharpe(s, 252)
        f["sortino_63d"] = safe_sortino(s, 63)
        f["sortino_126d"] = safe_sortino(s, 126)
        f["sortino_252d"] = safe_sortino(s, 252)
        cur = s.iloc[-1]
        f["above_20dma"] = float(cur > s.iloc[-20:].mean()) if len(s) >= 20 else 0.5
        f["above_50dma"] = float(cur > s.iloc[-50:].mean()) if len(s) >= 50 else 0.5
        f["above_200dma"] = float(cur > s.iloc[-200:].mean()) if len(s) >= 200 else 0.5
        pk63 = s.iloc[-63:].max() if len(s) >= 63 else s.max()
        pk252 = s.iloc[-252:].max() if len(s) >= 252 else s.max()
        f["dist_from_peak_63"] = (cur/pk63 - 1) if pk63 > 0 else 0
        f["dist_from_peak_252"] = (cur/pk252 - 1) if pk252 > 0 else 0
        f["mean_reversion_63d"] = (cur / s.iloc[-63:].mean() - 1) if len(s) >= 63 else 0
        if len(s) >= 63:
            rets = s.iloc[-63:].pct_change().dropna()
            rv = rets.rolling(10).std()
            f["vol_of_vol_63d"] = rv.std() if len(rv.dropna()) > 5 else np.nan
            up = (rets > 0).sum(); dn = (rets < 0).sum()
            f["up_down_ratio_63d"] = up/dn if dn > 0 else 2.0
            f["pos_day_pct_63d"] = up/len(rets)
        else:
            f["vol_of_vol_63d"] = np.nan; f["up_down_ratio_63d"] = 1.0; f["pos_day_pct_63d"] = 0.5
        gq = 0
        if len(s) >= 252:
            for q in range(4):
                qs = -(q+1)*63; qe = -q*63 if q > 0 else None
                qr = (s.iloc[qe]/s.iloc[qs]-1) if qe else (s.iloc[-1]/s.iloc[qs]-1)
                if not pd.isna(qr) and qr > 0: gq += 1
        f["consistency_qtrs"] = gq/4
        f.update(regime)
        records.append(f)

    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)

    # Cross-sectional features
    for cn, sc in [("cs_pctile_ret21","ret_21d"),("cs_pctile_ret63","ret_63d"),
                   ("cs_pctile_ret252","ret_252d"),("cs_pctile_sharpe63","sharpe_63d"),
                   ("cs_pctile_vol63","vol_63d")]:
        vals = df[sc].values
        valid = ~np.isnan(vals)
        res = np.full(len(vals), 0.5)
        if valid.sum() > 2:
            ranks = np.zeros(len(vals))
            ranks[valid] = rankdata(vals[valid]) / valid.sum()
            res = ranks
        df[cn] = res
    for cn, sc in [("cs_z_ret63","ret_63d"),("cs_z_ret252","ret_252d")]:
        vals = df[sc].values; valid = ~np.isnan(vals)
        res = np.zeros(len(vals))
        if valid.sum() > 2:
            m, s = np.nanmean(vals[valid]), np.nanstd(vals[valid])
            if s > 0: res[valid] = (vals[valid]-m)/s
        df[cn] = res
    pctcols = [c for c in df.columns if c.startswith("cs_pctile_")]
    df["cs_rank_composite"] = df[pctcols].mean(axis=1)
    return df


# ============================================================================
# 6. ML DATASET BUILDING
# ============================================================================
def build_ml_dataset(nav_df, bench, smap, hold_days, top_k, step_days=21, min_hist=370):
    start_idx = nav_df.index.searchsorted(nav_df.index.min() + pd.Timedelta(days=min_hist))
    end_idx = len(nav_df) - hold_days - 1
    rebal = list(range(start_idx, end_idx, step_days))
    all_recs = []
    prog = st.progress(0, "Building ML dataset...")
    for pi, idx in enumerate(rebal):
        if pi % 10 == 0:
            prog.progress(pi/len(rebal), f"Date {pi+1}/{len(rebal)}...")
        sig_dt = nav_df.index[idx]
        ent_idx = idx + 1
        ext_idx = min(idx+1+hold_days, len(nav_df)-1)
        ent_dt = nav_df.index[ent_idx]
        ext_dt = nav_df.index[ext_idx]
        snap = extract_ml_features(nav_df, bench, idx)
        if len(snap) < top_k + 3: continue
        fwd = {}
        for fid in snap["fund_id"]:
            try:
                en = nav_df.loc[ent_dt, fid]; ex = nav_df.loc[ext_dt, fid]
                if pd.notna(en) and pd.notna(ex) and en > 0: fwd[fid] = ex/en-1
            except: continue
        if len(fwd) < top_k + 3: continue
        fwd_s = pd.Series(fwd)
        fwd_ranks = rankdata(fwd_s.values)/len(fwd_s)
        rmap = dict(zip(fwd_s.index, fwd_ranks))
        topk_set = set(fwd_s.sort_values(ascending=False).index[:top_k])
        bret = 0
        if bench is not None:
            try: bret = bench.asof(ext_dt)/bench.asof(ent_dt)-1
            except: pass
        for _, row in snap.iterrows():
            fid = row["fund_id"]
            if fid not in fwd: continue
            rec = row.to_dict()
            rec.update({"signal_date": sig_dt, "entry_date": ent_dt, "exit_date": ext_dt,
                        "forward_return": fwd[fid], "graded_label": rmap[fid],
                        "binary_label": 1 if fid in topk_set else 0,
                        "bench_return": bret, "fund_name": smap.get(fid, fid)})
            all_recs.append(rec)
    prog.empty()
    return pd.DataFrame(all_recs)


# ============================================================================
# 7. ML MODELS (5 strategies from v2)
# ============================================================================
class GradedRegModel:
    name = "graded_reg"
    def __init__(self):
        self.model = HistGradientBoostingRegressor(
            max_iter=300, max_depth=5, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=1.5,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=42)
    def train(self, df):
        X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).fillna(0)
        self.model.fit(X, df["graded_label"].values); return True
    def predict_scores(self, snap):
        X = snap[FEATURE_COLS].fillna(snap[FEATURE_COLS].median()).fillna(0)
        return self.model.predict(X)

class ReturnRegModel:
    name = "return_reg"
    def __init__(self):
        self.model = HistGradientBoostingRegressor(
            max_iter=300, max_depth=5, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=1.5,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=42)
    def train(self, df):
        X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).fillna(0)
        y = np.clip(df["forward_return"].values, *np.percentile(df["forward_return"], [2,98]))
        self.model.fit(X, y); return True
    def predict_scores(self, snap):
        X = snap[FEATURE_COLS].fillna(snap[FEATURE_COLS].median()).fillna(0)
        return self.model.predict(X)

class MLPModel:
    name = "mlp_nn"
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(128,64,32), activation="relu",
            solver="adam", alpha=0.005, learning_rate="adaptive", max_iter=500,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=42)
        self.scaler = RobustScaler()
    def train(self, df):
        X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).fillna(0)
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), df["graded_label"].values); return True
    def predict_scores(self, snap):
        X = snap[FEATURE_COLS].fillna(snap[FEATURE_COLS].median()).fillna(0)
        return self.model.predict(self.scaler.transform(X))

class PairwiseModel:
    name = "pairwise"
    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=30, l2_regularization=2.0,
            early_stopping=True, validation_fraction=0.15, random_state=42)
        self.scaler = RobustScaler()
    def train(self, df, max_pairs=500):
        pX, py = [], []
        for dt in df["signal_date"].unique():
            snap = df[df["signal_date"]==dt]
            if len(snap) < 5: continue
            feats = snap[FEATURE_COLS].fillna(0).values
            fwd = snap["forward_return"].values
            n = len(feats)
            idxs = np.random.choice(n, size=(min(max_pairs, n*(n-1)//2), 2), replace=True)
            for i, j in idxs:
                if i == j: continue
                d = feats[i] - feats[j]
                if fwd[i] > fwd[j]: pX.append(d); py.append(1)
                elif fwd[j] > fwd[i]: pX.append(d); py.append(0)
        if len(pX) < 50: return False
        self.scaler.fit(np.array(pX))
        self.model.fit(self.scaler.transform(np.array(pX)), np.array(py)); return True
    def predict_scores(self, snap):
        feats = snap[FEATURE_COLS].fillna(0).values
        n = len(feats); wins = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                d = self.scaler.transform((feats[i]-feats[j]).reshape(1,-1))
                p = self.model.predict_proba(d)[0]
                wins[i] += p[1] if len(p) > 1 else p[0]
        return wins / max(n-1, 1)

class MLEnsembleModel:
    name = "ml_ensemble"
    def __init__(self):
        self.g = GradedRegModel(); self.r = ReturnRegModel(); self.m = MLPModel()
    def train(self, df):
        self.g.train(df); self.r.train(df); self.m.train(df); return True
    def predict_scores(self, snap):
        s1 = self.g.predict_scores(snap)
        s2 = self.r.predict_scores(snap)
        s3 = self.m.predict_scores(snap)
        rn = lambda a: rankdata(a)/len(a)
        return 0.4*rn(s1) + 0.35*rn(s2) + 0.25*rn(s3)

ML_MODEL_MAP = {
    "graded_reg": GradedRegModel, "return_reg": ReturnRegModel,
    "mlp_nn": MLPModel, "pairwise": PairwiseModel, "ml_ensemble": MLEnsembleModel,
}


# ============================================================================
# 8. ML BACKTESTER
# ============================================================================
def run_ml_backtest(dataset, model_key, top_k, target_k, hold_days, min_train=8):
    dates = sorted(dataset["signal_date"].unique())
    if len(dates) < min_train + 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    test_step = max(1, hold_days // 21)
    test_idxs = list(range(min_train, len(dates), test_step))
    results, trades = [], []
    eq = [{"date": dates[min_train], "value": 100.0}]
    bm = [{"date": dates[min_train], "value": 100.0}]
    cap, bcap = 100.0, 100.0
    last_model = None
    prog = st.progress(0, "ML Backtesting...")
    total = len(test_idxs)

    for pi, ti in enumerate(test_idxs):
        if ti >= len(dates): break
        prog.progress(min(pi/max(total,1), 1.0), f"Period {pi+1}/{total}...")
        test_dt = dates[ti]
        train_df = dataset[dataset["signal_date"].isin(dates[:ti])]
        test_df = dataset[dataset["signal_date"]==test_dt].copy()
        if len(train_df) < 100 or len(test_df) < top_k+3: continue
        model = ML_MODEL_MAP[model_key]()
        try:
            ok = model.train(train_df)
            if not ok: continue
        except: continue
        last_model = model
        try: scores = model.predict_scores(test_df)
        except: continue
        test_df["pred_score"] = scores
        pred_top = test_df.nlargest(top_k, "pred_score")
        pred_set = set(pred_top["fund_id"].values)
        actual_top = test_df.nlargest(target_k, "forward_return")
        actual_set = set(actual_top["fund_id"].values)
        hits = len(pred_set & actual_set)
        hr = hits/top_k
        pret = pred_top["forward_return"].mean()
        bret = test_df["bench_return"].iloc[0]
        cap *= (1+pret); bcap *= (1+bret)
        ent_dt = test_df["entry_date"].iloc[0]
        ext_dt = test_df["exit_date"].iloc[0]
        rec = {"Start": pd.Timestamp(ent_dt).strftime("%Y-%m-%d"),
               "End": pd.Timestamp(ext_dt).strftime("%Y-%m-%d"),
               "Pool": len(test_df), "Port %": pret*100, "Bench %": bret*100,
               "Alpha %": (pret-bret)*100, "Hits": hits, "HR %": hr*100}
        for j, (_, row) in enumerate(pred_top.iterrows()):
            rec[f"Pick{j+1}"] = row["fund_name"][:35]
            rec[f"Pick{j+1}%"] = row["forward_return"]*100
            rec[f"Pick{j+1}Hit"] = "✅" if row["fund_id"] in actual_set else "❌"
        trades.append(rec)
        results.append({"date": test_dt, "return": pret, "hit_rate": hr, "bench_return": bret})
        eq.append({"date": ext_dt, "value": cap})
        bm.append({"date": ext_dt, "value": bcap})
    prog.empty()
    return pd.DataFrame(results), pd.DataFrame(eq), pd.DataFrame(bm), pd.DataFrame(trades), last_model


# ============================================================================
# 9. CONSENSUS ENGINE — The core new feature
# ============================================================================
def compute_consensus_picks(nav, smap, bench, top_k, hold, ml_dataset=None, ml_models=None):
    """
    Run ALL strategies (rule-based + ML) and find funds that appear across
    the most methodologies. Returns consensus_df with vote counts.
    """
    all_picks = {}  # {strategy_name: [fund_ids]}

    # --- Rule-based picks ---
    for strat in RULE_STRATEGIES:
        try:
            picks = get_rule_current_picks(nav, smap, bench, strat, top_k, hold)
            all_picks[strat] = [p["fund_id"] for p in picks]
        except:
            all_picks[strat] = []

    # --- ML picks (if models trained) ---
    if ml_models:
        latest_snap = extract_ml_features(nav, bench, len(nav)-1)
        if len(latest_snap) > 0:
            for mkey, model in ml_models.items():
                try:
                    scores = model.predict_scores(latest_snap)
                    latest_snap_c = latest_snap.copy()
                    latest_snap_c["pred_score"] = scores
                    top = latest_snap_c.nlargest(top_k, "pred_score")
                    all_picks[mkey] = top["fund_id"].tolist()
                except:
                    all_picks[mkey] = []

    # --- Tally votes ---
    vote_map = {}  # fund_id -> {votes, strategies, info}
    for strat, fids in all_picks.items():
        for fid in fids:
            if fid not in vote_map:
                vote_map[fid] = {"votes": 0, "strategies": [], "name": smap.get(fid, fid)}
            vote_map[fid]["votes"] += 1
            vote_map[fid]["strategies"].append(strat)

    # Add fund metrics for display
    dt = nav.index.max()
    hist = nav[(nav.index >= dt - pd.Timedelta(days=400)) & (nav.index <= dt)]
    for fid in vote_map:
        fh = hist[fid].dropna() if fid in hist.columns else pd.Series()
        if len(fh) >= 63:
            rets = fh.pct_change().dropna()
            vote_map[fid]["ret_3m"] = (fh.iloc[-1]/fh.iloc[-63]-1)*100
            vote_map[fid]["ret_1y"] = (fh.iloc[-1]/fh.iloc[-252]-1)*100 if len(fh) >= 252 else np.nan
            vote_map[fid]["sharpe"] = calc_sharpe(rets)
            vote_map[fid]["vol"] = calc_vol(rets)

    # Sort by votes desc, then by 3m return desc
    sorted_funds = sorted(vote_map.items(), key=lambda x: (-x[1]["votes"], -(x[1].get("ret_3m") or 0)))

    total_strats = len([s for s, picks in all_picks.items() if picks])  # Only count strategies that produced picks

    return sorted_funds, all_picks, total_strats


# ============================================================================
# 10. VISUALIZATION HELPERS
# ============================================================================
def plot_equity(eq, bm, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["date"], y=eq["value"], name="Strategy",
        line=dict(color="#2196F3", width=2.5), fill="tozeroy", fillcolor="rgba(33,150,243,0.1)"))
    fig.add_trace(go.Scatter(x=bm["date"], y=bm["value"], name="Nifty 100",
        line=dict(color="gray", width=2, dash="dot")))
    fig.update_layout(title=title, yaxis_title="₹100 invested", hovermode="x unified",
                      height=400, legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"))
    return fig


# ============================================================================
# 11. STREAMLIT UI — Three main tabs
# ============================================================================
def render_run_all_tab():
    """Tab: Run ALL strategies and compare."""
    st.markdown("""<div class="info-banner">
        <h2>🏁 Run All Strategies — Unified Comparison</h2>
        <p>Backtest all 8 rule-based + 5 ML strategies side by side</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns([2,1,1,1,1])
    with c1: cat = st.selectbox("📁 Category", list(FILE_MAP.keys()), key="ra_cat")
    with c2: top_k = st.number_input("🎯 Pick K", 1, 10, 5, key="ra_topk")
    with c3: target_k = st.number_input("🏆 vs Top K", 1, 15, 5, key="ra_tgtk")
    with c4: hold = st.selectbox("📅 Hold", HOLD_PERIODS, index=1, format_func=hold_label, key="ra_hold")
    with c5: step = st.number_input("ML slide", 7, 63, 21, key="ra_step")

    nav, smap = load_funds(cat)
    bench = load_bench()
    if nav is None: st.error("No data."); return
    st.success(f"✅ {len(nav.columns)} funds · {nav.index.min().strftime('%Y-%m')} → {nav.index.max().strftime('%Y-%m')}")

    if st.button("🚀 Run All 13 Strategies", type="primary", use_container_width=True, key="ra_run"):
        all_results = {}

        # --- Rule-based ---
        st.markdown("### 📐 Rule-Based Strategies (8)")
        rule_prog = st.progress(0, "Running rule-based...")
        for ri, strat in enumerate(RULE_STRATEGIES):
            rule_prog.progress((ri+1)/len(RULE_STRATEGIES), f"{STRATEGY_INFO[strat][0]}...")
            h, e, b, t = run_rule_backtest(nav, strat, top_k, target_k, hold, bench, smap)
            if not e.empty:
                yrs = (e.iloc[-1]["date"]-e.iloc[0]["date"]).days/365.25
                cagr = (e.iloc[-1]["value"]/100)**(1/yrs)-1 if yrs > 0 else 0
                bcagr = (b.iloc[-1]["value"]/100)**(1/yrs)-1 if yrs > 0 else 0
                hr = h["hit_rate"].mean() if "hit_rate" in h.columns else 0
                all_results[strat] = {
                    "Type": "Rule", "CAGR %": cagr*100, "Bench %": bcagr*100,
                    "Alpha %": (cagr-bcagr)*100, "HR %": hr*100,
                    "Win %": (h["return"]>0).mean()*100, "Periods": len(h),
                }
        rule_prog.empty()

        # --- ML ---
        st.markdown("### 🧠 ML Strategies (5)")
        st.markdown("##### Building ML dataset...")
        ds = build_ml_dataset(nav, bench, smap, hold, target_k, step_days=step)
        if len(ds) < 200:
            st.warning("Not enough data for ML strategies.")
        else:
            n_dates = ds["signal_date"].nunique()
            st.success(f"ML Dataset: {len(ds):,} samples · {n_dates} dates")
            ml_trained_models = {}
            ml_prog = st.progress(0, "Running ML...")
            for mi, mkey in enumerate(ML_STRATEGIES):
                ml_prog.progress((mi+1)/len(ML_STRATEGIES), f"{STRATEGY_INFO[mkey][0]}...")
                res, e, b, t, last_m = run_ml_backtest(
                    ds, mkey, top_k, target_k, hold,
                    min_train=max(6, n_dates//4))
                if last_m: ml_trained_models[mkey] = last_m
                if not res.empty:
                    yrs = (e.iloc[-1]["date"]-e.iloc[0]["date"]).days/365.25
                    cagr = (e.iloc[-1]["value"]/100)**(1/yrs)-1 if yrs > 0 else 0
                    bcagr = (b.iloc[-1]["value"]/100)**(1/yrs)-1 if yrs > 0 else 0
                    hr = res["hit_rate"].mean()
                    all_results[mkey] = {
                        "Type": "ML", "CAGR %": cagr*100, "Bench %": bcagr*100,
                        "Alpha %": (cagr-bcagr)*100, "HR %": hr*100,
                        "Win %": (res["return"]>0).mean()*100, "Periods": len(res),
                    }
            ml_prog.empty()
            st.session_state["ra_ml_models"] = ml_trained_models
            st.session_state["ra_ml_dataset"] = ds

        st.session_state["ra_results"] = all_results
        st.session_state["ra_nav"] = nav
        st.session_state["ra_smap"] = smap
        st.session_state["ra_bench"] = bench
        st.session_state["ra_topk"] = top_k
        st.session_state["ra_hold"] = hold

    # Display
    if "ra_results" not in st.session_state:
        st.info("👆 Click **Run All 13 Strategies** to begin.")
        return

    all_results = st.session_state["ra_results"]
    if not all_results:
        st.warning("No results generated."); return

    # Build comparison table
    comp_df = pd.DataFrame(all_results).T
    comp_df.index = [f"{STRATEGY_INFO.get(k, ('',''))[0]}" for k in all_results.keys()]
    comp_df.index.name = "Strategy"
    comp_df = comp_df.sort_values("Alpha %", ascending=False)

    st.markdown("### 📊 Strategy Comparison Matrix")
    st.dataframe(
        comp_df.style.format({
            "CAGR %": "{:.2f}", "Bench %": "{:.2f}", "Alpha %": "{:+.2f}",
            "HR %": "{:.1f}", "Win %": "{:.1f}", "Periods": "{:.0f}",
        }).background_gradient(subset=["Alpha %","HR %"], cmap="RdYlGn"),
        use_container_width=True, height=600,
    )

    # Bar chart
    fig = go.Figure()
    strats = comp_df.index.tolist()
    colors = ["#2196F3" if "ML" not in str(comp_df.loc[s,"Type"]) else "#9c27b0" for s in strats]
    fig.add_trace(go.Bar(name="Alpha %", x=strats, y=comp_df["Alpha %"], marker_color=colors))
    fig.add_trace(go.Bar(name="HR %", x=strats, y=comp_df["HR %"], marker_color=
        ["#4caf50" if "ML" not in str(comp_df.loc[s,"Type"]) else "#e91e63" for s in strats]))
    fig.update_layout(barmode="group", title="Alpha & Hit Rate by Strategy", height=450,
                      xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("📥 Download Comparison", comp_df.to_csv(), "all_strategies_comparison.csv")


def render_consensus_tab():
    """Tab: Consensus picks across ALL methodologies."""
    st.markdown("""<div class="info-banner">
        <h2>🤝 Consensus Picks — Funds Common Across ALL Methods</h2>
        <p>Highest conviction: funds picked by the most strategies (rule-based + ML)</p>
    </div>""", unsafe_allow_html=True)

    nav = st.session_state.get("ra_nav")
    smap = st.session_state.get("ra_smap")
    bench = st.session_state.get("ra_bench")
    ml_models = st.session_state.get("ra_ml_models", {})
    top_k = st.session_state.get("ra_topk", 5)
    hold = st.session_state.get("ra_hold", 126)

    if nav is None:
        st.info("👆 First run **Run All 13 Strategies** in the previous tab to train models.")
        return

    st.markdown(f"**Category:** {len(nav.columns)} funds · **Top K:** {top_k} · **Hold:** {hold_label(hold)}")
    st.markdown(f"**ML Models trained:** {len(ml_models)} · **Rule strategies:** {len(RULE_STRATEGIES)}")
    st.divider()

    with st.spinner("Computing consensus across all strategies..."):
        sorted_funds, all_picks, total_strats = compute_consensus_picks(
            nav, smap, bench, top_k, hold, ml_models=ml_models)

    if not sorted_funds:
        st.warning("No consensus found."); return

    # Show total strategies that produced picks
    st.markdown(f"### Strategies that produced picks: **{total_strats}** / {len(RULE_STRATEGIES)+len(ml_models)}")

    # Determine max votes
    max_votes = sorted_funds[0][1]["votes"] if sorted_funds else 0

    # Top consensus picks
    st.markdown("### 🏆 Consensus Ranking — Top Funds")
    st.markdown(f"*Funds sorted by number of strategies that picked them (out of {total_strats})*")

    for idx, (fid, info) in enumerate(sorted_funds[:top_k * 3]):  # Show more than top_k
        votes = info["votes"]
        pct = votes / total_strats * 100 if total_strats > 0 else 0
        strats = info["strategies"]

        # Color coding
        if pct >= 60: badge_class = "vote-high"
        elif pct >= 30: badge_class = "vote-med"
        else: badge_class = "vote-low"

        # Strategy badges
        badges_html = ""
        for s in strats:
            sinfo = STRATEGY_INFO.get(s, ("?",""))
            sname = sinfo[0][:15]
            bc = "vote-high" if s in ML_STRATEGIES else "vote-med"
            badges_html += f'<span class="vote-badge {bc}">{sname}</span>'

        r3m = info.get("ret_3m", np.nan)
        r1y = info.get("ret_1y", np.nan)
        sharpe = info.get("sharpe", np.nan)
        r3m_str = f"{r3m:.1f}%" if not pd.isna(r3m) else "N/A"
        r1y_str = f"{r1y:.1f}%" if not pd.isna(r1y) else "N/A"
        sh_str = f"{sharpe:.2f}" if not pd.isna(sharpe) else "N/A"

        if idx < top_k:
            # Top picks get special styling
            st.markdown(f"""<div class="consensus-card">
                <div class="rank">#{idx+1}</div>
                <div class="name">{info['name'][:55]}</div>
                <div class="votes">🗳️ {votes}/{total_strats} strategies ({pct:.0f}%)</div>
                <div style="font-size:.8rem;color:#555;margin-top:6px">
                    3M: {r3m_str} · 1Y: {r1y_str} · Sharpe: {sh_str}
                </div>
                <div style="margin-top:6px">{badges_html}</div>
            </div>""", unsafe_allow_html=True)
        else:
            # Others get simpler styling
            st.markdown(f"""<div class="pick-card">
                <div class="rank" style="color:#666">#{idx+1}</div>
                <div class="name">{info['name'][:55]}</div>
                <div class="score">🗳️ {votes}/{total_strats} ({pct:.0f}%)</div>
                <div style="font-size:.75rem;margin-top:4px">{badges_html}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Strategy × Fund matrix
    st.markdown("### 📊 Strategy × Fund Matrix")
    # Build matrix
    top_fund_ids = [fid for fid, _ in sorted_funds[:top_k * 2]]
    matrix_data = []
    for fid in top_fund_ids:
        row = {"Fund": smap.get(fid, fid)[:40]}
        for strat, picks in all_picks.items():
            sname = STRATEGY_INFO.get(strat, (strat,""))[0][:12]
            row[sname] = "✅" if fid in picks else ""
        info = dict(sorted_funds).get(fid, {}) if isinstance(sorted_funds, list) else {}
        # Find info
        for f, i in sorted_funds:
            if f == fid: info = i; break
        row["Votes"] = info.get("votes", 0)
        matrix_data.append(row)

    matrix_df = pd.DataFrame(matrix_data).sort_values("Votes", ascending=False)

    def color_check(v):
        if v == "✅": return "background-color: #c8e6c9; text-align: center"
        return "text-align: center"

    strat_cols = [c for c in matrix_df.columns if c not in ("Fund", "Votes")]
    sty = matrix_df.style.format({"Votes": "{:.0f}"})
    for c in strat_cols:
        sty = sty.map(color_check, subset=[c])
    sty = sty.background_gradient(subset=["Votes"], cmap="Greens")
    st.dataframe(sty, use_container_width=True, height=600)

    st.download_button("📥 Download Consensus", matrix_df.to_csv(index=False), "consensus_picks.csv")


def render_individual_tab():
    """Tab: Run individual strategies with detailed backtest."""
    st.markdown("""<div class="info-banner">
        <h2>🔬 Individual Strategy Backtest</h2>
        <p>Deep dive into any single rule-based or ML strategy</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns([2,2,1,1,1])
    with c1: cat = st.selectbox("📁 Category", list(FILE_MAP.keys()), key="ind_cat")
    with c2:
        all_opts = [(k, STRATEGY_INFO[k][0]) for k in ALL_STRATEGIES]
        strat_key = st.selectbox("🤖 Strategy", all_opts, format_func=lambda x: x[1], key="ind_strat")[0]
    with c3: top_k = st.number_input("🎯 Pick K", 1, 10, 5, key="ind_topk")
    with c4: target_k = st.number_input("🏆 vs Top K", 1, 15, 5, key="ind_tgtk")
    with c5: hold = st.selectbox("📅 Hold", HOLD_PERIODS, index=1, format_func=hold_label, key="ind_hold")

    nav, smap = load_funds(cat)
    bench = load_bench()
    if nav is None: st.error("No data."); return

    is_ml = strat_key in ML_STRATEGIES

    if st.button(f"🚀 Run {STRATEGY_INFO[strat_key][0]}", type="primary", use_container_width=True, key="ind_run"):
        if is_ml:
            st.markdown("#### Building ML dataset...")
            ds = build_ml_dataset(nav, bench, smap, hold, target_k, step_days=21)
            if len(ds) < 200: st.error("Insufficient data."); return
            nd = ds["signal_date"].nunique()
            st.success(f"Dataset: {len(ds):,} samples · {nd} dates")
            res, eq, bm, trd, last_m = run_ml_backtest(ds, strat_key, top_k, target_k, hold, max(6, nd//4))
        else:
            res, eq, bm, trd = run_rule_backtest(nav, strat_key, top_k, target_k, hold, bench, smap)

        if eq.empty: st.error("No results."); return

        yrs = (eq.iloc[-1]["date"]-eq.iloc[0]["date"]).days/365.25
        cagr = (eq.iloc[-1]["value"]/100)**(1/yrs)-1 if yrs > 0 else 0
        bcagr = (bm.iloc[-1]["value"]/100)**(1/yrs)-1 if yrs > 0 else 0
        hr = res["hit_rate"].mean() if "hit_rate" in res.columns else 0
        wr = (res["return"]>0).mean() if "return" in res.columns else 0
        vals = eq["value"].values
        mdd = ((vals-np.maximum.accumulate(vals))/np.maximum.accumulate(vals)).min()

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("CAGR", f"{cagr*100:.2f}%")
        c2.metric("Bench", f"{bcagr*100:.2f}%")
        c3.metric("Alpha", f"{(cagr-bcagr)*100:+.2f}%")
        c4.metric("Hit Rate", f"{hr*100:.1f}%")
        c5.metric("Win Rate", f"{wr*100:.1f}%")
        c6.metric("Max DD", f"{mdd*100:.1f}%")

        t1, t2 = st.tabs(["📈 Equity Curve", "📋 Trade Details"])
        with t1:
            st.plotly_chart(plot_equity(eq, bm, STRATEGY_INFO[strat_key][0]), use_container_width=True)
        with t2:
            if not trd.empty:
                pct = [c for c in trd.columns if "%" in c]
                fmt = {c: "{:.2f}" for c in pct}
                fmt["Hits"] = "{:.0f}"; fmt["Pool"] = "{:.0f}"
                hit_cols = [c for c in trd.columns if "Hit" in c and "%" not in c]
                sty = trd.style.format(fmt, na_rep="")
                for c in hit_cols:
                    sty = sty.map(lambda v: "background-color:#c8e6c9" if v=="✅"
                                  else "background-color:#ffcdd2" if v=="❌" else "", subset=[c])
                st.dataframe(sty, use_container_width=True, height=600)
                st.download_button("📥 Download", trd.to_csv(index=False),
                                   f"{strat_key}_trades.csv", key="ind_dl")


# ============================================================================
# 12. MAIN
# ============================================================================
def main():
    st.markdown("""<div style="text-align:center;padding:10px 0 15px">
        <h1 style="margin:0;border:none">🧠 Unified Fund Predictor</h1>
        <p style="color:#666">8 Rule-Based + 5 ML Strategies · Run All · Consensus Picks</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🏁 Run All & Compare",
        "🤝 Consensus Picks",
        "🔬 Individual Backtest",
    ])

    with tab1: render_run_all_tab()
    with tab2: render_consensus_tab()
    with tab3: render_individual_tab()

    st.caption("Unified Fund Predictor | 8 Rule-Based + 5 ML | Consensus Engine | Walk-Forward Backtest")


if __name__ == "__main__":
    main()
