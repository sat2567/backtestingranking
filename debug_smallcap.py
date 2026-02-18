"""
ML Fund Pro v6 - Deep Learning Ranking Edition (PyTorch)
========================================================
KEY FEATURES:
  - Deep Learning: PyTorch LSTM & GRU implementations.
  - Learning-to-Rank: Pairwise Margin Ranking Loss (optimizes for sorting).
  - Alpha Targeting: Trains on excess returns (Alpha) rather than raw returns.
  - Attention Mechanism: Weighs important time steps in the sequence.
  - Dual Metrics: Tracks both 'Win Rate' and 'Top-5 Precision'.

Run: streamlit run ml_fund_pro_v6.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
import os
import warnings
import random

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

st.set_page_config(page_title="Deep Ranker Pro v6", page_icon="🧠", layout="wide")

# ============================================================================
# 1. CONFIG & DATA LOADING
# ============================================================================

DATA_DIR = "data"
FILE_MAPPING = {"Large Cap": "largecap_merged.xlsx"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, FILE_MAPPING["Large Cap"])
    if not os.path.exists(path): return None, None, None
    try:
        df = pd.read_excel(path, header=None)
        fund_names = df.iloc[2, 1:].tolist()
        data_df = df.iloc[4:, :].copy()
        # Cleanup
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
        
        # Benchmark Load
        b_path = os.path.join(DATA_DIR, "nifty100_data.csv")
        bench = None
        if os.path.exists(b_path):
            b_df = pd.read_csv(b_path)
            b_df.columns = [c.lower().strip() for c in b_df.columns]
            b_df['date'] = pd.to_datetime(b_df['date'])
            b_df = b_df.set_index('date').sort_index()
            bench = b_df['nav'] if 'nav' in b_df.columns else b_df.iloc[:,0]
            bench = bench.reindex(nav_wide.index).ffill()
            
        # Clean Data
        nav_wide = nav_wide.sort_index().ffill().dropna(how='all')
        if bench is not None: bench = bench.fillna(method='ffill')
            
        return nav_wide, scheme_map, bench
    except Exception as e:
        st.error(f"Data Error: {e}")
        return None, None, None

# ============================================================================
# 2. ADVANCED FEATURE ENGINEERING (TENSORS)
# ============================================================================

def compute_rolling_features(nav_series, bench_series, window=60):
    """
    Computes rolling sequence features for LSTM.
    Returns: (Seq_Len, N_Features) numpy array
    """
    # Calculate returns
    f_ret = nav_series.pct_change().fillna(0)
    b_ret = bench_series.pct_change().fillna(0)
    
    # 1. Relative Strength (Fund Ret - Bench Ret)
    rel_strength = f_ret - b_ret
    
    # 2. Rolling Volatility (21 days)
    roll_vol = f_ret.rolling(21).std().fillna(0)
    
    # 3. Rolling Beta (63 days)
    # Simplified beta: Correlation * (FundVol / BenchVol)
    # We use rolling correlation approximation for speed
    roll_corr = f_ret.rolling(63).corr(b_ret).fillna(0)
    roll_beta = roll_corr * (roll_vol / (b_ret.rolling(21).std().fillna(1e-6)))
    
    # 4. Momentum (Price / MA_50 - 1)
    ma_50 = nav_series.rolling(50).mean()
    mom_50 = (nav_series / ma_50 - 1).fillna(0)
    
    # Stack features: [RelStrength, Vol, Beta, Momentum]
    # Shape: (Total_Days, 4)
    features = np.column_stack([
        rel_strength.values, 
        roll_vol.values, 
        roll_beta.values, 
        mom_50.values
    ])
    
    # Replace infs
    features = np.nan_to_num(features)
    return features

def create_sequences(nav_df, benchmark, seq_length=60, pred_horizon=63):
    """
    Creates (X, y) pairs for PyTorch.
    X: Sequence of past features (Batch, Seq_Len, N_Features)
    y: Future Alpha (Batch, 1) -> "Alpha" is better for ranking than raw return
    """
    X_list, y_list, dates_list, fund_ids = [], [], [], []
    
    # Pre-calculate features for all funds to speed up
    fund_features = {}
    for fid in nav_df.columns:
        fund_features[fid] = compute_rolling_features(nav_df[fid], benchmark)
        
    # Valid indices (must have enough history and future)
    valid_indices = range(seq_length, len(nav_df) - pred_horizon, 21) # Step 21 (Monthly)
    
    for idx in valid_indices:
        dt = nav_df.index[idx]
        
        # Calculate Future Returns for Ranking
        # Target: Jensen's Alpha Proxy (Fund Ret - Bench Ret)
        b_future = benchmark.iloc[idx + pred_horizon] / benchmark.iloc[idx] - 1
        
        for fid in nav_df.columns:
            # Check data validity
            if np.isnan(nav_df[fid].iloc[idx]) or nav_df[fid].iloc[idx] == 0: continue
            
            # Future Return
            f_future = nav_df[fid].iloc[idx + pred_horizon] / nav_df[fid].iloc[idx] - 1
            alpha_target = f_future - b_future # This is what we want to predict!
            
            # Input Sequence
            # Slice: [idx - seq_len : idx]
            seq_data = fund_features[fid][idx-seq_length:idx]
            
            X_list.append(seq_data)
            y_list.append(alpha_target)
            dates_list.append(dt)
            fund_ids.append(fid)
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    return X, y, dates_list, fund_ids

# ============================================================================
# 3. PYTORCH MODEL (LSTM/GRU WITH ATTENTION)
# ============================================================================

class FundRanker(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2, model_type='GRU'):
        super(FundRanker, self).__init__()
        self.model_type = model_type
        
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            
        # Attention Mechanism: Weights time steps
        self.attn = nn.Linear(hidden_dim, 1)
        
        # Ranking Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1) # Outputs a raw score (unbounded)
        )
        
    def forward(self, x):
        # x: (Batch, Seq_Len, Features)
        out, _ = self.rnn(x) 
        
        # Attention: Which time step matters most?
        # weights = softmax(attn(out))
        attn_weights = torch.softmax(self.attn(out), dim=1)
        
        # Context Vector: Weighted sum of outputs
        context = torch.sum(attn_weights * out, dim=1)
        
        # Score
        score = self.fc(context)
        return score

# ============================================================================
# 4. PAIRWISE RANKING LOSS
# ============================================================================

def pairwise_ranking_loss(pred_scores, targets, margin=0.1):
    """
    Optimizes for Order: If Target A > Target B, then Score A > Score B + Margin
    """
    # Create all pairs in the batch
    # This is O(N^2), so batch size shouldn't be massive (e.g., < 200 funds)
    
    # 1. Comparison Matrices
    # pred_diff[i, j] = score[i] - score[j]
    pred_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)
    
    # target_diff[i, j] = target[i] - target[j]
    target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
    
    # 2. Mask: Where should i be ranked higher than j?
    # We only care about pairs where the actual return difference is significant
    s_ij = torch.sign(target_diff)
    
    # 3. Margin Ranking Loss
    # Loss = max(0, -s_ij * (pred_diff) + margin)
    # We only enforce margin if targets are different
    
    loss = torch.clamp(margin - s_ij * pred_diff, min=0)
    
    # Mask out diagonal (i==j) and where targets are equal
    mask = (target_diff.abs() > 1e-4).float()
    loss = loss * mask
    
    return loss.sum() / (mask.sum() + 1e-6)

# ============================================================================
# 5. TRAINING & BACKTESTING ENGINE
# ============================================================================

def train_model(X_train, y_train, input_dim, config):
    # Prepare Data
    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    
    model = FundRanker(
        input_dim=input_dim, 
        hidden_dim=config['hidden_dim'], 
        num_layers=config['layers'],
        model_type=config['type']
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    model.train()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        scores = model(X_t).squeeze()
        
        # Loss: 70% Ranking Loss + 30% MSE (Hybrid is usually best)
        loss_rank = pairwise_ranking_loss(scores, y_t)
        loss_mse = nn.MSELoss()(scores, y_t) 
        
        total_loss = 0.7 * loss_rank + 0.3 * loss_mse
        
        total_loss.backward()
        optimizer.step()
        
    return model

def run_backtest(X, y, dates, fund_ids, config, top_n=5):
    """
    Walk-Forward Validation
    Train [0...T], Predict [T+1]
    """
    unique_dates = sorted(list(set(dates)))
    start_idx = int(config['train_window_years'] * 12)
    
    results = []
    
    progress = st.progress(0)
    
    # Map data to indices for speed
    date_to_indices = {}
    for i, d in enumerate(dates):
        if d not in date_to_indices: date_to_indices[d] = []
        date_to_indices[d].append(i)
    
    for i in range(start_idx, len(unique_dates)):
        test_date = unique_dates[i]
        
        # 1. Define Train/Test Split (Expanding Window)
        # Train on all dates BEFORE test_date
        train_indices = []
        for d in unique_dates[:i]:
            train_indices.extend(date_to_indices[d])
            
        test_indices = date_to_indices[test_date]
        
        if len(train_indices) < 100 or len(test_indices) < 5: continue
            
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        ids_test = [fund_ids[k] for k in test_indices]
        
        # 2. Train Model (Re-train every step to avoid lookahead)
        # Note: In production, we might incrementally train. Here we retrain for purity.
        input_dim = X.shape[2]
        model = train_model(X_train, y_train, input_dim, config)
        
        # 3. Predict
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            scores = model(X_test_t).squeeze().cpu().numpy()
            
        # 4. Rank & Select
        # Combine results into DataFrame
        period_df = pd.DataFrame({
            'fund_id': ids_test,
            'score': scores,
            'actual_alpha': y_test
        })
        
        # Sort by predicted score
        period_df = period_df.sort_values('score', ascending=False)
        
        # Our Picks
        top_picks = period_df.head(top_n)
        
        # Actual Best (Ground Truth)
        actual_best = period_df.nlargest(top_n, 'actual_alpha')
        
        # 5. Calculate Metrics
        # Exact Precision: How many picks match actual best?
        hits = len(set(top_picks['fund_id']) & set(actual_best['fund_id']))
        precision = hits / top_n
        
        # Win Rate: Did we beat the median fund?
        median_alpha = period_df['actual_alpha'].median()
        win_count = (top_picks['actual_alpha'] > median_alpha).sum()
        win_rate = win_count / top_n
        
        # Returns
        avg_alpha_captured = top_picks['actual_alpha'].mean()
        
        results.append({
            'date': test_date,
            'precision': precision,
            'win_rate': win_rate,
            'alpha_captured': avg_alpha_captured,
            'n_funds': len(period_df)
        })
        
        progress.progress((i - start_idx) / (len(unique_dates) - start_idx))
        
    return pd.DataFrame(results)

# ============================================================================
# 6. DASHBOARD UI
# ============================================================================

def main():
    st.markdown("## 🧠 Deep Learning Fund Ranker (v6)")
    st.markdown("""
    **Architecture:** PyTorch LSTM/GRU with Attention.
    **Loss Function:** Pairwise Margin Ranking Loss (Optimizes Sort Order).
    **Target:** Future Jensen's Alpha (Excess Return).
    """)
    
    # 1. Hyperparameters
    with st.expander("⚙️ Model Configuration", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        model_type = c1.selectbox("RNN Type", ["GRU", "LSTM"], index=0)
        hidden_dim = c2.selectbox("Hidden Neurons", [32, 64, 128], index=1)
        epochs = c3.slider("Training Epochs", 1, 20, 5)
        top_n = c4.number_input("Top N Picks", 1, 10, 5)
        
        config = {
            'type': model_type,
            'hidden_dim': hidden_dim,
            'layers': 2,
            'lr': 0.005,
            'epochs': epochs,
            'train_window_years': 3
        }
        
    # 2. Data Load
    nav, _, bench = load_data()
    if nav is None: return
    
    if st.button("🚀 Train & Backtest"):
        st.info("Creating Tensors & Sequences... (This may take a moment)")
        
        # Create (X, y)
        X, y, dates, fids = create_sequences(nav, bench, seq_length=60, pred_horizon=63)
        st.write(f"Tensor Shape: {X.shape} (Samples, Sequence, Features)")
        
        # Run Backtest
        res = run_backtest(X, y, dates, fids, config, top_n)
        
        if res.empty:
            st.error("Backtest failed. Check data history.")
            return
            
        # Results
        res['cum_alpha'] = res['alpha_captured'].cumsum()
        
        avg_prec = res['precision'].mean() * 100
        avg_win = res['win_rate'].mean() * 100
        total_alpha = res['cum_alpha'].iloc[-1] * 100
        
        # Visualization
        
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Top-5 Precision (Exact)", f"{avg_prec:.1f}%", help="How often we picked the EXACT top 5 funds.")
        m2.metric("Win Rate (Broad)", f"{avg_win:.1f}%", help="How often our picks beat the median.")
        m3.metric("Total Alpha Generated", f"{total_alpha:+.1f}%")
        
        t1, t2 = st.tabs(["📈 Ranking Performance", "📋 Logs"])
        
        with t1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['date'], y=res['precision'], name='Precision', line=dict(color='#FF5252', width=2), mode='lines+markers'))
            fig.add_trace(go.Scatter(x=res['date'], y=res['win_rate'], name='Win Rate', line=dict(color='#4CAF50', width=2)))
            fig.update_layout(title="Ranking Accuracy Over Time", yaxis_title="Score (0-1)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Cumulative Alpha")
            fig2 = go.Figure()
            fig2.add_trace(go.Area(x=res['date'], y=res['cum_alpha'], name='Cumulative Alpha', line=dict(color='#2196F3')))
            st.plotly_chart(fig2, use_container_width=True)
            
        with t2:
            st.dataframe(res)

if __name__ == "__main__":
    main()
