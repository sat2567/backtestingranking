import pandas as pd
import numpy as np

# Load Nifty 100 data
nifty = pd.read_csv('../data/nifty100_fileter_data.csv')
nifty['Date'] = pd.to_datetime(nifty['Date'])
nifty = nifty.sort_values('Date').set_index('Date')
nifty['Close'] = nifty['Close'].astype(str).str.replace(' ', '').astype(float)

# Load summary to get rebalance dates
summary = pd.read_csv('../output/backtest_results_summary.csv')
summary['rebalance_date'] = pd.to_datetime(summary['rebalance_date'])
rebalance_dates = summary['rebalance_date'].tolist()

HOLDING_PERIOD = 126

results = []
for date in rebalance_dates:
    # Find closest index in nifty
    idx = nifty.index.get_indexer([date], method='nearest')[0]
    
    # Backward 6-month return
    backward_ret = np.nan
    if idx - HOLDING_PERIOD >= 0:
        start = nifty.iloc[idx - HOLDING_PERIOD]['Close']
        end = nifty.iloc[idx]['Close']
        backward_ret = (end / start) - 1
    
    # Forward 6-month return (already in summary, but compute for consistency)
    forward_ret = np.nan
    if idx + HOLDING_PERIOD < len(nifty):
        start = nifty.iloc[idx]['Close']
        end = nifty.iloc[idx + HOLDING_PERIOD]['Close']
        forward_ret = (end / start) - 1
    
    results.append({
        'rebalance_date': date.strftime('%Y-%m-%d'),
        'backward_6m_return': backward_ret,
        'forward_6m_return': forward_ret
    })

df = pd.DataFrame(results)
print(df)

# Averages
avg_backward = df['backward_6m_return'].mean()
avg_forward = df['forward_6m_return'].mean()

print(f"\nAverage Backward 6M Return: {avg_backward*100:.2f}%")
print(f"Average Forward 6M Return: {avg_forward*100:.2f}%")
print(f"Difference: {(avg_forward - avg_backward)*100:.2f}%")
