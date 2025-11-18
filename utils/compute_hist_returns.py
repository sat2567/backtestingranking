import pandas as pd
import numpy as np

# Load Nifty 100 data
nifty = pd.read_csv('../data/nifty100_fileter_data.csv')
nifty['Date'] = pd.to_datetime(nifty['Date'])
nifty = nifty.sort_values('Date').set_index('Date')
nifty['Close'] = nifty['Close'].astype(str).str.replace(' ', '').astype(float)

# Resample to 6-month ends (end of each 6 months)
resampled = nifty['Close'].resample('6ME').last()

# Historical 6M returns: (current / previous) - 1
hist_returns = resampled / resampled.shift(1) - 1
hist_returns = hist_returns.dropna()

print("Historical 6M Returns (resampled):")
for date, ret in hist_returns.items():
    print(f"{date.strftime('%Y-%m-%d')}: {ret*100:.2f}%")

avg_hist = hist_returns.mean()
print(f"\nAverage Historical 6M Return: {avg_hist*100:.2f}%")
