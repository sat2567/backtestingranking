"""
Quick fix for Nifty download
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
import os

data_dir = r"c:\Users\ab528\OneDrive\Desktop\Largecap\data"

print("Downloading Nifty 100 data...")

try:
    ticker = "^CNX100"
    nifty = yf.download(ticker, start="2006-01-01", end=datetime.now().strftime("%Y-%m-%d"), progress=False)
    
    if nifty.empty:
        print("Trying ^NSEI...")
        ticker = "^NSEI"
        nifty = yf.download(ticker, start="2006-01-01", end=datetime.now().strftime("%Y-%m-%d"), progress=False)
    
    if not nifty.empty:
        # Create CSV properly
        nifty_df = pd.DataFrame({
            'date': nifty.index,
            'nav': nifty['Close'].values.flatten()
        })
        
        output_path = os.path.join(data_dir, "nifty100_data.csv")
        nifty_df.to_csv(output_path, index=False)
        print(f"SUCCESS: Nifty data saved to {output_path}")
        print(f"Data range: {nifty.index.min()} to {nifty.index.max()}")
    else:
        print("ERROR: Could not download Nifty data")
        
except Exception as e:
    print(f"ERROR: {e}")
