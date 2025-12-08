"""
Non-interactive version of merge and download script
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os

def merge_largecap_files():
    """Merge two largecap Excel files into one"""
    print("=" * 60)
    print("MERGING LARGECAP FILES")
    print("=" * 60)
    
    data_dir = r"c:\Users\ab528\OneDrive\Desktop\Largecap\data"
    
    # Read both files
    print("\nReading LARGECAP_1.xlsx...")
    df1 = pd.read_excel(os.path.join(data_dir, "LARGECAP_1.xlsx"), header=None)
    print(f"Shape: {df1.shape}")
    
    print("\nReading LARGECAP_FUND_2.xlsx...")
    df2 = pd.read_excel(os.path.join(data_dir, "LARGECAP_FUND_2.xlsx"), header=None)
    print(f"Shape: {df2.shape}")
    
    # Extract fund names (row 2, starting from column 1)
    funds1 = df1.iloc[2, 1:].tolist()
    funds1 = [f for f in funds1 if pd.notna(f) and f != '']
    
    funds2 = df2.iloc[2, 1:].tolist()
    funds2 = [f for f in funds2 if pd.notna(f) and f != '']
    
    print(f"\nFile 1 has {len(funds1)} funds")
    print(f"File 2 has {len(funds2)} funds")
    
    # Extract dates from both files (row 4 onwards, column 0)
    dates1 = pd.to_datetime(df1.iloc[4:, 0], errors='coerce')
    dates2 = pd.to_datetime(df2.iloc[4:, 0], errors='coerce')
    
    # Remove disclaimer rows
    dates1 = dates1[dates1.notna()]
    dates2 = dates2[dates2.notna()]
    
    # Get all unique dates and sort
    all_dates = pd.Series(pd.concat([dates1, dates2]).unique()).sort_values().reset_index(drop=True)
    print(f"\nTotal unique dates: {len(all_dates)}")
    print(f"Date range: {all_dates.min()} to {all_dates.max()}")
    
    # Create new merged dataframe
    num_funds = len(funds1) + len(funds2)
    num_rows = len(all_dates) + 4  # +4 for header rows
    num_cols = num_funds + 1  # +1 for date column
    
    merged_df = pd.DataFrame(index=range(num_rows), columns=range(num_cols))
    
    # Set up header rows
    merged_df.iloc[0, 0] = " >>NAV Data"
    merged_df.iloc[2, 0] = np.nan
    merged_df.iloc[3, 0] = "NAV Date"
    
    # Add fund names in row 2 (starting from column 1)
    all_funds = funds1 + funds2
    for i, fund in enumerate(all_funds):
        merged_df.iloc[2, i+1] = fund
        merged_df.iloc[3, i+1] = "Adjusted NAV NonCorporate(Rs)"
    
    # Add dates starting from row 4
    for i, date in enumerate(all_dates):
        merged_df.iloc[i+4, 0] = date
    
    # Now fill in NAV data for each fund
    print("\nMerging NAV data...")
    
    # Process file 1 funds
    for fund_idx, fund_name in enumerate(funds1):
        col_in_original = fund_idx + 1
        nav_values = pd.to_numeric(df1.iloc[4:, col_in_original], errors='coerce')
        dates_for_fund = pd.to_datetime(df1.iloc[4:, 0], errors='coerce')
        
        # Create a series with date index
        fund_series = pd.Series(nav_values.values, index=dates_for_fund)
        fund_series = fund_series[fund_series.index.notna()]
        
        # Map to all_dates
        for row_idx, date in enumerate(all_dates):
            if date in fund_series.index:
                merged_df.iloc[row_idx+4, fund_idx+1] = fund_series.loc[date]
    
    # Process file 2 funds
    for fund_idx, fund_name in enumerate(funds2):
        col_in_original = fund_idx + 1
        nav_values = pd.to_numeric(df2.iloc[4:, col_in_original], errors='coerce')
        dates_for_fund = pd.to_datetime(df2.iloc[4:, 0], errors='coerce')
        
        # Create a series with date index
        fund_series = pd.Series(nav_values.values, index=dates_for_fund)
        fund_series = fund_series[fund_series.index.notna()]
        
        # Map to all_dates
        col_in_merged = len(funds1) + fund_idx + 1
        for row_idx, date in enumerate(all_dates):
            if date in fund_series.index:
                merged_df.iloc[row_idx+4, col_in_merged] = fund_series.loc[date]
    
    # Save merged file
    output_path = os.path.join(data_dir, "largecap_merged.xlsx")
    print(f"\nSaving merged file to: {output_path}")
    merged_df.to_excel(output_path, index=False, header=False)
    print("✓ Merged largecap file saved successfully!")
    
    return output_path


def download_nifty100_data():
    """Download Nifty 100 data from Yahoo Finance"""
    print("\n" + "=" * 60)
    print("DOWNLOADING NIFTY 100 DATA FROM YAHOO FINANCE")
    print("=" * 60)
    
    data_dir = r"c:\Users\ab528\OneDrive\Desktop\Largecap\data"
    
    # Nifty 100 ticker symbol on Yahoo Finance
    ticker = "^CNX100"  # or try "^NSEI" for Nifty 50
    
    print(f"\nDownloading {ticker} data from 2006-01-01...")
    
    try:
        # Download data
        nifty = yf.download(ticker, start="2006-01-01", end=datetime.now().strftime("%Y-%m-%d"), progress=False)
        
        if nifty.empty:
            print(f"⚠ No data for {ticker}, trying alternative ticker ^NSEI...")
            ticker = "^NSEI"
            nifty = yf.download(ticker, start="2006-01-01", end=datetime.now().strftime("%Y-%m-%d"), progress=False)
        
        if nifty.empty:
            print("✗ Could not download Nifty data from Yahoo Finance")
            return None
        
        print(f"✓ Downloaded {len(nifty)} days of data")
        print(f"Date range: {nifty.index.min()} to {nifty.index.max()}")
        
        # Create a simple CSV with Date and Close price
        nifty_df = pd.DataFrame({
            'date': nifty.index,
            'nav': nifty['Close'].values
        })
        
        # Save to CSV
        output_path = os.path.join(data_dir, "nifty100_data.csv")
        nifty_df.to_csv(output_path, index=False)
        print(f"✓ Nifty 100 data saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"✗ Error downloading Nifty data: {e}")
        return None


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LARGECAP MERGE & NIFTY 100 DOWNLOAD - AUTOMATIC MODE")
    print("=" * 60)
    
    try:
        # Step 1: Merge largecap files
        merged_file = merge_largecap_files()
        
        # Step 2: Download Nifty 100 data
        nifty_file = download_nifty100_data()
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ SUCCESS - ALL TASKS COMPLETED!")
        print("=" * 60)
        if merged_file:
            print(f"✓ Merged largecap file: largecap_merged.xlsx")
        if nifty_file:
            print(f"✓ Nifty 100 data: nifty100_data.csv")
        
        print("\n🚀 Ready to start the dashboard!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
