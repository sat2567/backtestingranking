"""
Merge INTERNATIONALFUNDS1.xlsx and INTERNATIONALFUNDS2.xlsx into one file
"""

import pandas as pd
import numpy as np
import os

def merge_international_files():
    """Merge two international Excel files into one"""
    print("=" * 60)
    print("MERGING INTERNATIONAL FUNDS FILES")
    print("=" * 60)
    
    data_dir = r"c:\Users\ab528\OneDrive\Desktop\Largecap\data"
    
    # Read both files
    print("\nReading INTERNATIONALFUNDS1.xlsx...")
    df1 = pd.read_excel(os.path.join(data_dir, "INTERNATIONALFUNDS1.xlsx"), header=None)
    print(f"Shape: {df1.shape}")
    
    print("\nReading INTERNATIONALFUNDS2.xlsx...")
    df2 = pd.read_excel(os.path.join(data_dir, "INTERNATIONALFUNDS2.xlsx"), header=None)
    print(f"Shape: {df2.shape}")
    
    # Extract fund names (row 2, starting from column 1)
    funds1 = df1.iloc[2, 1:].tolist()
    funds1 = [f for f in funds1 if pd.notna(f) and f != '']
    
    funds2 = df2.iloc[2, 1:].tolist()
    funds2 = [f for f in funds2 if pd.notna(f) and f != '']
    
    print(f"\nFile 1 has {len(funds1)} funds")
    print(f"File 2 has {len(funds2)} funds")
    print(f"Total: {len(funds1) + len(funds2)} funds")
    
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
    output_path = os.path.join(data_dir, "international_merged.xlsx")
    print(f"\nSaving merged file to: {output_path}")
    merged_df.to_excel(output_path, index=False, header=False)
    print("SUCCESS: Merged international file saved!")
    
    return output_path


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("INTERNATIONAL FUNDS MERGE SCRIPT")
    print("=" * 60)
    
    try:
        merged_file = merge_international_files()
        
        print("\n" + "=" * 60)
        print("SUCCESS - MERGE COMPLETED!")
        print("=" * 60)
        print(f"Merged file: international_merged.xlsx")
        print("\nNext: The dashboard will be updated to include this category.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
