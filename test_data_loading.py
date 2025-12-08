import pandas as pd
import os

# Test Nifty Data
nifty_path = r"c:\Users\ab528\OneDrive\Desktop\Largecap\data\nifty100_funds.csv"
print("=" * 60)
print("NIFTY 100 DATA")
print("=" * 60)
df_nifty = pd.read_csv(nifty_path)
print(f"Columns: {df_nifty.columns.tolist()}")
print(f"Total rows: {len(df_nifty)}")
print(f"\nFirst 3 rows:")
print(df_nifty.head(3))
print(f"\nLast 3 rows:")
print(df_nifty.tail(3))

# Parse dates
df_nifty['date'] = pd.to_datetime(df_nifty['date'], format='%d-%m-%Y', errors='coerce')
print(f"\nDate range: {df_nifty['date'].min()} to {df_nifty['date'].max()}")
print(f"NaN dates: {df_nifty['date'].isna().sum()}")
print(f"Days of data: {(df_nifty['date'].max() - df_nifty['date'].min()).days}")

# Test Largecap Data
print("\n" + "=" * 60)
print("LARGECAP FUNDS DATA")
print("=" * 60)
largecap_path = r"c:\Users\ab528\OneDrive\Desktop\Largecap\data\largecap_funds.csv"
df_large = pd.read_csv(largecap_path)
print(f"Columns: {df_large.columns.tolist()}")
print(f"Total rows: {len(df_large)}")
print(f"\nFirst 3 rows:")
print(df_large.head(3))

# Parse dates
df_large['date'] = pd.to_datetime(df_large['date'], format='%d-%m-%Y', errors='coerce')
print(f"\nDate range: {df_large['date'].min()} to {df_large['date'].max()}")
print(f"NaN dates: {df_large['date'].isna().sum()}")
print(f"Days of data: {(df_large['date'].max() - df_large['date'].min()).days}")
print(f"Unique funds: {df_large['scheme_name'].nunique()}")

# Generate scheme codes
df_large['scheme_code'] = df_large['scheme_name'].apply(lambda x: str(abs(hash(x)) % (10 ** 8)))
print(f"Unique scheme codes: {df_large['scheme_code'].nunique()}")

# Pivot to wide format
nav_wide = df_large.pivot(index='date', columns='scheme_code', values='nav')
nav_wide = nav_wide.sort_index()
print(f"\nPivoted data shape: {nav_wide.shape}")
print(f"Date range after pivot: {nav_wide.index.min()} to {nav_wide.index.max()}")

# Check backtest requirements
start_date_required = nav_wide.index.min() + pd.Timedelta(days=370)
holding_days = 126
end_idx = len(nav_wide) - holding_days - 1
try:
    start_idx = nav_wide.index.searchsorted(start_date_required)
except:
    start_idx = 0

print(f"\nBacktest Analysis:")
print(f"Required start date (min + 370 days): {start_date_required}")
print(f"Start index: {start_idx}")
print(f"End index (len - holding - lag): {end_idx}")
print(f"Can backtest: {start_idx < end_idx}")
print(f"Available rebalance periods: {(end_idx - start_idx) // holding_days if start_idx < end_idx else 0}")
