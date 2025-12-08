import pandas as pd

# Test the fixed loading logic
largecap_path = r"c:\Users\ab528\OneDrive\Desktop\Largecap\data\largecap_funds.csv"
df = pd.read_csv(largecap_path)
df.columns = [c.lower().strip() for c in df.columns]

print("Original data:")
print(f"  Total rows: {len(df)}")

# Parse dates
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
df = df.dropna(subset=['date', 'nav'])

print(f"  After cleaning: {len(df)}")

# Remove duplicates by scheme_name + date
df = df.sort_values('date').drop_duplicates(subset=['scheme_name', 'date'], keep='last')
print(f"  After deduplication: {len(df)}")

# Generate scheme codes
df['scheme_code'] = df['scheme_name'].apply(lambda x: str(abs(hash(x)) % (10 ** 8)))

# Final check
df = df.drop_duplicates(subset=['scheme_code', 'date'], keep='last')
print(f"  After final check: {len(df)}")

# Try pivot
try:
    nav_wide = df.pivot(index='date', columns='scheme_code', values='nav')
    nav_wide = nav_wide.ffill(limit=2)
    print(f"\n[SUCCESS] Pivoted data shape: {nav_wide.shape}")
    print(f"   Date range: {nav_wide.index.min()} to {nav_wide.index.max()}")
    print(f"   Number of funds: {nav_wide.shape[1]}")
    
    # Check backtest feasibility
    start_date_required = nav_wide.index.min() + pd.Timedelta(days=370)
    holding_days = 126
    end_idx = len(nav_wide) - holding_days - 1
    start_idx = nav_wide.index.searchsorted(start_date_required)
    
    print(f"\n[Backtest Analysis]:")
    print(f"   Required start: {start_date_required}")
    print(f"   Start index: {start_idx}")
    print(f"   End index: {end_idx}")
    print(f"   Can backtest: {start_idx < end_idx}")
    if start_idx < end_idx:
        num_periods = (end_idx - start_idx) // holding_days
        print(f"   Rebalance periods: {num_periods}")
        print(f"   [OK] SUFFICIENT DATA FOR BACKTEST!")
    else:
        print(f"   [FAIL] INSUFFICIENT DATA")
        
except Exception as e:
    print(f"\n[ERROR]: {e}")
