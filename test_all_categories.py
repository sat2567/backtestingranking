import pandas as pd
import os

base_path = r"c:\Users\ab528\OneDrive\Desktop\Largecap\data"

categories = {
    "largecap": f"{base_path}\\largecap_funds.csv",
    "smallcap": f"{base_path}\\smallcap_funds.csv",
    "midcap": f"{base_path}\\midcap_funds.csv",
    "large_and_midcap": f"{base_path}\\large_and_midcap_funds.csv",
    "multicap": f"{base_path}\\multicap_funds.csv",
    "international": f"{base_path}\\international_funds.csv",
}

print("=" * 80)
print("TESTING ALL FUND CATEGORIES")
print("=" * 80)

for cat_name, file_path in categories.items():
    print(f"\n{cat_name.upper()}:")
    print("-" * 80)
    
    if not os.path.exists(file_path):
        print(f"  [ERROR] File not found: {file_path}")
        continue
    
    try:
        # Load and process like the dashboard does
        df = pd.read_csv(file_path)
        df.columns = [c.lower().strip() for c in df.columns]
        
        print(f"  Original rows: {len(df):,}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Parse dates - Handle multiple formats
        date_original = df['date'].copy()
        
        df['date'] = pd.to_datetime(date_original, format='%d-%m-%Y', errors='coerce')
        if df['date'].isna().all() or df['date'].isna().sum() > len(df) * 0.5:
            df['date'] = pd.to_datetime(date_original, format='%Y-%m-%d', errors='coerce')
        if df['date'].isna().all() or df['date'].isna().sum() > len(df) * 0.5:
            df['date'] = pd.to_datetime(date_original, errors='coerce')
        
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df = df.dropna(subset=['date', 'nav'])
        
        # Remove duplicates
        df = df.sort_values('date').drop_duplicates(subset=['scheme_name', 'date'], keep='last')
        
        print(f"  After dedup: {len(df):,}")
        print(f"  Unique funds: {df['scheme_name'].nunique()}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Generate scheme codes and pivot
        df['scheme_code'] = df['scheme_name'].apply(lambda x: str(abs(hash(x)) % (10 ** 8)))
        df = df.drop_duplicates(subset=['scheme_code', 'date'], keep='last')
        
        nav_wide = df.pivot(index='date', columns='scheme_code', values='nav')
        nav_wide = nav_wide.ffill(limit=2)
        
        print(f"  Pivoted shape: {nav_wide.shape[0]} days x {nav_wide.shape[1]} funds")
        
        # Check backtest feasibility
        start_date_required = nav_wide.index.min() + pd.Timedelta(days=370)
        holding_days = 126
        end_idx = len(nav_wide) - holding_days - 1
        start_idx = nav_wide.index.searchsorted(start_date_required)
        
        if start_idx < end_idx:
            num_periods = (end_idx - start_idx) // holding_days
            print(f"  [OK] Can backtest with {num_periods} rebalance periods")
        else:
            print(f"  [WARN] Insufficient data for backtest (start={start_idx}, end={end_idx})")
            
    except Exception as e:
        print(f"  [ERROR] {str(e)}")

print("\n" + "=" * 80)
print("SUMMARY: Check which categories work and which need attention")
print("=" * 80)
