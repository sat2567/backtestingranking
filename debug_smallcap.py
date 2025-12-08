import pandas as pd

file_path = r"c:\Users\ab528\OneDrive\Desktop\Largecap\data\smallcap_funds.csv"
df = pd.read_csv(file_path)
df.columns = [c.lower().strip() for c in df.columns]

print("Step 1 - Original data:")
print(f"  Total rows: {len(df):,}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Sample dates (raw): {df['date'].head(10).tolist()}")

# Parse dates - step by step
print("\nStep 2 - Try DD-MM-YYYY format:")
df['date_test1'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
print(f"  Successful parses: {df['date_test1'].notna().sum():,}/{len(df):,}")
print(f"  % parsed: {df['date_test1'].notna().sum() / len(df) * 100:.2f}%")

print("\nStep 3 - Try YYYY-MM-DD format:")
df['date_test2'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
print(f"  Successful parses: {df['date_test2'].notna().sum():,}/{len(df):,}")
print(f"  % parsed: {df['date_test2'].notna().sum() / len(df) * 100:.2f}%")

# Use correct format
df['date'] = df['date_test2']

print("\nStep 4 - After date parsing:")
print(f"  Non-null dates: {df['date'].notna().sum():,}")

df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
print(f"  Non-null NAV: {df['nav'].notna().sum():,}")

df = df.dropna(subset=['date', 'nav'])
print(f"  After dropna: {len(df):,}")

print("\nStep 5 - Check for duplicates:")
dup_count = df.groupby(['scheme_name', 'date']).size()
print(f"  Unique (scheme_name, date) pairs: {len(dup_count):,}")
print(f"  Total rows: {len(df):,}")
print(f"  Duplicates: {(dup_count > 1).sum():,}")

# Show some examples
if len(df) > 0:
    print(f"\nSample data after parsing:")
    print(df[['scheme_name', 'date', 'nav']].head(10))
    
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique funds: {df['scheme_name'].nunique()}")
