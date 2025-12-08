import pandas as pd

file_path = r"c:\Users\ab528\OneDrive\Desktop\Largecap\data\smallcap_funds.csv"
df = pd.read_csv(file_path)
df.columns = [c.lower().strip() for c in df.columns]

print("=== STEP BY STEP TRACE ===\n")

print(f"1. After CSV load: {len(df):,} rows")

# Parse dates - Handle multiple formats
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
print(f"2. After DD-MM-YYYY parse: {df['date'].notna().sum():,} non-null dates")

if df['date'].isna().all() or df['date'].isna().sum() > len(df) * 0.5:
    print("   -> Trying YYYY-MM-DD format...")
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    print(f"   -> After YYYY-MM-DD parse: {df['date'].notna().sum():,} non-null dates")
    
if df['date'].isna().all() or df['date'].isna().sum() > len(df) * 0.5:
    print("   -> Trying auto-detection...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print(f"   -> After auto-detection: {df['date'].notna().sum():,} non-null dates")

df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
print(f"3. After NAV parse: {df['nav'].notna().sum():,} non-null NAVs")

df = df.dropna(subset=['date', 'nav'])
print(f"4. After dropna: {len(df):,} rows")

# Remove duplicates
print(f"5. Before dedup: {len(df):,} rows")
print(f"   Columns: {df.columns.tolist()}")
print(f"   Checking subset: ['scheme_name', 'date']")

df_sorted = df.sort_values('date')
print(f"   After sort: {len(df_sorted):,} rows")

df_dedup = df_sorted.drop_duplicates(subset=['scheme_name', 'date'], keep='last')
print(f"6. After dedup: {len(df_dedup):,} rows")

if len(df_dedup) > 0:
    print(f"\nSUCCESS! Unique funds: {df_dedup['scheme_name'].nunique()}")
    print(f"Date range: {df_dedup['date'].min()} to {df_dedup['date'].max()}")
else:
    print(f"\nFAIL! All rows removed during deduplication")
    print(f"Let me check the original df more carefully...")
    print(f"Scheme_name column type: {df['scheme_name'].dtype}")
    print(f"Date column type: {df['date'].dtype}")
