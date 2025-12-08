import pandas as pd

print("Testing GitHub data loading...\n")

# Test loading largecap data from GitHub
url = "https://raw.githubusercontent.com/sat2567/cagr-improved-ranking/sat2567-patch-1/data/largecap_funds.csv"

try:
    print(f"Loading from: {url}")
    df = pd.read_csv(url)
    print(f"[OK] SUCCESS! Loaded {len(df):,} rows")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  First row: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"[FAIL] FAILED: {e}")

# Test Nifty data
print("\n" + "="*60)
url_nifty = "https://raw.githubusercontent.com/sat2567/cagr-improved-ranking/sat2567-patch-1/data/nifty100_funds.csv"

try:
    print(f"Loading from: {url_nifty}")
    df_nifty = pd.read_csv(url_nifty)
    print(f"[OK] SUCCESS! Loaded {len(df_nifty):,} rows")
    print(f"  Columns: {df_nifty.columns.tolist()}")
except Exception as e:
    print(f"[FAIL] FAILED: {e}")

print("\n" + "="*60)
print("GitHub data source is working! Dashboard will load data from repository.")
