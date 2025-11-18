import pandas as pd
import os

fund_files = [
    'data/largecap_funds.csv',
    'data/midcap_funds.csv',
    'data/smallcap_funds.csv',
    'data/multicap_funds.csv',
    'data/large_and_midcap_funds.csv',
    'data/international_funds.csv'
]

issues = []

for file in fund_files:
    if os.path.exists(file):
        try:
            df = pd.read_csv(file)
            # Check columns
            expected_cols = ['scheme_code', 'scheme_name', 'date', 'nav']
            if list(df.columns) != expected_cols:
                issues.append(f"{file}: Wrong columns {df.columns}")
                continue
            
            # Check for nulls
            nulls = df.isnull().sum().sum()
            if nulls > 0:
                issues.append(f"{file}: {nulls} null values")
            
            # Check nav > 0, convert to float
            try:
                df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
                invalid_nav = (df['nav'] <= 0).sum()
                if invalid_nav > 0:
                    issues.append(f"{file}: {invalid_nav} NAV <= 0")
                nan_nav = df['nav'].isnull().sum()
                if nan_nav > 0:
                    issues.append(f"{file}: {nan_nav} NAV NaN after conversion")
            except Exception as e:
                issues.append(f"{file}: NAV conversion error - {e}")
            
            # Check date format (dd-mm-yyyy)
            df['date_parsed'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
            bad_dates = df['date_parsed'].isnull().sum()
            if bad_dates > 0:
                issues.append(f"{file}: {bad_dates} invalid dates")
            
            # Check duplicates on scheme_code + date
            dups = df.duplicated(['scheme_code', 'date']).sum()
            if dups > 0:
                issues.append(f"{file}: {dups} duplicate scheme_code-date pairs")
            
            # Check scheme_name unique per code
            name_uniques = df.groupby('scheme_code')['scheme_name'].nunique()
            bad_names = (name_uniques > 1).sum()
            if bad_names > 0:
                issues.append(f"{file}: {bad_names} scheme_codes with multiple names")
            
            # Check dates descending (assuming loaded in order)
            df = df.sort_values('date_parsed', ascending=False)
            if not df['date_parsed'].is_monotonic_decreasing:
                issues.append(f"{file}: Dates not in descending order")
            
            print(f"{file}: {len(df)} rows, {df['scheme_code'].nunique()} unique schemes")
        except Exception as e:
            issues.append(f"{file}: Error reading - {e}")
    else:
        issues.append(f"{file}: File not found")

if issues:
    print("Issues found:")
    for iss in issues:
        print(iss)
else:
    print("No issues found in any files.")
