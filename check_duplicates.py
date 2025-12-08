import pandas as pd

# Check for duplicates
largecap_path = r"c:\Users\ab528\OneDrive\Desktop\Largecap\data\largecap_funds.csv"
df = pd.read_csv(largecap_path)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

print("Checking for duplicates...")
print(f"Total rows: {len(df)}")

# Check scheme_name + date duplicates
duplicates = df.groupby(['scheme_name', 'date']).size()
dup_count = (duplicates > 1).sum()
print(f"Duplicate (scheme_name, date) pairs: {dup_count}")

if dup_count > 0:
    print("\nExamples of duplicates:")
    dup_examples = duplicates[duplicates > 1].head(10)
    for (name, date), count in dup_examples.items():
        print(f"  {name[:50]}... on {date}: {count} entries")
        subset = df[(df['scheme_name'] == name) & (df['date'] == date)]
        print(f"    NAV values: {subset['nav'].tolist()}")
        print(f"    Plan types: {subset['plan_type'].tolist()}")
