"""
Test script to verify data cleaning (weekdays only, date ranges)
"""

import sys
sys.path.insert(0, r'c:\Users\ab528\OneDrive\Desktop\Largecap')

from backtest_strategy import load_fund_data, load_nifty_data, MAX_DATA_DATE, LAST_REBALANCE_DATE
import pandas as pd

print("=" * 70)
print("DATA CLEANING VERIFICATION TEST")
print("=" * 70)

print(f"\nConfiguration:")
print(f"  Max Data Date: {MAX_DATA_DATE.strftime('%Y-%m-%d')}")
print(f"  Last Rebalance Date: {LAST_REBALANCE_DATE.strftime('%Y-%m-%d')}")

# Test Large Cap
print("\n" + "-" * 70)
print("Testing Large Cap Data...")
print("-" * 70)
nav_data, scheme_map = load_fund_data('largecap')

if nav_data is not None:
    print(f"✓ Loaded {len(scheme_map)} funds")
    print(f"✓ Date range: {nav_data.index.min().strftime('%Y-%m-%d')} to {nav_data.index.max().strftime('%Y-%m-%d')}")
    print(f"✓ Total dates: {len(nav_data)}")
    
    # Check for weekends
    weekdays = nav_data.index.dayofweek
    has_saturdays = (weekdays == 5).any()
    has_sundays = (weekdays == 6).any()
    
    print(f"\nWeekend Check:")
    print(f"  Contains Saturdays: {'YES - ERROR!' if has_saturdays else 'NO - Good!'}")
    print(f"  Contains Sundays: {'YES - ERROR!' if has_sundays else 'NO - Good!'}")
    
    # Check max date
    exceeds_max = nav_data.index.max() > MAX_DATA_DATE
    print(f"\nDate Range Check:")
    print(f"  Exceeds Dec 1, 2025: {'YES - ERROR!' if exceeds_max else 'NO - Good!'}")
    
    # Sample fund data
    sample_fund = list(scheme_map.keys())[0]
    sample_name = scheme_map[sample_fund]
    sample_data = nav_data[sample_fund].dropna()
    print(f"\nSample Fund: {sample_name}")
    print(f"  Data points: {len(sample_data)}")
    print(f"  Last 5 dates and NAVs:")
    for date, nav in sample_data.tail(5).items():
        day_name = date.strftime('%A')
        print(f"    {date.strftime('%Y-%m-%d')} ({day_name}): {nav:.2f}")

# Test Nifty 100
print("\n" + "-" * 70)
print("Testing Nifty 100 Data...")
print("-" * 70)
nifty_data = load_nifty_data()

if nifty_data is not None:
    print(f"✓ Date range: {nifty_data.index.min().strftime('%Y-%m-%d')} to {nifty_data.index.max().strftime('%Y-%m-%d')}")
    print(f"✓ Total dates: {len(nifty_data)}")
    
    # Check for weekends
    weekdays = nifty_data.index.dayofweek
    has_saturdays = (weekdays == 5).any()
    has_sundays = (weekdays == 6).any()
    
    print(f"\nWeekend Check:")
    print(f"  Contains Saturdays: {'YES - ERROR!' if has_saturdays else 'NO - Good!'}")
    print(f"  Contains Sundays: {'YES - ERROR!' if has_sundays else 'NO - Good!'}")
    
    # Check max date
    exceeds_max = nifty_data.index.max() > MAX_DATA_DATE
    print(f"\nDate Range Check:")
    print(f"  Exceeds Dec 1, 2025: {'YES - ERROR!' if exceeds_max else 'NO - Good!'}")
    
    print(f"\nLast 5 dates and values:")
    for date, value in nifty_data.tail(5).items():
        day_name = date.strftime('%A')
        print(f"  {date.strftime('%Y-%m-%d')} ({day_name}): {value:.2f}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
