"""
Check last dates for all fund categories
"""

import sys
sys.path.insert(0, r'c:\Users\ab528\OneDrive\Desktop\Largecap')

from backtest_strategy import load_fund_data

categories = ['largecap', 'smallcap', 'midcap', 'large_and_midcap', 'multicap', 'international']

print('LAST DATE FOR EACH CATEGORY:')
print('=' * 50)

for cat in categories:
    try:
        nav_data, scheme_map = load_fund_data(cat)
        if nav_data is not None:
            last_date = nav_data.index.max().strftime('%Y-%m-%d')
            total_days = len(nav_data)
            fund_count = len(scheme_map)
            print(f'{cat.upper():20}: {last_date} ({total_days} days, {fund_count} funds)')
        else:
            print(f'{cat.upper():20}: No data')
    except Exception as e:
        print(f'{cat.upper():20}: Error - {str(e)[:30]}')

print('=' * 50)
