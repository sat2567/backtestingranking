# Setup Instructions for Updated Dashboard

## What's New
1. ✅ Merged LARGECAP_1.xlsx and LARGECAP_FUND_2.xlsx into one file
2. ✅ Added Multi Cap category
3. ✅ Nifty 100 data download from Yahoo Finance (starting Jan 1, 2006)
4. ✅ Updated dashboard to use new data files

## Steps to Complete Setup

### Step 1: Close All Excel Files
**IMPORTANT**: Close any open Excel files in the `data` folder before proceeding.

### Step 2: Run the Merge & Download Script
Open a terminal/command prompt in the project folder and run:

```bash
python merge_and_download.py
```

This script will:
- Merge LARGECAP_1.xlsx and LARGECAP_FUND_2.xlsx → `largecap_merged.xlsx`
- Download Nifty 100 data from Yahoo Finance → `nifty100_data.csv`

Expected output files in the `data` folder:
- ✓ `largecap_merged.xlsx` (merged largecap funds)
- ✓ `nifty100_data.csv` (Nifty 100 benchmark from 2006-01-01)

### Step 3: Run the Dashboard
After the merge script completes successfully, start the dashboard:

```bash
streamlit run backtest_strategy.py
```

## Available Categories
- **Large Cap** (merged from 2 files)
- **Small Cap**
- **Mid Cap**
- **Large & Mid Cap**
- **Multi Cap** (newly added)

## Troubleshooting

### If you see "Permission denied" error:
- Close all Excel files in the `data` folder
- Close any Excel application that might be accessing the files
- Try running the script again

### If Nifty 100 download fails:
- Check your internet connection
- The script will try `^CNX100` ticker first, then fallback to `^NSEI` (Nifty 50)
- You can manually download the data from Yahoo Finance if needed

### If dashboard shows "File not found":
- Make sure `merge_and_download.py` completed successfully
- Check that `largecap_merged.xlsx` exists in the `data` folder
- Check that `nifty100_data.csv` exists in the `data` folder

## Data Files Summary

### Input Files (Original):
- LARGECAP_1.xlsx
- LARGECAP_FUND_2.xlsx
- smallcap.xlsx
- midcap.xlsx
- large_and_midcap_fund.xlsx
- MULTICAP.xlsx

### Output Files (Generated):
- largecap_merged.xlsx (created by merge script)
- nifty100_data.csv (downloaded from Yahoo Finance)

### Dashboard Uses:
- largecap_merged.xlsx (for Large Cap analysis)
- smallcap.xlsx (for Small Cap analysis)
- midcap.xlsx (for Mid Cap analysis)
- large_and_midcap_fund.xlsx (for Large & Mid Cap analysis)
- MULTICAP.xlsx (for Multi Cap analysis)
- nifty100_data.csv (benchmark for all analyses)

## Notes
- The merged largecap file will combine all funds from both source files
- Date ranges will be aligned to cover all available dates from both files
- Nifty 100 data is downloaded from Yahoo Finance starting 2006-01-01
- The dashboard uses Nifty 100 as the benchmark for all categories
