# Large Cap Mutual Fund Selection Strategy - Backtest System

## Overview
This project implements a systematic fund selection strategy based on risk-adjusted performance metrics. The strategy selects top mutual funds every 6 months using a composite scoring system that combines Sharpe ratio, Sortino ratio, and Maximum Drawdown.

## Project Structure

```
Largecap/
├── data/                          # Input data files
│   ├── nifty100_fileter_data.csv  # Benchmark data (Nifty 100)
│   ├── largecap_funds.csv         # Large cap fund NAV data
│   ├── smallcap_funds.csv         # Small cap fund NAV data
│   ├── midcap_funds.csv           # Mid cap fund NAV data
│   ├── large_and_midcap_funds.csv # Large & Mid cap fund NAV data
│   ├── multicap_funds.csv         # Multi cap fund NAV data
│   └── international_funds.csv    # International fund NAV data
│
├── output/                        # Generated results
│   ├── backtest_results_summary_*.csv    # Summary results per category
│   ├── backtest_results_detailed_*.csv   # Detailed results per category
│   └── backtest_analysis.png              # Analysis visualization
│
├── utils/                         # Utility scripts
│   ├── check_fund_data.py         # Data validation utility
│   ├── compute_returns.py         # Compute backward/forward returns
│   ├── compute_hist_returns.py    # Compute historical 6M returns
│   └── open_dashboard.py          # Browser launcher for Streamlit
│
├── backtest_strategy.py           # Main backtesting script
├── analyze_results.py             # Results analysis and visualization
├── streamlit_app.py              # Interactive dashboard
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place your fund NAV data files in the `data/` directory:
- Format: CSV with columns `scheme_code`, `scheme_name`, `date` (DD-MM-YYYY), `nav`
- Example: `data/largecap_funds.csv`

### 3. Validate Data (Optional)
```bash
python utils/check_fund_data.py
```

### 4. Run Backtest
```bash
# Run for a specific category
python backtest_strategy.py --category largecap

# Or specify custom funds file
python backtest_strategy.py --funds-csv data/largecap_funds.csv --top-n 15
```

### 5. Analyze Results
```bash
python analyze_results.py
```

### 6. Launch Dashboard
```bash
streamlit run streamlit_app.py
```

## Main Scripts

### `backtest_strategy.py`
Main backtesting engine that:
- Loads fund NAV data and benchmark data
- Computes risk metrics (Sharpe, Sortino, MDD, etc.)
- Selects top N funds using composite scoring
- Evaluates forward returns over 6-month holding periods
- Generates summary and detailed results CSV files

**Usage:**
```bash
python backtest_strategy.py --category largecap --top-n 15
```

**Parameters:**
- `--category`: Fund category (largecap, smallcap, midcap, large_and_midcap, multicap, international)
- `--funds-csv`: Custom path to funds CSV file
- `--top-n`: Number of funds to select (default: 15)

### `streamlit_app.py`
Interactive web dashboard for exploring backtest results:
- Fund rankings by various criteria
- Performance metrics and visualizations
- Historical selection frequency
- Criteria performance comparison

**Usage:**
```bash
streamlit run streamlit_app.py
```

### `analyze_results.py`
Generates comprehensive analysis report:
- Overall performance statistics
- Return analysis
- Selection accuracy distribution
- Time-series trends
- Most frequently selected funds
- Risk metrics analysis
- Visualization dashboard (saved to `output/backtest_analysis.png`)

**Usage:**
```bash
python analyze_results.py
```

## Utility Scripts

### `utils/check_fund_data.py`
Validates fund data files for:
- Column structure
- Missing values
- Invalid NAVs
- Date format consistency
- Duplicate entries

### `utils/compute_returns.py`
Computes backward and forward 6-month returns for rebalance dates.

### `utils/compute_hist_returns.py`
Computes historical 6-month returns from benchmark data.

### `utils/open_dashboard.py`
Helper script to automatically open the Streamlit dashboard in browser.

## Methodology

### Strategy Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Lookback Period | 252 days (~1 year) | Historical data for metric calculation |
| Holding Period | 126 days (~6 months) | Time between rebalancing |
| Top N | 15 funds (default) | Number of funds selected per period |
| Risk-free Rate | 5% annual | Used for Sharpe/Sortino calculation |

### Risk Metrics
For each rebalance date, using the past 1-year data:
- **Sharpe Ratio**: (Excess Return) / (Total Volatility) - annualized
- **Sortino Ratio**: (Excess Return) / (Downside Volatility) - annualized
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Annual Return**: Mean daily return × 252
- **Annual Volatility**: Std daily return × √252
- **Tracking Error**: Std of active returns × √252
- **Information Ratio**: Mean active return × √252 / Tracking Error
- **Beta**: Covariance with benchmark / Variance of benchmark

### Fund Selection Process
1. Calculate risk metrics for all funds
2. Normalize metrics using z-score transformation (direction-aware)
3. Compute composite score using weighted-available method
4. Select top N funds by composite score

### Composite Scoring
The default composite score uses weighted z-scores:
- **Composite (Balanced)**: Sharpe 0.25, Sortino 0.25, IR 0.20, Ann Return 0.20, MDD 0.10, Ann Vol 0.10, TE 0.10
- **Momentum**: Momentum 6M 0.4, Ann Return 0.4, Ann Vol 0.2
- **Sharpe-Focused**: Sharpe 0.40, Sortino 0.40, MDD 0.20
- **Risk-Adjusted**: IR 0.30, Sortino 0.40, MDD 0.20, Ann Vol 0.10
- **Consistency**: Sortino 0.40, IR 0.30, MDD 0.20, Ann Vol 0.10

## Output Files

### Summary Results (`output/backtest_results_summary_*.csv`)
One row per rebalance period:
- `rebalance_date`: Date of rebalancing
- `overlap_count`: Number of selected funds in actual top N
- `overlap_accuracy`: Overlap count / Top N
- `avg_rank_of_selected`: Average forward rank of selected funds
- `mean_future_return`: Average 6M forward return of selections
- `bench_forward_return`: Benchmark 6M forward return

### Detailed Results (`output/backtest_results_detailed_*.csv`)
One row per fund per rebalance:
- All risk metrics (Sharpe, Sortino, MDD, etc.)
- Composite score and rank
- Forward return and forward rank
- Whether fund was in actual top N

## Dependencies

- `streamlit>=1.33` - Web dashboard
- `pandas>=2.0` - Data manipulation
- `numpy>=1.24` - Numerical computations
- `matplotlib>=3.7` - Visualizations

## Notes

- The strategy uses **weighted-available scoring**: missing metrics don't penalize funds, but funds with all metrics missing rank last
- **Direction-aware z-scores**: Metrics where lower is better (MDD, Volatility, TE, Beta) are inverted before z-scoring
- Results are saved per category with suffix `_{category}` (e.g., `backtest_results_summary_largecap.csv`)
- For largecap category, results are also saved without suffix for backward compatibility

## Limitations

- Metrics depend on historical data availability; short histories produce NaN for some metrics
- Z-scores are relative to the peer group and date; they don't convey absolute attractiveness
- Single benchmark (Nifty 100) is used across categories unless replaced with category-specific benchmarks
- No transaction costs, taxes, or liquidity constraints are modeled

## License

This project is for educational and research purposes.
