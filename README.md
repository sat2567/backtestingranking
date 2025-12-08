# Fund Analysis Dashboard

A comprehensive Streamlit-based application for backtesting and analyzing mutual fund selection strategies. Compare Sharpe, Sortino, Momentum, and custom strategies against benchmarks with interactive visualizations and deep-dive snapshots.

## 🚀 Features

### Core Capabilities
- **Multi-Strategy Backtesting**: Test Sharpe Ratio, Sortino Ratio, Momentum, and Custom algorithms
- **Interactive Dashboard**: Real-time equity curves, performance metrics, and benchmark comparisons
- **Deep-Dive Analysis**: Point-in-time snapshots showing fund rankings and forward returns
- **Flexible Momentum**: Configurable weights for 3M, 6M, and 1Y returns with optional risk adjustment
- **Custom Strategy Builder**: Weight multiple metrics (Sharpe, Sortino, Momentum, Volatility, Info Ratio) to create personalized algorithms

### Key Metrics
- Strategy CAGR and Total Return
- Benchmark (Nifty 100) comparison
- Winning periods ratio
- Forward return analysis
- Risk-adjusted performance

## 📊 Strategies

### 1. Sharpe Ratio
Ranks funds by annualized excess return per unit of volatility:
```
Sharpe = (Annual Return - Risk-Free Rate) / Annual Volatility
```

### 2. Sortino Ratio
Similar to Sharpe but uses downside deviation instead of total volatility:
```
Sortino = (Annual Return - Risk-Free Rate) / Downside Volatility
```

### 3. Momentum Strategy
Composite score based on weighted trailing returns:
```
Momentum Score = (w_3M × 3M_Return) + (w_6M × 6M_Return) + (w_1Y × 1Y_Return)
```
- Optional risk adjustment: divides by 1Y volatility
- Customizable weights via sidebar

### 4. Custom Strategy
Combines multiple metrics with user-defined weights:
- Sharpe Ratio
- Sortino Ratio
- Momentum
- Low Volatility (inverted rank)
- Information Ratio (vs Nifty 100)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone repository
git clone https://github.com/sat2567/cagr-improved-ranking.git
cd cagr-improved-ranking

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run backtest_strategy.py
```

### Dependencies
- `streamlit>=1.33`: Web framework
- `pandas>=2.0`: Data manipulation
- `numpy>=1.24`: Numerical computations
- `plotly>=5.0`: Interactive charts

## 📁 Data Requirements

### Fund Data Format
CSV files in `data/` folder with columns:
- `date`: Date in DD-MM-YYYY format (robust parsing handles variations)
- `scheme_code`: Unique fund identifier (converted to string)
- `scheme_name`: Human-readable fund name
- `nav`: Net Asset Value (numeric)

### Benchmark Data
`data/nifty100_funds.csv` with:
- `date`: Date column
- `nav` or `close`: Benchmark price/index value

### Supported Categories
- Large Cap
- Small Cap
- Mid Cap
- Large & Mid Cap
- Multi Cap
- International

## 🎯 Usage

### Basic Operation
1. **Select Category**: Choose fund category from sidebar
2. **Configure Parameters**:
   - Top N Funds: Number of funds to select per period
   - Rebalance Days: Holding period between rebalances
   - Momentum Weights: 3M, 6M, 1Y return weights
   - Risk Adjust: Toggle volatility normalization for momentum
3. **Choose Strategy Tab**: Sharpe, Sortino, Momentum, or Custom
4. **Analyze Results**: View metrics, charts, and deep-dive tables

### Custom Strategy
1. Adjust weights for different metrics (0-100%)
2. Click "🚀 Run Custom Strategy"
3. Results persist in session until weights change

### Deep-Dive Snapshot
- Select any rebalance date
- View fund rankings by strategy score
- Compare with actual future performance
- Download snapshot data

## 🏗️ Code Structure

### Main Components

#### 1. Data Loading (`load_fund_data`, `load_nifty_data`)
- Robust CSV parsing with column standardization
- Forward-fill missing NAVs (max 2 days)
- Scheme code normalization

#### 2. Metric Calculations
- `calculate_sharpe_ratio`: Risk-adjusted returns
- `calculate_sortino_ratio`: Downside risk focus
- `calculate_flexible_momentum`: Weighted trailing returns
- `calculate_volatility`: Annualized standard deviation
- `calculate_information_ratio`: Active return vs benchmark

#### 3. Backtesting Engine (`run_backtest`)
- Date-range filtering (requires ~1 year history)
- Strategy score calculation per period
- Portfolio simulation with equal weighting
- Equity curve generation

#### 4. Analysis Tables (`generate_snapshot_table`)
- Point-in-time fund scoring
- Forward return calculation
- Ranking by strategy vs actual performance

#### 5. UI Components (`main`)
- Sidebar configuration
- Tabbed strategy interface
- Interactive charts and tables
- Session state management

### Key Parameters
- `DEFAULT_HOLDING = 126`: ~6 months trading days
- `DEFAULT_TOP_N = 5`: Funds per selection
- `RISK_FREE_RATE = 0.06`: Annual risk-free rate
- `EXECUTION_LAG = 1`: T+1 trading assumption
- `TRADING_DAYS_YEAR = 252`: Annual trading days

## 📈 Methodology

### Backtesting Process
1. **Initialization**: Load data and set parameters
2. **Rebalance Loop**:
   - Calculate analysis date
   - Score all funds using selected strategy
   - Select top N funds
   - Simulate holding period returns
   - Update portfolio value
3. **Results**: Generate metrics and visualizations

### Risk Management
- Minimum data requirements (126+ days)
- NaN handling for missing data
- Volatility normalization options
- Benchmark alignment for IR calculations

### Performance Assumptions
- Equal-weighted portfolios
- No transaction costs (except optional exit loads)
- Daily rebalancing at specified intervals
- T+1 execution lag

## 🔧 Customization

### Adding New Strategies
1. Implement metric function in calculations section
2. Add routing logic in `calculate_strategy_score`
3. Update UI tabs in `main()`

### Modifying Parameters
- Edit constants at top of file
- Adjust default weights in sidebar
- Modify date ranges and lookbacks

### Data Pipeline
- Extend `CATEGORY_MAP` for new categories
- Add CSV files following naming convention
- Update `FUNDS_CSV_MAP` accordingly

## 📊 Output Formats

### Charts
- Interactive Plotly equity curves
- Benchmark overlay
- Range slider for date navigation

### Tables
- Performance metrics dashboard
- Deep-dive fund rankings
- CSV export functionality

### Metrics
- CAGR: Compound Annual Growth Rate
- Total Return: Absolute percentage gain
- Benchmark Comparison: Nifty 100 performance

## 🚨 Known Limitations

- Requires sufficient historical data (1+ years minimum)
- Assumes equal-weighted rebalancing
- Benchmark data must align with fund dates
- No consideration of fund expenses beyond exit loads
- Momentum requires 2+ years for full calculation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

This project is part of the CAGR Improved Ranking repository. See repository license for details.

## 📞 Support

For issues or questions:
- Check data file formats
- Verify dependencies installation
- Review error logs in Streamlit

---

**Built with Streamlit for interactive financial analysis.**
