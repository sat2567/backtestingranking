# Large Cap Mutual Fund Selection Strategy - Backtest Report

## Overview
This backtest evaluates a systematic fund selection strategy based on risk-adjusted performance metrics. The strategy selects the top 5 large-cap mutual funds every 6 months using a composite scoring system that combines Sharpe ratio, Sortino ratio, and Maximum Drawdown.

## Methodology

### 1. Data Preparation
- **Data Source**: `largecap_funds.csv` containing NAV data for 31 large-cap funds
- **Date Range**: November 2015 - October 2025 (~10 years)
- **Final Universe**: 13 funds with sufficient data quality
- **Data Cleaning**: 
  - Removed duplicate entries
  - Forward-filled missing NAVs up to 5 days
  - Excluded funds with >20% missing data

### 2. Strategy Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Lookback Period | 252 days (~1 year) | Historical data for metric calculation |
| Holding Period | 126 days (~6 months) | Time between rebalancing |
| Top N | 5 funds | Number of funds selected per period |
| Risk-free Rate | 5% annual | Used for Sharpe/Sortino calculation |
| Sharpe Weight | 0.4 | Weight in composite score |
| Sortino Weight | 0.4 | Weight in composite score |
| Max Drawdown Weight | -0.2 | Weight in composite score (negative = lower is better) |

### 3. Risk Metrics Calculation
For each rebalance date, using the past 1-year data:
- **Sharpe Ratio**: (Excess Return) / (Total Volatility) - annualized
- **Sortino Ratio**: (Excess Return) / (Downside Volatility) - annualized
- **Maximum Drawdown**: Largest peak-to-trough decline

### 4. Fund Selection Process
1. Calculate risk metrics for all funds
2. Normalize metrics using z-score transformation
3. Compute composite score: `0.4×Sharpe + 0.4×Sortino - 0.2×MDD`
4. Select top 5 funds by composite score

### 5. Performance Evaluation
For each selection:
- Measure 6-month forward returns
- Compare selected funds to actual top performers
- Calculate overlap accuracy and average rank

---

## Key Findings

### Overall Performance Metrics
- **Total Rebalance Periods**: 17 (Nov 2016 - Dec 2024)
- **Average 6M Return**: 7.57%
- **Median 6M Return**: 7.68%
- **Best Period**: +26.88% (June 2020)
- **Worst Period**: -15.42% (Nov 2019)
- **Win Rate**: 82.4% (14 out of 17 periods positive)

### Selection Accuracy
- **Average Overlap Accuracy**: 47.06%
  - This means on average, 2.35 out of 5 selected funds ended up in the actual top 5 performers
- **Average Rank of Selected Funds**: 6.53
  - Out of 13 total funds, our selections averaged in the top half

### Accuracy Distribution
| Overlap | Frequency | Percentage |
|---------|-----------|------------|
| 1/5 funds correct | 2 periods | 11.8% |
| 2/5 funds correct | 8 periods | 47.1% |
| 3/5 funds correct | 6 periods | 35.3% |
| 4/5 funds correct | 1 period | 5.9% |

### Temporal Performance
| Period | Avg Overlap Accuracy | Avg Return |
|--------|---------------------|------------|
| First Half (2016-2020) | 42.50% | 6.06% |
| Second Half (2020-2024) | 51.11% | 8.92% |

**Insight**: Strategy improved over time, showing better selection accuracy and returns in recent years.

---

## Most Frequently Selected Funds

| Rank | Fund Name | Times Selected | Selection Rate | Avg 6M Return |
|------|-----------|----------------|----------------|---------------|
| 1 | HDFC Large Cap Fund | 11 | 64.7% | 7.41% |
| 2 | BANDHAN Large Cap Fund | 10 | 58.8% | 8.06% |
| 3 | CANARA ROBECO Large Cap Fund | 9 | 52.9% | 7.52% |
| 4 | Axis Large Cap Fund | 8 | 47.1% | 8.09% |
| 5 | UTI Large Cap Fund | 8 | 47.1% | 9.55% |

**Insight**: HDFC and BANDHAN Large Cap funds consistently met the risk-adjusted criteria, appearing in ~60% of rebalances.

---

## Risk Metrics Analysis

### Average Metrics of Selected Funds
- **Sharpe Ratio**: 0.969
- **Sortino Ratio**: 1.316
- **Maximum Drawdown**: -13.13%

### Correlation with Forward Returns
| Metric | Correlation |
|--------|-------------|
| Sharpe Ratio | -0.269 |
| Sortino Ratio | -0.281 |
| Max Drawdown | -0.511 |
| Composite Score | 0.003 |

**Critical Insight**: Surprisingly, historical risk metrics showed **negative correlation** with future returns:
- Past Sharpe/Sortino ratios were poor predictors of future performance
- Max Drawdown showed the strongest (negative) correlation
- The composite score had near-zero predictive power

This suggests **mean reversion** in fund performance - funds with exceptional past risk-adjusted returns tend to underperform going forward.

---

## Best Performing Selections

Top 10 individual fund returns (from selected portfolio):

| Date | Fund | 6M Return | Rank | In Top 5? |
|------|------|-----------|------|-----------|
| 2020-06-02 | UTI Large Cap Fund | 29.22% | 2 | Yes |
| 2020-06-02 | Canara Robeco Large Cap | 26.73% | 6 | No |
| 2020-06-02 | Axis Large Cap Fund | 26.45% | 7 | No |
| 2023-12-22 | JM Large Cap Fund | 22.43% | 1 | Yes |
| 2023-06-19 | JM Large Cap Fund | 21.20% | 1 | Yes |

**Insight**: June 2020 (post-COVID recovery) provided exceptional returns across most selections.

---

## Returns by Overlap Accuracy

| Overlap | Avg Return | # Periods |
|---------|------------|-----------|
| 1/5 correct | +12.15% | 2 |
| 2/5 correct | +4.36% | 8 |
| 3/5 correct | +8.49% | 6 |
| 4/5 correct | +18.62% | 1 |

**Interesting Finding**: Higher overlap accuracy correlates with better returns, BUT the relationship is non-linear. The 1/5 overlap periods had strong returns, suggesting that even when most selections were "wrong" vs the top 5, they still captured market upside.

---

## Strategy Strengths

1. **Consistent Positive Returns**: 82.4% win rate over 17 periods
2. **Risk Management**: Selected funds had reasonable drawdown profiles (-13.13% avg)
3. **Improved Over Time**: Better performance in recent years
4. **Diversification**: Naturally rotated across 13 different funds

---

## Strategy Limitations

1. **Limited Universe**: Only 13 funds after data quality filtering
2. **Low Predictive Power**: Historical risk metrics showed weak/negative correlation with future returns
3. **Modest Overlap Accuracy**: Only ~47% overlap with actual top performers
4. **Mean Reversion**: Past winners tend to become future laggards

---

## Recommendations for Improvement

### 1. Modify Weighting Scheme
Given the negative correlation, consider:
- Reducing weight on historical Sharpe/Sortino
- Incorporating momentum factors
- Adding fundamental metrics (expense ratio, AUM, fund manager tenure)

### 2. Extend Lookback Period
- Test 2-year or 3-year lookbacks instead of 1 year
- Use multiple timeframes and ensemble scoring

### 3. Include More Factors
- **Expense Ratio**: Lower costs compound over time
- **Portfolio Turnover**: Indicator of trading costs
- **Tracking Error**: Consistency vs benchmark
- **Fund Flows**: Recent inflows/outflows

### 4. Reduce Holding Period Frequency
- Consider annual rebalancing instead of 6-monthly
- Reduces transaction costs and mean reversion effects

### 5. Market Regime Detection
- Adjust weights based on market volatility regime
- Use different selection criteria for bull vs bear markets

---

## Files Generated

1. **backtest_strategy.py** - Main backtesting script (6 steps implementation)
2. **analyze_results.py** - Comprehensive analysis and visualization
3. **backtest_results_summary.csv** - Rebalance-level results (17 records)
4. **backtest_results_detailed.csv** - Fund-level results (85 records)
5. **backtest_analysis.png** - 6-panel visualization dashboard
6. **README.md** - This documentation

---

## Conclusion

The risk-adjusted momentum strategy showed **modest success** with consistent positive returns (7.57% avg) and reasonable selection accuracy (47.06%). However, the **near-zero predictive power** of historical risk metrics suggests that:

1. **Past performance is not predictive** of future results (even risk-adjusted metrics)
2. **Mean reversion dominates** in mutual fund performance
3. **Alternative factors** (fundamental, behavioral, or momentum-based) may be more predictive
4. **Simple equal-weight** or **passive index investing** might be competitive alternatives

For practical implementation, consider:
- Using this as a **screening tool** combined with qualitative analysis
- **Equal-weighting** selected funds to reduce concentration risk
- **Quarterly review** instead of mechanical 6-month rebalancing
- Combining with **factor-based** or **fundamental** filters

---

## How to Run

```bash
# 1. Run the main backtest
python backtest_strategy.py

# 2. Generate analysis and visualizations  
python analyze_results.py

# 3. View results
# - backtest_results_summary.csv (17 rebalance periods)
# - backtest_results_detailed.csv (85 fund selections)
# - backtest_analysis.png (visualization dashboard)
```

---

**Backtest Period**: November 2016 - December 2024  
**Universe**: 13 Large Cap Mutual Funds  
**Strategy**: Risk-Adjusted Composite Scoring (Sharpe + Sortino - MDD)  
**Rebalancing**: Semi-Annual (6 months)
