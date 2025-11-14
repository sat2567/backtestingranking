# Executive Summary - Fund Selection Strategy Backtest

## Quick Stats

| Metric | Value |
|--------|-------|
| **Strategy Period** | Nov 2016 - Dec 2024 (8+ years) |
| **Number of Rebalances** | 17 periods |
| **Average 6-Month Return** | **7.57%** |
| **Win Rate** | **82.4%** (14/17 positive) |
| **Best Period** | +26.88% (Jun 2020) |
| **Worst Period** | -15.42% (Nov 2019) |
| **Selection Accuracy** | 47.06% overlap with top 5 |

## Strategy in One Sentence
**Select 5 large-cap funds every 6 months based on a composite score combining Sharpe ratio (0.4), Sortino ratio (0.4), and inverted Max Drawdown (-0.2) calculated over the past year.**

---

## Key Insights

### ✅ What Worked
1. **Consistent Returns**: Positive returns in 14 out of 17 periods
2. **Risk Management**: Average drawdown of -13.13% for selected funds
3. **Improving Over Time**: Better accuracy and returns in 2020-2024 vs 2016-2020
4. **Natural Diversification**: Rotated across 13 different funds

### ❌ What Didn't Work
1. **Weak Predictive Power**: Historical risk metrics showed near-zero correlation with future returns
2. **Mean Reversion Effect**: Past high-performers tended to underperform going forward
3. **Limited Universe**: Only 13 funds qualified after data cleaning
4. **Modest Overlap**: Less than 50% overlap with actual top performers

---

## Top Performers

### Most Reliable Funds (Frequently Selected)
1. **HDFC Large Cap** - 11/17 selections (64.7%) | 7.41% avg return
2. **BANDHAN Large Cap** - 10/17 selections (58.8%) | 8.06% avg return
3. **CANARA ROBECO Large Cap** - 9/17 selections (52.9%) | 7.52% avg return

### Best Individual Returns
1. **UTI Large Cap** (Jun 2020) - **29.22%** return
2. **JM Large Cap** (Dec 2023) - **22.43%** return
3. **JM Large Cap** (Jun 2023) - **21.20%** return

---

## Critical Finding: Mean Reversion

**Historical risk metrics showed NEGATIVE correlation with future returns:**
- Sharpe Ratio → Future Returns: **-0.269**
- Sortino Ratio → Future Returns: **-0.281**
- Max Drawdown → Future Returns: **-0.511**
- Composite Score → Future Returns: **+0.003**

**Translation**: Funds with great past Sharpe/Sortino ratios underperformed going forward. This suggests the mutual fund market exhibits strong mean reversion.

---

## Should You Use This Strategy?

### ✅ Yes, if you...
- Want a systematic, rules-based approach
- Prefer active rotation over buy-and-hold
- Can tolerate 47% selection "accuracy" (better than random)
- Value risk-adjusted metrics for screening

### ⚠️ Consider alternatives if you...
- Expect past winners to keep winning (they don't)
- Want >80% accuracy in selecting top performers
- Prefer simpler equal-weight or index strategies
- Can't accept 8-10% volatility in returns

---

## Recommendations

### Immediate Improvements
1. **Add Momentum Filters** - Recent 3-month performance
2. **Include Expense Ratios** - Lower costs matter
3. **Extend Lookback** - Try 2-3 year windows
4. **Reduce Rebalancing** - Annual instead of 6-monthly

### Strategy Enhancements
1. **Market Regime Detection** - Different criteria for bull/bear markets
2. **Factor Combinations** - Blend value, momentum, quality
3. **Equal Weight Top 10** - Reduce concentration risk
4. **Combine with Index** - 70% index + 30% strategy

---

## The Bottom Line

**Performance**: Solid but not exceptional (7.57% per 6 months ≈ 15-16% annualized)

**Predictability**: Low - historical metrics don't reliably predict winners

**Practical Value**: 
- ✅ Better than random selection
- ✅ Provides systematic discipline
- ❌ Not a "holy grail" strategy
- ❌ Passive indexing might be competitive after costs

**Best Use Case**: As a **screening tool** combined with qualitative analysis, NOT as a fully automated selection system.

---

## Files for Review

1. **README.md** - Full methodology and findings (detailed)
2. **backtest_results_summary.csv** - 17 rebalance periods
3. **backtest_results_detailed.csv** - 85 individual fund selections
4. **backtest_analysis.png** - 6-panel visual dashboard
5. **backtest_strategy.py** - Reproducible code (all 6 steps)

---

## Next Steps

1. **Review Detailed Results**: Open CSVs to see specific funds and dates
2. **Study Visualizations**: `backtest_analysis.png` shows trends clearly
3. **Test Modifications**: Edit parameters in `backtest_strategy.py` and re-run
4. **Compare to Benchmark**: Compare 15-16% annualized to Nifty 50 returns
5. **Consider Implementation**: If satisfied, start with small allocation

---

**Remember**: Past performance ≠ Future results. This backtest shows the strategy worked historically, but mean reversion suggests caution in expecting similar results going forward.
