import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load results
summary = pd.read_csv('output/backtest_results_summary.csv')
detailed = pd.read_csv('output/backtest_results_detailed.csv')

# Convert dates
summary['rebalance_date'] = pd.to_datetime(summary['rebalance_date'])
detailed['rebalance_date'] = pd.to_datetime(detailed['rebalance_date'])

print("="*70)
print("BACKTEST ANALYSIS REPORT")
print("="*70)

# ============================================================================
# 1. OVERALL PERFORMANCE METRICS
# ============================================================================
print("\n1. OVERALL PERFORMANCE METRICS")
print("-" * 70)
print(f"Total rebalance periods: {len(summary)}")
print(f"Total fund selections: {len(detailed)}")
print(f"Unique funds selected: {detailed['scheme_code'].nunique()}")
print(f"\nAverage overlap accuracy: {summary['overlap_accuracy'].mean()*100:.2f}%")
print(f"Median overlap accuracy: {summary['overlap_accuracy'].median()*100:.2f}%")
print(f"Best overlap accuracy: {summary['overlap_accuracy'].max()*100:.2f}%")
print(f"Worst overlap accuracy: {summary['overlap_accuracy'].min()*100:.2f}%")
print(f"\nAverage rank of selected funds: {summary['avg_rank_of_selected'].mean():.2f}")
print(f"Median rank of selected funds: {summary['avg_rank_of_selected'].median():.2f}")

# ============================================================================
# 2. RETURN ANALYSIS
# ============================================================================
print("\n2. RETURN ANALYSIS")
print("-" * 70)
print(f"Average 6M forward return: {summary['mean_future_return'].mean()*100:.2f}%")
print(f"Median 6M forward return: {summary['mean_future_return'].median()*100:.2f}%")
print(f"Best 6M forward return: {summary['mean_future_return'].max()*100:.2f}%")
print(f"Worst 6M forward return: {summary['mean_future_return'].min()*100:.2f}%")
print(f"Standard deviation: {summary['mean_future_return'].std()*100:.2f}%")
print(f"\nPositive return periods: {(summary['mean_future_return'] > 0).sum()} ({(summary['mean_future_return'] > 0).mean()*100:.1f}%)")
print(f"Negative return periods: {(summary['mean_future_return'] < 0).sum()} ({(summary['mean_future_return'] < 0).mean()*100:.1f}%)")

# ============================================================================
# 3. SELECTION ACCURACY DISTRIBUTION
# ============================================================================
print("\n3. SELECTION ACCURACY DISTRIBUTION")
print("-" * 70)
overlap_dist = summary['overlap_count'].value_counts().sort_index()
for count in range(6):
    pct = (summary['overlap_count'] == count).sum() / len(summary) * 100
    if pct > 0:
        print(f"{count}/5 funds in top 5: {int((summary['overlap_count'] == count).sum())} periods ({pct:.1f}%)")

# ============================================================================
# 4. TIME-SERIES TRENDS
# ============================================================================
print("\n4. PERFORMANCE OVER TIME")
print("-" * 70)
summary_sorted = summary.sort_values('rebalance_date')
first_half = summary_sorted.iloc[:len(summary_sorted)//2]
second_half = summary_sorted.iloc[len(summary_sorted)//2:]

print(f"First half ({first_half['rebalance_date'].min().year}-{first_half['rebalance_date'].max().year}):")
print(f"  Avg overlap accuracy: {first_half['overlap_accuracy'].mean()*100:.2f}%")
print(f"  Avg return: {first_half['mean_future_return'].mean()*100:.2f}%")
print(f"\nSecond half ({second_half['rebalance_date'].min().year}-{second_half['rebalance_date'].max().year}):")
print(f"  Avg overlap accuracy: {second_half['overlap_accuracy'].mean()*100:.2f}%")
print(f"  Avg return: {second_half['mean_future_return'].mean()*100:.2f}%")

# ============================================================================
# 5. MOST FREQUENTLY SELECTED FUNDS
# ============================================================================
print("\n5. MOST FREQUENTLY SELECTED FUNDS")
print("-" * 70)
fund_counts = detailed.groupby(['scheme_code', 'scheme_name']).size().sort_values(ascending=False)
print("Top 10 most selected funds:")
for idx, (fund, count) in enumerate(fund_counts.head(10).items(), 1):
    scheme_code, scheme_name = fund
    pct = count / len(summary) * 100
    avg_return = detailed[detailed['scheme_code'] == scheme_code]['forward_return'].mean() * 100
    print(f"{idx:2d}. {scheme_name[:50]:<50} - Selected {count:2d}x ({pct:4.1f}%) | Avg 6M return: {avg_return:6.2f}%")

# ============================================================================
# 6. BEST PERFORMING SELECTIONS
# ============================================================================
print("\n6. BEST PERFORMING SELECTIONS")
print("-" * 70)
top_returns = detailed.nlargest(10, 'forward_return')[['rebalance_date', 'scheme_name', 'forward_return', 'forward_rank', 'is_in_actual_top']]
print("Top 10 individual fund returns:")
for idx, row in enumerate(top_returns.itertuples(), 1):
    in_top = "Y" if row.is_in_actual_top else "N"
    print(f"{idx:2d}. {row.rebalance_date.strftime('%Y-%m-%d')} | {row.scheme_name[:40]:<40} | Return: {row.forward_return*100:6.2f}% | Rank: {row.forward_rank:2.0f} | In Top 5: {in_top}")

# ============================================================================
# 7. RISK METRICS ANALYSIS
# ============================================================================
print("\n7. RISK METRICS OF SELECTED FUNDS")
print("-" * 70)
print(f"Average Sharpe ratio: {detailed['sharpe'].mean():.3f}")
print(f"Average Sortino ratio: {detailed['sortino'].mean():.3f}")
print(f"Average Max Drawdown: {detailed['mdd'].mean()*100:.2f}%")
print(f"\nCorrelation between metrics and forward returns:")
print(f"  Sharpe ratio: {detailed['sharpe'].corr(detailed['forward_return']):.3f}")
print(f"  Sortino ratio: {detailed['sortino'].corr(detailed['forward_return']):.3f}")
print(f"  Max Drawdown: {detailed['mdd'].corr(detailed['forward_return']):.3f}")
print(f"  Composite score: {detailed['composite_score'].corr(detailed['forward_return']):.3f}")

# ============================================================================
# 8. PERFORMANCE BY OVERLAP ACCURACY
# ============================================================================
print("\n8. RETURNS BY OVERLAP ACCURACY")
print("-" * 70)
for overlap in sorted(summary['overlap_accuracy'].unique()):
    subset = summary[summary['overlap_accuracy'] == overlap]
    avg_return = subset['mean_future_return'].mean() * 100
    count = len(subset)
    print(f"Overlap {int(overlap*5)}/5: Avg return = {avg_return:6.2f}% ({count} periods)")

# ============================================================================
# 9. VISUALIZATION
# ============================================================================
print("\n9. GENERATING VISUALIZATIONS...")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Backtest Analysis Dashboard', fontsize=16, fontweight='bold')

# 1. Returns over time
ax1 = axes[0, 0]
ax1.plot(summary_sorted['rebalance_date'], summary_sorted['mean_future_return'] * 100, 
         marker='o', linewidth=2, markersize=6, color='#2E86AB')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.set_title('6-Month Forward Returns Over Time', fontweight='bold')
ax1.set_xlabel('Rebalance Date')
ax1.set_ylabel('Return (%)')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Overlap accuracy over time
ax2 = axes[0, 1]
ax2.plot(summary_sorted['rebalance_date'], summary_sorted['overlap_accuracy'] * 100, 
         marker='s', linewidth=2, markersize=6, color='#A23B72')
ax2.set_title('Selection Accuracy Over Time', fontweight='bold')
ax2.set_xlabel('Rebalance Date')
ax2.set_ylabel('Overlap Accuracy (%)')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. Distribution of overlap counts
ax3 = axes[0, 2]
overlap_counts = summary['overlap_count'].value_counts().sort_index()
colors = ['#F18F01' if x < 3 else '#C73E1D' if x == 3 else '#6A994E' for x in overlap_counts.index]
ax3.bar(overlap_counts.index, overlap_counts.values, color=colors, edgecolor='black')
ax3.set_title('Distribution of Overlap Counts', fontweight='bold')
ax3.set_xlabel('Number of Funds in Top 5')
ax3.set_ylabel('Frequency')
ax3.set_xticks(range(6))
ax3.grid(True, alpha=0.3, axis='y')

# 4. Return distribution
ax4 = axes[1, 0]
ax4.hist(summary['mean_future_return'] * 100, bins=15, color='#2E86AB', 
         edgecolor='black', alpha=0.7)
ax4.axvline(x=summary['mean_future_return'].mean() * 100, color='red', 
            linestyle='--', linewidth=2, label=f'Mean: {summary["mean_future_return"].mean()*100:.2f}%')
ax4.set_title('Distribution of Forward Returns', fontweight='bold')
ax4.set_xlabel('6-Month Return (%)')
ax4.set_ylabel('Frequency')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. Rank vs overlap accuracy
ax5 = axes[1, 1]
scatter = ax5.scatter(summary['avg_rank_of_selected'], summary['overlap_accuracy'] * 100,
                      c=summary['mean_future_return'] * 100, cmap='RdYlGn',
                      s=100, edgecolors='black', alpha=0.7)
ax5.set_title('Avg Rank vs Overlap Accuracy', fontweight='bold')
ax5.set_xlabel('Average Rank of Selected Funds')
ax5.set_ylabel('Overlap Accuracy (%)')
ax5.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax5, label='6M Return (%)')

# 6. Cumulative returns
ax6 = axes[1, 2]
cumulative_returns = (1 + summary_sorted['mean_future_return']).cumprod() - 1
ax6.plot(summary_sorted['rebalance_date'], cumulative_returns * 100, 
         linewidth=2.5, color='#6A994E')
ax6.fill_between(summary_sorted['rebalance_date'], 0, cumulative_returns * 100, 
                  alpha=0.3, color='#6A994E')
ax6.set_title('Cumulative Returns', fontweight='bold')
ax6.set_xlabel('Rebalance Date')
ax6.set_ylabel('Cumulative Return (%)')
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('output/backtest_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: output/backtest_analysis.png")

# ============================================================================
# 10. DETAILED REBALANCE TABLE
# ============================================================================
print("\n10. DETAILED REBALANCE HISTORY")
print("-" * 70)
print(f"{'Date':<12} {'Overlap':<8} {'Avg Rank':<10} {'Return':<10} {'Selected Funds'}")
print("-" * 70)
for _, row in summary_sorted.iterrows():
    date_str = row['rebalance_date'].strftime('%Y-%m-%d')
    overlap_str = f"{int(row['overlap_count'])}/5"
    rank_str = f"{row['avg_rank_of_selected']:.1f}"
    return_str = f"{row['mean_future_return']*100:+.2f}%"
    funds_str = row['selected_funds'].replace(',', ', ')[:50]
    print(f"{date_str:<12} {overlap_str:<8} {rank_str:<10} {return_str:<10} {funds_str}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
