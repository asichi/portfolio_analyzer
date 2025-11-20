# 📊 Portfolio Analyzer

A comprehensive Python tool for analyzing and combining multiple trading strategy backtests. Generates beautiful HTML reports with interactive equity curves, drawdown analysis, and performance metrics.

## 🎯 Features

- **Multi-Strategy Analysis**: Combine multiple trading strategies and analyze them together
- **Interactive Reports**: Beautiful HTML reports with Plotly charts
- **Comprehensive Metrics**: CAGR, win rate, profit factor, expectancy, drawdowns, and more
- **Drawdown Analysis**: 
  - Top 3 portfolio drawdowns with recovery times
  - Strategy contribution breakdown for each drawdown
  - Individual strategy drawdown analysis
- **Monthly Returns Matrix**: Heat-map style table showing returns by month and year
- **Flexible Organization**: Group strategies into folders (swing_systems, day_trades, etc.)
- **Single or Multi-Folder Processing**: Analyze all strategies or target specific groups

## 📋 Requirements

- Python 3.7+
- pandas
- numpy
- plotly (included via CDN in reports)

Install dependencies:
```bash
pip install pandas numpy
```

## 🚀 Quick Start

### 1. Setup Directory Structure

```
your_project/
├── portfolio_analyzer.py
├── datasets/
│   ├── swing_systems/
│   │   ├── ams_6.csv
│   │   ├── ams_7.csv
│   │   └── ams_8.csv
│   ├── gappers/
│   │   ├── ams_10.csv
│   │   ├── ams_11.csv
│   │   ├── ams_12.csv
│   │   └── ams_13.csv
│   └── day_trades/
│       ├── ams_1.csv
│       └── ams_2.csv
└── reports/          # Auto-created
```

### 2. CSV File Format

Your CSV files must contain these columns:
- `Symbol`: Stock ticker
- `Date`: Trade entry date
- `Ex. date`: Trade exit date
- `Profit`: Profit/loss for the trade
- `Cum. Profit`: Cumulative profit (for individual strategy tracking)

**Example CSV:**
```csv
Symbol,Date,Ex. date,Profit,Cum. Profit
AAPL,2020-01-15,2020-01-20,150.50,150.50
MSFT,2020-01-16,2020-01-22,-45.20,105.30
TSLA,2020-01-20,2020-01-25,320.00,425.30
```

### 3. Run Analysis

**Analyze all strategy groups:**
```bash
python portfolio_analyzer.py
```

**Analyze specific group:**
```bash
python portfolio_analyzer.py swing_systems
```

### 4. View Reports

Open the generated HTML files in `reports/` folder:
- `swing_systems.html`
- `gappers.html`
- `day_trades.html`

## 📊 Report Contents

Each HTML report includes:

### Portfolio Summary
- Total Return ($ and %)
- CAGR (Compound Annual Growth Rate)
- Maximum Drawdown ($ and %)
- Win Rate & Trade Statistics
- Profit Factor
- Expectancy per Trade

### Interactive Equity Curve
- Combined portfolio equity over time
- Maximum drawdown visualization
- Peak and trough markers
- Baseline starting capital overlay

### Strategy Breakdown Table
Individual metrics for each strategy:
- Total returns
- CAGR
- Max drawdown
- Win rate
- Profit factor
- Expectancy
- Number of trades

### Top 3 Drawdown Periods
For the combined portfolio:
- Peak, trough, and recovery dates
- Drawdown amount and percentage
- Time to trough and recovery
- Total duration

### Drawdown Strategy Breakdown
Shows which strategies contributed to each major drawdown:
- Individual strategy losses during the period
- Percentage contribution to total drawdown
- Helps identify weak strategies during market stress

### Individual Strategy Drawdowns
Top 3 drawdown periods for each strategy independently:
- Identifies each strategy's worst periods
- May differ from combined drawdown timing
- Useful for understanding uncorrelated risks

### Monthly Returns Matrix
Calendar-style returns table:
- Returns by month and year
- Average returns per month
- Annual totals
- Color-coded for quick visual analysis

## ⚙️ Configuration

Edit these constants at the top of `portfolio_analyzer.py`:

```python
INITIAL_CAPITAL = 100000      # Starting capital for analysis
DATASET_FOLDER = "datasets"    # Where to find CSV files
REPORTS_FOLDER = "reports"     # Where to save HTML reports
```

## 📁 Folder Organization Tips

Organize strategies by type for clearer analysis:

**By Strategy Type:**
- `swing_systems/` - Position trading strategies
- `day_trades/` - Intraday strategies  
- `gappers/` - Gap trading strategies
- `mean_reversion/` - Mean reversion strategies

**By Market:**
- `equities/`
- `crypto/`
- `forex/`

**By Development Stage:**
- `live/` - Currently trading
- `development/` - Under testing
- `archived/` - Deprecated strategies

## 🎨 Example Output

```
============================================================
Portfolio Analyzer - Trading Strategy Combination Tool
============================================================
📁 Found 3 strategy group(s) in datasets/
  📁 Found 3 strategy file(s)
  ⏳ Loading ams_6.csv... ✓ (2223 trades)
  ⏳ Loading ams_7.csv... ✓ (2075 trades)
  ⏳ Loading ams_8.csv... ✓ (3027 trades)
  📊 Combined: $479,603 (479.6%) | CAGR: 5.9% | 7325 trades
  ✓ Report: swing_systems.html
============================================================
✅ ANALYSIS COMPLETE - 3 report(s) generated
============================================================
```

## 🔧 Troubleshooting

### Error: "No subfolders found"
- Create subfolders inside `datasets/`
- Place your CSV files inside those subfolders (not directly in `datasets/`)

### Error: "Missing required columns"
- Ensure your CSV has: Symbol, Date, Ex. date, Profit, Cum. Profit
- Column names are case-sensitive

### Report shows incorrect data
- Verify date formats are parseable (YYYY-MM-DD recommended)
- Check that Cum. Profit is truly cumulative for each strategy
- Ensure Profit values are numeric (not strings)

## 📈 Use Cases

1. **Compare Strategy Performance**: See which strategies perform best
2. **Portfolio Diversification**: Analyze how strategies work together
3. **Risk Management**: Identify drawdown periods and contributing strategies
4. **Performance Tracking**: Monitor strategy evolution over time
5. **Presentation**: Generate professional reports for stakeholders

## 🤝 Contributing

Feel free to submit issues or pull requests for:
- Bug fixes
- New features
- Documentation improvements
- Performance optimizations

## 📝 License

GPL-3.0 License - This software is free and open source. You can redistribute and modify it under the terms of the GNU General Public License v3.0. 

## ⚠️ Disclaimer

This tool is for analysis purposes only. Past performance does not guarantee future results. Trading involves risk of loss.

---

**Created by:** AMS Engineering
**Version:** 1.0  
**Last Updated:** November 2025