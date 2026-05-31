# 📊 Portfolio Analyzer

Analyze and combine multiple trading strategy backtests with professional HTML reports featuring interactive charts, drawdown analysis, and capital usage tracking.

## Features

- **Multi-Format CSV Support** - Auto-detects StrategyEngine (C# backtests) and PortfolioAnalyzer formats
- **Multi-Strategy Analysis** - Combine strategies and analyze performance together
- **All Strategies Report** - Master report combining all strategies across all folders
- **Interactive HTML Reports** - Plotly charts with equity curves and capital usage
- **Comprehensive Metrics** - Corrected annualized profit, drawdowns, win rate, profit factor, recovery factors
- **Capital Usage Analysis** - Track deployment with percentile and tail risk metrics
- **Drawdown Attribution** - Identify which strategies contribute to portfolio drawdowns
- **Individual Strategy Drawdowns** - Top 3 drawdowns for each strategy
- **CLI Support** - Analyze all strategies or target specific groups

## Installation

```bash
pip install pandas numpy
```

## Quick Start

### 1. Project Structure

```text
your_project/
├── main.py
├── config.py
├── csv_adapter.py          # NEW: Format detection
├── data_loader.py
├── metrics.py
├── report_generator/
│   ├── __init__.py
│   ├── chart_builder.py
│   ├── table_builder.py
│   ├── html_builder.py
│   ├── formatters.py
│   └── processor.py
├── styles.css
├── datasets/
│   ├── swing_systems/
│   │   ├── AmsD1_20260530_0721.csv
│   │   ├── AmsD6_20260530_0722.csv
│   │   └── ...
│   ├── day_trades/
│   │   ├── AmsA1_20260531_0819.csv
│   │   ├── AmsA2_20260531_0820.csv
│   │   └── ...
│   ├── gappers/
│   │   └── AmsG1-4_*.csv
│   └── weekly_systems/
│       └── AmsB1-2_*.csv
└── reports/          # Auto-generated
    ├── swing_systems.html
    ├── day_trades.html
    ├── gappers.html
    ├── weekly_systems.html
    └── all_strategies.html
```

### 2. CSV Formats Supported

#### StrategyEngine Format (C# Backtests)

With optional metadata headers:

```csv
# === BACKTEST METADATA ===
# StrategyName: AmsA1
# BacktestDate: 2026-05-31
# CapitalAllocation: 150000
# RiskAmount: 750
# MaxPositionValue: 15000
# === TRADES ===
Symbol,EntryDate,EntryPrice,ExitDate,ExitPrice,PnL,ExitReason,Shares
AAPL,2020-01-15,150.00,2020-01-20,155.00,125.00,Target,1
MSFT,2020-01-16,200.00,2020-01-22,199.10,-90.00,StopLoss,1
```

#### PortfolioAnalyzer Format (Legacy)

```csv
Symbol,Date,Ex. date,Profit,Position value
AAPL,2020-01-15,2020-01-20,150.50,10000
MSFT,2020-01-16,2020-01-22,-45.20,10000
```

**Format auto-detected - no configuration needed!**

### 3. Run

```bash
# Analyze all folders + generate combined "all_strategies" report
python main.py

# Analyze specific folder only (no combined report)
python main.py swing_systems
```

Reports saved to `reports/` folder.

## Configuration

Edit `config.py`:

```python
INITIAL_CAPITAL = 100000
DATASET_FOLDER = "datasets"
REPORTS_FOLDER = "reports"

# Auto-detected formats:
# STRATEGYENGINE_REQUIRED = ["Symbol", "EntryDate", "EntryPrice", "ExitDate", "ExitPrice", "PnL", "ExitReason", "Shares"]
# REQUIRED_COLUMNS = ["Symbol", "Date", "Ex. date", "Profit", "Position value"]  # PortfolioAnalyzer
```

## Report Contents

Each report includes:

- **Portfolio Summary** - Returns, Annual Profits (period-based), drawdowns, win rate, profit factor, recovery factors
- **Equity Curve** - Interactive dual-panel chart (equity + drawdown visualization)
- **Capital Usage** - Daily deployment chart with percentile analysis and tail risk metrics
- **Strategy Breakdown** - Individual metrics for each strategy
- **Top Drawdowns** - Peak/trough dates, recovery times, strategy contributions sorted by impact
- **Individual Strategy Drawdowns** - Top 3 drawdowns for each strategy
- **Monthly Returns** - Calendar heatmap with yearly totals

### All Strategies Report

When running without arguments, an additional `all_strategies.html` report is generated that:

- Combines all unique CSV files across all folders
- Provides portfolio-wide view of all strategies together
- Shows how different strategy groups interact during stress periods
- Useful for overall capital allocation and risk management decisions

## Metrics & Corrections

### Annualized Profit Calculation

**Fixed in v2.1:** Annualized profit now uses **period-based calculation** (correct):

```text
Annualized Profit = Total Profit / Actual Years
```

**Before:** Used yearly average which diluted partial years (incorrect)

- Example: 3,360 days = 9.20 years (not 10 calendar years)
- Result: +8% correction in reported annualized returns

## Drawdown Calculation

Drawdowns use the **"max pain" perspective**:

- A drawdown only ends when equity reaches a **new all-time high**
- Partial recoveries don't split the drawdown into separate episodes
- Tracks the worst continuous pain from peak to ultimate trough
- Ensures Portfolio Summary and Top Drawdowns table show consistent maximum drawdown values
- Drawdown % calculated relative to initial capital (not peak) for practical risk assessment

## Architecture

**`config.py`** - Configuration constants and format detection patterns  
**`csv_adapter.py`** - Multi-format CSV detection and normalization

- Auto-detects StrategyEngine vs PortfolioAnalyzer format
- Parses metadata headers from StrategyEngine CSVs
- Normalizes both formats to standard schema  
**`data_loader.py`** - CSV validation and ingestion (uses csv_adapter)  
**`metrics.py`** - Performance calculations, equity curves, capital usage, drawdown attribution
- `debug_annualized_calculation()` - Debug utility for validating annualized metrics  
**`report_generator/`** - Modular HTML report generation package
- `chart_builder.py` - Plotly chart generation (equity curve, capital usage)
- `table_builder.py` - HTML tables (drawdowns, monthly returns, capital usage)
- `html_builder.py` - HTML structure, headers, and summary sections
- `formatters.py` - Formatting utilities for numbers and dates
- `processor.py` - Main processing logic for folders and combined reports  
**`main.py`** - CLI entry point and orchestration  
**`styles.css`** - Report styling  

## Troubleshooting

**"Cannot detect CSV format"** - Ensure CSV has all required columns for one of the supported formats

**"No subfolders found"** - Create subfolders inside `datasets/` for your CSV files

**"Missing required columns"** - Verify column names (case-sensitive)

**Incorrect data** - Verify date formats (YYYY-MM-DD) and numeric values

**Duplicate strategy names** - If same CSV filename exists in multiple folders, only first occurrence is used in all_strategies report

## Changelog

### v2.1 (December 2025)

- **NEW:** Multi-format CSV support (StrategyEngine + PortfolioAnalyzer auto-detection)
- **FIX:** Annualized profit calculation now uses period-based method (was yearly-average)
- **ADD:** csv_adapter.py module for format detection and normalization
- **ADD:** debug_annualized_calculation() utility function

### v2.0

- Initial release with comprehensive portfolio analysis

## License

GPL-3.0 License

## Disclaimer

For analysis purposes only. Past performance does not guarantee future results.

---

**Version:** 2.1 | **Last Updated:** May 2026
