# 📊 Portfolio Analyzer

Analyze and combine multiple trading strategy backtests with professional HTML reports featuring interactive charts, drawdown analysis, capital usage tracking, and allocation-compliance testing.

## Features

- **Multi-Format CSV Support** - Auto-detects StrategyEngine (C# backtests) and PortfolioAnalyzer formats
- **Multi-Strategy Analysis** - Combine strategies and analyze performance together
- **All Strategies Report** - Master report combining all strategies across all folders
- **Interactive HTML Reports** - Plotly charts with equity curves and capital usage
- **Comprehensive Metrics** - Corrected annualized profit, drawdowns, win rate, profit factor, monthly GPR, and recovery factors
- **Capital Usage Analysis** - Track deployment with percentile and tail risk metrics
- **Swing Allocation Compliance Test** - Simulates configured swing/weekly allocation caps against raw trades
- **Monthly Gain-to-Pain Ratio** - Schwager-style monthly GPR calculated from monthly portfolio returns
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
├── csv_adapter.py
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

Edit `config.py`.

### Core Settings

```python
INITIAL_CAPITAL = 100000
DATASET_FOLDER = "datasets"
REPORTS_FOLDER = "reports"
```

### CSV Format Detection

These constants define the supported CSV schemas used by the analyzer:

```python
STRATEGYENGINE_REQUIRED = [
    "Symbol",
    "EntryDate",
    "EntryPrice",
    "ExitDate",
    "ExitPrice",
    "PnL",
    "ExitReason",
    "Shares",
]

REQUIRED_COLUMNS = [
    "Symbol",
    "Date",
    "Ex. date",
    "Profit",
    "Position value",
]

COLUMN_MAPPINGS = {
    "strategyengine": {
        "EntryDate": "Date",
        "ExitDate": "Ex. date",
        "PnL": "Profit",
    },
    "portfolioanalyzer": {},
}
```

### Swing Allocation Compliance Test

These settings control the swing/weekly allocation simulation shown in the HTML report:

```python
SWING_CAP_GROUPS = [
    "swing_systems",
    "weekly_systems",
]

SWING_CAP_LEVELS = [
    75_000,
    90_000,
    100_000,
    110_000,
    125_000,
    150_000,
    None,
]

SWING_CAP_ENABLED = True
```

`SWING_CAP_GROUPS` must match folder names under `datasets/`.

`SWING_CAP_LEVELS` defines the tested allocation caps. `None` means uncapped baseline.

The `$100,000` row represents the intended total allocation for swing and weekly systems.

## Report Contents

Each report includes:

- **Portfolio Summary** - Returns, annual profits (period-based), drawdowns, win rate, profit factor, monthly GPR, expectancy, and recovery factors
- **Equity Curve** - Interactive dual-panel chart with equity and drawdown visualization
- **Capital Usage** - Daily deployment chart with percentile analysis and tail-risk metrics
- **Swing Allocation Compliance Test** - Shows how configured swing/weekly allocation caps affect P&L, drawdown, recovery factor, and capital usage
- **Strategy Breakdown** - Individual metrics for each strategy
- **Top Drawdowns** - Peak/trough dates, recovery times, and strategy contributions sorted by impact
- **Individual Strategy Drawdowns** - Top 3 drawdowns for each strategy
- **Monthly Returns** - Calendar table with yearly totals and averages

### All Strategies Report

When running without arguments, an additional `all_strategies.html` report is generated that:

- Combines all unique CSV files across all folders
- Provides a portfolio-wide view of all strategies together
- Shows how different strategy groups interact during stress periods
- Tests configured swing/weekly allocation caps against the full portfolio
- Supports capital allocation and risk management decisions

## Metrics & Corrections

### Annualized Profit Calculation

**Fixed in v2.1:** Annualized profit uses **period-based calculation**:

```text
Annualized Profit = Total Profit / Actual Years
```

The previous yearly-average method diluted partial years.

Example:

```text
3,360 days = 9.20 years, not 10 calendar years
```

### Monthly Gain-to-Pain Ratio

Monthly GPR is calculated using Schwager-style monthly returns:

```text
Monthly GPR = sum(positive monthly returns) / abs(sum(negative monthly returns))
```

This is different from trade-level Profit Factor. Profit Factor is calculated from individual winning and losing trades. Monthly GPR measures equity-curve quality by comparing positive months against losing months.

### Drawdown Calculation

Drawdowns use the **max pain** perspective:

- A drawdown only ends when equity reaches a **new all-time high**
- Partial recoveries do not split the drawdown into separate episodes
- Tracks the worst continuous pain from peak to ultimate trough
- Ensures Portfolio Summary and Top Drawdowns table show consistent maximum drawdown values
- Drawdown percentage is shown relative to both peak equity and initial capital where appropriate

### Swing Allocation Compliance

The Swing Allocation Compliance Test simulates a first-come-first-served capital gate for configured swing groups.

The simulator:

- Applies only to folders listed in `SWING_CAP_GROUPS`
- Leaves non-swing groups, such as day trades and gappers, unthrottled
- Tracks active accepted swing/weekly capital by entry and exit date
- Skips new swing/weekly trades when accepting them would exceed the tested cap
- Rebuilds portfolio P&L, drawdown, recovery factor, profit factor, win rate, and capital usage from the accepted trades

The `None` cap is the uncapped baseline.

## Architecture

**`config.py`** - Core settings, CSV format detection constants, and swing allocation compliance settings  
**`csv_adapter.py`** - Multi-format CSV detection and normalization

- Auto-detects StrategyEngine vs PortfolioAnalyzer format
- Parses metadata headers from StrategyEngine CSVs
- Preserves valid ticker symbols such as `NA`
- Normalizes both formats to the standard internal schema
- Normalizes position value as absolute notional exposure

**`data_loader.py`** - CSV validation and ingestion using `csv_adapter.py`

- Converts normalized adapter columns to PortfolioAnalyzer column names
- Tags trades with `StrategyGroup` from the dataset folder name

**`metrics.py`** - Performance calculations, equity curves, capital usage, drawdown attribution, swing allocation simulation, and monthly GPR

- `simulate_swing_cap_sweep()` - Allocation-compliance simulator for configured swing/weekly groups
- `calculate_monthly_gain_to_pain()` - Schwager-style monthly GPR
- `debug_annualized_calculation()` - Debug utility for validating annualized metrics

**`report_generator/`** - Modular HTML report generation package

- `chart_builder.py` - Plotly chart generation for equity curve and capital usage
- `table_builder.py` - HTML tables for drawdowns, monthly returns, capital usage, and swing allocation compliance
- `html_builder.py` - HTML structure, headers, footer, and summary sections
- `formatters.py` - Formatting utilities for numbers and dates
- `processor.py` - Main processing logic for folders and combined reports

**`main.py`** - CLI entry point and orchestration  
**`styles.css`** - Report styling

## Troubleshooting

**"Cannot detect CSV format"** - Ensure CSV has all required columns for one of the supported formats.

**"No subfolders found"** - Create subfolders inside `datasets/` for your CSV files.

**"Missing required columns"** - Verify column names are present and case-sensitive.

**"Column 'Symbol' contains null values"** - Check for truly blank symbols. Valid tickers such as `NA` are preserved by the CSV reader.

**Incorrect data** - Verify date formats and numeric values.

**Duplicate strategy names** - If the same CSV filename exists in multiple folders, only the first occurrence is used in the `all_strategies.html` report.

**Unexpected allocation test results** - Confirm `SWING_CAP_GROUPS` exactly matches the intended dataset folder names.

## Changelog

### v2.2 (May 2026)

- **NEW:** Swing Allocation Compliance Test for configured swing/weekly capital caps
- **NEW:** Monthly Gain-to-Pain Ratio in portfolio summary
- **ADD:** StrategyGroup tagging by dataset folder for group-level allocation testing
- **FIX:** CSV reader preserves valid ticker symbols like `NA`
- **FIX:** Position value normalized as absolute notional exposure for long and short trades

### v2.1 (May 2026)

- **NEW:** Multi-format CSV support with StrategyEngine and PortfolioAnalyzer auto-detection
- **FIX:** Annualized profit calculation now uses period-based method instead of yearly average
- **ADD:** `csv_adapter.py` module for format detection and normalization
- **ADD:** `debug_annualized_calculation()` utility function

### v2.0

- Initial release with comprehensive portfolio analysis

## License

GPL-3.0 License

## Disclaimer

For analysis purposes only. Past performance does not guarantee future results.

---

**Version:** 2.2 | **Last Updated:** May 2026
