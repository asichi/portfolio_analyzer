# 📊 Portfolio Analyzer

Analyze and combine multiple trading strategy backtests with professional HTML reports featuring interactive charts, drawdown analysis, and capital usage tracking.

## Features

- **Multi-Strategy Analysis** - Combine strategies and analyze performance together
- **All Strategies Report** - Master report combining all strategies across all folders
- **Interactive HTML Reports** - Plotly charts with equity curves and capital usage
- **Comprehensive Metrics** - Annual Profit, drawdowns, win rate, profit factor, recovery factors
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
```
your_project/
├── main.py
├── config.py
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
│   │   ├── ams_3.csv
│   │   ├── ams_4.csv
│   │   ├── ams_5.csv
│   │   └── ams_6.csv
│   ├── day_trades/
│   │   ├── ams_1.csv
│   │   └── ams_2.csv
│   ├── gappers/
│   │   └── ...
│   └── weekly_systems/
│       └── ...
└── reports/          # Auto-generated
    ├── swing_systems.html
    ├── day_trades.html
    ├── gappers.html
    ├── weekly_systems.html
    └── all_strategies.html
```

### 2. CSV Format

Required columns: `Symbol`, `Date`, `Ex. date`, `Profit`, `Position value`
```csv
Symbol,Date,Ex. date,Profit,Position value
AAPL,2020-01-15,2020-01-20,150.50,10000
MSFT,2020-01-16,2020-01-22,-45.20,10000
```

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
REQUIRED_COLUMNS = ["Symbol", "Date", "Ex. date", "Profit", "Position value"]
```

## Report Contents

Each report includes:

- **Portfolio Summary** - Returns, Annual Profits, drawdowns, win rate, profit factor, recovery factors
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

## Drawdown Calculation

Drawdowns use the **"max pain" perspective**:
- A drawdown only ends when equity reaches a **new all-time high**
- Partial recoveries don't split the drawdown into separate episodes
- Tracks the worst continuous pain from peak to ultimate trough
- Ensures Portfolio Summary and Top Drawdowns table show consistent maximum drawdown values
- Drawdown % calculated relative to initial capital (not peak) for practical risk assessment

## Architecture

**`config.py`** - Configuration constants  
**`data_loader.py`** - CSV validation and ingestion  
**`metrics.py`** - Performance calculations, equity curves, capital usage, drawdown attribution  
**`report_generator/`** - Modular HTML report generation package
  - `chart_builder.py` - Plotly chart generation (equity curve, capital usage)
  - `table_builder.py` - HTML tables (drawdowns, monthly returns, capital usage)
  - `html_builder.py` - HTML structure, headers, and summary sections
  - `formatters.py` - Formatting utilities for numbers and dates
  - `processor.py` - Main processing logic for folders and combined reports  
**`main.py`** - CLI entry point and orchestration  
**`styles.css`** - Report styling  

## Troubleshooting

**"No subfolders found"** - Create subfolders inside `datasets/` for your CSV files

**"Missing required columns"** - Ensure CSV has all required columns (case-sensitive)

**Incorrect data** - Verify date formats (YYYY-MM-DD) and numeric Profit values

**Duplicate strategy names** - If same CSV filename exists in multiple folders, only first occurrence is used in all_strategies report

## License

GPL-3.0 License

## Disclaimer

For analysis purposes only. Past performance does not guarantee future results.

---

**Version:** 2.1 | **Last Updated:** December 2025