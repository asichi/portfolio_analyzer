# 📊 Portfolio Analyzer

Analyze and combine multiple trading strategy backtests with professional HTML reports featuring interactive charts, drawdown analysis, and capital usage tracking.

## Features

- **Multi-Strategy Analysis** - Combine strategies and analyze performance together
- **Interactive HTML Reports** - Plotly charts with equity curves and capital usage
- **Comprehensive Metrics** - Annual Profit, drawdowns, win rate, profit factor, recovery factors
- **Capital Usage Analysis** - Track deployment with percentile and tail risk metrics
- **Drawdown Attribution** - Identify which strategies contribute to portfolio drawdowns
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
├── report_html_generator.py
├── styles.css
├── datasets/
│   ├── swing_systems/
│   │   ├── strategy1.csv
│   │   └── strategy2.csv
│   └── day_trades/
│       └── strategy3.csv
└── reports/          # Auto-generated
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
# Analyze all folders
python main.py

# Analyze specific folder
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

- **Portfolio Summary** - Returns, Annual Profits, drawdowns, win rate, profit factor, recovery factors
- **Equity Curve** - Interactive chart with drawdown visualization
- **Capital Usage** - Daily deployment chart with percentile analysis and tail risk metrics
- **Strategy Breakdown** - Individual metrics for each strategy
- **Top Drawdowns** - Peak/trough dates, recovery times, strategy contributions
- **Monthly Returns** - Calendar heatmap with yearly totals

## Architecture

**`config.py`** - Configuration constants  
**`data_loader.py`** - CSV validation and ingestion  
**`metrics.py`** - Performance calculations, equity curves, capital usage  
**`report_html_generator.py`** - HTML report generation with Plotly charts  
**`main.py`** - CLI entry point and orchestration  
**`styles.css`** - Report styling  

## Troubleshooting

**"No subfolders found"** - Create subfolders inside `datasets/` for your CSV files

**"Missing required columns"** - Ensure CSV has all required columns (case-sensitive)

**Incorrect data** - Verify date formats (YYYY-MM-DD) and numeric Profit values

## License

GPL-3.0 License

## Disclaimer

For analysis purposes only. Past performance does not guarantee future results.

---

**Version:** 2.0 | **Last Updated:** December 2025
