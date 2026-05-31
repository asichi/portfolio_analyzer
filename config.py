# config.py

# ============================================================
# EXISTING SETTINGS (KEEP AS-IS)
# ============================================================

INITIAL_CAPITAL = 100000
DATASET_FOLDER = "datasets"
REPORTS_FOLDER = "reports"

# ============================================================
# NEW: Format detection patterns (ADD ONLY THIS)
# ============================================================

# StrategyEngine required columns
STRATEGYENGINE_REQUIRED = [
    "Symbol", "EntryDate", "EntryPrice", "ExitDate", "ExitPrice", "PnL", "ExitReason", "Shares"
]

# PortfolioAnalyzer required columns (ORIGINAL)
REQUIRED_COLUMNS = [
    "Symbol", "Date", "Ex. date", "Profit", "Position value"
]

# Column mapping for normalization
COLUMN_MAPPINGS = {
    'strategyengine': {
        'EntryDate': 'Date',
        'ExitDate': 'Ex. date',
        'PnL': 'Profit'
    },
    'portfolioanalyzer': {}  # No mapping needed
}