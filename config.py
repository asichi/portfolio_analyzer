# config.py

INITIAL_CAPITAL = 100000
DATASET_FOLDER = "datasets"
REPORTS_FOLDER = "reports"

# ============================================================
# CSV FORMAT DETECTION
# ============================================================

# StrategyEngine required columns
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

# PortfolioAnalyzer required columns
REQUIRED_COLUMNS = [
    "Symbol",
    "Date",
    "Ex. date",
    "Profit",
    "Position value",
]

# Column mapping for normalization
COLUMN_MAPPINGS = {
    "strategyengine": {
        "EntryDate": "Date",
        "ExitDate": "Ex. date",
        "PnL": "Profit",
    },
    "portfolioanalyzer": {},
}

# ============================================================
# SWING CAPITAL CAP WHAT-IF SETTINGS
# ============================================================

# Folder names treated as swing/overnight systems for capital-cap simulation.
# These names must match subfolder names under DATASET_FOLDER.
SWING_CAP_GROUPS = [
    "swing_systems",
    "weekly_systems",
]

# Cap sweep used by the what-if simulator.
# None means uncapped baseline.
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
