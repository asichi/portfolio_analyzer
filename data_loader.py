"""
data_loader.py
---------------
Handles CSV validation and data ingestion for trading strategies.

Responsibilities:
- Validate that input files contain required columns
- Load and clean strategy trade data into pandas DataFrames
- Provide standardized data structures for downstream metrics calculations
"""



import pandas as pd

from config import REQUIRED_COLUMNS


def validate_csv_file(filepath):
    """Validate that CSV has required columns"""
    try:
        df = pd.read_csv(filepath, nrows=1)
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{filepath.name} missing required columns: {missing_cols}"
            )
        return True
    except Exception as e:
        raise ValueError(f"Failed to read {filepath.name}: {str(e)}")


def load_strategy_data(filepath):
    """Load and clean strategy data from CSV"""
    try:
        # Read only needed columns
        df = pd.read_csv(filepath, usecols=REQUIRED_COLUMNS)

        # Convert dates
        df["Date"] = pd.to_datetime(df["Date"])
        df["Ex. date"] = pd.to_datetime(df["Ex. date"])

        # Extract strategy name from filename (e.g., AMS_7.csv -> AMS_7)
        strategy_name = filepath.stem
        df["Strategy"] = strategy_name

        # Keep only relevant columns
        df = df[["Strategy", "Symbol", "Date", "Ex. date", "Profit", "Position value"]]

        return df

    except Exception as e:
        raise ValueError(f"Error loading {filepath.name}: {str(e)}")
