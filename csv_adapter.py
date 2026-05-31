"""
csv_adapter.py
--------------
Handles detection and normalization of multiple CSV formats.

Detects whether a CSV is in:
  - StrategyEngine format (Symbol, EntryDate, EntryPrice, ExitDate, ExitPrice, PnL, ExitReason, Shares)
  - PortfolioAnalyzer format (Symbol, Date, Ex. date, Profit, Position value)

Normalizes both to a standard internal schema with optional metadata.

Responsibilities:
- Auto-detect CSV format
- Parse CSV metadata headers (StrategyEngine only)
- Normalize column names and data types
- Calculate missing values (e.g., PositionValue from EntryPrice × Shares)
- Return standardized DataFrame + metadata dict
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CSVMetadata:
    """Metadata extracted from CSV headers"""

    strategy_name: str
    backtest_date: Optional[str] = None
    capital_allocation: Optional[float] = None
    risk_amount: Optional[float] = None
    max_position_value: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class CSVFormatError(Exception):
    """Raised when CSV format cannot be detected or is invalid"""

    pass


def detect_csv_format(filepath: Path) -> str:
    """
    Detect which CSV format is being used.

    Returns:
        'strategyengine' - StrategyEngine backtest format
        'portfolioanalyzer' - Portfolio Analyzer format

    Raises:
        CSVFormatError - If format cannot be detected
    """
    try:
        # Read first few rows to detect format
        df = pd.read_csv(filepath, nrows=20, comment="#")
        columns = set(df.columns)

        # StrategyEngine format check
        strategyengine_required = {"Symbol", "EntryDate", "ExitDate", "PnL", "Shares"}
        if strategyengine_required.issubset(columns):
            return "strategyengine"

        # PortfolioAnalyzer format check
        portfolioanalyzer_required = {
            "Symbol",
            "Date",
            "Ex. date",
            "Profit",
            "Position value",
        }
        if portfolioanalyzer_required.issubset(columns):
            return "portfolioanalyzer"

        # If neither matches, raise error
        raise CSVFormatError(
            f"Cannot detect CSV format for {filepath.name}\n"
            f"Found columns: {sorted(columns)}\n"
            f"Expected StrategyEngine: {sorted(strategyengine_required)}\n"
            f"Or PortfolioAnalyzer: {sorted(portfolioanalyzer_required)}"
        )

    except pd.errors.ParserError as e:
        raise CSVFormatError(f"Failed to read {filepath.name}: {str(e)}")
    except Exception as e:
        raise CSVFormatError(f"Error detecting format for {filepath.name}: {str(e)}")


def parse_metadata_header(filepath: Path) -> CSVMetadata:
    """
    Parse metadata from CSV comment headers (StrategyEngine format only).

    Expected format:
        # === BACKTEST METADATA ===
        # StrategyName: AmsA1
        # BacktestDate: 2026-03-04
        # CapitalAllocation: 150000
        # RiskAmount: 750
        # MaxPositionValue: 15000
        # === TRADES ===

    Returns:
        CSVMetadata object with extracted values
    """
    metadata = {"strategy_name": filepath.stem}  # Default: use filename

    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()

                # Stop at end of metadata
                if line == "# === TRADES ===" or not line.startswith("#"):
                    break

                # Parse metadata lines
                if ":" in line:
                    key, value = line[1:].split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    if key == "StrategyName":
                        metadata["strategy_name"] = value
                    elif key == "BacktestDate":
                        metadata["backtest_date"] = value
                    elif key == "CapitalAllocation":
                        try:
                            metadata["capital_allocation"] = float(value)
                        except ValueError:
                            pass
                    elif key == "RiskAmount":
                        try:
                            metadata["risk_amount"] = float(value)
                        except ValueError:
                            pass
                    elif key == "MaxPositionValue":
                        try:
                            metadata["max_position_value"] = float(value)
                        except ValueError:
                            pass

    except Exception as e:
        print(f"  ⚠️  Warning: Could not parse metadata from {filepath.name}: {str(e)}")

    return CSVMetadata(**metadata)


def normalize_strategyengine_csv(filepath: Path) -> Tuple[pd.DataFrame, CSVMetadata]:
    """
    Normalize StrategyEngine format CSV to standard schema.

    Input columns: Symbol, EntryDate, EntryPrice, ExitDate, ExitPrice, PnL, ExitReason, Shares
    Output columns: Symbol, EntryDate, ExitDate, PnL, PositionValue

    Returns:
        (normalized_df, metadata)
    """
    # Parse metadata from headers
    metadata = parse_metadata_header(filepath)

    # Read CSV (skip comment lines)
    df = pd.read_csv(filepath, comment="#")

    # Ensure required columns exist
    required = {
        "Symbol",
        "EntryDate",
        "EntryPrice",
        "ExitDate",
        "ExitPrice",
        "PnL",
        "Shares",
    }
    missing = required - set(df.columns)
    if missing:
        raise CSVFormatError(f"Missing columns in {filepath.name}: {missing}")

    # Convert dates
    df["EntryDate"] = pd.to_datetime(df["EntryDate"])
    df["ExitDate"] = pd.to_datetime(df["ExitDate"])

    # Calculate PositionValue if not present
    df["PositionValue"] = df["EntryPrice"] * df["Shares"]

    # Select and rename to standard schema
    normalized = df[["Symbol", "EntryDate", "ExitDate", "PnL", "PositionValue"]].copy()

    # Add strategy name
    normalized["Strategy"] = metadata.strategy_name

    # Reorder columns
    normalized = normalized[
        ["Strategy", "Symbol", "EntryDate", "ExitDate", "PnL", "PositionValue"]
    ]

    return normalized, metadata


def normalize_portfolioanalyzer_csv(filepath: Path) -> Tuple[pd.DataFrame, CSVMetadata]:
    """
    Normalize PortfolioAnalyzer format CSV to standard schema.

    Input columns: Symbol, Date, Ex. date, Profit, Position value
    Output columns: Symbol, EntryDate, ExitDate, PnL, PositionValue

    Returns:
        (normalized_df, metadata)
    """
    # Create minimal metadata (no header parsing in this format)
    metadata = CSVMetadata(strategy_name=filepath.stem)

    # Read CSV
    df = pd.read_csv(filepath)

    # Ensure required columns exist
    required = {"Symbol", "Date", "Ex. date", "Profit", "Position value"}
    missing = required - set(df.columns)
    if missing:
        raise CSVFormatError(f"Missing columns in {filepath.name}: {missing}")

    # Convert dates
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ex. date"] = pd.to_datetime(df["Ex. date"])

    # Normalize to standard schema
    normalized = pd.DataFrame(
        {
            "Strategy": metadata.strategy_name,
            "Symbol": df["Symbol"],
            "EntryDate": df["Date"],
            "ExitDate": df["Ex. date"],
            "PnL": df["Profit"],
            "PositionValue": df["Position value"],
        }
    )

    return normalized, metadata


def load_and_normalize_csv(filepath: Path) -> Tuple[pd.DataFrame, CSVMetadata]:
    """
    Auto-detect CSV format and normalize to standard schema.

    Args:
        filepath: Path to CSV file

    Returns:
        (normalized_dataframe, metadata)

    Raises:
        CSVFormatError: If format cannot be detected or normalized
    """
    if not filepath.exists():
        raise CSVFormatError(f"File not found: {filepath}")

    try:
        # Detect format
        format_type = detect_csv_format(filepath)

        if format_type == "strategyengine":
            return normalize_strategyengine_csv(filepath)
        elif format_type == "portfolioanalyzer":
            return normalize_portfolioanalyzer_csv(filepath)
        else:
            raise CSVFormatError(f"Unknown format type: {format_type}")

    except CSVFormatError:
        raise  # Re-raise our custom exceptions
    except Exception as e:
        raise CSVFormatError(f"Error normalizing {filepath.name}: {str(e)}")


def validate_normalized_csv(df: pd.DataFrame) -> bool:
    """
    Validate that a normalized CSV has all required columns.

    Returns:
        True if valid, raises CSVFormatError if invalid
    """
    required = {"Strategy", "Symbol", "EntryDate", "ExitDate", "PnL", "PositionValue"}
    missing = required - set(df.columns)

    if missing:
        raise CSVFormatError(f"Normalized CSV missing columns: {missing}")

    # Check for empty DataFrame
    if len(df) == 0:
        raise CSVFormatError("CSV contains no trade data")

    # Check for null values in key columns
    key_columns = ["Symbol", "EntryDate", "ExitDate", "PnL"]
    for col in key_columns:
        if df[col].isnull().any():
            raise CSVFormatError(f"Column '{col}' contains null values")

    return True


# Example usage
if __name__ == "__main__":
    # Test the adapter
    from pathlib import Path

    test_file = Path("data/backtests/AmsA1_20260304_0839.csv")

    if test_file.exists():
        try:
            df, metadata = load_and_normalize_csv(test_file)
            print(f"✓ Successfully loaded: {test_file.name}")
            print(f"  Format detected: strategyengine")
            print(f"  Rows: {len(df)}")
            print(f"  Metadata: {metadata.to_dict()}")
            print(f"\nNormalized columns: {list(df.columns)}")
            print(f"\nFirst row:\n{df.iloc[0]}")
        except CSVFormatError as e:
            print(f"✗ Error: {e}")
    else:
        print(f"Test file not found: {test_file}")
