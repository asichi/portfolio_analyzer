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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass
class CSVMetadata:
    """Metadata extracted from CSV headers."""

    strategy_name: str
    backtest_date: Optional[str] = None
    capital_allocation: Optional[float] = None
    risk_amount: Optional[float] = None
    max_position_value: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        return {key: value for key, value in self.__dict__.items() if value is not None}


class CSVFormatError(Exception):
    """Raised when CSV format cannot be detected or is invalid."""

    pass


def read_trade_csv(filepath: Path, **kwargs) -> pd.DataFrame:
    """
    Read trade CSVs while preserving valid ticker symbols such as "NA".

    pandas treats strings like "NA" as missing by default. That is wrong for
    stock symbols. Keep default NA parsing disabled and treat only blank fields
    as missing.
    """
    return pd.read_csv(
        filepath,
        keep_default_na=False,
        na_values=[""],
        **kwargs,
    )


def detect_csv_format(filepath: Path) -> str:
    """
    Detect which CSV format is being used.

    Returns:
        "strategyengine" - StrategyEngine backtest format
        "portfolioanalyzer" - Portfolio Analyzer format

    Raises:
        CSVFormatError - If format cannot be detected
    """
    try:
        df = read_trade_csv(filepath, nrows=20, comment="#")
        columns = set(df.columns)

        strategyengine_required = {"Symbol", "EntryDate", "ExitDate", "PnL", "Shares"}
        if strategyengine_required.issubset(columns):
            return "strategyengine"

        portfolioanalyzer_required = {
            "Symbol",
            "Date",
            "Ex. date",
            "Profit",
            "Position value",
        }
        if portfolioanalyzer_required.issubset(columns):
            return "portfolioanalyzer"

        raise CSVFormatError(
            f"Cannot detect CSV format for {filepath.name}\n"
            f"Found columns: {sorted(columns)}\n"
            f"Expected StrategyEngine: {sorted(strategyengine_required)}\n"
            f"Or PortfolioAnalyzer: {sorted(portfolioanalyzer_required)}"
        )

    except pd.errors.ParserError as e:
        raise CSVFormatError(f"Failed to read {filepath.name}: {str(e)}")
    except CSVFormatError:
        raise
    except Exception as e:
        raise CSVFormatError(f"Error detecting format for {filepath.name}: {str(e)}")


def parse_metadata_header(filepath: Path) -> CSVMetadata:
    """
    Parse metadata from CSV comment headers.

    Expected StrategyEngine format:
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
    metadata = {"strategy_name": filepath.stem}

    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()

                if line == "# === TRADES ===" or not line.startswith("#"):
                    break

                if ":" not in line:
                    continue

                key, value = line[1:].split(":", 1)
                key = key.strip()
                value = value.strip()

                if key == "StrategyName":
                    metadata["strategy_name"] = value
                elif key == "BacktestDate":
                    metadata["backtest_date"] = value
                elif key == "CapitalAllocation":
                    metadata["capital_allocation"] = _try_parse_float(value)
                elif key == "RiskAmount":
                    metadata["risk_amount"] = _try_parse_float(value)
                elif key == "MaxPositionValue":
                    metadata["max_position_value"] = _try_parse_float(value)

    except Exception as e:
        print(f"  ⚠️  Warning: Could not parse metadata from {filepath.name}: {str(e)}")

    return CSVMetadata(**metadata)


def normalize_strategyengine_csv(filepath: Path) -> Tuple[pd.DataFrame, CSVMetadata]:
    """
    Normalize StrategyEngine format CSV to standard schema.

    Input columns:
        Symbol, EntryDate, EntryPrice, ExitDate, ExitPrice, PnL, ExitReason, Shares

    Output columns:
        Strategy, Symbol, EntryDate, ExitDate, PnL, PositionValue
    """
    metadata = parse_metadata_header(filepath)
    df = read_trade_csv(filepath, comment="#")

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
        raise CSVFormatError(f"Missing columns in {filepath.name}: {sorted(missing)}")

    df["EntryDate"] = pd.to_datetime(df["EntryDate"])
    df["ExitDate"] = pd.to_datetime(df["ExitDate"])

    df["EntryPrice"] = _to_numeric_required(df, "EntryPrice", filepath)
    df["Shares"] = _to_numeric_required(df, "Shares", filepath)
    df["PnL"] = _to_numeric_required(df, "PnL", filepath)

    # Capital usage is absolute notional exposure.
    # This is important for short systems where Shares may be negative.
    if "PositionValue" in df.columns:
        df["PositionValue"] = _to_numeric_required(df, "PositionValue", filepath).abs()
    elif "Position value" in df.columns:
        df["PositionValue"] = _to_numeric_required(df, "Position value", filepath).abs()
    else:
        df["PositionValue"] = (df["EntryPrice"].abs() * df["Shares"].abs()).astype(float)

    normalized = df[["Symbol", "EntryDate", "ExitDate", "PnL", "PositionValue"]].copy()
    normalized["Strategy"] = metadata.strategy_name

    normalized = normalized[
        ["Strategy", "Symbol", "EntryDate", "ExitDate", "PnL", "PositionValue"]
    ]

    validate_normalized_csv(normalized)
    return normalized, metadata


def normalize_portfolioanalyzer_csv(filepath: Path) -> Tuple[pd.DataFrame, CSVMetadata]:
    """
    Normalize PortfolioAnalyzer format CSV to standard schema.

    Input columns:
        Symbol, Date, Ex. date, Profit, Position value

    Output columns:
        Strategy, Symbol, EntryDate, ExitDate, PnL, PositionValue
    """
    metadata = CSVMetadata(strategy_name=filepath.stem)
    df = read_trade_csv(filepath)

    required = {"Symbol", "Date", "Ex. date", "Profit", "Position value"}
    missing = required - set(df.columns)
    if missing:
        raise CSVFormatError(f"Missing columns in {filepath.name}: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Ex. date"] = pd.to_datetime(df["Ex. date"])
    df["Profit"] = _to_numeric_required(df, "Profit", filepath)
    df["Position value"] = _to_numeric_required(df, "Position value", filepath).abs()

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

    validate_normalized_csv(normalized)
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
        format_type = detect_csv_format(filepath)

        if format_type == "strategyengine":
            return normalize_strategyengine_csv(filepath)
        if format_type == "portfolioanalyzer":
            return normalize_portfolioanalyzer_csv(filepath)

        raise CSVFormatError(f"Unknown format type: {format_type}")

    except CSVFormatError:
        raise
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
        raise CSVFormatError(f"Normalized CSV missing columns: {sorted(missing)}")

    if len(df) == 0:
        raise CSVFormatError("CSV contains no trade data")

    key_columns = ["Strategy", "Symbol", "EntryDate", "ExitDate", "PnL", "PositionValue"]
    for col in key_columns:
        if df[col].isnull().any():
            raise CSVFormatError(f"Column '{col}' contains null values")

    if (df["ExitDate"] < df["EntryDate"]).any():
        raise CSVFormatError("CSV contains trades with ExitDate before EntryDate")

    if (df["PositionValue"] < 0).any():
        raise CSVFormatError("Column 'PositionValue' contains negative values")

    return True


def _to_numeric_required(df: pd.DataFrame, column: str, filepath: Path) -> pd.Series:
    """Convert a required numeric column and fail clearly if values are invalid."""
    values = pd.to_numeric(df[column], errors="coerce")

    if values.isnull().any():
        bad_count = int(values.isnull().sum())
        raise CSVFormatError(
            f"Column '{column}' in {filepath.name} contains {bad_count} non-numeric value(s)"
        )

    return values.astype(float)


def _try_parse_float(value: str) -> Optional[float]:
    """Parse a metadata float value. Return None when blank or invalid."""
    try:
        if value == "":
            return None
        return float(value)
    except ValueError:
        return None


if __name__ == "__main__":
    test_file = Path("data/backtests/AmsA1_20260304_0839.csv")

    if test_file.exists():
        try:
            df, metadata = load_and_normalize_csv(test_file)
            print(f"✓ Successfully loaded: {test_file.name}")
            print(f"  Rows: {len(df)}")
            print(f"  Metadata: {metadata.to_dict()}")
            print(f"\nNormalized columns: {list(df.columns)}")
            print(f"\nFirst row:\n{df.iloc[0]}")
        except CSVFormatError as e:
            print(f"✗ Error: {e}")
    else:
        print(f"Test file not found: {test_file}")
