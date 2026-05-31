"""
data_loader.py
Validate CSV format + normalize columns
"""

from csv_adapter import load_and_normalize_csv, detect_csv_format, CSVFormatError


def validate_csv_file(filepath):
    """
    Validate CSV format.

    Auto-detects StrategyEngine or PortfolioAnalyzer format.
    """
    try:
        detect_csv_format(filepath)
        return True
    except CSVFormatError as e:
        raise ValueError(f"Failed to read {filepath.name}: {str(e)}")


def load_strategy_data(filepath, strategy_group=None):
    """
    Load and normalize strategy data from CSV.

    Args:
        filepath:
            CSV file path.

        strategy_group:
            Dataset subfolder name, such as swing_systems, weekly_systems,
            day_trades, or gappers. Used by portfolio what-if simulations
            to apply rules to specific strategy groups.

    Returns:
        DataFrame using PortfolioAnalyzer's standard column names:
        Strategy, Symbol, Date, Ex. date, Profit, Position value, StrategyGroup
    """
    try:
        df, metadata = load_and_normalize_csv(filepath)

        # Rename normalized adapter columns to match PortfolioAnalyzer schema.
        df = df.rename(
            columns={
                "EntryDate": "Date",
                "ExitDate": "Ex. date",
                "PnL": "Profit",
                "PositionValue": "Position value",
            }
        )

        # Add Strategy column if missing.
        if "Strategy" not in df.columns:
            df["Strategy"] = filepath.stem.split("_")[0]

        # Tag each trade with the dataset folder it came from.
        # This enables Option C: configurable group-based swing capital caps.
        df["StrategyGroup"] = strategy_group or "unknown"

        return df

    except CSVFormatError as e:
        raise ValueError(f"Error loading {filepath.name}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading {filepath.name}: {str(e)}")
