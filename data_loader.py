"""
data_loader.py
Validate CSV format + normalize columns
"""

from csv_adapter import load_and_normalize_csv, detect_csv_format, CSVFormatError


def validate_csv_file(filepath):
    """
    Validate CSV format (auto-detects StrategyEngine or PortfolioAnalyzer).
    
    CHANGED: Now uses csv_adapter for format detection
    """
    try:
        detect_csv_format(filepath)
        return True  # Valid if format detected
    except CSVFormatError as e:
        raise ValueError(f"Failed to read {filepath.name}: {str(e)}")


def load_strategy_data(filepath):
    """
    Load and normalize strategy data from CSV (any supported format).
    
    CHANGED: Uses csv_adapter to handle both formats
    RETURNS: Normalized df with standard columns
    """
    try:
        # Load and normalize using adapter
        df, metadata = load_and_normalize_csv(filepath)
        
        # Rename columns to match original schema
        df = df.rename(columns={
            'EntryDate': 'Date',
            'ExitDate': 'Ex. date',
            'PnL': 'Profit',
            'PositionValue': 'Position value'
        })
        
        # Add Strategy column if missing
        if 'Strategy' not in df.columns:
            df['Strategy'] = filepath.stem.split('_')[0]
        
        return df
        
    except CSVFormatError as e:
        raise ValueError(f"Error loading {filepath.name}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading {filepath.name}: {str(e)}")