"""
formatters.py
-------------
Utility functions for formatting numbers, dates, and CSS classes.
"""


def format_folder_name(folder_name):
    """Convert folder_name to Title Case with spaces"""
    return folder_name.replace("_", " ").title()


def format_currency(value, decimals=0):
    """Format value as currency with thousand separators"""
    return f"${value:,.{decimals}f}"


def format_percentage(value, decimals=1):
    """Format value as percentage"""
    return f"{value:.{decimals}f}%"


def get_pnl_class(value):
    """Return CSS class based on P&L value"""
    if value < 0:
        return "negative"
    elif value > 0:
        return "positive"
    else:
        return "neutral"


def format_date(date_obj):
    """Format date object to YYYY-MM-DD string"""
    import pandas as pd
    
    if date_obj is None:
        return None
    if isinstance(date_obj, str):
        return date_obj
    return pd.Timestamp(date_obj).strftime("%Y-%m-%d")
