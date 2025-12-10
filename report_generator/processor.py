"""
processor.py
------------
Main processing logic for folders and combined strategies.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import INITIAL_CAPITAL, REPORTS_FOLDER
from data_loader import load_strategy_data, validate_csv_file
from metrics import (
    calculate_metrics,
    calculate_daily_capital_usage,
    calculate_strategy_contributions,
    find_strategy_drawdowns,
    find_top_drawdowns,
)
from .html_builder import generate_html_report
from .formatters import format_folder_name


def process_folder(folder_path, folder_name):
    """Process a single strategy folder and generate report"""

    # Find all CSV files in this folder
    csv_files = list(folder_path.glob("*.csv"))

    if len(csv_files) == 0:
        print(f"  ⚠️  No CSV files found - skipping\n")
        return None

    print(f"  📁 Found {len(csv_files)} strategy file(s)")

    # Validate and load all strategy files
    all_trades = []
    strategy_names = []

    for csv_file in csv_files:
        try:
            print(f"  ⏳ Loading {csv_file.name}...", end=" ")
            validate_csv_file(csv_file)
            df = load_strategy_data(csv_file)
            all_trades.append(df)
            strategy_names.append(csv_file.stem)
            print(f"✓ ({len(df)} trades)")
        except Exception as e:
            print(f"\n  ❌ ERROR: {str(e)}")
            return None

    # Combine all trades
    combined_df = pd.concat(all_trades, ignore_index=True)
    combined_df = combined_df.sort_values("Ex. date").reset_index(drop=True)

    # Build combined equity curve
    combined_equity = combined_df.groupby("Ex. date")["Profit"].sum().reset_index()
    combined_equity["Cum_Profit"] = combined_equity["Profit"].cumsum()
    combined_equity["Equity"] = INITIAL_CAPITAL + combined_equity["Cum_Profit"]
    combined_equity = combined_equity.rename(columns={"Ex. date": "Exit_Date"})

    # Find top 3 drawdowns
    top_drawdowns = find_top_drawdowns(combined_equity, n=3)

    # Calculate strategy contributions for each drawdown
    drawdown_contributions = calculate_strategy_contributions(
        combined_df, top_drawdowns
    )

    # Calculate metrics for each strategy
    strategy_metrics = []
    for strategy in strategy_names:
        metrics = calculate_metrics(combined_df, strategy)
        strategy_metrics.append(metrics)

    # Calculate top drawdowns for each individual strategy
    strategy_drawdowns = {}
    for strategy in strategy_names:
        strategy_drawdowns[strategy] = find_strategy_drawdowns(
            combined_df, strategy, n=3
        )

    # Calculate combined metrics
    combined_metrics = calculate_metrics(
        combined_df, equity_curve=combined_equity["Equity"].values
    )
    strategy_metrics.append(combined_metrics)

    metrics_df = pd.DataFrame(strategy_metrics)

    # Compute daily capital usage
    # Combine list of DataFrames into one
    if isinstance(all_trades, list):
        all_trades = pd.concat(all_trades, ignore_index=True)

    # Normalize column names to avoid case/space mismatches
    all_trades.columns = all_trades.columns.str.strip().str.lower()

    # Now safe to call with lowercase
    usage, usage_stats, extra = calculate_daily_capital_usage(
        all_trades,
        sizing_col="position value",  # lowercase after normalization
        entry_col="date",  # lowercase after normalization
        exit_col="ex. date",  # lowercase after normalization
        inclusive_exit=True,
    )

    # Calculate monthly returns
    combined_equity["Year"] = combined_equity["Exit_Date"].dt.year
    combined_equity["Month"] = combined_equity["Exit_Date"].dt.month

    monthly_returns = (
        combined_equity.groupby(["Year", "Month"])["Profit"].sum().reset_index()
    )
    monthly_returns["Return %"] = (monthly_returns["Profit"] / INITIAL_CAPITAL) * 100

    # Create pivot table
    monthly_pivot = monthly_returns.pivot(
        index="Year", columns="Month", values="Return %"
    )

    # Ensure all 12 months are present
    for month in range(1, 13):
        if month not in monthly_pivot.columns:
            monthly_pivot[month] = np.nan

    monthly_pivot = monthly_pivot.sort_index(axis=1)
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    monthly_pivot.columns = month_names

    # Generate HTML report
    formatted_name = format_folder_name(folder_name)
    html_content = generate_html_report(
        all_trades,  # Pass normalized trades (with lowercase columns)
        combined_equity,
        metrics_df,
        combined_metrics,
        monthly_pivot,
        top_drawdowns,
        formatted_name,
        drawdown_contributions,
        strategy_drawdowns,
        usage_stats,
        usage,
    )

    # Create reports folder if needed
    reports_path = Path(REPORTS_FOLDER)
    reports_path.mkdir(exist_ok=True)

    # Save report with folder name
    report_filename = f"{folder_name}.html"
    report_path = reports_path / report_filename

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Print summary
    print(
        f"  📊 Combined: ${combined_metrics['Total Return $']:,.0f} ({combined_metrics['Total Return %']:.1f}%) | "
        f"CAGR: {combined_metrics['CAGR %']:.1f}% | {combined_metrics['Total Trades']:.0f} trades"
    )
    print(f"  ✓ Report: {report_filename}\n")

    return {
        "folder": folder_name,
        "strategies": len(strategy_names),
        "trades": combined_metrics["Total Trades"],
        "return": combined_metrics["Total Return $"],
        "report": report_filename,
    }


def process_all_strategies(dataset_path):
    """
    Collect all unique CSV files across all folders and generate a combined report.
    Returns summary dict or None if no files found.
    """
    print("\n" + "=" * 60)
    print("GENERATING COMBINED 'ALL STRATEGIES' REPORT")
    print("=" * 60)

    # Collect all CSV files from all subdirectories
    all_csv_files = []
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            all_csv_files.extend(list(folder.glob("*.csv")))

    if len(all_csv_files) == 0:
        print("  ⚠️  No CSV files found across all folders\n")
        return None

    # Remove duplicates by filename (keep first occurrence)
    unique_files = {}
    for csv_file in all_csv_files:
        strategy_name = csv_file.stem
        if strategy_name not in unique_files:
            unique_files[strategy_name] = csv_file

    print(f"  📁 Found {len(unique_files)} unique strategy file(s) across all folders")

    # Load all strategies
    all_trades = []
    strategy_names = []

    for strategy_name, csv_file in sorted(unique_files.items()):
        try:
            print(f"  ⏳ Loading {csv_file.name}...", end=" ")
            validate_csv_file(csv_file)
            df = load_strategy_data(csv_file)
            all_trades.append(df)
            strategy_names.append(strategy_name)
            print(f"✓ ({len(df)} trades)")
        except Exception as e:
            print(f"\n  ❌ ERROR: {str(e)}")
            continue

    if len(all_trades) == 0:
        print("  ⚠️  No valid strategy files loaded\n")
        return None

    # Combine all trades
    combined_df = pd.concat(all_trades, ignore_index=True)
    combined_df = combined_df.sort_values("Ex. date").reset_index(drop=True)

    # Build combined equity curve
    combined_equity = combined_df.groupby("Ex. date")["Profit"].sum().reset_index()
    combined_equity["Cum_Profit"] = combined_equity["Profit"].cumsum()
    combined_equity["Equity"] = INITIAL_CAPITAL + combined_equity["Cum_Profit"]
    combined_equity = combined_equity.rename(columns={"Ex. date": "Exit_Date"})

    # Find top 3 drawdowns
    top_drawdowns = find_top_drawdowns(combined_equity, n=3)

    # Calculate strategy contributions for each drawdown
    drawdown_contributions = calculate_strategy_contributions(
        combined_df, top_drawdowns
    )

    # Calculate metrics for each strategy
    strategy_metrics = []
    for strategy in strategy_names:
        metrics = calculate_metrics(combined_df, strategy)
        strategy_metrics.append(metrics)

    # Calculate top drawdowns for each individual strategy
    strategy_drawdowns = {}
    for strategy in strategy_names:
        strategy_drawdowns[strategy] = find_strategy_drawdowns(
            combined_df, strategy, n=3
        )

    # Calculate combined metrics
    combined_metrics = calculate_metrics(
        combined_df, equity_curve=combined_equity["Equity"].values
    )
    strategy_metrics.append(combined_metrics)

    metrics_df = pd.DataFrame(strategy_metrics)

    # Compute daily capital usage
    if isinstance(all_trades, list):
        all_trades_df = pd.concat(all_trades, ignore_index=True)
    else:
        all_trades_df = all_trades

    # Normalize column names
    all_trades_df.columns = all_trades_df.columns.str.strip().str.lower()

    usage, usage_stats, extra = calculate_daily_capital_usage(
        all_trades_df,
        sizing_col="position value",
        entry_col="date",
        exit_col="ex. date",
        inclusive_exit=True,
    )

    # Calculate monthly returns
    combined_equity["Year"] = combined_equity["Exit_Date"].dt.year
    combined_equity["Month"] = combined_equity["Exit_Date"].dt.month

    monthly_returns = (
        combined_equity.groupby(["Year", "Month"])["Profit"].sum().reset_index()
    )
    monthly_returns["Return %"] = (monthly_returns["Profit"] / INITIAL_CAPITAL) * 100

    # Create pivot table
    monthly_pivot = monthly_returns.pivot(
        index="Year", columns="Month", values="Return %"
    )

    # Ensure all 12 months are present
    for month in range(1, 13):
        if month not in monthly_pivot.columns:
            monthly_pivot[month] = np.nan

    monthly_pivot = monthly_pivot.sort_index(axis=1)
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    monthly_pivot.columns = month_names

    # Generate HTML report
    html_content = generate_html_report(
        all_trades_df,  # Pass normalized trades (with lowercase columns)
        combined_equity,
        metrics_df,
        combined_metrics,
        monthly_pivot,
        top_drawdowns,
        "All Strategies Combined",
        drawdown_contributions,
        strategy_drawdowns,
        usage_stats,
        usage,
    )

    # Save report
    reports_path = Path(REPORTS_FOLDER)
    reports_path.mkdir(exist_ok=True)

    report_filename = "all_strategies.html"
    report_path = reports_path / report_filename

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Print summary
    print(
        f"  📊 Combined: ${combined_metrics['Total Return $']:,.0f} ({combined_metrics['Total Return %']:.1f}%) | "
        f"CAGR: {combined_metrics['CAGR %']:.1f}% | {combined_metrics['Total Trades']:.0f} trades"
    )
    print(f"  ✓ Report: {report_filename}\n")

    return {
        "folder": "all_strategies",
        "strategies": len(strategy_names),
        "trades": combined_metrics["Total Trades"],
        "return": combined_metrics["Total Return $"],
        "report": report_filename,
    }
