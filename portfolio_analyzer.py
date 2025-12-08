"""
Portfolio Analyzer - Trading Strategy Combination Tool
Author: Anthony's Trading System
Date: 2025-11-14

Combines multiple trading strategy CSV files and generates comprehensive performance analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import argparse

# Configuration
INITIAL_CAPITAL = 100000
DATASET_FOLDER = "datasets"
REPORTS_FOLDER = "reports"

# Required columns in CSV files
REQUIRED_COLUMNS = ["Symbol", "Date", "Ex. date", "Profit"]


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
        df = df[["Strategy", "Symbol", "Date", "Ex. date", "Profit"]]

        return df

    except Exception as e:
        raise ValueError(f"Error loading {filepath.name}: {str(e)}")


def calculate_metrics(trades_df, strategy_name=None, equity_curve=None):
    """Calculate performance metrics for a strategy or combined portfolio"""

    if strategy_name:
        trades = trades_df[trades_df["Strategy"] == strategy_name].copy()
    else:
        trades = trades_df.copy()

    if len(trades) == 0:
        return None

    # Basic stats
    total_profit = trades["Profit"].sum()
    n_trades = len(trades)

    # Win/Loss analysis
    wins = trades[trades["Profit"] > 0]
    losses = trades[trades["Profit"] <= 0]

    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades if n_trades > 0 else 0

    avg_win = wins["Profit"].mean() if n_wins > 0 else 0
    avg_loss = abs(losses["Profit"].mean()) if n_losses > 0 else 0

    # Expected Value per Trade: (Win% × Avg Win) - (Loss% × Avg Loss)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Profit Factor
    gross_profit = wins["Profit"].sum() if n_wins > 0 else 0
    gross_loss = abs(losses["Profit"].sum()) if n_losses > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

    # Build equity curve
    if equity_curve is None:
        equity_df = build_equity_curve(trades_df, strategy_name)
        equity = equity_df["Equity"].values
    else:
        equity = equity_curve
    # Max Drawdown calculation
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = drawdown.min()
    max_dd_idx = np.argmin(drawdown)

    # Calculate drawdown as % of deployed capital ($100k), not peak equity
    max_dd_pct = (max_dd / INITIAL_CAPITAL) * 100

    # Recovery Time (calendar days from max drawdown to recovery)
    recovery_days = np.nan
    if max_dd_idx < len(equity) - 1:
        recovery_idx = np.where(equity[max_dd_idx:] >= peak[max_dd_idx])[0]
        if len(recovery_idx) > 0:
            trough_date = trades["Ex. date"].iloc[max_dd_idx]
            recovery_date = trades["Ex. date"].iloc[max_dd_idx + recovery_idx[0]]
            recovery_days = (recovery_date - trough_date).days

    # CAGR calculation
    start_date = trades["Ex. date"].min()
    end_date = trades["Ex. date"].max()
    days = (end_date - start_date).days
    years = days / 365.25
    final_value = INITIAL_CAPITAL + total_profit
    cagr = (
        (((final_value / INITIAL_CAPITAL) ** (1 / years)) - 1) * 100 if years > 0 else 0
    )

    # Recovery Factor (total profit / max drawdown)
    recovery_factor = (total_profit / abs(max_dd)) if max_dd != 0 else np.nan

    # Annualized Recovery Factor (normalize by years)
    recovery_factor_per_year = (recovery_factor / years) if years > 0 else np.nan

    return {
        "Strategy": strategy_name if strategy_name else "COMBINED",
        "Total Return $": total_profit,
        "Total Return %": (total_profit / INITIAL_CAPITAL) * 100,
        "CAGR %": cagr,
        "Max Drawdown $": max_dd,
        "Drawdown % Peak": (abs(max_dd) / peak[max_dd_idx]) * 100,
        "Drawdown % Initial": (abs(max_dd) / INITIAL_CAPITAL) * 100,
        "Recovery Days": recovery_days,
        "Win Rate %": win_rate * 100,
        "Avg Win $": avg_win,
        "Avg Loss $": avg_loss,
        "Profit Factor": profit_factor,
        "Expectancy $": expectancy,
        "Total Trades": n_trades,
        "Winning Trades": n_wins,
        "Losing Trades": n_losses,
        "Start Date": start_date.strftime("%Y-%m-%d"),
        "End Date": end_date.strftime("%Y-%m-%d"),
        "Days": days,
        "Recovery Factor": recovery_factor,
        "Recovery Factor / Year": recovery_factor_per_year,
    }


def build_equity_curve(trades_df, strategy_name=None, initial_capital=INITIAL_CAPITAL):
    """Return equity curve DataFrame for a strategy or combined portfolio."""
    if strategy_name:
        trades = trades_df[trades_df["Strategy"] == strategy_name].copy()
    else:
        trades = trades_df.copy()

    trades = trades.sort_values("Ex. date").reset_index(drop=True)
    trades["Cum_Profit"] = trades["Profit"].cumsum()
    equity_curve = pd.DataFrame(
        {
            "Exit_Date": trades["Ex. date"],
            "Equity": initial_capital + trades["Cum_Profit"],
        }
    )
    return equity_curve


def find_top_drawdowns(equity_curve_df, n=3):
    """Find the top N drawdowns and their recovery information"""

    equity = equity_curve_df["Equity"].values
    dates = equity_curve_df["Exit_Date"].values

    # Calculate running peak and drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak

    drawdowns = []
    in_drawdown = False
    dd_start_idx = 0
    dd_peak_value = 0

    for i in range(len(equity)):
        # Start of drawdown (at peak)
        if not in_drawdown and (i == 0 or equity[i] == peak[i]):
            if i < len(equity) - 1 and equity[i + 1] < equity[i]:
                in_drawdown = True
                dd_start_idx = i
                dd_peak_value = equity[i]

        # During drawdown
        elif in_drawdown:
            if equity[i] >= dd_peak_value:
                # Drawdown ended
                dd_segment = equity[dd_start_idx:i]
                dd_trough_idx = dd_start_idx + np.argmin(dd_segment)
                dd_trough_value = equity[dd_trough_idx]

                dd_amount = dd_peak_value - dd_trough_value
                dd_pct_peak = (dd_amount / dd_peak_value) * 100  # standard convention
                dd_pct_initial = (
                    dd_amount / INITIAL_CAPITAL
                ) * 100  # your current style

                days_to_trough = dd_trough_idx - dd_start_idx
                days_to_recovery = i - dd_trough_idx
                total_days = i - dd_start_idx

                drawdowns.append(
                    {
                        "Peak Date": dates[dd_start_idx],
                        "Trough Date": dates[dd_trough_idx],
                        "Recovery Date": dates[i],
                        "Peak Value": dd_peak_value,
                        "Trough Value": dd_trough_value,
                        "Drawdown $": dd_amount,
                        "Drawdown % Peak": dd_pct_peak,
                        "Drawdown % Initial": dd_pct_initial,
                        "Days to Trough": days_to_trough,
                        "Days to Recovery": days_to_recovery,
                        "Total Days": total_days,
                    }
                )

                in_drawdown = False

    # Handle ongoing drawdown at the end
    if in_drawdown:
        dd_segment = equity[dd_start_idx:]
        dd_trough_idx = dd_start_idx + np.argmin(dd_segment)
        dd_trough_value = equity[dd_trough_idx]

        dd_amount = dd_peak_value - dd_trough_value
        dd_pct_peak = (dd_amount / dd_peak_value) * 100
        dd_pct_initial = (dd_amount / INITIAL_CAPITAL) * 100

        days_to_trough = dd_trough_idx - dd_start_idx

        drawdowns.append(
            {
                "Peak Date": dates[dd_start_idx],
                "Trough Date": dates[dd_trough_idx],
                "Recovery Date": None,
                "Peak Value": dd_peak_value,
                "Trough Value": dd_trough_value,
                "Drawdown $": dd_amount,
                "Drawdown % Peak": dd_pct_peak,
                "Drawdown % Initial": dd_pct_initial,
                "Days to Trough": days_to_trough,
                "Days to Recovery": None,
                "Total Days": None,
            }
        )

    # Sort by magnitude and take top N
    drawdowns_sorted = sorted(drawdowns, key=lambda x: x["Drawdown $"], reverse=True)
    return drawdowns_sorted[:n]


def find_strategy_drawdowns(all_trades, strategy_name, n=3):
    """Find top N drawdowns for a specific strategy"""

    # Build equity curve using the shared helper
    equity_curve = build_equity_curve(all_trades, strategy_name)

    # Use existing find_top_drawdowns function
    return find_top_drawdowns(equity_curve, n=n)


def calculate_strategy_contributions(all_trades, drawdown_periods):
    """Calculate how much each strategy contributed to each drawdown period"""

    contributions = []

    for dd in drawdown_periods:
        peak_date = pd.Timestamp(dd["Peak Date"])
        trough_date = pd.Timestamp(dd["Trough Date"])

        # Get all trades that exited during this drawdown period
        period_trades = all_trades[
            (all_trades["Ex. date"] >= peak_date)
            & (all_trades["Ex. date"] <= trough_date)
        ]

        # Calculate loss per strategy during this period
        strategy_breakdown = []
        total_loss = dd["Drawdown $"]  # Negative value

        for strategy in sorted(all_trades["Strategy"].unique()):
            strategy_trades = period_trades[period_trades["Strategy"] == strategy]
            strategy_loss = strategy_trades["Profit"].sum()
            strategy_loss_pct = (strategy_loss / INITIAL_CAPITAL) * 100
            contribution_pct = (
                (strategy_loss / total_loss * 100) if total_loss != 0 else 0
            )

            strategy_breakdown.append(
                {
                    "Strategy": strategy,
                    "Loss $": strategy_loss,
                    "Loss %": strategy_loss_pct,
                    "Contribution %": contribution_pct,
                }
            )

        contributions.append({"drawdown": dd, "strategies": strategy_breakdown})

    return contributions


def generate_html_report(
    all_trades,
    combined_equity,
    metrics_df,
    combined_metrics,
    monthly_returns_table,
    top_drawdowns,
    folder_name,
    drawdown_contributions,
    strategy_drawdowns,
):
    """Generate beautiful HTML report"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build sections
    header = _build_html_header(timestamp)
    summary = _build_summary_section(metrics_df, combined_metrics, folder_name)
    equity_chart = _build_equity_chart_section()
    strategy_table = _build_strategy_breakdown(metrics_df)
    top_dd_table = _build_top_drawdowns_table(top_drawdowns)
    dd_breakdown = _build_drawdown_breakdown(drawdown_contributions)
    individual_dd = _build_individual_strategy_drawdowns(strategy_drawdowns)
    monthly_table = _build_monthly_returns_table(monthly_returns_table)
    footer = _build_html_footer(timestamp)
    chart_script = _build_plotly_script(combined_equity, combined_metrics)

    # Combine all sections
    return (
        header
        + summary
        + equity_chart
        + strategy_table
        + top_dd_table
        + dd_breakdown
        + individual_dd
        + monthly_table
        + footer
        + chart_script
    )


def _build_html_header(timestamp):
    """Build HTML head and CSS styles"""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Analysis Report - {timestamp}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        {_get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
"""


def _get_css_styles():
    """Return CSS stylesheet as string"""
    return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
        }
        .summary-box {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 30px;
            font-size: 14px;
            color: #2c3e50;
        }
        .summary-box strong {
            color: #2980b9;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-label {
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 14px;
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .positive {
            color: #27ae60;
            font-weight: 600;
        }
        .negative {
            color: #e74c3c;
            font-weight: 600;
        }
        .dd-summary {
            margin-top: 10px;
            margin-bottom: 20px;
            font-size: 0.95em;
            line-height: 1.4;
            background-color: #f9f9f9;
            padding: 10px 15px;
            border-left: 4px solid #3498db;
            border-radius: 6px;
        }
        .dd-summary p {
            margin: 5px 0;
        }
        .dd-summary .worsened {
            color: #e74c3c;   /* red */
            font-weight: 600;
        }
        .dd-summary .offset {
            color: #27ae60;   /* green */
            font-weight: 600;
        }
        .chart-container {
            margin: 30px 0;
        }
        .timestamp {
            text-align: right;
            color: #7f8c8d;
            font-size: 12px;
            margin-top: 20px;
        }
"""


def _build_summary_section(metrics_df, combined_metrics, folder_name):
    """Build portfolio summary section with key metrics"""
    total_strategies = len(metrics_df) - 1
    total_trades = int(combined_metrics["Total Trades"])
    date_range = f"{combined_metrics['Start Date']} to {combined_metrics['End Date']}"
    days_covered = int(combined_metrics["Days"])

    html = f"""
        <h1>📊 {folder_name} Analysis Report</h1>
        
        <div class="summary-box">
            <strong>Analysis Summary:</strong> Combined {total_strategies} strategies | 
            <strong>{total_trades:,}</strong> total trades | 
            <strong>Period:</strong> {date_range} ({days_covered:,} days)
        </div>
        
        <h2>Portfolio Summary</h2>
        <div class="metrics-grid">
"""

    # Add metric cards
    key_metrics = [
        (
            "Total Return",
            f"${combined_metrics['Total Return $']:,.0f}",
            f"{combined_metrics['Total Return %']:.1f}%",
        ),
        ("CAGR", f"{combined_metrics['CAGR %']:.1f}%", ""),
        (
            "Max Drawdown",
            f"${combined_metrics['Max Drawdown $']:,.0f}",
            f"{combined_metrics['Drawdown % Peak']:.1f}% of peak / {combined_metrics['Drawdown % Initial']:.1f}% of initial",
        ),
        (
            "Win Rate",
            f"{combined_metrics['Win Rate %']:.1f}%",
            f"{combined_metrics['Winning Trades']:.0f}W / {combined_metrics['Losing Trades']:.0f}L",
        ),
        ("Profit Factor", f"{combined_metrics['Profit Factor']:.2f}", ""),
        ("Expectancy/Trade", f"${combined_metrics['Expectancy $']:.2f}", ""),
        (
            "Recovery Factor",
            f"{combined_metrics['Recovery Factor']:.2f}",
            f"{combined_metrics['Recovery Factor / Year']:.2f} per year",
        ),
    ]

    for label, value, subtitle in key_metrics:
        html += _build_metric_card(label, value, subtitle)

    html += """
        </div>
"""
    return html


def _build_metric_card(label, value, subtitle=""):
    """Build a single metric card"""
    subtitle_html = f'<div class="metric-label">{subtitle}</div>' if subtitle else ""
    return f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {subtitle_html}
            </div>
"""


def _build_equity_chart_section():
    """Build equity curve chart placeholder"""
    return """
        <h2>Combined Equity Curve</h2>
        <div id="equity_curve" class="chart-container"></div>
"""


def _build_strategy_breakdown(metrics_df):
    """Build strategy breakdown table"""
    html = """
        <h2>Strategy Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Total Return</th>
                    <th>CAGR %</th>
                    <th>Max DD</th>
                    <th>Win Rate %</th>
                    <th>Profit Factor</th>
                    <th>Expectancy</th>
                    <th>Trades</th>
                    <th>Recovery Factor</th>
                    <th>Recovery Factor / Year</th>
                </tr>
            </thead>
            <tbody>
"""

    for _, row in metrics_df.iterrows():
        html += _build_strategy_row(row)

    html += """
            </tbody>
        </table>
"""
    return html


def _build_strategy_row(row):
    """Build single strategy table row"""
    return_class = "positive" if row["Total Return $"] > 0 else "negative"
    return f"""
                <tr>
                    <td><strong>{row['Strategy']}</strong></td>
                    <td class="{return_class}">${row['Total Return $']:,.0f} ({row['Total Return %']:.1f}%)</td>
                    <td>{row['CAGR %']:.1f}%</td>
                    <td class="negative">
                        ${row['Max Drawdown $']:,.0f} 
                        ({row['Drawdown % Peak']:.1f}% of peak / {row['Drawdown % Initial']:.1f}% of initial)
                    </td>
                    <td>{row['Win Rate %']:.1f}%</td>
                    <td>{row['Profit Factor']:.2f}</td>
                    <td>${row['Expectancy $']:.2f}</td>
                    <td>{row['Total Trades']:.0f}</td>
                    <td>{row['Recovery Factor']:.2f}</td>
                    <td>{row['Recovery Factor / Year']:.2f}</td>
                </tr>
    """


def _build_top_drawdowns_table(top_drawdowns):
    """Build top 3 drawdowns table"""
    html = """
        <h2>Top 3 Drawdown Periods</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Peak Date</th>
                    <th>Trough Date</th>
                    <th>Recovery Date</th>
                    <th>Drawdown</th>
                    <th>Days to Trough</th>
                    <th>Days to Recover</th>
                    <th>Total Duration</th>
                </tr>
            </thead>
            <tbody>
"""

    for rank, dd in enumerate(top_drawdowns, 1):
        html += _build_drawdown_row(rank, dd)

    html += """
            </tbody>
        </table>
"""
    return html


def _build_drawdown_row(rank, dd):
    """Build single drawdown table row with both % conventions"""
    peak_date_str = pd.Timestamp(dd["Peak Date"]).strftime("%Y-%m-%d")
    trough_date_str = pd.Timestamp(dd["Trough Date"]).strftime("%Y-%m-%d")
    recovery_date = (
        pd.Timestamp(dd["Recovery Date"]).strftime("%Y-%m-%d")
        if dd["Recovery Date"] is not None
        else '<span style="color: #e67e22;">Ongoing</span>'
    )
    days_to_recovery = (
        f"{dd['Days to Recovery']:.0f}" if dd["Days to Recovery"] is not None else "—"
    )
    total_days = f"{dd['Total Days']:.0f}" if dd["Total Days"] is not None else "—"

    return f"""
                <tr>
                    <td><strong>#{rank}</strong></td>
                    <td>{peak_date_str}</td>
                    <td>{trough_date_str}</td>
                    <td>{recovery_date}</td>
                    <td class="negative">
                        ${dd['Drawdown $']:,.0f} 
                        ({dd['Drawdown % Peak']:.1f}% of peak / {dd['Drawdown % Initial']:.1f}% of initial)
                    </td>
                    <td>{dd['Days to Trough']:.0f}</td>
                    <td>{days_to_recovery}</td>
                    <td>{total_days}</td>
                </tr>
    """


def _build_drawdown_breakdown(drawdown_contributions):
    """Build drawdown strategy breakdown section"""
    html = """
        <h2>Drawdown Strategy Breakdown</h2>
        <p style="color: #7f8c8d; font-size: 14px; margin-bottom: 20px;">
            Shows how much each strategy contributed to the top 3 combined drawdown periods.
        </p>
          Legend: <span class="negative">▲ Negative % = worsened DD</span>, 
            <span class="positive">▼ Positive % = offset DD</span>
        </p>

"""

    for idx, dd_contrib in enumerate(drawdown_contributions, 1):
        html += _build_single_drawdown_breakdown(idx, dd_contrib)

    return html


def _build_single_drawdown_breakdown(idx, dd_contrib):
    """Build breakdown for a single drawdown period with worsened vs offset grouping"""
    dd = dd_contrib["drawdown"]
    strategies = dd_contrib["strategies"]

    peak_date_str = pd.Timestamp(dd["Peak Date"]).strftime("%Y-%m-%d")
    trough_date_str = pd.Timestamp(dd["Trough Date"]).strftime("%Y-%m-%d")

    html = f"""
        <h3 style="color: #e74c3c; margin-top: 30px; margin-bottom: 10px;">
           Rank #{idx}: {peak_date_str} to {trough_date_str} | Total: ${dd['Drawdown $']:,.0f} ({dd['Drawdown % Peak']:.1f}% of peak / {dd['Drawdown % Initial']:.1f}% of initial)
        </h3>
        <table style="margin-bottom: 10px;">
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>P&L</th>
                    <th>P&L % of capital</th>
                    <th>Contribution % (signed)</th>
                </tr>
            </thead>
            <tbody>
    """

    worsened = []
    offset = []

    for strat in strategies:
        pnl_class = "negative" if strat["Loss $"] < 0 else "positive"
        contrib_class = "negative" if strat["Contribution %"] < 0 else "positive"

        html += f"""
                <tr>
                    <td><strong>{strat['Strategy']}</strong></td>
                    <td class="{pnl_class}">${strat['Loss $']:,.0f}</td>
                    <td class="{pnl_class}">{strat['Loss %']:.1f}%</td>
                    <td class="{contrib_class}">{strat['Contribution %']:.1f}%</td>
                </tr>
        """

        if strat["Contribution %"] < 0:
            worsened.append(f"▲ {strat['Strategy']} ({strat['Contribution %']:.1f}%)")
        elif strat["Contribution %"] > 0:
            offset.append(f"▼ {strat['Strategy']} ({strat['Contribution %']:.1f}%)")

    html += """
            </tbody>
        </table>
    """

    # Add grouping summary
    html += f"""
        <div class="dd-summary">
            <p><strong>Worsened DD:</strong> 
                <span class="worsened">{", ".join(worsened) if worsened else "None"}</span>
            </p>
            <p><strong>Offset DD:</strong> 
                <span class="offset">{", ".join(offset) if offset else "None"}</span>
            </p>
        </div>
"""

    return html


def _build_individual_strategy_drawdowns(strategy_drawdowns):
    """Build individual strategy drawdowns section"""
    html = """
        <h2>Individual Strategy Drawdowns</h2>
        <p style="color: #7f8c8d; font-size: 14px; margin-bottom: 20px;">
            Top 3 drawdown periods for each strategy independently.
        </p>
        <p style="color: #7f8c8d; font-size: 13px; margin-bottom: 20px;">
        Legend: <span class="negative">▲ Negative % = worsened DD</span>, 
            <span class="positive">▼ Positive % = offset DD</span>
        </p>
"""

    for strategy in sorted(strategy_drawdowns.keys()):
        drawdowns = strategy_drawdowns[strategy]
        if len(drawdowns) > 0:
            html += _build_strategy_drawdown_table(strategy, drawdowns)

    return html


def _build_strategy_drawdown_table(strategy, drawdowns):
    """Build drawdown table for a single strategy with both % conventions"""
    worst_dd = drawdowns[0]

    html = f"""
        <h3 style="color: #3498db; margin-top: 30px; margin-bottom: 10px;">
            {strategy} — Worst DD: ${worst_dd['Drawdown $']:,.0f} 
            ({worst_dd['Drawdown % Peak']:.1f}% of peak / {worst_dd['Drawdown % Initial']:.1f}% of initial)
        </h3>
        <table style="margin-bottom: 30px;">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Peak Date</th>
                    <th>Trough Date</th>
                    <th>Recovery Date</th>
                    <th>Drawdown</th>
                    <th>Days to Trough</th>
                    <th>Days to Recover</th>
                    <th>Total Duration</th>
                </tr>
            </thead>
            <tbody>
    """

    for rank, dd in enumerate(drawdowns, 1):
        html += _build_drawdown_row(rank, dd)

    html += """
            </tbody>
        </table>
"""
    return html


def _build_monthly_returns_table(monthly_returns_table):
    """Build monthly returns table"""
    html = """
        <h2>Monthly Returns (%)</h2>
        <table>
            <thead>
                <tr>
                    <th style="text-align: left;">Year</th>
"""

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    for month in month_names:
        html += f'                    <th style="text-align: right;">{month}</th>\n'

    html += '                    <th style="text-align: right; background-color: #2980b9;">Year Total</th>\n'
    html += """                </tr>
            </thead>
            <tbody>
"""

    # Add data rows
    for year in monthly_returns_table.index:
        yearly_total = monthly_returns_table.loc[year].sum()
        total_class = "positive" if yearly_total > 0 else "negative"

        html += f"                <tr>\n"
        html += f"                    <td><strong>{int(year)}</strong></td>\n"

        for month in month_names:
            value = monthly_returns_table.loc[year, month]
            if pd.isna(value) or value == 0:
                html += f'                    <td style="text-align: right; color: #bdc3c7;">—</td>\n'
            else:
                cell_class = "positive" if value > 0 else "negative"
                cell_style = "text-align: right;"
                if abs(value) > 2:
                    cell_style += " font-weight: 600;"
                html += f'                    <td class="{cell_class}" style="{cell_style}">{value:.1f}%</td>\n'

        html += f'                    <td class="{total_class}" style="text-align: right; font-weight: 700; background-color: #ecf0f1;">{yearly_total:.1f}%</td>\n'
        html += "                </tr>\n"

    # Calculate averages
    month_averages = []
    for month in month_names:
        month_values = monthly_returns_table[month].dropna()
        month_values = month_values[month_values != 0]
        avg = month_values.mean() if len(month_values) > 0 else 0
        month_averages.append(avg)

    yearly_totals = [
        monthly_returns_table.loc[year].sum() for year in monthly_returns_table.index
    ]
    avg_year_total = (
        sum(yearly_totals) / len(yearly_totals) if len(yearly_totals) > 0 else 0
    )

    # Add average row
    html += """
                <tr style="background-color: #d5dbdb; border-top: 3px solid #2980b9;">
                    <td><strong>AVG</strong></td>
"""

    for avg in month_averages:
        avg_class = "positive" if avg > 0 else "negative" if avg < 0 else ""
        html += f'                    <td class="{avg_class}" style="text-align: right; font-weight: 700;">{avg:.1f}%</td>\n'

    avg_total_class = "positive" if avg_year_total > 0 else "negative"
    html += f'                    <td class="{avg_total_class}" style="text-align: right; font-weight: 700; background-color: #bdc3c7;">{avg_year_total:.1f}%</td>\n'
    html += """                </tr>
            </tbody>
        </table>
"""

    return html


def _build_html_footer(timestamp):
    """Build HTML footer"""
    return f"""
        <div class="timestamp">
            Report generated: {timestamp}
        </div>
    </div>
"""


def _build_plotly_script(combined_equity, combined_metrics):
    """Build Plotly chart JavaScript"""

    # Calculate max drawdown points for visualization
    equity_values = combined_equity["Equity"].values
    peak = np.maximum.accumulate(equity_values)
    drawdown = equity_values - peak
    max_dd_idx = np.argmin(drawdown)

    # Find the peak before max drawdown
    peak_idx = 0
    for i in range(max_dd_idx, -1, -1):
        if equity_values[i] == peak[max_dd_idx]:
            peak_idx = i
            break

    peak_date = combined_equity["Exit_Date"].iloc[peak_idx].strftime("%Y-%m-%d")
    trough_date = combined_equity["Exit_Date"].iloc[max_dd_idx].strftime("%Y-%m-%d")
    peak_value = equity_values[peak_idx]
    trough_value = equity_values[max_dd_idx]

    return f"""
    <script>
        // Equity Curve
        var equity_data = {{
            x: {combined_equity['Exit_Date'].dt.strftime('%Y-%m-%d').tolist()},
            y: {combined_equity['Equity'].tolist()},
            type: 'scatter',
            mode: 'lines',
            name: 'Portfolio Equity',
            line: {{
                color: '#3498db',
                width: 2
            }},
            fill: 'tonexty',
            fillcolor: 'rgba(52, 152, 219, 0.1)'
        }};
        
        var baseline = {{
            x: {combined_equity['Exit_Date'].dt.strftime('%Y-%m-%d').tolist()},
            y: Array({len(combined_equity)}).fill({INITIAL_CAPITAL}),
            type: 'scatter',
            mode: 'lines',
            name: 'Starting Capital',
            line: {{
                color: '#95a5a6',
                width: 1,
                dash: 'dash'
            }}
        }};

        // Max Drawdown Markers
        var dd_markers = {{
            x: ['{peak_date}', '{trough_date}'],
            y: [{peak_value}, {trough_value}],
            mode: 'markers+text',
            name: 'Max Drawdown',
            marker: {{
                color: ['#e74c3c', '#e74c3c'],
                size: 12,
                symbol: ['triangle-down', 'triangle-up'],
                line: {{
                    color: 'white',
                    width: 2
                }}
            }},
            text: ['Peak', 'Trough'],
            textposition: 'top center',
            textfont: {{
                color: '#e74c3c',
                size: 10,
                family: 'Arial, sans-serif'
            }},
            hovertemplate: '<b>%{{text}}</b><br>Date: %{{x}}<br>Equity: $%{{y:,.0f}}<extra></extra>'
        }};
        
        var equity_layout = {{
            title: 'Portfolio Equity Curve ($100,000 Starting Capital)',
            xaxis: {{
                title: 'Date',
                showgrid: true,
                gridcolor: '#ecf0f1'
            }},
            yaxis: {{
                title: 'Equity ($)',
                showgrid: true,
                gridcolor: '#ecf0f1',
                tickformat: '$,.0f'
            }},
            hovermode: 'x unified',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            height: 500,
            shapes: [
                {{
                    type: 'rect',
                    xref: 'x',
                    yref: 'paper',
                    x0: '{peak_date}',
                    x1: '{trough_date}',
                    y0: 0,
                    y1: 1,
                    fillcolor: 'rgba(231, 76, 60, 0.1)',
                    line: {{
                        width: 0
                    }},
                    layer: 'below'
                }}
            ],
            annotations: [
                {{
                    x: '{trough_date}',
                    y: {trough_value},
                    xref: 'x',
                    yref: 'y',
                    text: 'Max DD: ${combined_metrics['Max Drawdown $']:,.0f}<br>({combined_metrics['Drawdown % Peak']:.1f}% of peak / {combined_metrics['Drawdown % Initial']:.1f}% of initial)',
                    showarrow: true,
                    arrowhead: 2,
                    arrowsize: 1,
                    arrowwidth: 2,
                    arrowcolor: '#e74c3c',
                    ax: 0,
                    ay: -60,
                    bgcolor: 'rgba(255, 255, 255, 0.9)',
                    bordercolor: '#e74c3c',
                    borderwidth: 2,
                    borderpad: 4,
                    font: {{
                        size: 11,
                        color: '#2c3e50'
                    }}
                }}
            ]
        }};
        
        Plotly.newPlot('equity_curve', [baseline, equity_data, dd_markers], equity_layout, {{responsive: true}});
    </script>
</body>
</html>
"""


def format_folder_name(folder_name):
    """Convert folder_name to Title Case with spaces"""
    # Replace underscores with spaces and title case
    return folder_name.replace("_", " ").title()


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
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    monthly_pivot.columns = month_names

    # Generate HTML report
    formatted_name = format_folder_name(folder_name)
    html_content = generate_html_report(
        combined_df,
        combined_equity,
        metrics_df,
        combined_metrics,
        monthly_pivot,
        top_drawdowns,
        formatted_name,
        drawdown_contributions,
        strategy_drawdowns,
    )

    # Create reports folder if needed
    reports_path = Path(REPORTS_FOLDER)
    reports_path.mkdir(exist_ok=True)

    # Save report with folder name and timestamp
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


def main():
    """Main execution function"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze trading strategy performance")
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Specific subfolder to analyze (default: all subfolders)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Portfolio Analyzer - Trading Strategy Combination Tool")
    print("=" * 60 + "\n")

    # Check if dataset folder exists
    dataset_path = Path(DATASET_FOLDER)
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset folder '{DATASET_FOLDER}' not found!")
        print(f"   Please create the folder and add your CSV files.")
        sys.exit(1)

    # Find all subfolders (ignore files directly in dataset/)
    subfolders = [d for d in dataset_path.iterdir() if d.is_dir()]

    if len(subfolders) == 0:
        print(f"❌ ERROR: No subfolders found in '{DATASET_FOLDER}' folder!")
        print(f"   Please create subfolders like: swing_systems/, day_trades/, etc.")
        sys.exit(1)

    # If specific folder requested, validate it
    if args.folder:
        requested_folder = dataset_path / args.folder
        if not requested_folder.exists() or not requested_folder.is_dir():
            print(f"❌ ERROR: Folder '{args.folder}' not found in '{DATASET_FOLDER}'!")
            print(f"\n   Available folders:")
            for folder in subfolders:
                print(f"     • {folder.name}")
            sys.exit(1)
        folders_to_process = [requested_folder]
    else:
        folders_to_process = subfolders

    print(f"📁 Found {len(subfolders)} strategy group(s) in {DATASET_FOLDER}/\n")
    if args.folder:
        print(f"🎯 Processing only: {args.folder}\n")

    # Single timestamp for all reports
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Process each folder
    results = []
    for folder_path in folders_to_process:
        try:
            result = process_folder(folder_path, folder_path.name)
            if result:
                results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️  Analysis interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"  ❌ ERROR processing {folder_path.name}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    # Final summary
    print("=" * 60)
    if len(results) == 0:
        print("⚠️  No reports generated - all folders were empty or had errors")
    else:
        print(f"✅ ANALYSIS COMPLETE - {len(results)} report(s) generated")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
