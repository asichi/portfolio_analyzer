"""
report_html_generator.py
-------------------------
Generates the HTML portfolio analysis report.

Responsibilities:
- Assemble summary cards, tables, and charts into a styled HTML document
- Integrate external CSS (styles.css) for consistent UI
- Support interactive features such as tabbed navigation
- Present capital usage, drawdown breakdowns, and strategy performance
"""

import numpy as np
import pandas as pd
import json

from datetime import datetime
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
    usage_stats,
    usage,
):
    """Generate beautiful HTML report"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build sections
    header = _build_html_header(timestamp)
    summary = _build_summary_section(metrics_df, combined_metrics, folder_name)
    equity_chart = _build_equity_chart_section()
    capital_usage_chart = _build_capital_usage_chart(usage)
    capital_usage_table = _build_capital_usage_table(usage, all_trades)
    top_dd_table = _build_top_drawdowns_table(top_drawdowns)
    dd_breakdown = _build_drawdown_breakdown(drawdown_contributions)
    individual_dd = _build_individual_strategy_drawdowns(strategy_drawdowns)
    monthly_table = _build_monthly_returns_table(monthly_returns_table)
    footer = _build_html_footer(timestamp)
    chart_script = _build_plotly_script(combined_equity, usage, combined_metrics)

    # Combine all sections
    return (
        header
        + summary
        + equity_chart
        + capital_usage_chart
        + capital_usage_table
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
    <link rel="stylesheet" type="text/css" href="../styles.css">
</head>
<body>
    <div class="container">
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
        (
            "Average Annual Profit",
            f"${combined_metrics['Avg Annual Profit $']:,.0f}",
            f"{combined_metrics['Avg Annual Profit %']:.1f}% of initial",
        ),
        (
            "Median Annual Profit",
            f"${combined_metrics['Median Annual Profit $']:,.0f}",
            f"{combined_metrics['Median Annual Profit %']:.1f}% of initial",
        ),
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


def _build_capital_usage_chart(usage: pd.Series) -> str:
    """
    Build a standalone Plotly chart for daily capital usage.
    Expects `usage` to be a Pandas Series with a DatetimeIndex.
    """
    x_values = usage.index.strftime("%Y-%m-%d").tolist()
    y_values = usage.values.tolist()

    trace = {
        "x": x_values,
        "y": y_values,
        "type": "scatter",
        "mode": "lines",
        "name": "Capital Usage",
        "line": {"color": "#e67e22", "width": 2},  # Match your orange from original
        "fill": "tozeroy",
        "fillcolor": "rgba(230, 126, 34, 0.1)",
    }

    trace_json = json.dumps(trace)

    return f"""
    <h2>Daily Capital Usage</h2>
    <div id="capital-usage-chart" class="chart-container"></div>
    <script>
        var capitalUsageData = [{trace_json}];
        var capitalLayout = {{
            xaxis: {{title: 'Date', showgrid: true, gridcolor: '#ecf0f1'}},
            yaxis: {{title: 'Capital Usage ($)', showgrid: true, gridcolor: '#ecf0f1', tickformat: '$,.0f'}},
            hovermode: 'x unified',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            height: 500
        }};
        Plotly.newPlot('capital-usage-chart', capitalUsageData, capitalLayout, {{responsive: true}});
    </script>
    """


def _build_top_drawdowns_table(top_drawdowns):
    """Build HTML table for top drawdowns."""
    html = """
        <h2>Top Drawdowns</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Peak Date</th>
                    <th>Trough Date</th>
                    <th>Recovery Date</th>
                    <th>Drawdown</th>
                    <th>% of Initial</th>
                    <th>% of Peak</th>
                    <th>Days to Trough</th>
                    <th>Days to Recover</th>
                </tr>
            </thead>
            <tbody>
    """

    for rank, dd in enumerate(top_drawdowns, 1):
        peak_val = dd["Peak Value"]
        trough_val = dd["Trough Value"]
        abs_dd = peak_val - trough_val
        pct_initial = (abs_dd / INITIAL_CAPITAL) * 100
        pct_peak = (abs_dd / peak_val) * 100

        html += (
            f"<tr>"
            f"<td><strong>#{rank}</strong></td>"
            f"<td>{pd.Timestamp(dd['Peak Date']).strftime('%Y-%m-%d')}</td>"
            f"<td>{pd.Timestamp(dd['Trough Date']).strftime('%Y-%m-%d')}</td>"
            f"<td>{pd.Timestamp(dd['Recovery Date']).strftime('%Y-%m-%d') if dd['Recovery Date'] else 'ONGOING'}</td>"
            f"<td>${abs_dd:,.2f}</td>"
            f"<td>{pct_initial:.2f}%</td>"
            f"<td>{pct_peak:.2f}%</td>"
            f"<td>{dd['Days to Trough']}</td>"
            f"<td>{dd['Days to Recovery'] if dd['Days to Recovery'] is not None else 'ONGOING'}</td>"
            f"</tr>"
        )

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

    # Sort strategies by Contribution % ascending (negative first, positive last)
    strategies_sorted = sorted(strategies, key=lambda s: s["Contribution %"])

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

    for strat in strategies_sorted:
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


def _build_plotly_script(combined_equity, usage, combined_metrics):
    """Build Plotly chart JavaScript with equity curve and drawdown markers (no usage overlay)."""

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
            line: {{color: '#3498db', width: 2}},
            fill: 'tonexty',
            fillcolor: 'rgba(52, 152, 219, 0.1)'
        }};
        
        var baseline = {{
            x: {combined_equity['Exit_Date'].dt.strftime('%Y-%m-%d').tolist()},
            y: Array({len(combined_equity)}).fill({INITIAL_CAPITAL}),
            type: 'scatter',
            mode: 'lines',
            name: 'Starting Capital',
            line: {{color: '#95a5a6', width: 1, dash: 'dash'}}
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
                line: {{color: 'white', width: 2}}
            }},
            text: ['Peak', 'Trough'],
            textposition: 'top center',
            textfont: {{color: '#e74c3c', size: 10, family: 'Arial, sans-serif'}},
            hovertemplate: '<b>%{{text}}</b><br>Date: %{{x}}<br>Equity: $%{{y:,.0f}}<extra></extra>'
        }};
        
        var equity_layout = {{
            title: 'Portfolio Equity Curve',
            xaxis: {{title: 'Date', showgrid: true, gridcolor: '#ecf0f1'}},
            yaxis: {{title: 'Equity ($)', showgrid: true, gridcolor: '#ecf0f1', tickformat: '$,.0f'}},
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
                    line: {{width: 0}},
                    layer: 'below'
                }}
            ],
            annotations: [
                {{
                    x: '{trough_date}',
                    y: {trough_value},
                    xref: 'x',
                    yref: 'y',
                    text: 'Max DD: ${combined_metrics["Max Drawdown $"]:,.0f}<br>'
                          + '({combined_metrics["Drawdown % Peak"]:.1f}% of peak / '
                          + '{combined_metrics["Drawdown % Initial"]:.1f}% of initial)',
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
                    font: {{size: 11, color: '#2c3e50'}}
                }}
            ]
        }};
        
        Plotly.newPlot('equity_curve', [baseline, equity_data, dd_markers], equity_layout, {{responsive: true}});
    </script>
    """


def _build_capital_usage_table(usage, trades_df):
    """
    Build HTML table of capital usage percentiles, max, tail analysis, and P&L context.

    Parameters
    ----------
    usage : pandas.Series
        Daily capital usage values in dollars (indexed by Date).
    trades_df : pandas.DataFrame
        Trades with 'Ex. date' and 'Profit' columns.
    """
    import numpy as np
    import pandas as pd

    # Percentiles
    median = np.percentile(usage, 50)
    p75 = np.percentile(usage, 75)
    p95 = np.percentile(usage, 95)
    p99 = np.percentile(usage, 99)
    max_val = usage.max()

    # Tail metrics
    tail_ratio = max_val / p95 if p95 > 0 else float("nan")
    days_total = len(usage)
    days_above_95 = (usage > p95).sum()
    days_above_99 = (usage > p99).sum()
    freq_95 = (days_above_95 / days_total) * 100 if days_total > 0 else 0
    freq_99 = (days_above_99 / days_total) * 100 if days_total > 0 else 0

    # Daily P&L series
    daily_pnl = trades_df.groupby("Ex. date")["Profit"].sum()
    daily_pnl = daily_pnl.reindex(usage.index, fill_value=0.0)

    # Tail P&L stats
    pnl_95 = daily_pnl[usage > p95]
    pnl_99 = daily_pnl[usage > p99]

    avg_pnl_95 = pnl_95.mean() if len(pnl_95) else 0.0
    avg_pnl_99 = pnl_99.mean() if len(pnl_99) else 0.0
    worst_pnl_95 = pnl_95.min() if len(pnl_95) else 0.0
    best_pnl_95 = pnl_95.max() if len(pnl_95) else 0.0

    return f"""
        <h2>Capital Usage Percentiles, Tail Analysis & P&L Context</h2>
        <table>
            <thead>
                <tr><th>Metric</th><th>Value</th></tr>
            </thead>
            <tbody>
                <tr><td>Median (50th)</td><td class="neutral">${median:,.0f}</td></tr>
                <tr><td>75th percentile</td><td class="neutral">${p75:,.0f}</td></tr>
                <tr><td>95th percentile</td><td class="negative">${p95:,.0f}</td></tr>
                <tr><td>99th percentile</td><td class="negative">${p99:,.0f}</td></tr>
                <tr class="tail-row"><td>Max</td><td class="negative">${max_val:,.0f}</td></tr>
                <tr class="tail-row"><td>Tail Ratio (Max ÷ 95th)</td><td>{tail_ratio:.2f}×</td></tr>
                <tr class="tail-row"><td>Days >95th</td><td>{freq_95:.1f}% ({days_above_95} days)</td></tr>
                <tr class="tail-row"><td>Days >99th</td><td>{freq_99:.1f}% ({days_above_99} days)</td></tr>
                <tr class="tail-row"><td>Avg P&L >95th</td><td>${avg_pnl_95:,.0f}</td></tr>
                <tr class="tail-row"><td>Avg P&L >99th</td><td>${avg_pnl_99:,.0f}</td></tr>
                <tr class="tail-row"><td>Worst P&L >95th</td><td class="negative">${worst_pnl_95:,.0f}</td></tr>
                <tr class="tail-row"><td>Best P&L >95th</td><td class="positive">${best_pnl_95:,.0f}</td></tr>
            </tbody>
        </table>
        <p style="font-size: 0.9em; color: #7f8c8d;">
            Tail metrics highlight rare but extreme usage days and their associated profit/loss outcomes.
        </p>
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
        usage_stats,
        usage,
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
    html_content = generate_html_report(
        combined_df,
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
