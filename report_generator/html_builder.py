"""
html_builder.py
---------------
Core HTML structure generation - headers, footers, summary sections.
"""

from datetime import datetime


def build_html_header(timestamp):
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


def build_html_footer(timestamp):
    """Build HTML footer"""
    return f"""
        <div class="footer">
            <p>Generated: {timestamp}</p>
            <p>Portfolio Analyzer v2.1 | For analysis purposes only</p>
        </div>
    </div>
</body>
</html>
"""


def build_metric_card(label, value, subtitle=""):
    """Build a single metric card"""
    subtitle_html = f'<div class="metric-label">{subtitle}</div>' if subtitle else ""
    return f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {subtitle_html}
            </div>
"""


def build_summary_section(metrics_df, combined_metrics, folder_name):
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
        html += build_metric_card(label, value, subtitle)

    html += """
        </div>
"""
    return html


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
    """Generate complete HTML report by assembling all sections"""
    
    from .chart_builder import (
        build_equity_chart_section,
        build_capital_usage_chart,
        build_plotly_script,
    )
    from .table_builder import (
        build_top_drawdowns_table,
        build_drawdown_breakdown,
        build_individual_strategy_drawdowns,
        build_monthly_returns_table,
        build_capital_usage_table,
    )
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build sections
    header = build_html_header(timestamp)
    summary = build_summary_section(metrics_df, combined_metrics, folder_name)
    equity_chart = build_equity_chart_section()
    capital_usage_chart = build_capital_usage_chart(usage)
    capital_usage_table = build_capital_usage_table(usage, all_trades)
    top_dd_table = build_top_drawdowns_table(top_drawdowns)
    dd_breakdown = build_drawdown_breakdown(drawdown_contributions)
    individual_dd = build_individual_strategy_drawdowns(strategy_drawdowns)
    monthly_table = build_monthly_returns_table(monthly_returns_table)
    footer = build_html_footer(timestamp)
    chart_script = build_plotly_script(combined_equity, usage, combined_metrics)

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
