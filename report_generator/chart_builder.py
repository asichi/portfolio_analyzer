"""
chart_builder.py
----------------
Plotly chart generation for equity curves and capital usage.
"""

import json


def build_equity_chart_section():
    """Build equity curve chart placeholder"""
    return """
        <h2>Combined Equity Curve</h2>
        <div id="equity_curve" class="chart-container"></div>
"""


def build_capital_usage_chart(usage):
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
        "line": {"color": "#e67e22", "width": 2},
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


def build_plotly_script(combined_equity, usage, combined_metrics):
    """Generate Plotly JavaScript for equity curve"""
    
    from config import INITIAL_CAPITAL

    # Prepare equity data
    dates = combined_equity["Exit_Date"].dt.strftime("%Y-%m-%d").tolist()
    equity_values = combined_equity["Equity"].tolist()

    # Calculate drawdown as % of initial capital (more meaningful for personal analysis)
    peak = combined_equity["Equity"].cummax()
    drawdown_pct = ((combined_equity["Equity"] - peak) / INITIAL_CAPITAL * 100).tolist()

    # Prepare data as JSON
    equity_trace = {
        "x": dates,
        "y": equity_values,
        "type": "scatter",
        "mode": "lines",
        "name": "Equity",
        "yaxis": "y",  # Equity uses y1
        "line": {"color": "#2ecc71", "width": 2},
        "hovertemplate": "Date: %{x}<br>Equity: $%{y:,.0f}<extra></extra>",
    }

    drawdown_trace = {
        "x": dates,
        "y": drawdown_pct,
        "type": "scatter",
        "mode": "lines",
        "name": "Drawdown %",
        "yaxis": "y2",  # Drawdown uses y2
        "line": {"color": "#e74c3c", "width": 1},
        "fill": "tozeroy",
        "fillcolor": "rgba(231, 76, 60, 0.3)",
        "hovertemplate": "Drawdown: %{y:.1f}%<extra></extra>",
    }

    layout = {
        "title": f"Portfolio Equity Curve - ${combined_metrics['Total Return $']:,.0f} ({combined_metrics['Total Return %']:.1f}%)",
        "xaxis": {
            "title": "Date",
            "showgrid": True,
            "gridcolor": "#ecf0f1",
            "domain": [0, 1],  # Full width
        },
        "yaxis": {
            "title": "",  # No label
            "domain": [0.28, 1.0],  # Equity panel (top)
            "showgrid": True,
            "gridcolor": "#ecf0f1",
            "tickformat": "$,.0f",
            "rangemode": "tozero",
        },
        "yaxis2": {
            "title": "DD %",
            "domain": [0, 0.28],  # DD panel (bottom)
            "showgrid": True,
            "gridcolor": "#ecf0f1",
            "tickformat": ".0f",
            "rangemode": "tozero",
        },
        "hovermode": "x unified",
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "height": 600,
    }

    equity_json = json.dumps(equity_trace)
    drawdown_json = json.dumps(drawdown_trace)
    layout_json = json.dumps(layout)

    return f"""
<script>
    // Equity curve with drawdown overlay
    var equityData = [{equity_json}, {drawdown_json}];
    var equityLayout = {layout_json};
    var config = {{
        responsive: true,
        modeBarButtonsToRemove: ['zoomIn2d', 'zoomOut2d', 'autoScale2d'],
        displaylogo: false
    }};
    Plotly.newPlot('equity_curve', equityData, equityLayout, config);
</script>
"""
