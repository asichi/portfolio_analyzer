"""
table_builder.py
----------------
HTML table generation for drawdowns, monthly returns, and capital usage.
"""

import pandas as pd
import numpy as np

from config import INITIAL_CAPITAL


def build_top_drawdowns_table(top_drawdowns):
    """Build HTML table for top drawdowns"""
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


def build_drawdown_row(rank, dd):
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


def build_drawdown_breakdown(drawdown_contributions):
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
        html += build_single_drawdown_breakdown(idx, dd_contrib)

    return html


def build_single_drawdown_breakdown(idx, dd_contrib):
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


def build_individual_strategy_drawdowns(strategy_drawdowns):
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
            html += build_strategy_drawdown_table(strategy, drawdowns)

    return html


def build_strategy_drawdown_table(strategy, drawdowns):
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
        html += build_drawdown_row(rank, dd)

    html += """
            </tbody>
        </table>
"""
    return html


def build_monthly_returns_table(monthly_returns_table):
    """Build monthly returns table"""
    html = """
        <h2>Monthly Returns (%)</h2>
        <table>
            <thead>
                <tr>
                    <th style="text-align: left;">Year</th>
"""

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
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


def build_capital_usage_table(usage, trades_df):
    """
    Build HTML table of capital usage percentiles, max, tail analysis, and P&L context.

    Parameters
    ----------
    usage : pandas.Series
        Daily capital usage values in dollars (indexed by Date).
    trades_df : pandas.DataFrame
        Trades with 'Ex. date' and 'Profit' columns.
    """
    
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
    # Note: columns are already normalized to lowercase in processor.py
    daily_pnl = trades_df.groupby("ex. date")["profit"].sum()
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
