"""
metrics.py
-----------
Core performance calculations for strategies and portfolios.

Responsibilities:
- Build equity curves
- Compute performance metrics (CAGR, Sharpe, drawdowns, recovery, etc.)
- Track daily capital usage and trade density
- Attribute contributions by strategy for benchmarking and tail analysis
"""

import pandas as pd
import numpy as np

from config import INITIAL_CAPITAL


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

    # Expected Value per Trade
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

    # Max Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = drawdown.min()
    max_dd_idx = np.argmin(drawdown)

    max_dd_pct = (max_dd / INITIAL_CAPITAL) * 100

    # Recovery Time
    recovery_days = np.nan
    if max_dd_idx < len(equity) - 1:
        recovery_idx = np.where(equity[max_dd_idx:] >= peak[max_dd_idx])[0]
        if len(recovery_idx) > 0:
            trough_date = trades["Ex. date"].iloc[max_dd_idx]
            recovery_date = trades["Ex. date"].iloc[max_dd_idx + recovery_idx[0]]
            recovery_days = (recovery_date - trough_date).days

    # Period length
    start_date = trades["Ex. date"].min()
    end_date = trades["Ex. date"].max()
    days = (end_date - start_date).days
    years = days / 365.25

    final_value = INITIAL_CAPITAL + total_profit

    # CAGR (keep for optional use)
    cagr = (
        (((final_value / INITIAL_CAPITAL) ** (1 / years)) - 1) * 100 if years > 0 else 0
    )

    # Recovery Factor
    recovery_factor = (total_profit / abs(max_dd)) if max_dd != 0 else np.nan
    recovery_factor_per_year = (recovery_factor / years) if years > 0 else np.nan

    # --- NEW: Cash-flow metrics ---
    trades["Year"] = trades["Ex. date"].dt.year
    yearly_profits = trades.groupby("Year")["Profit"].sum()

    avg_annual_profit = yearly_profits.mean() if len(yearly_profits) > 0 else 0
    median_annual_profit = yearly_profits.median() if len(yearly_profits) > 0 else 0

    avg_annual_profit_pct = (avg_annual_profit / INITIAL_CAPITAL) * 100
    median_annual_profit_pct = (median_annual_profit / INITIAL_CAPITAL) * 100

    return {
        "Strategy": strategy_name if strategy_name else "COMBINED",
        "Total Return $": total_profit,
        "Total Return %": (total_profit / INITIAL_CAPITAL) * 100,
        "CAGR %": cagr,  # keep for optional use
        "Avg Annual Profit $": avg_annual_profit,
        "Avg Annual Profit %": avg_annual_profit_pct,
        "Median Annual Profit $": median_annual_profit,
        "Median Annual Profit %": median_annual_profit_pct,
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
    """Find the top N drawdowns - max pain perspective
    
    A drawdown only ends when equity reaches a NEW all-time high.
    Partial recoveries don't end the drawdown - tracks worst continuous pain.
    """
    equity = equity_curve_df["Equity"].values
    dates = equity_curve_df["Exit_Date"].values
    peak = np.maximum.accumulate(equity)
    
    drawdowns = []
    in_dd = False
    dd_start = dd_peak = dd_trough_idx = dd_trough_val = None
    
    for i in range(len(equity)):
        at_peak = (equity[i] == peak[i])
        
        if at_peak:
            if in_dd:
                # Drawdown recovered - close it
                dd_amt = dd_peak - dd_trough_val
                drawdowns.append({
                    "Peak Date": dates[dd_start],
                    "Trough Date": dates[dd_trough_idx],
                    "Recovery Date": dates[i],
                    "Peak Value": dd_peak,
                    "Trough Value": dd_trough_val,
                    "Drawdown $": dd_amt,
                    "Drawdown % Peak": (dd_amt / dd_peak) * 100,
                    "Drawdown % Initial": (dd_amt / INITIAL_CAPITAL) * 100,
                    "Days to Trough": dd_trough_idx - dd_start,
                    "Days to Recovery": i - dd_trough_idx,
                    "Total Days": i - dd_start,
                })
                in_dd = False
        else:
            # Below peak
            if not in_dd:
                # Start new drawdown
                peak_idx = np.where(equity[:i+1] == peak[i])[0][-1]
                dd_start = peak_idx
                dd_peak = peak[i]
                dd_trough_idx = i
                dd_trough_val = equity[i]
                in_dd = True
            elif equity[i] < dd_trough_val:
                # Update trough
                dd_trough_idx = i
                dd_trough_val = equity[i]
    
    # Ongoing drawdown
    if in_dd:
        dd_amt = dd_peak - dd_trough_val
        drawdowns.append({
            "Peak Date": dates[dd_start],
            "Trough Date": dates[dd_trough_idx],
            "Recovery Date": None,
            "Peak Value": dd_peak,
            "Trough Value": dd_trough_val,
            "Drawdown $": dd_amt,
            "Drawdown % Peak": (dd_amt / dd_peak) * 100,
            "Drawdown % Initial": (dd_amt / INITIAL_CAPITAL) * 100,
            "Days to Trough": dd_trough_idx - dd_start,
            "Days to Recovery": None,
            "Total Days": None,
        })
    
    drawdowns.sort(key=lambda x: x["Drawdown $"], reverse=True)
    return drawdowns[:n]


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

        for strategy in all_trades["Strategy"].unique():
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


def calculate_daily_capital_usage(
    trades_df: pd.DataFrame,
    per_trade_capital: float = 100_000.0,
    sizing_col: str | None = None,
    entry_col: str = "Date",
    exit_col: str = "Ex. date",
    strategy_col: str = "Strategy",
    inclusive_exit: bool = True,
):
    """
    Compute daily capital usage across all strategies (vectorized).
    """

    if trades_df.empty:
        idx = pd.Index([], name="Date")
        empty = pd.Series(dtype=float, index=idx)
        return (
            empty,
            {
                "Max Usage": 0.0,
                "Min Usage": 0.0,
                "Avg Usage": 0.0,
                "Median Usage": 0.0,
                "Std Usage": 0.0,
            },
            {"Baseline Capital": 0.0, "Usage % Series": empty},
        )

    df = trades_df.copy()
    df[entry_col] = pd.to_datetime(df[entry_col]).dt.normalize()
    df[exit_col] = pd.to_datetime(df[exit_col]).dt.normalize()

    if (df[exit_col] < df[entry_col]).any():
        raise ValueError("Found trades with exit before entry. Fix input data.")

    # Determine per-trade capital
    if sizing_col and sizing_col in df.columns:
        cap = df[sizing_col].astype(float).clip(lower=0.0)
    else:
        cap = pd.Series(per_trade_capital, index=df.index, dtype=float)

    # Build event series: +cap at entry, -cap at exit+1 (if inclusive)
    events = {}
    for entry, exit_, c in zip(df[entry_col], df[exit_col], cap):
        # exit+1 if inclusive, else exit day is excluded
        end_day = exit_ + pd.Timedelta(days=1) if inclusive_exit else exit_
        events[entry] = events.get(entry, 0.0) + c
        events[end_day] = events.get(end_day, 0.0) - c

    # Convert events dict to Series
    events_series = pd.Series(events).sort_index()

    # Build full calendar index
    all_days = pd.date_range(df[entry_col].min(), df[exit_col].max(), freq="D")

    # Reindex events to full calendar, fill missing with 0, then cumsum
    usage = events_series.reindex(all_days, fill_value=0.0).cumsum()
    usage.index.name = "Date"

    # Baseline capital
    if strategy_col in df.columns:
        num_strategies = df[strategy_col].nunique()
        baseline_capital = (
            float(num_strategies) * float(per_trade_capital)
            if sizing_col is None
            else float(df[sizing_col].groupby(df[strategy_col]).max().sum())
        )
    else:
        baseline_capital = usage.max()

    # Stats
    nonzero = usage[usage > 0]
    stats = {
        "Max Usage": float(usage.max()),
        "Min Usage": float(nonzero.min()) if len(nonzero) else 0.0,
        "Avg Usage": float(usage.mean()),
        "Median Usage": float(usage.median()),
        "Std Usage": float(usage.std(ddof=1)) if len(usage) > 1 else 0.0,
    }

    usage_pct = (
        (usage / baseline_capital * 100.0)
        if baseline_capital > 0
        else pd.Series(0.0, index=usage.index)
    )
    extra = {"Baseline Capital": float(baseline_capital), "Usage % Series": usage_pct}

    return usage, stats, extra
