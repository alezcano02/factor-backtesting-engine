# The backtest loop, vectorised: weights decided at the end of month t earn the returns of month t+1

import pandas as pd


def run_backtest(prices: pd.DataFrame, weights: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    """Monthly returns of a strategy.

    prices  : month-end prices (rows = dates, columns = stocks)
    weights : target weights decided at each month end (same shape)
    cost_bps: transaction cost in basis points per unit of turnover (buy + sell)

    Returns a DataFrame with gross return, cost, net return and turnover of each month.
    """
    ret = prices.pct_change(fill_method=None).fillna(0.0)
    w = weights.reindex_like(prices).fillna(0.0)
    held = w.shift(1).fillna(0.0)                     # weights decided at t-1 are held during month t
    gross = (held * ret).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()          # first month: we build the portfolio from cash
    cost = (turnover * cost_bps / 10000).shift(1).fillna(0.0)   # we pay the trades of t-1 in month t
    out = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "turnover": turnover})
    return out.iloc[1:]
