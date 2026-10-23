# Signals: each function takes monthly prices (rows = month end, columns = stocks) and returns a score
# known at the end of each month. The backtest uses the score of month t to invest in month t+1,
# so a signal must never look at prices after t.

import numpy as np
import pandas as pd


def momentum(prices: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """Return of the last `lookback` months skipping the most recent `skip` (classic 12-1 momentum)."""
    return prices.shift(skip) / prices.shift(lookback) - 1


def low_volatility(prices: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Minus the volatility of the monthly returns, so a higher score means a less volatile stock."""
    ret = prices.pct_change(fill_method=None)
    return -ret.rolling(window, min_periods=window).std()


def short_term_reversal(prices: pd.DataFrame) -> pd.DataFrame:
    """Minus the return of the last month: last month's losers tend to bounce back."""
    return -(prices / prices.shift(1) - 1)


def zscore(score: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score, to combine signals with different scales."""
    m = score.mean(axis=1)
    s = score.std(axis=1).replace(0, np.nan)
    return score.sub(m, axis=0).div(s, axis=0)
