# Performance metrics of a series of monthly returns

import numpy as np
import pandas as pd


def max_drawdown(ret: pd.Series) -> float:
    wealth = (1 + ret).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def summary(ret: pd.Series, rf: pd.Series | None = None) -> pd.Series:
    """Annual return, volatility, Sharpe, max drawdown and share of positive months."""
    ret = ret.dropna()
    ex = ret - (rf.reindex(ret.index).fillna(0) if rf is not None else 0)
    n = len(ret)
    return pd.Series({
        "annual_return": float((1 + ret).prod() ** (12 / n) - 1),
        "annual_vol": float(ret.std() * np.sqrt(12)),
        "sharpe": float(ex.mean() / ex.std() * np.sqrt(12)) if ex.std() > 0 else np.nan,
        "max_drawdown": max_drawdown(ret),
        "hit_rate": float((ret > 0).mean()),
        "months": n,
    })
