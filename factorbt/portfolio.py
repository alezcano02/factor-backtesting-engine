# From scores to portfolio weights

import numpy as np
import pandas as pd


def quantile_weights(score: pd.DataFrame, quantiles: int = 5, long_short: bool = True) -> pd.DataFrame:
    """Equal weights in the top quantile (long) and, if long_short, minus equal weights in the bottom one.

    Long-only weights sum 1. Long-short weights sum 0 (+1 long, -1 short).
    Stocks without score in a month get weight 0.
    """
    rank = score.rank(axis=1, pct=True)
    top = (rank > 1 - 1 / quantiles).astype(float)
    bottom = (rank <= 1 / quantiles).astype(float)
    w = top.div(top.sum(axis=1).replace(0, np.nan), axis=0)
    if long_short:
        w = w - bottom.div(bottom.sum(axis=1).replace(0, np.nan), axis=0)
    return w.fillna(0.0)
