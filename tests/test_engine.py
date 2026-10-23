# Tests of the engine with small synthetic data, where we know the right answer

import numpy as np
import pandas as pd
import pytest

from factorbt import momentum, quantile_weights, run_backtest, summary
from factorbt.metrics import max_drawdown


@pytest.fixture
def precios():
    fechas = pd.date_range("2020-01-31", periods=24, freq="ME")
    # stock A grows 2% every month, B is flat, C falls 1% every month
    return pd.DataFrame({"A": 100 * 1.02 ** np.arange(24), "B": 100.0, "C": 100 * 0.99 ** np.arange(24)}, index=fechas)


def test_momentum_does_not_look_ahead(precios):
    s1 = momentum(precios)
    cambiado = precios.copy()
    cambiado.iloc[-1] = cambiado.iloc[-1] * 10          # a shock in the last month
    s2 = momentum(cambiado)
    # the score of the last month uses prices until t-1 (skip = 1), so it must not change
    pd.testing.assert_series_equal(s1.iloc[-1], s2.iloc[-1])


def test_long_only_weights_sum_one(precios):
    w = quantile_weights(momentum(precios).dropna(), quantiles=3, long_short=False)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_long_short_weights_sum_zero(precios):
    w = quantile_weights(momentum(precios).dropna(), quantiles=3, long_short=True)
    assert np.allclose(w.sum(axis=1), 0.0)
    assert np.allclose(w.abs().sum(axis=1), 2.0)


def test_weights_earn_next_month_return(precios):
    # always 100% in A: every month must earn exactly 2% (before costs)
    w = pd.DataFrame({"A": 1.0, "B": 0.0, "C": 0.0}, index=precios.index)
    bt = run_backtest(precios, w, cost_bps=0)
    assert np.allclose(bt["gross"], 0.02)


def test_costs_reduce_returns(precios):
    w = quantile_weights(momentum(precios).fillna(0), quantiles=3, long_short=True)
    sin = run_backtest(precios, w, cost_bps=0)["net"].sum()
    con = run_backtest(precios, w, cost_bps=50)["net"].sum()
    assert con < sin


def test_max_drawdown():
    r = pd.Series([0.10, -0.50, 0.20])
    # wealth 1.1 -> 0.55 -> 0.66, the worst fall from the peak is -50%
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_summary_of_constant_return():
    r = pd.Series([0.01] * 12)
    s = summary(r)
    assert s["annual_return"] == pytest.approx(1.01 ** 12 - 1)
    assert s["max_drawdown"] == 0
