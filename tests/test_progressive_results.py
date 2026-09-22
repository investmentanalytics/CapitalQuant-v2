import numpy as np
import pandas as pd

from core.types import BacktestConfig
from engine.backtester import BacktestEngine
from optimization.monte_carlo import MonteCarloAnalyzer


def _ohlcv(n=140):
    idx = pd.date_range('2025-01-01', periods=n, freq='D')
    close = 100 + np.cumsum(np.sin(np.arange(n) / 5.0) * 0.2 + 0.15)
    return pd.DataFrame({
        'open': close - 0.1, 'high': close + 0.6,
        'low': close - 0.6, 'close': close,
        'volume': np.full(n, 1000.0),
        'signal': np.where(np.arange(n) % 12 == 0, 1, 0),
    }, index=idx)


def test_backtest_emits_progress_snapshots():
    df = _ohlcv()
    seen = []
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, commission=0, slippage=0))
    result = engine.run(df, asset='TEST', timeframe='1d', progress_callback=seen.append, progress_every=20)
    assert seen
    assert seen[-1]['bar_index'] == len(df) - 1
    assert len(result.equity_curve) == len(df)
    assert isinstance(seen[-1]['trades'], list)


def test_monte_carlo_emits_progress_snapshots():
    df = _ohlcv()
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, commission=0, slippage=0))
    result = engine.run(df, asset='TEST', timeframe='1d')
    seen = []
    mc = MonteCarloAnalyzer(result, n_simulations=25, seed=7)
    final = mc.run(progress_callback=lambda done, total, partial: seen.append((done, total, partial)), chunk_size=10)
    assert seen
    assert seen[-1][0] == 25
    assert final.n_simulations == 25
    assert len(final.final_capitals) == 25
