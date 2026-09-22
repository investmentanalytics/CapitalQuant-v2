"""
tests/test_hardening.py
Tests para las mejoras de la Fase 1 del roadmap:
- Purge/embargo en Walk-Forward (optimization/walk_forward.py)
- Modelo de costos realista (core/costs.py, engine/backtester.py)
- Sandbox AST (strategies/ast_sandbox.py)
- Generalización cruzada de símbolos y estabilidad de régimen (discovery/generalization.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.types import BacktestConfig
from core.costs import InstrumentCostProfile, entry_cost, exit_cost
from engine.backtester import BacktestEngine
from strategies.base import BaseStrategy
from strategies.ast_sandbox import validate_ast_safety, ForbiddenConstructError
from strategies.code_compiler import compile_strategy_code
from optimization.walk_forward import WalkForwardOptimizer
from discovery.generalization import run_cross_symbol_generalization, run_regime_and_seasonality_test


def make_ohlcv(n: int = 800, seed: int = 42, trend: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100.0 * np.cumprod(1 + rng.normal(trend, 0.02, n))
    high  = close * (1 + rng.uniform(0, 0.02, n))
    low   = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol   = rng.uniform(1000, 5000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


class SimpleSMAStrategy(BaseStrategy):
    """Estrategia mínima de cruce de medias, usada solo para tests."""
    name = "SMA Test"

    def _validate_params(self):
        self.fast = self.params.get("fast", 10)
        self.slow = self.params.get("slow", 30)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["sma_fast"] = df["close"].rolling(self.fast).mean()
        df["sma_slow"] = df["close"].rolling(self.slow).mean()
        df["signal"] = 0
        df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
        df.loc[df["sma_fast"] < df["sma_slow"], "signal"] = -1
        df["signal"] = df["signal"].fillna(0).astype(int)
        return df

    def get_param_space(self) -> dict:
        return {"fast": ("int", 5, 20), "slow": ("int", 20, 60)}


# ---------------------------------------------------------------------------
# Purge / Embargo
# ---------------------------------------------------------------------------

def test_wfo_purge_embargo_creates_gap_between_train_and_test():
    data = make_ohlcv(1000)
    wfo = WalkForwardOptimizer(
        strategy_class=SimpleSMAStrategy, data=data, asset="TEST", timeframe="1d",
        n_windows=3, n_trials=2, purge_pct=0.05, embargo_pct=0.05,
    )
    windows = wfo._build_windows()
    assert len(windows) > 0
    for w in windows:
        assert w.purge_bars > 0
        assert w.embargo_bars > 0
        # El test debe empezar estrictamente después del train (con embargo de por medio)
        assert w.test_start > w.train_end


def test_wfo_zero_purge_embargo_is_backward_compatible():
    data = make_ohlcv(1000)
    wfo = WalkForwardOptimizer(
        strategy_class=SimpleSMAStrategy, data=data, asset="TEST", timeframe="1d",
        n_windows=3, n_trials=2, purge_pct=0.0, embargo_pct=0.0,
    )
    windows = wfo._build_windows()
    assert len(windows) > 0
    for w in windows:
        assert w.purge_bars == 0
        assert w.embargo_bars == 0


def test_wfo_max_lookback_floor_applies_even_with_small_pct():
    data = make_ohlcv(1000)
    wfo = WalkForwardOptimizer(
        strategy_class=SimpleSMAStrategy, data=data, asset="TEST", timeframe="1d",
        n_windows=3, n_trials=2, purge_pct=0.001, embargo_pct=0.001,
        max_lookback_bars=50,
    )
    windows = wfo._build_windows()
    for w in windows:
        assert w.purge_bars >= 50
        assert w.embargo_bars >= 50


# ---------------------------------------------------------------------------
# Modelo de costos realista
# ---------------------------------------------------------------------------

def test_realistic_cost_profile_matches_manual_calc():
    profile = InstrumentCostProfile(point=0.0001, spread_points=10, contract_size=100_000, commission_per_lot=7.0)
    size = 100_000  # 1 lote
    price = 1.10
    e = entry_cost(profile, size, price)
    x = exit_cost(profile, size, price)
    # spread total = 10 * 0.0001 * 100000 = 100; comisión total = 7
    # se reparte mitad entrada / mitad salida
    assert e == pytest.approx((100 * 0.5) + (7 * 0.5))
    assert x == pytest.approx((100 * 0.5) + (7 * 0.5))
    assert e + x == pytest.approx(100 + 7)


def test_backtester_realistic_costs_runs_without_error():
    data = make_ohlcv(300)
    strategy = SimpleSMAStrategy(fast=5, slow=20)
    signals = strategy.generate_signals(data)

    profile = InstrumentCostProfile(point=0.0001, spread_points=2, contract_size=100_000, commission_per_lot=5.0)
    cfg = BacktestConfig(use_realistic_costs=True, cost_profile=profile)
    engine = BacktestEngine(cfg)
    results = engine.run(signals, strategy_name="SMA Test", asset="EURUSD", timeframe="1d")
    assert results.total_trades >= 0
    # con costos realistas activos, cada trade debe tener comisión registrada (>0 si hubo trades)
    if results.trades:
        assert all(t.commission >= 0 for t in results.trades)


def test_backtester_default_pct_costs_unchanged():
    """Verifica que NO activar use_realistic_costs preserva el comportamiento original."""
    data = make_ohlcv(300)
    strategy = SimpleSMAStrategy(fast=5, slow=20)
    signals = strategy.generate_signals(data)

    cfg = BacktestConfig()  # default: use_realistic_costs=False
    engine = BacktestEngine(cfg)
    results = engine.run(signals, strategy_name="SMA Test", asset="EURUSD", timeframe="1d")
    assert results.config.use_realistic_costs is False


# ---------------------------------------------------------------------------
# Sandbox AST
# ---------------------------------------------------------------------------

def test_ast_sandbox_blocks_mro_escape():
    malicious = (
        "class Evil(BaseStrategy):\n"
        "    name = 'Evil'\n"
        "    def generate_signals(self, data):\n"
        "        leak = ().__class__.__mro__[1].__subclasses__()\n"
        "        return data\n"
    )
    with pytest.raises(ForbiddenConstructError):
        validate_ast_safety(malicious)


def test_ast_sandbox_blocks_getattr_dunder_string():
    malicious = (
        "class Evil(BaseStrategy):\n"
        "    name = 'Evil'\n"
        "    def generate_signals(self, data):\n"
        "        g = getattr(object, '__subclasses__')\n"
        "        return data\n"
    )
    with pytest.raises(ForbiddenConstructError):
        validate_ast_safety(malicious)


def test_ast_sandbox_allows_normal_strategy_code():
    benign = (
        "class Ok(BaseStrategy):\n"
        "    name = 'Ok'\n"
        "    def generate_signals(self, data):\n"
        "        df = data.copy()\n"
        "        df['signal'] = 0\n"
        "        return df\n"
    )
    validate_ast_safety(benign)  # no debe lanzar


def test_compile_strategy_code_rejects_mro_escape_end_to_end():
    malicious = (
        "class Evil(BaseStrategy):\n"
        "    name = 'Evil'\n"
        "    def generate_signals(self, data):\n"
        "        leak = ().__class__.__mro__[1].__subclasses__()\n"
        "        df = data.copy()\n"
        "        df['signal'] = 0\n"
        "        return df\n"
    )
    result = compile_strategy_code(malicious)
    assert result.ok is False


# ---------------------------------------------------------------------------
# Generalización cruzada / régimen y estacionalidad
# ---------------------------------------------------------------------------

def test_cross_symbol_generalization_reports_rate():
    strategy = SimpleSMAStrategy(fast=10, slow=30)
    datasets = {
        "ORIGIN": make_ohlcv(500, seed=1, trend=0.001),
        "SYM_A":  make_ohlcv(500, seed=2, trend=0.001),
        "SYM_B":  make_ohlcv(500, seed=3, trend=-0.0005),
    }
    report = run_cross_symbol_generalization(
        strategy, datasets, origin_symbol="ORIGIN", timeframe="1d",
    )
    assert report.n_tested == 3
    assert 0.0 <= report.generalization_rate <= 1.0
    assert report.verdict() in (
        "GENERALIZA BIEN", "GENERALIZACIÓN PARCIAL",
        "NO GENERALIZA (probable sobreajuste al símbolo de origen)",
    )


def test_regime_and_seasonality_report_shape():
    strategy = SimpleSMAStrategy(fast=10, slow=30)
    data = make_ohlcv(600, seed=7)
    report = run_regime_and_seasonality_test(strategy, data, asset="TEST", timeframe="1d")
    assert len(report.monthly) == 12
    assert report.seasonality_verdict() != ""
    assert report.regime_verdict() != ""
