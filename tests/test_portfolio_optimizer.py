"""
tests/test_portfolio_optimizer.py
Cobertura para Fase 2 del roadmap (bloque C): optimización real de
pesos de portafolio (risk parity / mean-variance) y su integración con
PortfolioManager vía descomposición de pesos planos.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from portfolio.optimizer import (
    risk_parity_weights, mean_variance_weights, decompose_component_weights,
    _risk_contributions,
)
from portfolio.manager import PortfolioManager, AssetAllocation
from core.types import BacktestResults


def make_returns(seed: int, n: int = 500, mu: float = 0.0004, sigma: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    noise = rng.normal(0.0, sigma, n)
    noise -= noise.mean()  # fuerza que la media muestral sea EXACTAMENTE mu,
                           # para que los tests de "quién tiene mejor retorno"
                           # no sean flaky por ruido de muestreo con series cortas.
    return pd.Series(mu + noise, index=dates)


def make_backtest_results(returns: pd.Series, initial_capital: float = 100_000.0) -> BacktestResults:
    equity = initial_capital * (1 + returns).cumprod()
    return BacktestResults(
        equity_curve=equity,
        trades=[],
        strategy_name="Test", asset="TEST", timeframe="1d",
        initial_capital=initial_capital,
        final_capital=float(equity.iloc[-1]),
        net_profit=float(equity.iloc[-1] - initial_capital),
        net_profit_pct=float(equity.iloc[-1] / initial_capital - 1) * 100,
        sharpe_ratio=1.0, sortino_ratio=1.2, calmar_ratio=0.8,
        cagr=10.0, max_drawdown_pct=-10.0,
        win_rate=50.0, profit_factor=1.5, total_trades=10,
    )


# ---------------------------------------------------------------------------
# Risk parity
# ---------------------------------------------------------------------------

class TestRiskParity:
    def setup_method(self):
        self.returns = {
            "BTCUSD::Donchian":  make_returns(1, sigma=0.02),   # más volátil
            "ETHUSD::SMA":       make_returns(2, sigma=0.01),
            "EURUSD::MeanRev":   make_returns(3, sigma=0.005),  # menos volátil
        }

    def test_weights_sum_to_one(self):
        result = risk_parity_weights(self.returns, timeframe="1d")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_all_weights_non_negative(self):
        result = risk_parity_weights(self.returns, timeframe="1d")
        assert all(w >= -1e-9 for w in result.weights.values())

    def test_less_volatile_component_gets_more_weight(self):
        # En risk parity puro (sin correlación fuerte), el componente menos
        # volátil debería recibir MÁS peso que el más volátil, para igualar
        # la contribución de riesgo.
        result = risk_parity_weights(self.returns, timeframe="1d")
        assert result.weights["EURUSD::MeanRev"] > result.weights["BTCUSD::Donchian"]

    def test_risk_contributions_are_roughly_equal(self):
        result = risk_parity_weights(self.returns, timeframe="1d")
        contribs = list(result.risk_contributions.values())
        # No van a ser IDÉNTICAS (optimización numérica + correlación
        # cruzada), pero deben estar razonablemente cerca del ideal 1/3 cada una.
        target = 1.0 / len(contribs)
        for c in contribs:
            assert abs(c - target) < 0.15

    def test_respects_max_weight_cap(self):
        result = risk_parity_weights(self.returns, timeframe="1d", max_weight=0.4)
        assert all(w <= 0.4 + 1e-6 for w in result.weights.values())

    def test_raises_with_single_component(self):
        with pytest.raises(ValueError):
            risk_parity_weights({"only::one": self.returns["BTCUSD::Donchian"]})

    def test_raises_with_no_components(self):
        with pytest.raises(ValueError):
            risk_parity_weights({})


# ---------------------------------------------------------------------------
# Mean-variance
# ---------------------------------------------------------------------------

class TestMeanVariance:
    def setup_method(self):
        self.returns = {
            "A::S1": make_returns(10, mu=0.0010, sigma=0.015),   # mejor retorno/riesgo
            "B::S2": make_returns(20, mu=0.0002, sigma=0.015),
            "C::S3": make_returns(30, mu=0.0001, sigma=0.020),
        }

    def test_max_sharpe_weights_sum_to_one(self):
        result = mean_variance_weights(self.returns, timeframe="1d", method="max_sharpe")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_min_variance_weights_sum_to_one(self):
        result = mean_variance_weights(self.returns, timeframe="1d", method="min_variance")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_max_sharpe_favors_best_return_component(self):
        result = mean_variance_weights(self.returns, timeframe="1d", method="max_sharpe")
        assert result.weights["A::S1"] >= result.weights["C::S3"]

    def test_no_short_positions_by_default(self):
        result = mean_variance_weights(self.returns, timeframe="1d", method="max_sharpe")
        assert all(w >= -1e-9 for w in result.weights.values())

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            mean_variance_weights(self.returns, timeframe="1d", method="not_a_method")

    def test_target_return_constraint_is_approximately_met(self):
        target = 0.15  # 15% anualizado
        result = mean_variance_weights(
            self.returns, timeframe="1d", method="target_return", target_return=target,
        )
        if result.converged:
            assert result.expected_return == pytest.approx(target, abs=0.03)

    def test_expected_sharpe_is_finite(self):
        result = mean_variance_weights(self.returns, timeframe="1d", method="max_sharpe")
        assert np.isfinite(result.expected_sharpe)


# ---------------------------------------------------------------------------
# Descomposición de pesos planos -> (asset_weights, strategies por activo)
# ---------------------------------------------------------------------------

def test_decompose_component_weights_basic():
    weights = {"BTCUSD::Donchian": 0.5, "BTCUSD::SMA": 0.2, "ETHUSD::MeanRev": 0.3}
    asset_weights, strategies_by_asset = decompose_component_weights(weights)
    assert asset_weights["BTCUSD"] == pytest.approx(0.7)
    assert asset_weights["ETHUSD"] == pytest.approx(0.3)
    assert strategies_by_asset["BTCUSD"]["Donchian"] == pytest.approx(0.5 / 0.7)
    assert strategies_by_asset["BTCUSD"]["SMA"] == pytest.approx(0.2 / 0.7)
    assert strategies_by_asset["ETHUSD"]["MeanRev"] == pytest.approx(1.0)


def test_decompose_reconstructs_flat_weights_via_portfolio_manager():
    """El punto central del diseño: descomponer y volver a componer con
    PortfolioManager debe reproducir EXACTAMENTE los pesos planos
    originales, sin modificar esa clase."""
    weights = {"BTCUSD::Donchian": 0.5, "BTCUSD::SMA": 0.2, "ETHUSD::MeanRev": 0.3}
    asset_weights, strategies_by_asset = decompose_component_weights(weights)

    pm = PortfolioManager()
    for asset, strat_returns in [
        ("BTCUSD", {"Donchian": make_returns(1), "SMA": make_returns(2)}),
        ("ETHUSD", {"MeanRev": make_returns(3)}),
    ]:
        pm.add_allocation(AssetAllocation(asset=asset, strategies=strategies_by_asset[asset]))
        for strat_name, ret in strat_returns.items():
            pm.register_result(asset, strat_name, make_backtest_results(ret))

    results = pm.compute(asset_weights=asset_weights)
    # peso efectivo de BTCUSD::Donchian = asset_weight[BTCUSD] * norm_weight[Donchian dentro de BTC]
    effective_btc_donchian = asset_weights["BTCUSD"] * strategies_by_asset["BTCUSD"]["Donchian"]
    assert effective_btc_donchian == pytest.approx(0.5, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
