"""
tests/test_discovery_generalization_pipeline.py
Cubre el camino que usa el botón "Validar generalización" agregado a
ui/views/discovery.py (Roadmap Fase 1, punto A.2 del prompt de
continuación): una definición de regla del Hall of Fame se traduce a
código real vía discovery/codegen.py y se compila EN MEMORIA a una clase
BaseStrategy real vía discovery/codegen.py::compile_definition_to_class
(NO vía strategies/code_compiler.py — ese sandbox es para texto libre
tecleado por un humano en el Constructor y rechaza cualquier `import`,
mientras que el código de codegen.py siempre trae `import pandas as pd` /
`import numpy as np` de cabecera; es el mismo nivel de confianza que ya
usan save_user_strategy_file + refresh_registry para persistir a disco).
Esa instancia se corre a través de discovery/generalization.py.

Antes de esta sesión, discovery/generalization.py solo se probaba con una
BaseStrategy escrita a mano en el test (SimpleSMAStrategy en
test_hardening.py). Este archivo agrega cobertura para el camino real que
usa la UI: una definición de regla generada por el motor genético.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.types import BacktestConfig
from discovery.codegen import compile_definition_to_class
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


def _rsi_reversion_definition() -> dict:
    """Definición mínima válida, con la misma forma que produce
    genetic_discovery.individual_to_definition — condiciones reales de
    CONDITION_LIBRARY, no inventadas."""
    return {
        "long_rule": [["rsi14_below_30"]],
        "short_rule": [["rsi14_above_70"]],
        "stop_loss_atr": 2.0,
        "take_profit_atr": 3.0,
        "atr_period": 14,
        "families_used": ["RSI"],
        "long_rule_text": "RSI(14) < 30",
        "short_rule_text": "RSI(14) > 70",
    }


def _compile_definition(class_name: str = "TestGenStrategy"):
    strategy_class = compile_definition_to_class(
        _rsi_reversion_definition(), class_name=class_name,
        symbol="EURUSD", timeframe="1d",
    )
    assert strategy_class is not None
    return strategy_class


class TestCodegenCompilesToRealStrategy:
    def test_generated_code_compiles_ok(self):
        strategy_class = _compile_definition()
        assert strategy_class is not None

    def test_generated_strategy_produces_signals(self):
        strategy_class = _compile_definition()
        strategy = strategy_class()
        df = make_ohlcv(300)
        signals = strategy.generate_signals(df)
        assert "signal" in signals.columns
        assert set(signals["signal"].unique()).issubset({-1, 0, 1})

    def test_generated_strategy_class_name_matches(self):
        strategy_class = _compile_definition()
        assert strategy_class.__name__ == "TestGenStrategy"


class TestGeneralizationPipelineEndToEnd:
    """El mismo camino que dispara el botón '🧪 Validar generalización'."""

    def setup_method(self):
        self.strategy_class = _compile_definition()
        self.strategy = self.strategy_class()
        self.datasets = {
            "EURUSD": make_ohlcv(500, seed=1),
            "GBPUSD": make_ohlcv(500, seed=2),
            "USDJPY": make_ohlcv(500, seed=3),
        }
        self.config = BacktestConfig(initial_capital=100_000.0)

    def test_cross_symbol_generalization_report_shape(self):
        report = run_cross_symbol_generalization(
            self.strategy, self.datasets, origin_symbol="EURUSD",
            config=self.config, timeframe="1d",
        )
        assert report.n_tested == 3
        assert 0.0 <= report.generalization_rate <= 1.0
        assert report.verdict() in (
            "GENERALIZA BIEN", "GENERALIZACIÓN PARCIAL",
            "NO GENERALIZA (probable sobreajuste al símbolo de origen)",
        )

    def test_regime_and_seasonality_report_shape(self):
        report = run_regime_and_seasonality_test(
            self.strategy, self.datasets["EURUSD"], config=self.config,
            asset="EURUSD", timeframe="1d",
        )
        assert len(report.monthly) == 12
        assert len(report.regimes) <= 2
        assert report.seasonality_verdict() != ""
        assert report.regime_verdict() != ""

    def test_generalization_handles_missing_symbol_gracefully(self):
        # Un dataset vacío (símbolo sin datos) no debe tumbar el reporte,
        # solo marcarse como no-ok en esa fila.
        datasets = dict(self.datasets)
        datasets["XAUUSD"] = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        report = run_cross_symbol_generalization(
            self.strategy, datasets, origin_symbol="EURUSD",
            config=self.config, timeframe="1d",
        )
        xau_result = [r for r in report.results if r.symbol == "XAUUSD"][0]
        assert xau_result.ok is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
