"""
tests/test_engine.py
Tests básicos del motor de backtesting.

Verifican que las métricas sean correctas y consistentes
antes y después de la refactorización.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.types import BacktestConfig
from core.metrics import (
    cagr, sharpe_ratio, sortino_ratio, profit_factor,
    max_streak, compute_drawdown, deflated_sharpe_ratio, expected_max_sharpe_null,
)
from engine.backtester import BacktestEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Genera datos OHLCV sintéticos para tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    high  = close * (1 + rng.uniform(0, 0.02, n))
    low   = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol   = rng.uniform(1000, 5000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


def make_signals(df: pd.DataFrame, every: int = 20) -> pd.DataFrame:
    """Genera señales alternadas simples cada N barras."""
    df = df.copy()
    df["signal"] = 0
    df["stop_loss"] = np.nan
    df["take_profit"] = np.nan
    direction = 1
    for i in range(0, len(df), every):
        df.iloc[i, df.columns.get_loc("signal")] = direction
        df.iloc[i, df.columns.get_loc("stop_loss")] = df.iloc[i]["close"] * (
            0.97 if direction == 1 else 1.03
        )
        df.iloc[i, df.columns.get_loc("take_profit")] = df.iloc[i]["close"] * (
            1.06 if direction == 1 else 0.94
        )
        direction *= -1
    return df


# ---------------------------------------------------------------------------
# Tests de métricas puras
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_cagr_positive(self):
        # cagr() usa 365.25 dias/anio (correcto, contempla anios bisiestos),
        # por lo que 365 dias no son exactamente 1 anio -> ~50.04%, no 50.0%.
        result = cagr(100_000, 150_000, 365)
        assert abs(result - 50.0) < 0.1

    def test_cagr_negative(self):
        result = cagr(100_000, 80_000, 365)
        assert result < 0

    def test_sharpe_flat_returns(self):
        """Retornos constantes = Sharpe infinito o muy alto."""
        returns = pd.Series([0.001] * 252)
        sr = sharpe_ratio(returns, "1d")
        assert sr > 5

    def test_sharpe_zero_std(self):
        """Si no hay volatilidad, sharpe debe manejar división por cero."""
        returns = pd.Series([0.0] * 100)
        sr = sharpe_ratio(returns, "1d")
        assert sr == 0.0

    def test_sortino_no_losses(self):
        """Sin pérdidas, sortino debe ser 0 (sin downside)."""
        returns = pd.Series([0.001] * 50)
        sr = sortino_ratio(returns, "1d")
        assert sr == 0.0  # sin downside std

    def test_profit_factor_all_wins(self):
        pnls = [100.0, 200.0, 50.0]
        pf = profit_factor(pnls)
        assert pf == float("inf")

    def test_profit_factor_all_losses(self):
        pnls = [-100.0, -50.0]
        pf = profit_factor(pnls)
        assert pf == 0.0

    def test_profit_factor_mixed(self):
        pnls = [100.0, -50.0]
        pf = profit_factor(pnls)
        assert abs(pf - 2.0) < 0.001

    def test_max_streak_wins(self):
        pnls = [10, 20, -5, 15, 30, 25, -10]
        assert max_streak(pnls, win=True) == 3

    def test_max_streak_losses(self):
        pnls = [10, -5, -10, -15, 20, -5]
        assert max_streak(pnls, win=False) == 3

    def test_drawdown_series(self):
        equity = pd.Series([100, 110, 105, 95, 100, 108])
        dd_series, dd_abs, dd_pct = compute_drawdown(equity)
        assert dd_pct < 0
        assert dd_abs < 0


class TestDeflatedSharpe:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) — corrige el
    Sharpe reportado por el numero de combinaciones que el genetico probo."""

    def test_expected_max_sharpe_grows_with_more_trials(self):
        """A mas intentos, el 'liston' que debe superar un Sharpe para no
        ser explicable por azar sube — mas tests, mas facil encontrar un
        Sharpe alto por pura suerte."""
        low_n = expected_max_sharpe_null(n_trials=10, sharpe_std=0.5)
        high_n = expected_max_sharpe_null(n_trials=10_000, sharpe_std=0.5)
        assert high_n > low_n > 0

    def test_expected_max_sharpe_zero_with_one_trial(self):
        assert expected_max_sharpe_null(n_trials=1, sharpe_std=0.5) == 0.0

    def test_dsr_high_with_few_trials_and_strong_sharpe(self):
        """Un Sharpe muy por encima del listón, con pocos intentos detrás,
        debe dar un DSR alto (creible, no explicable por suerte)."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=0.15, sharpe_std_across_trials=0.02,
            n_trials=5, n_periods=2000, skew=0.0, kurtosis=3.0,
        )
        assert dsr > 0.9

    def test_dsr_low_with_many_trials_and_marginal_sharpe(self):
        """El mismo Sharpe, pero encontrado tras probar decenas de miles de
        combinaciones con alta dispersion entre ellas, debe dar un DSR bajo
        — es exactamente el escenario que produce el genetico si no se
        corrige por tests multiples."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=0.15, sharpe_std_across_trials=0.5,
            n_trials=50_000, n_periods=2000, skew=0.0, kurtosis=3.0,
        )
        assert dsr < 0.5

    def test_dsr_in_valid_probability_range(self):
        for sr in (-1.0, 0.0, 0.5, 2.0, 5.0):
            dsr = deflated_sharpe_ratio(sr, 0.3, 1000, 500)
            assert 0.0 <= dsr <= 1.0


class TestWalkForwardFolds:
    """_fold_boundaries: particion secuencial usada por el genetico para
    validar cada regla en multiples ventanas temporales en vez de un unico
    split train/holdout."""

    def test_folds_cover_full_range_contiguously(self):
        from discovery.genetic_discovery import _fold_boundaries
        n_bars, n_folds, warmup = 1000, 4, 50
        bounds = _fold_boundaries(n_bars, n_folds, warmup)
        assert len(bounds) == n_folds
        assert bounds[0][1] == 0
        assert bounds[-1][2] == n_bars
        for k in range(1, n_folds):
            assert bounds[k][1] == bounds[k - 1][2]  # sin huecos ni solapes

    def test_each_fold_borrows_warmup_from_before_its_segment(self):
        from discovery.genetic_discovery import _fold_boundaries
        bounds = _fold_boundaries(1000, 4, 50)
        for data_start, seg_start, seg_end in bounds[1:]:
            assert seg_start - data_start == 50

    def test_single_fold_degenerates_to_whole_range(self):
        from discovery.genetic_discovery import _fold_boundaries
        bounds = _fold_boundaries(500, 1, 50)
        assert len(bounds) == 1
        assert bounds[0] == (0, 0, 500)


# ---------------------------------------------------------------------------
# Tests del motor de backtesting
# ---------------------------------------------------------------------------

class TestBacktestEngine:

    def setup_method(self):
        self.df = make_ohlcv(300)
        self.signals = make_signals(self.df)
        self.config = BacktestConfig(initial_capital=100_000, commission=0.001)
        self.engine = BacktestEngine(self.config)

    def test_run_returns_results(self):
        results = self.engine.run(self.signals, asset="TEST", timeframe="1d")
        assert results is not None
        assert results.total_trades >= 0

    def test_equity_curve_length(self):
        results = self.engine.run(self.signals, asset="TEST", timeframe="1d")
        assert len(results.equity_curve) == len(self.signals)

    def test_initial_capital_preserved_no_trades(self):
        """Si no hay señales, el capital no cambia."""
        df_no_signals = self.df.copy()
        df_no_signals["signal"] = 0
        results = self.engine.run(df_no_signals, asset="TEST", timeframe="1d")
        assert results.total_trades == 0
        assert abs(results.final_capital - self.config.initial_capital) < 1.0

    def test_win_rate_in_range(self):
        results = self.engine.run(self.signals, asset="TEST", timeframe="1d")
        if results.total_trades > 0:
            assert 0.0 <= results.win_rate <= 100.0

    def test_sharpe_is_float(self):
        results = self.engine.run(self.signals, asset="TEST", timeframe="1d")
        assert isinstance(results.sharpe_ratio, float)
        assert not np.isnan(results.sharpe_ratio)

    def test_trades_df_columns(self):
        results = self.engine.run(self.signals, asset="TEST", timeframe="1d")
        if results.total_trades > 0:
            assert "net_pnl" in results.trades_df.columns
            assert "entry_date" in results.trades_df.columns
            assert "exit_date" in results.trades_df.columns

    def test_max_drawdown_negative_or_zero(self):
        results = self.engine.run(self.signals, asset="TEST", timeframe="1d")
        assert results.max_drawdown_pct <= 0

    def test_no_short_when_disabled(self):
        config = BacktestConfig(initial_capital=100_000, allow_short=False)
        engine = BacktestEngine(config)
        results = engine.run(self.signals, asset="TEST", timeframe="1d")
        if results.total_trades > 0:
            directions = results.trades_df.get("direction", pd.Series(["Long"]))
            assert (directions == "Short").sum() == 0


# ---------------------------------------------------------------------------
# Tests de estrategias
# ---------------------------------------------------------------------------

class TestStrategies:

    def test_all_strategies_discoverable(self):
        from strategies import STRATEGY_REGISTRY
        assert len(STRATEGY_REGISTRY) > 0

    def test_each_strategy_generates_signals(self):
        from strategies import STRATEGY_REGISTRY
        df = make_ohlcv(300)
        for name, cls in STRATEGY_REGISTRY.items():
            strategy = cls()
            result = strategy.generate_signals(df)
            assert "signal" in result.columns, f"{name} no genera columna 'signal'"
            assert result["signal"].isin([-1, 0, 1]).all(), f"{name} tiene señales inválidas"

    def test_each_strategy_has_param_space(self):
        from strategies import STRATEGY_REGISTRY
        for name, cls in STRATEGY_REGISTRY.items():
            space = cls().get_param_space()
            assert isinstance(space, dict), f"{name} get_param_space() no retorna dict"


# ---------------------------------------------------------------------------
# Consistencia entre motores: engine/backtester.py (real) vs.
# discovery/backtest_engine.py (simulador rapido del genetico).
#
# El genetico NO usa engine/backtester.py durante la evolucion (por
# velocidad), sino su propio simulador. Si ambos motores no comparten la
# misma semantica de ejecucion (rezago de 1 vela, sizing, y sobre todo la
# prohibicion de reabrir una posicion en la misma vela en que se cerro por
# SL/TP), el fitness que guia la busqueda evolutiva deja de predecir lo que
# el Hall of Fame obtendra al verificarse con el motor real — exactamente el
# bug que motivo este test (ver discovery/backtest_engine.py, seccion sobre
# bloqueo de reentrada intravela).
# ---------------------------------------------------------------------------

class TestEngineConsistency:

    def _sl_reentry_scenario(self):
        """Construye una serie de precios donde una entrada long dispara el
        SL casi de inmediato y, en la MISMA vela del SL, la señal ya apunta
        de nuevo a largo — el escenario exacto que distingue 'reabre en la
        misma vela' (bug) de 'reabre recien en la siguiente vela' (correcto)."""
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        price = np.full(n, 100.0)
        df = pd.DataFrame({
            "open": price.copy(), "high": price.copy(),
            "low": price.copy(), "close": price.copy(),
            "volume": np.full(n, 1000.0),
        }, index=idx)

        # Vela 10: entra long a 100 (open). SL fijado a 2 ATR de distancia.
        # Vela 10 tambien perfora hacia abajo lo suficiente para tocar el SL.
        # La señal en la vela 10 (que decide la entrada de la vela 11) sigue
        # en largo, para forzar el intento de reentrada inmediata.
        df.loc[idx[9], ["open", "high", "low", "close"]] = [100, 100.5, 99.5, 100]
        df.loc[idx[10], ["open", "high", "low", "close"]] = [100, 100.2, 90.0, 91.0]
        df.loc[idx[11], ["open", "high", "low", "close"]] = [91.5, 92, 91, 91.5]
        for j in range(12, n):
            df.iloc[j] = df.iloc[11]
        return df

    def test_no_same_bar_reentry_after_stop_real_engine(self):
        """engine/backtester.py: un cierre por SL/TP no puede reabrir en la
        misma vela."""
        df = self._sl_reentry_scenario()
        sig = pd.Series(0, index=df.index)
        sig.iloc[9] = 1
        sig.iloc[10] = 1  # señal sigue larga en la vela que dispara el SL

        exec_df = df.copy()
        exec_df["signal"] = sig
        atr_guess = 2.0
        exec_df["stop_loss"] = df["close"].iloc[9] - atr_guess
        exec_df["take_profit"] = df["close"].iloc[9] + 20

        config = BacktestConfig(initial_capital=100_000, commission=0.0, slippage=0.0)
        engine = BacktestEngine(config)
        results = engine.run(exec_df, asset="TEST", timeframe="1h")

        # La entrada original (vela 10) y una eventual reentrada NO pueden
        # compartir la misma vela de entrada con la vela de salida del trade
        # anterior.
        entry_dates = [t.entry_date for t in results.trades]
        exit_dates = [t.exit_date for t in results.trades]
        for exit_dt in exit_dates:
            assert exit_dt not in entry_dates or entry_dates.index(exit_dt) == exit_dates.index(exit_dt), (
                "Se reabrio una posicion en la misma vela en que se cerro por SL/TP"
            )

    def test_fast_ga_simulator_matches_real_engine_reentry_rule(self):
        """discovery/backtest_engine.py debe compartir la misma regla de
        no-reentrada-en-la-misma-vela que engine/backtester.py. Si alguien
        vuelve a divergir los dos motores, este test debe fallar."""
        from discovery.backtest_engine import run_backtest_signal

        df = self._sl_reentry_scenario()
        sig = pd.Series(0, index=df.index)
        sig.iloc[9] = 1
        sig.iloc[10] = 1

        result = run_backtest_signal(
            "t1", sig, df, stop_loss_atr=1.0, take_profit_atr=10.0,
            atr_period=5, warmup=9, timeframe="H1",
            initial_capital=100_000, commission=0.0, slippage=0.0,
        )

        sl_exits = [t for t in result.trades if t.exit_reason == "sl"]
        for sl_trade in sl_exits:
            reentries_same_bar = [
                t for t in result.trades
                if t.entry_time == sl_trade.exit_time and t is not sl_trade
            ]
            assert not reentries_same_bar, (
                "El simulador rapido del genetico reabrio una posicion en la "
                "misma vela en que otra se cerro por SL/TP (deberia esperar a "
                "la siguiente vela, igual que engine/backtester.py)"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
