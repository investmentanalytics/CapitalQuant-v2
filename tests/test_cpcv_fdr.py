"""
tests/test_cpcv_fdr.py
Cobertura para el cierre de Fase 1 (bloque B del roadmap):
- FDR / Benjamini-Hochberg (core/multiple_testing.py)
- CPCV completo + PBO (optimization/cpcv.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.multiple_testing import benjamini_hochberg, dsr_to_pvalue
from core.types import BacktestConfig
from strategies.base import BaseStrategy
from optimization.cpcv import (
    build_cpcv_splits, generate_candidate_params, CPCVEngine, _subtract_ranges, _merge_ranges,
)


def make_ohlcv(n: int = 1200, seed: int = 42, trend: float = 0.001) -> pd.DataFrame:
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
    """Estrategia mínima de cruce de medias, usada solo para tests (misma
    que tests/test_hardening.py, duplicada a propósito para no acoplar
    archivos de test entre sí)."""
    name = "SMA Test CPCV"

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
# Benjamini-Hochberg / FDR
# ---------------------------------------------------------------------------

def test_bh_known_example_from_literature():
    # Ejemplo clásico: 5 p-valores, alpha=0.05.
    # p = [0.01, 0.02, 0.03, 0.04, 0.30] -> thresholds i/5*0.05 = [.01,.02,.03,.04,.05]
    # los primeros 4 (0.01..0.04) cumplen p_(i) <= threshold_i, el 5to no -> los 4 primeros significativos
    p_values = [0.04, 0.01, 0.30, 0.03, 0.02]
    result = benjamini_hochberg(p_values, alpha=0.05)
    assert result.n_tested == 5
    assert result.n_significant == 4
    # El valor grande (0.30, índice 2) nunca debe ser significativo
    assert result.rejected[2] is False
    # Los 4 chicos sí
    for i in (0, 1, 3, 4):
        assert result.rejected[i] is True


def test_bh_all_high_pvalues_reject_none():
    p_values = [0.9, 0.8, 0.95, 0.99]
    result = benjamini_hochberg(p_values, alpha=0.10)
    assert result.n_significant == 0
    assert all(r is False for r in result.rejected)


def test_bh_all_tiny_pvalues_accept_all():
    p_values = [0.0001, 0.0002, 0.0003]
    result = benjamini_hochberg(p_values, alpha=0.10)
    assert result.n_significant == 3


def test_bh_handles_none_entries_without_breaking():
    p_values = [0.01, None, 0.02, None, 0.5]
    result = benjamini_hochberg(p_values, alpha=0.10)
    assert result.n_tested == 3
    # Las posiciones None nunca se marcan significativas
    assert result.rejected[1] is False
    assert result.rejected[3] is False
    assert result.adjusted_p_values[1] == 1.0


def test_bh_empty_input():
    result = benjamini_hochberg([], alpha=0.10)
    assert result.n_tested == 0
    assert result.n_significant == 0


def test_bh_adjusted_pvalues_are_monotonic_with_sorted_input():
    p_values = [0.001, 0.2, 0.01, 0.4, 0.03]
    result = benjamini_hochberg(p_values, alpha=0.10)
    order = np.argsort(p_values)
    sorted_adjusted = [result.adjusted_p_values[i] for i in order]
    # Los ajustados en orden de p-valor creciente nunca deben decrecer
    for a, b in zip(sorted_adjusted, sorted_adjusted[1:]):
        assert b >= a - 1e-9


def test_dsr_to_pvalue_conversion():
    assert dsr_to_pvalue(0.95) == pytest.approx(0.05)
    assert dsr_to_pvalue(1.0) == pytest.approx(0.0)
    assert dsr_to_pvalue(0.0) == pytest.approx(1.0)
    assert dsr_to_pvalue(None) is None


# ---------------------------------------------------------------------------
# CPCV — construcción de particiones (purga/embargo, sin fuga)
# ---------------------------------------------------------------------------

def test_subtract_ranges_removes_middle_chunk():
    pieces = _subtract_ranges((0, 100), [(40, 60)])
    assert pieces == [(0, 40), (60, 100)]


def test_subtract_ranges_removes_full_overlap():
    pieces = _subtract_ranges((0, 100), [(0, 100)])
    assert pieces == []


def test_merge_ranges_combines_overlapping():
    merged = _merge_ranges([(0, 10), (5, 15), (20, 30)])
    assert merged == [(0, 15), (20, 30)]


def test_cpcv_splits_count_matches_combinatorics():
    from math import comb
    splits = build_cpcv_splits(n_bars=1200, n_groups=6, n_test_groups=2, purge_bars=5, embargo_bars=5)
    assert len(splits) == comb(6, 2) == 15


def test_cpcv_train_and_test_never_overlap():
    splits = build_cpcv_splits(n_bars=1200, n_groups=6, n_test_groups=2, purge_bars=10, embargo_bars=10)
    for split in splits:
        train_idx = set()
        for s, e in split.train_start_idxs:
            train_idx.update(range(s, e))
        test_idx = set()
        for s, e in split.test_start_idxs:
            test_idx.update(range(s, e))
        assert train_idx.isdisjoint(test_idx)


def test_cpcv_purge_embargo_creates_gap_around_test_blocks():
    purge, embargo = 15, 20
    splits = build_cpcv_splits(n_bars=1200, n_groups=6, n_test_groups=1, purge_bars=purge, embargo_bars=embargo)
    for split in splits:
        test_start, test_end = split.test_start_idxs[0]
        for tr_start, tr_end in split.train_start_idxs:
            # Ningún tramo de train puede caer dentro de [test_start-purge, test_end+embargo)
            assert tr_end <= test_start - purge or tr_start >= test_end + embargo


def test_cpcv_splits_raises_nothing_with_minimal_groups():
    splits = build_cpcv_splits(n_bars=500, n_groups=4, n_test_groups=1, purge_bars=5, embargo_bars=5)
    assert len(splits) == 4


# ---------------------------------------------------------------------------
# CPCV — generación de candidatos
# ---------------------------------------------------------------------------

def test_generate_candidate_params_respects_bounds():
    candidates = generate_candidate_params(SimpleSMAStrategy, n_candidates=15, seed=7)
    assert len(candidates) == 15
    for c in candidates:
        assert 5 <= c["fast"] <= 20
        assert 20 <= c["slow"] <= 60


def test_generate_candidate_params_reproducible_with_same_seed():
    a = generate_candidate_params(SimpleSMAStrategy, n_candidates=10, seed=99)
    b = generate_candidate_params(SimpleSMAStrategy, n_candidates=10, seed=99)
    assert a == b


# ---------------------------------------------------------------------------
# CPCV — motor completo end-to-end
# ---------------------------------------------------------------------------

class TestCPCVEngineEndToEnd:
    def setup_method(self):
        self.data = make_ohlcv(1200, seed=11)
        self.config = BacktestConfig(initial_capital=100_000.0)

    def test_cpcv_run_produces_valid_pbo(self):
        engine = CPCVEngine(
            strategy_class=SimpleSMAStrategy, data=self.data, config=self.config,
            asset="TEST", timeframe="1d", n_groups=6, n_test_groups=2,
        )
        results = engine.run(objective="sharpe", n_candidates=8, seed=3)
        assert 0.0 <= results.pbo <= 1.0
        assert results.n_splits == 15
        assert len(results.per_result) > 0
        assert results.best_candidate_by_is is not None
        assert results.best_candidate_by_oos is not None

    def test_cpcv_summary_and_verdict_are_strings(self):
        engine = CPCVEngine(
            strategy_class=SimpleSMAStrategy, data=self.data, config=self.config,
            asset="TEST", timeframe="1d", n_groups=6, n_test_groups=2,
        )
        results = engine.run(objective="sharpe", n_candidates=6, seed=1)
        assert isinstance(results.summary(), str)
        assert results.verdict() in (
            "ALTO RIESGO DE SOBREAJUSTE", "RIESGO MODERADO", "BAJO RIESGO DE SOBREAJUSTE",
        )

    def test_cpcv_rejects_invalid_objective(self):
        engine = CPCVEngine(
            strategy_class=SimpleSMAStrategy, data=self.data, config=self.config,
            asset="TEST", timeframe="1d",
        )
        with pytest.raises(ValueError):
            engine.run(objective="no_existe")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
