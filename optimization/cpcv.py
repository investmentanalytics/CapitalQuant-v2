"""
optimization/cpcv.py
Combinatorial Purged Cross-Validation (CPCV) + Probabilidad de
Sobreajuste de Backtest (PBO).

Referencias:
- López de Prado, M. (2018), *Advances in Financial Machine Learning*,
  cap. 11-12 (Cross-Validation en Finanzas / CPCV).
- Bailey, D.H., Borwein, J., López de Prado, M. & Zhu, Q.J. (2016),
  "The Probability of Backtest Overfitting", Journal of Computational
  Finance.

Por qué esto es distinto de `optimization/walk_forward.py`
------------------------------------------------------------
`WalkForwardOptimizer` ya implementa purga/embargo, pero solo sobre
ventanas SECUENCIALES: un único camino train→test que avanza en el
tiempo. Esto da una sola estimación out-of-sample por ventana — útil,
pero no permite estimar cuán *probable* es que el proceso de selección
de parámetros esté sobreajustando, porque solo hay un tren/test por
ventana y no hay forma de comparar "qué tan bien generaliza el mejor
candidato in-sample" contra la distribución completa de resultados
out-of-sample posibles.

CPCV resuelve esto dividiendo los datos en N grupos y evaluando TODAS
las combinaciones posibles de k grupos como test (los N-k restantes
como train, con purga/embargo en cada frontera) — C(N,k) particiones en
vez de una sola. Sobre cada partición se evalúa el MISMO conjunto fijo
de configuraciones candidatas (no se re-optimiza con una búsqueda
distinta por partición, a propósito: así el candidato "mejor in-sample"
de cada partición es comparable entre particiones), lo que permite
calcular la Probabilidad de Sobreajuste de Backtest (PBO): la fracción
de particiones en las que el candidato que mejor rindió in-sample
resultó estar por DEBAJO de la mediana out-of-sample — evidencia directa
de que elegir "el mejor in-sample" no habría generalizado.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from typing import Type, List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy
from core.types import BacktestConfig, BacktestResults
from engine.backtester import BacktestEngine
from optimization.objectives import OBJECTIVES


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class CPCVSplit:
    """Una partición combinatoria: qué grupos son test, con sus rangos."""
    split_id: int
    test_groups: Tuple[int, ...]
    train_start_idxs: List[Tuple[int, int]] = field(default_factory=list)  # [(start,end)] en posiciones
    test_start_idxs: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class CPCVCandidateResult:
    """Resultado de un candidato de parámetros sobre una partición."""
    split_id: int
    candidate_id: int
    params: dict
    is_score: float     # in-sample (train)
    oos_score: float     # out-of-sample (test)


@dataclass
class CPCVResults:
    """Resultado completo de la corrida CPCV."""
    n_groups: int = 0
    n_test_groups: int = 0
    n_splits: int = 0
    n_candidates: int = 0
    objective: str = "sharpe"

    candidate_params: List[dict] = field(default_factory=list)
    per_result: List[CPCVCandidateResult] = field(default_factory=list)

    # Probabilidad de Sobreajuste de Backtest (Bailey et al., 2016)
    pbo: float = 0.0
    logits: List[float] = field(default_factory=list)

    # Distribución OOS del candidato elegido "mejor in-sample" en cada partición
    oos_scores_of_is_best: List[float] = field(default_factory=list)

    # Mejor candidato global por criterio (para comparar cuánto difieren)
    best_candidate_by_is: Optional[dict] = None
    best_candidate_by_oos: Optional[dict] = None
    mean_oos_by_candidate: Dict[int, float] = field(default_factory=dict)

    def verdict(self) -> str:
        if self.pbo >= 0.5:
            return "ALTO RIESGO DE SOBREAJUSTE"
        elif self.pbo >= 0.25:
            return "RIESGO MODERADO"
        else:
            return "BAJO RIESGO DE SOBREAJUSTE"

    def summary(self) -> str:
        return (
            f"PBO={self.pbo*100:.1f}% ({self.verdict()}) sobre {self.n_splits} "
            f"particiones combinatorias ({self.n_groups} grupos, {self.n_test_groups} "
            f"de test por partición) y {self.n_candidates} configuraciones candidatas."
        )


# ---------------------------------------------------------------------------
# Constructor de particiones combinatorias
# ---------------------------------------------------------------------------

def _build_groups(n_bars: int, n_groups: int) -> List[Tuple[int, int]]:
    """Divide [0, n_bars) en n_groups bloques contiguos casi iguales.
    Devuelve lista de (start, end) en posiciones (end exclusivo)."""
    edges = np.linspace(0, n_bars, n_groups + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def build_cpcv_splits(
    n_bars: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> List[CPCVSplit]:
    """
    Construye todas las C(n_groups, n_test_groups) particiones combinatorias.

    Para cada partición: los grupos de test pueden ser NO contiguos entre
    sí. El train se arma con el resto de los grupos, EXCLUYENDO además una
    banda de `purge_bars` velas antes de cada bloque de test y
    `embargo_bars` velas después — en ambos lados, porque en CPCV un
    bloque de train puede quedar tanto antes como después de un bloque de
    test en el eje temporal (a diferencia de walk-forward secuencial,
    donde train siempre precede a test).
    """
    groups = _build_groups(n_bars, n_groups)
    splits: List[CPCVSplit] = []

    for split_id, test_group_ids in enumerate(combinations(range(n_groups), n_test_groups)):
        test_ranges = [groups[g] for g in test_group_ids]

        # Bandas de exclusión (purga + embargo) alrededor de cada bloque de test
        exclude_ranges = []
        for (t_start, t_end) in test_ranges:
            ex_start = max(0, t_start - purge_bars)
            ex_end = min(n_bars, t_end + embargo_bars)
            exclude_ranges.append((ex_start, ex_end))

        # Train = todos los grupos que no son de test, recortados contra
        # las bandas de exclusión de TODOS los bloques de test.
        train_ranges: List[Tuple[int, int]] = []
        for g_id, (g_start, g_end) in enumerate(groups):
            if g_id in test_group_ids:
                continue
            train_ranges.extend(_subtract_ranges((g_start, g_end), exclude_ranges))

        splits.append(CPCVSplit(
            split_id=split_id,
            test_groups=test_group_ids,
            train_start_idxs=_merge_ranges(train_ranges),
            test_start_idxs=_merge_ranges(test_ranges),
        ))

    return splits


def _subtract_ranges(rng: Tuple[int, int], exclude: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Resta una lista de rangos de exclusión de un rango base."""
    pieces = [rng]
    for (e_start, e_end) in exclude:
        new_pieces = []
        for (p_start, p_end) in pieces:
            if e_end <= p_start or e_start >= p_end:
                new_pieces.append((p_start, p_end))
                continue
            if e_start > p_start:
                new_pieces.append((p_start, e_start))
            if e_end < p_end:
                new_pieces.append((e_end, p_end))
        pieces = new_pieces
    return [p for p in pieces if p[1] > p[0]]


def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _slice_by_ranges(data: pd.DataFrame, ranges: List[Tuple[int, int]]) -> pd.DataFrame:
    if not ranges:
        return data.iloc[0:0]
    parts = [data.iloc[start:end] for start, end in ranges]
    return pd.concat(parts) if len(parts) > 1 else parts[0]


# ---------------------------------------------------------------------------
# Generación de candidatos (fijos, comparables entre todas las particiones)
# ---------------------------------------------------------------------------

def generate_candidate_params(
    strategy_class: Type[BaseStrategy],
    n_candidates: int = 20,
    seed: int = 42,
) -> List[dict]:
    """
    Genera un conjunto FIJO de configuraciones candidatas muestreando el
    espacio de parámetros de la estrategia, uniforme al azar. Se usa el
    MISMO conjunto en todas las particiones CPCV — es lo que hace que el
    PBO sea comparable entre particiones (Bailey et al., 2016): si cada
    partición usara candidatos distintos, "el mejor in-sample" de una
    partición no sería comparable con el de otra.
    """
    space = strategy_class().get_param_space()
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(n_candidates):
        params = {}
        for name, spec in space.items():
            ptype = spec[0]
            if ptype == "int":
                params[name] = int(rng.integers(spec[1], spec[2] + 1))
            elif ptype == "float":
                params[name] = float(rng.uniform(spec[1], spec[2]))
            elif ptype == "categorical":
                params[name] = rng.choice(list(spec[1]))
            elif ptype == "bool":
                params[name] = bool(rng.choice([True, False]))
        candidates.append(params)
    return candidates


# ---------------------------------------------------------------------------
# Motor CPCV
# ---------------------------------------------------------------------------

class CPCVEngine:
    """
    Ejecuta CPCV completo: para cada partición combinatoria (train/test con
    purga+embargo) evalúa el mismo pool fijo de candidatos, y calcula la
    Probabilidad de Sobreajuste de Backtest (PBO).

    Uso:
        engine = CPCVEngine(
            strategy_class=DonchianBreakout, data=df,
            asset="BTCUSD", timeframe="1d",
            n_groups=6, n_test_groups=2,
        )
        results = engine.run(objective="sharpe", n_candidates=20)
        print(results.summary())
    """

    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
        asset: str = "",
        timeframe: str = "",
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_pct: float = 0.02,
        embargo_pct: float = 0.02,
        max_lookback_bars: Optional[int] = None,
    ):
        self.strategy_class = strategy_class
        self.data = data.copy()
        self.config = config or BacktestConfig()
        self.asset = asset
        self.timeframe = timeframe
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.max_lookback_bars = max_lookback_bars
        self.engine = BacktestEngine(self.config)

    def run(
        self,
        objective: str = "sharpe",
        n_candidates: int = 20,
        seed: int = 42,
        min_bars_per_side: int = 50,
        progress_callback: Optional[callable] = None,
    ) -> CPCVResults:
        if objective not in OBJECTIVES:
            raise ValueError(f"Objetivo '{objective}' no válido. Disponibles: {list(OBJECTIVES.keys())}")
        obj_fn = OBJECTIVES[objective]

        n_bars = len(self.data)
        group_size = n_bars // self.n_groups
        purge_bars = int(round(group_size * self.purge_pct))
        embargo_bars = int(round(group_size * self.embargo_pct))
        if self.max_lookback_bars:
            purge_bars = max(purge_bars, self.max_lookback_bars)
            embargo_bars = max(embargo_bars, self.max_lookback_bars)

        splits = build_cpcv_splits(
            n_bars, self.n_groups, self.n_test_groups, purge_bars, embargo_bars,
        )
        if not splits:
            raise ValueError("No se pudieron construir particiones CPCV. Verifica n_groups/n_test_groups.")

        candidates = generate_candidate_params(self.strategy_class, n_candidates, seed)
        if not candidates or not candidates[0]:
            raise ValueError(
                f"{self.strategy_class.name} no tiene espacio de parámetros optimizable "
                "(get_param_space() vacío) — CPCV no aplica."
            )

        results = CPCVResults(
            n_groups=self.n_groups, n_test_groups=self.n_test_groups,
            n_splits=len(splits), n_candidates=len(candidates), objective=objective,
            candidate_params=candidates,
        )

        # Matriz [split][candidato] -> (is_score, oos_score)
        is_matrix = np.full((len(splits), len(candidates)), np.nan)
        oos_matrix = np.full((len(splits), len(candidates)), np.nan)

        for s_idx, split in enumerate(splits):
            train_data = _slice_by_ranges(self.data, split.train_start_idxs)
            test_data = _slice_by_ranges(self.data, split.test_start_idxs)

            if len(train_data) < min_bars_per_side or len(test_data) < min_bars_per_side:
                logger.warning(f"[CPCV] Partición {s_idx} con datos insuficientes tras purga/embargo, omitiendo.")
                if progress_callback:
                    try:
                        progress_callback(s_idx + 1, len(splits), results)
                    except TypeError:
                        progress_callback(s_idx + 1, len(splits))
                continue

            for c_idx, params in enumerate(candidates):
                is_score = self._score(params, train_data, obj_fn)
                oos_score = self._score(params, test_data, obj_fn)
                is_matrix[s_idx, c_idx] = is_score
                oos_matrix[s_idx, c_idx] = oos_score
                results.per_result.append(CPCVCandidateResult(
                    split_id=s_idx, candidate_id=c_idx, params=params,
                    is_score=is_score, oos_score=oos_score,
                ))

            if progress_callback:
                try:
                    progress_callback(s_idx + 1, len(splits), results)
                except TypeError:
                    progress_callback(s_idx + 1, len(splits))

        # ------------------------------------------------------------
        # PBO — Probabilidad de Sobreajuste de Backtest (Bailey et al. 2016)
        # ------------------------------------------------------------
        logits = []
        oos_of_is_best = []
        for s_idx in range(len(splits)):
            row_is = is_matrix[s_idx]
            row_oos = oos_matrix[s_idx]
            if np.all(np.isnan(row_is)):
                continue
            best_c = int(np.nanargmax(row_is))
            oos_best = row_oos[best_c]
            if np.isnan(oos_best):
                continue
            # Rango relativo (omega) del candidato "mejor in-sample" dentro
            # de la distribución OOS de esta partición.
            valid_oos = row_oos[~np.isnan(row_oos)]
            rank = float(np.sum(valid_oos <= oos_best))  # cuántos quedan igual o por debajo
            omega = rank / (len(valid_oos) + 1)
            omega = min(max(omega, 1e-6), 1 - 1e-6)
            logit = float(np.log(omega / (1 - omega)))
            logits.append(logit)
            oos_of_is_best.append(float(oos_best))

        results.logits = logits
        results.oos_scores_of_is_best = oos_of_is_best
        results.pbo = float(np.mean([1 if l <= 0 else 0 for l in logits])) if logits else 0.0

        # Mejor candidato global por IS promedio y por OOS promedio
        mean_is = np.nanmean(is_matrix, axis=0)
        mean_oos = np.nanmean(oos_matrix, axis=0)
        results.mean_oos_by_candidate = {i: float(v) for i, v in enumerate(mean_oos) if not np.isnan(v)}
        if not np.all(np.isnan(mean_is)):
            best_is_idx = int(np.nanargmax(mean_is))
            results.best_candidate_by_is = {
                "candidate_id": best_is_idx, "params": candidates[best_is_idx],
                "mean_is_score": float(mean_is[best_is_idx]),
                "mean_oos_score": float(mean_oos[best_is_idx]) if not np.isnan(mean_oos[best_is_idx]) else None,
            }
        if not np.all(np.isnan(mean_oos)):
            best_oos_idx = int(np.nanargmax(mean_oos))
            results.best_candidate_by_oos = {
                "candidate_id": best_oos_idx, "params": candidates[best_oos_idx],
                "mean_oos_score": float(mean_oos[best_oos_idx]),
                "mean_is_score": float(mean_is[best_oos_idx]) if not np.isnan(mean_is[best_oos_idx]) else None,
            }

        logger.info(f"[CPCV] {results.summary()}")
        return results

    def _score(self, params: dict, data: pd.DataFrame, obj_fn) -> float:
        try:
            strategy = self.strategy_class(**params)
            signals = strategy.generate_signals(data)
            res: BacktestResults = self.engine.run(
                signals, strategy_name=self.strategy_class.name,
                asset=self.asset, timeframe=self.timeframe,
            )
            return float(obj_fn(res))
        except Exception as e:
            logger.debug(f"[CPCV] Candidato falló: {e}")
            return -999.0
