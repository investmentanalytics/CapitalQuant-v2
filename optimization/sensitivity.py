"""
optimization/sensitivity.py
Análisis de sensibilidad de parámetros.

Genera mapas 2D y 3D que muestran cómo varía el rendimiento
de la estrategia al cambiar uno o dos parámetros simultáneamente.
Permite identificar zonas de robustez vs zonas de alta sensibilidad.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Type, Optional, List
import numpy as np
import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy
from core.types import BacktestConfig
from engine.backtester import BacktestEngine
from optimization.objectives import OBJECTIVES


@dataclass
class SensitivityResult:
    """Resultado del análisis de sensibilidad."""
    param1_name: str
    param1_values: List
    param2_name: Optional[str]
    param2_values: Optional[List]
    scores: np.ndarray  # 1D si un parámetro, 2D si dos parámetros
    objective: str
    best_params: dict = field(default_factory=dict)
    best_score: float = 0.0


class SensitivityAnalyzer:
    """
    Análisis de sensibilidad de uno o dos parámetros.

    Usa los mejores parámetros encontrados en la optimización
    como referencia y varía uno o dos de ellos para ver el impacto.

    Uso:
        # Sensibilidad 1D
        sa = SensitivityAnalyzer(DonchianBreakout, df, config)
        result = sa.analyze_1d(
            base_params=best_params,
            param="entry_period",
            values=range(10, 61, 5),
            objective="sharpe",
        )

        # Sensibilidad 2D
        result = sa.analyze_2d(
            base_params=best_params,
            param1="entry_period", values1=range(10, 61, 5),
            param2="atr_multiplier", values2=[1.0, 1.5, 2.0, 2.5, 3.0],
            objective="sharpe",
        )
    """

    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
        asset: str = "",
        timeframe: str = "",
    ):
        self.strategy_class = strategy_class
        self.data = data
        self.config = config or BacktestConfig()
        self.asset = asset
        self.timeframe = timeframe
        self.engine = BacktestEngine(self.config)

    def analyze_1d(
        self,
        base_params: dict,
        param: str,
        values: list,
        objective: str = "sharpe",
        progress_callback=None,
    ) -> SensitivityResult:
        """Analiza sensibilidad variando un parámetro."""
        obj_fn = OBJECTIVES[objective]
        scores = []
        values = list(values)

        logger.info(f"[Sensitivity 1D] {param} × {len(values)} valores")

        for idx, val in enumerate(values, start=1):
            params = {**base_params, param: val}
            score = self._evaluate(params, obj_fn)
            scores.append(score)
            if progress_callback:
                progress_callback(idx, len(values), {"value": val, "score": float(score), "scores": list(scores)})

        scores_arr = np.array(scores)
        best_idx = np.argmax(scores_arr)

        return SensitivityResult(
            param1_name=param,
            param1_values=values,
            param2_name=None,
            param2_values=None,
            scores=scores_arr,
            objective=objective,
            best_params={**base_params, param: values[best_idx]},
            best_score=float(scores_arr[best_idx]),
        )

    def analyze_2d(
        self,
        base_params: dict,
        param1: str,
        values1: list,
        param2: str,
        values2: list,
        objective: str = "sharpe",
        progress_callback=None,
    ) -> SensitivityResult:
        """Analiza sensibilidad variando dos parámetros simultáneamente."""
        obj_fn = OBJECTIVES[objective]
        values1, values2 = list(values1), list(values2)
        n1, n2 = len(values1), len(values2)
        scores = np.zeros((n1, n2))

        total = n1 * n2
        logger.info(f"[Sensitivity 2D] {param1}×{param2} — {total} combinaciones")

        done = 0
        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                params = {**base_params, param1: v1, param2: v2}
                scores[i, j] = self._evaluate(params, obj_fn)
                done += 1
                if progress_callback:
                    progress_callback(done, total, {"value1": v1, "value2": v2, "score": float(scores[i, j]), "scores": scores.copy()})

        best_flat = np.argmax(scores)
        best_i, best_j = np.unravel_index(best_flat, scores.shape)

        return SensitivityResult(
            param1_name=param1,
            param1_values=values1,
            param2_name=param2,
            param2_values=values2,
            scores=scores,
            objective=objective,
            best_params={
                **base_params,
                param1: values1[best_i],
                param2: values2[best_j],
            },
            best_score=float(scores[best_i, best_j]),
        )

    def _evaluate(self, params: dict, obj_fn) -> float:
        """Ejecuta un backtest y retorna el score del objetivo."""
        try:
            strategy = self.strategy_class(**params)
            signals = strategy.generate_signals(self.data)
            results = self.engine.run(
                signals,
                strategy_name=self.strategy_class.name,
                asset=self.asset,
                timeframe=self.timeframe,
            )
            return float(obj_fn(results))
        except Exception as e:
            logger.debug(f"[Sensitivity] Error evaluando {params}: {e}")
            return -999.0
