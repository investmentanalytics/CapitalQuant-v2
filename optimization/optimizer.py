"""
optimization/optimizer.py
Motor de optimización bayesiana con Optuna.

Refactorizado:
- Funciones objetivo separadas en objectives.py
- Sin dependencias de Streamlit
- get_top_trials() con tipado correcto
- Guardado de resultados con metadatos completos
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Type, Optional, Callable, List

import numpy as np
import optuna
import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy
from core.types import BacktestConfig, BacktestResults
from engine.backtester import BacktestEngine
from optimization.objectives import OBJECTIVES, OBJECTIVES_LABELS
from config.settings import OPT_DIR

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class StrategyOptimizer:
    """
    Optimizador de parámetros con búsqueda bayesiana (Optuna TPE).

    Uso:
        optimizer = StrategyOptimizer(
            strategy_class=DonchianBreakout,
            data=df,
            config=config,
            asset="BTCUSD",
            timeframe="1d",
        )
        best_params = optimizer.optimize(n_trials=100, objective="sharpe")
        top = optimizer.get_top_trials(10)
    """

    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
        asset: str = "",
        timeframe: str = "",
        regime_mask: Optional[pd.Series] = None,
    ):
        self.strategy_class = strategy_class
        self.data = data
        self.config = config or BacktestConfig()
        self.asset = asset
        self.timeframe = timeframe
        self.regime_mask = regime_mask
        self.engine = BacktestEngine(self.config)
        self._best_results: Optional[BacktestResults] = None
        self._study: Optional[optuna.Study] = None

    def optimize(
        self,
        n_trials: int = 100,
        objective: str = "sharpe",
        n_jobs: int = 1,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> dict:
        """
        Ejecuta la optimización bayesiana.

        Parameters
        ----------
        n_trials : int
            Número de combinaciones a probar
        objective : str
            Métrica a maximizar — ver OBJECTIVES en objectives.py
        n_jobs : int
            Número de workers paralelos
        progress_callback : callable(trial_num, total, best_value)
            Función para actualizar UI durante la optimización

        Returns
        -------
        dict : Mejores parámetros encontrados
        """
        if objective not in OBJECTIVES:
            raise ValueError(
                f"Objetivo '{objective}' no válido. "
                f"Disponibles: {list(OBJECTIVES.keys())}"
            )

        obj_fn = OBJECTIVES[objective]
        param_space = self.strategy_class().get_param_space()

        if not param_space:
            logger.warning(f"{self.strategy_class.name} no tiene espacio de parámetros.")
            return {}

        logger.info(
            f"Optimizando [{self.strategy_class.name}|{self.asset}|{self.timeframe}] "
            f"objetivo={objective} trials={n_trials}"
        )

        self._study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=15),
        )

        def objective_fn(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, param_space)
            try:
                strategy = self.strategy_class(**params)
                signals = strategy.generate_signals(self.data)
                if self.regime_mask is not None and "signal" in signals.columns:
                    signals = signals.copy()
                    m = self.regime_mask.reindex(signals.index).fillna(False)
                    signals.loc[~m, "signal"] = 0
                results = self.engine.run(
                    signals,
                    strategy_name=self.strategy_class.name,
                    asset=self.asset,
                    timeframe=self.timeframe,
                )
                score = obj_fn(results)

                # Guardar métricas en el trial para análisis posterior
                trial.set_user_attr("net_profit_pct", results.net_profit_pct)
                trial.set_user_attr("win_rate",        results.win_rate)
                trial.set_user_attr("total_trades",    results.total_trades)
                trial.set_user_attr("max_dd_pct",      results.max_drawdown_pct)
                trial.set_user_attr("sharpe",          results.sharpe_ratio)
                trial.set_user_attr("calmar",          results.calmar_ratio)
                trial.set_user_attr("profit_factor",   min(results.profit_factor, 10.0))

                # Actualizar mejor resultado
                current_best = trial.study.best_value if trial.study.trials else -np.inf
                if self._best_results is None or score > current_best:
                    self._best_results = results

                if progress_callback:
                    best_val = trial.study.best_value if trial.study.trials else score
                    progress_callback(trial.number + 1, n_trials, best_val)

                return score
            except Exception as e:
                logger.debug(f"Trial {trial.number} falló: {e}")
                return -999.0

        self._study.optimize(
            objective_fn,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )

        best_params = self._study.best_params
        best_value = self._study.best_value

        logger.info(f"Mejor {objective}: {best_value:.4f} | params={best_params}")
        self._save_results(best_params, best_value, objective, n_trials)

        return best_params

    def get_optimization_history(self) -> Optional[pd.DataFrame]:
        """Historial completo de trials como DataFrame."""
        if self._study is None:
            return None
        rows = []
        for t in self._study.trials:
            if t.value is None:
                continue
            row = {"trial": t.number, "value": t.value}
            row.update(t.params)
            row.update(t.user_attrs)
            rows.append(row)
        return pd.DataFrame(rows) if rows else None

    def get_top_trials(self, n: int = 10) -> List[optuna.trial.FrozenTrial]:
        """Retorna los N mejores trials ordenados por score."""
        if self._study is None:
            return []
        completed = [t for t in self._study.trials if t.value is not None]
        completed.sort(key=lambda x: x.value, reverse=True)
        return completed[:n]

    def load_best_params(self) -> Optional[dict]:
        """Carga los mejores parámetros guardados para este activo/timeframe."""
        path = self._result_path()
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data.get("best_params")
        return None

    @property
    def best_results(self) -> Optional[BacktestResults]:
        return self._best_results

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest_params(trial: optuna.Trial, space: dict) -> dict:
        params = {}
        for name, spec in space.items():
            ptype = spec[0]
            if ptype == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif ptype == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])
            elif ptype == "bool":
                params[name] = trial.suggest_categorical(name, [True, False])
        return params

    def _result_path(self) -> Path:
        strategy_slug = self.strategy_class.name.lower().replace(" ", "_")
        return OPT_DIR / f"{strategy_slug}_{self.asset}_{self.timeframe}.json"

    def _save_results(self, params: dict, score: float, objective: str, n_trials: int) -> None:
        data = {
            "strategy":   self.strategy_class.name,
            "asset":      self.asset,
            "timeframe":  self.timeframe,
            "objective":  objective,
            "best_score": round(score, 6),
            "best_params": params,
            "n_trials":   n_trials,
            "completed_trials": len(self._study.trials) if self._study else 0,
        }
        path = self._result_path()
        OPT_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Resultados guardados  {path}")


# ---------------------------------------------------------------------------
# Utilidades standalone
# ---------------------------------------------------------------------------

def list_saved_optimizations() -> list:
    """Lista todas las optimizaciones guardadas."""
    results = []
    for f in sorted(OPT_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                results.append(json.load(fp))
        except Exception:
            pass
    return results
