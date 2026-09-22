"""
optimization/walk_forward.py
Walk-Forward Optimization (WFO).

Evita el overfitting optimizando en ventanas de entrenamiento
y validando en períodos out-of-sample no vistos por el optimizador.

Métricas clave:
- Efficiency Ratio: test_score / train_score
   >0.7 indica estrategia robusta
   <0.3 indica overfitting severo
- Parameter Stability: variación de parámetros entre ventanas
   Bajo CV indica parámetros estables (menor riesgo de overfitting)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Type, List, Optional, Callable
import numpy as np
import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy
from core.types import BacktestConfig, BacktestResults
from engine.backtester import BacktestEngine
from optimization.optimizer import StrategyOptimizer


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class WFOWindow:
    """Resultado de una ventana individual de Walk-Forward."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict = field(default_factory=dict)
    train_score: float = 0.0
    test_score: float = 0.0
    test_results: Optional[BacktestResults] = None
    train_results: Optional[BacktestResults] = None
    purge_bars: int = 0     # velas eliminadas del final de train (evita fuga por lookback)
    embargo_bars: int = 0   # velas eliminadas al inicio de test (evita fuga por autocorrelación)


@dataclass
class WFOResults:
    """Resultados completos del Walk-Forward."""
    windows: List[WFOWindow] = field(default_factory=list)
    combined_oos_equity: pd.Series = field(default_factory=pd.Series)

    # Métricas de robustez
    efficiency_ratio: float = 0.0       # test/train score promedio
    parameter_stability: dict = field(default_factory=dict)  # {param: CV}
    avg_train_score: float = 0.0
    avg_test_score: float = 0.0

    # Mejor configuración encontrada globalmente
    best_params: dict = field(default_factory=dict)
    best_window_id: int = 0

    def robustness_label(self) -> str:
        er = self.efficiency_ratio
        if er >= 0.75:
            return "ROBUSTA"
        elif er >= 0.50:
            return "MODERADA"
        elif er >= 0.25:
            return "FRÁGIL"
        else:
            return "OVERFITTING"


# ---------------------------------------------------------------------------
# Walk-Forward Optimizer
# ---------------------------------------------------------------------------

class WalkForwardOptimizer:
    """
    Optimización Walk-Forward con ventanas deslizantes.

    Uso:
        wfo = WalkForwardOptimizer(
            strategy_class=DonchianBreakout,
            data=df,
            config=config,
            asset="BTCUSD",
            timeframe="1d",
            train_pct=0.70,     # 70% entrenamiento, 30% test
            n_windows=5,        # 5 ventanas deslizantes
            n_trials=50,        # 50 trials Optuna por ventana
        )
        results = wfo.run(objective="sharpe")
    """

    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
        asset: str = "",
        timeframe: str = "",
        train_pct: float = 0.70,
        n_windows: int = 5,
        n_trials: int = 50,
        purge_pct: float = 0.02,
        embargo_pct: float = 0.02,
        max_lookback_bars: Optional[int] = None,
    ):
        """
        purge_pct : float
            Fracción del tamaño de la ventana de TRAIN que se descarta al
            final de train, justo antes del corte train/test. Evita que
            un indicador de lookback largo (ej. SMA200) calculado cerca
            del borde "vea" implícitamente datos que se solapan con test
            (purga, ver López de Prado, *Advances in Financial ML*).
        embargo_pct : float
            Fracción del tamaño de ventana que se descarta al INICIO de
            test, inmediatamente después del corte. Evita fuga por
            autocorrelación serial de los retornos entre el último dato
            de entrenamiento y el primero de test.
        max_lookback_bars : int, opcional
            Si se conoce el lookback máximo real de los indicadores de la
            estrategia (ej. 200 para una SMA200), se usa como PISO del
            purge/embargo en velas, en vez de solo el porcentaje — un
            0.02 de una ventana pequeña puede ser menor que el lookback
            real del indicador más lento, lo cual purgaría de menos.
        """
        self.strategy_class = strategy_class
        self.data = data.copy()
        self.config = config or BacktestConfig()
        self.asset = asset
        self.timeframe = timeframe
        self.train_pct = train_pct
        self.n_windows = n_windows
        self.n_trials = n_trials
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.max_lookback_bars = max_lookback_bars
        self.engine = BacktestEngine(self.config)

    def run(
        self,
        objective: str = "sharpe",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> WFOResults:
        """
        Ejecuta el Walk-Forward completo.

        Parameters
        ----------
        objective : str
            Métrica a maximizar durante la optimización de entrenamiento.
        progress_callback : callable(window_id, total_windows)

        Returns
        -------
        WFOResults con equity out-of-sample combinada y métricas de robustez.
        """
        windows = self._build_windows()
        if not windows:
            raise ValueError("No se pudieron construir ventanas WFO. "
                             "Verifica que haya suficientes datos.")

        results = WFOResults()
        oos_equity_pieces: List[pd.Series] = []
        param_history: dict[str, list] = {}
        best_test_score = -np.inf

        for w in windows:
            logger.info(
                f"[WFO] Ventana {w.window_id}/{len(windows)} | "
                f"Train: {w.train_start.date()}{w.train_end.date()} "
                f"(purge={w.purge_bars} velas) | "
                f"Test: {w.test_start.date()}{w.test_end.date()} "
                f"(embargo={w.embargo_bars} velas)"
            )

            train_data = self.data.loc[w.train_start:w.train_end].copy()
            test_data  = self.data.loc[w.test_start:w.test_end].copy()

            if len(train_data) < 50 or len(test_data) < 10:
                logger.warning(f"[WFO] Ventana {w.window_id} con datos insuficientes, omitiendo.")
                continue

            # --- Optimizar en entrenamiento ---
            optimizer = StrategyOptimizer(
                strategy_class=self.strategy_class,
                data=train_data,
                config=self.config,
                asset=self.asset,
                timeframe=self.timeframe,
            )
            try:
                best_params = optimizer.optimize(
                    n_trials=self.n_trials,
                    objective=objective,
                )
                w.best_params = best_params
                w.train_score = optimizer._study.best_value

                # Guardar historial de parámetros
                for param, val in best_params.items():
                    param_history.setdefault(param, []).append(val)

                # --- Validar en out-of-sample ---
                strategy = self.strategy_class(**best_params)
                signals = strategy.generate_signals(test_data)
                test_res = self.engine.run(
                    signals,
                    strategy_name=self.strategy_class.name,
                    asset=self.asset,
                    timeframe=self.timeframe,
                )
                w.test_results = test_res
                w.test_score = test_res.sharpe_ratio

                oos_equity_pieces.append(test_res.equity_curve)

                # Mejor configuración global (por test score)
                if w.test_score > best_test_score:
                    best_test_score = w.test_score
                    results.best_params = best_params
                    results.best_window_id = w.window_id

            except Exception as e:
                logger.error(f"[WFO] Error en ventana {w.window_id}: {e}")

            results.windows.append(w)
            if progress_callback:
                try:
                    progress_callback(w.window_id, len(windows), w, results)
                except TypeError:
                    progress_callback(w.window_id, len(windows))

        # --- Equity OOS combinada ---
        if oos_equity_pieces:
            # Re-encadenar equity: cada ventana parte del capital final de la anterior
            base = self.config.initial_capital
            chains = []
            for piece in oos_equity_pieces:
                scaled = piece / piece.iloc[0] * base
                chains.append(scaled)
                base = float(scaled.iloc[-1])
            results.combined_oos_equity = pd.concat(chains)

        # --- Métricas de robustez ---
        valid_windows = [w for w in results.windows if w.train_score and w.test_results]

        if valid_windows:
            train_scores = [w.train_score for w in valid_windows]
            test_scores  = [w.test_score  for w in valid_windows]
            results.avg_train_score = float(np.mean(train_scores))
            results.avg_test_score  = float(np.mean(test_scores))
            results.efficiency_ratio = (
                results.avg_test_score / results.avg_train_score
                if results.avg_train_score != 0 else 0.0
            )

        # Estabilidad de parámetros (Coeficiente de Variación)
        results.parameter_stability = {}
        for param, values in param_history.items():
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val  = np.std(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else 0.0
                results.parameter_stability[param] = round(float(cv), 4)

        logger.info(
            f"[WFO] Completado | ER={results.efficiency_ratio:.2f} "
            f"({results.robustness_label()}) | "
            f"Train={results.avg_train_score:.2f} Test={results.avg_test_score:.2f}"
        )

        return results

    # ------------------------------------------------------------------
    # Construcción de ventanas
    # ------------------------------------------------------------------

    def _build_windows(self) -> List[WFOWindow]:
        """Construye ventanas deslizantes de entrenamiento y prueba."""
        n = len(self.data)
        # Tamaño mínimo de ventana para tener suficientes datos
        min_window = max(100, n // (self.n_windows * 2))
        window_size = max(n // self.n_windows, min_window)
        train_size = int(window_size * self.train_pct)
        test_size  = window_size - train_size

        if test_size < 10:
            logger.warning("[WFO] test_size muy pequeño, aumentando train_pct.")
            test_size = max(10, window_size // 4)
            train_size = window_size - test_size

        # Purge y embargo en velas: el máximo entre el % de la ventana y el
        # lookback real conocido del indicador más lento de la estrategia,
        # para no purgar de menos cuando la ventana es chica pero el
        # indicador es lento (ej. SMA200 sobre una ventana de 500 velas).
        purge_bars = int(round(window_size * self.purge_pct))
        embargo_bars = int(round(window_size * self.embargo_pct))
        if self.max_lookback_bars:
            purge_bars = max(purge_bars, self.max_lookback_bars)
            embargo_bars = max(embargo_bars, self.max_lookback_bars)

        windows = []
        for i in range(self.n_windows):
            start_idx      = i * test_size
            train_end_idx  = start_idx + train_size
            # --- Purga: recortar el final de train ---
            purged_train_end_idx = max(start_idx + 1, train_end_idx - purge_bars)
            # --- Embargo: saltar velas al inicio de test ---
            test_start_idx = train_end_idx + embargo_bars
            test_end_idx   = train_end_idx + test_size

            if test_end_idx > n or test_start_idx >= test_end_idx:
                break

            w = WFOWindow(
                window_id=i + 1,
                train_start=self.data.index[start_idx],
                train_end=self.data.index[purged_train_end_idx - 1],
                test_start=self.data.index[test_start_idx],
                test_end=self.data.index[min(test_end_idx - 1, n - 1)],
                purge_bars=train_end_idx - purged_train_end_idx,
                embargo_bars=embargo_bars,
            )
            windows.append(w)

        return windows
