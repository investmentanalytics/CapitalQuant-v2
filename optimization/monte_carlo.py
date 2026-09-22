"""
optimization/monte_carlo.py
Análisis Monte Carlo sobre equity curves y distribución de retornos.

Simula miles de trayectorias posibles basándose en la distribución
empírica de retornos del backtest. Permite visualizar el rango
de resultados posibles, no solo la trayectoria histórica puntual.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

from core.types import BacktestResults


@dataclass
class MonteCarloResults:
    """Resultados del análisis Monte Carlo."""
    n_simulations: int = 0
    n_periods: int = 0
    initial_capital: float = 100_000.0

    # Distribución de resultados finales
    final_capitals: np.ndarray = field(default_factory=lambda: np.array([]))
    p5:  float = 0.0   # Percentil 5  (escenario pesimista)
    p25: float = 0.0   # Percentil 25
    p50: float = 0.0   # Mediana
    p75: float = 0.0   # Percentil 75
    p95: float = 0.0   # Percentil 95 (escenario optimista)

    # Curvas de percentiles para visualización
    equity_p5:  Optional[pd.Series] = None
    equity_p25: Optional[pd.Series] = None
    equity_p50: Optional[pd.Series] = None
    equity_p75: Optional[pd.Series] = None
    equity_p95: Optional[pd.Series] = None

    # Métricas de riesgo
    prob_profit: float = 0.0       # Probabilidad de terminar en positivo
    prob_ruin: float = 0.0         # Probabilidad de pérdida >50%
    avg_max_drawdown: float = 0.0  # Drawdown máximo promedio de todas las simulaciones
    expected_return: float = 0.0   # Retorno esperado (media de simulaciones)

    def summary(self) -> dict:
        return {
            "Simulaciones":         self.n_simulations,
            "Capital inicial":      f"${self.initial_capital:,.2f}",
            "Percentil 5%":         f"${self.p5:,.2f}",
            "Percentil 25%":        f"${self.p25:,.2f}",
            "Mediana":              f"${self.p50:,.2f}",
            "Percentil 75%":        f"${self.p75:,.2f}",
            "Percentil 95%":        f"${self.p95:,.2f}",
            "Prob. ganancia":       f"{self.prob_profit:.1f}%",
            "Prob. pérdida >50%":   f"{self.prob_ruin:.1f}%",
            "DD máximo promedio":   f"{self.avg_max_drawdown:.2f}%",
            "Retorno esperado":     f"{self.expected_return:.2f}%",
        }


class MonteCarloAnalyzer:
    """
    Análisis Monte Carlo por remuestreo de retornos (bootstrap).

    No asume distribución normal — usa la distribución empírica
    real de los retornos del backtest.

    Uso:
        mc = MonteCarloAnalyzer(backtest_results, n_simulations=5000)
        mc_results = mc.run()
    """

    def __init__(
        self,
        results: BacktestResults,
        n_simulations: int = 5000,
        n_periods: Optional[int] = None,
        seed: int = 42,
    ):
        self.results = results
        self.n_simulations = n_simulations
        self.seed = seed

        returns = results.equity_curve.pct_change().dropna()
        self.returns = returns.values.astype(float)
        self.n_periods = n_periods or len(self.returns)
        self.initial_capital = results.initial_capital

    def run(self, progress_callback=None, chunk_size: int = 500) -> MonteCarloResults:
        """Ejecuta las simulaciones Monte Carlo y permite publicar snapshots.

        El remuestreo conserva la misma semántica bootstrap; se procesa por
        bloques para que la interfaz pueda mostrar resultados parciales sin
        esperar a las miles de simulaciones finales.
        """
        logger.info(
            f"[MonteCarlo] Iniciando {self.n_simulations} simulaciones "
            f"({self.n_periods} períodos cada una)..."
        )

        rng = np.random.default_rng(self.seed)
        mc_results = MonteCarloResults(
            n_simulations=self.n_simulations,
            n_periods=self.n_periods,
            initial_capital=self.initial_capital,
        )

        chunk_size = max(1, int(chunk_size))
        chunks = []
        completed = 0
        for start in range(0, self.n_simulations, chunk_size):
            count = min(chunk_size, self.n_simulations - start)
            sampled = rng.choice(
                self.returns, size=(self.n_periods, count), replace=True,
            )
            chunk_equity = np.cumprod(1 + sampled, axis=0) * self.initial_capital
            chunk_equity = np.vstack([
                np.full((1, count), self.initial_capital), chunk_equity,
            ])
            chunks.append(chunk_equity)
            completed += count

            if progress_callback:
                partial_matrix = np.hstack(chunks)
                partial_final = partial_matrix[-1, :]
                partial = MonteCarloResults(
                    n_simulations=completed, n_periods=self.n_periods,
                    initial_capital=self.initial_capital, final_capitals=partial_final,
                    p5=float(np.percentile(partial_final, 5)),
                    p25=float(np.percentile(partial_final, 25)),
                    p50=float(np.percentile(partial_final, 50)),
                    p75=float(np.percentile(partial_final, 75)),
                    p95=float(np.percentile(partial_final, 95)),
                    equity_p5=pd.Series(np.percentile(partial_matrix, 5, axis=1)),
                    equity_p25=pd.Series(np.percentile(partial_matrix, 25, axis=1)),
                    equity_p50=pd.Series(np.percentile(partial_matrix, 50, axis=1)),
                    equity_p75=pd.Series(np.percentile(partial_matrix, 75, axis=1)),
                    equity_p95=pd.Series(np.percentile(partial_matrix, 95, axis=1)),
                    prob_profit=float(np.mean(partial_final > self.initial_capital) * 100),
                    prob_ruin=float(np.mean(partial_final < self.initial_capital * 0.5) * 100),
                    expected_return=float((np.mean(partial_final) / self.initial_capital - 1) * 100),
                )
                peaks_partial = np.maximum.accumulate(partial_matrix, axis=0)
                partial.avg_max_drawdown = float(np.mean(((partial_matrix - peaks_partial) / peaks_partial * 100).min(axis=0)))
                progress_callback(completed, self.n_simulations, partial)

        # Equity curves de todas las simulaciones
        equity_matrix = np.hstack(chunks)
        final_capitals = equity_matrix[-1, :]
        mc_results.final_capitals = final_capitals

        # Percentiles de capital final
        mc_results.p5  = float(np.percentile(final_capitals, 5))
        mc_results.p25 = float(np.percentile(final_capitals, 25))
        mc_results.p50 = float(np.percentile(final_capitals, 50))
        mc_results.p75 = float(np.percentile(final_capitals, 75))
        mc_results.p95 = float(np.percentile(final_capitals, 95))

        # Curvas de percentiles para visualización
        idx = range(self.n_periods + 1)
        mc_results.equity_p5  = pd.Series(np.percentile(equity_matrix, 5,  axis=1), index=idx)
        mc_results.equity_p25 = pd.Series(np.percentile(equity_matrix, 25, axis=1), index=idx)
        mc_results.equity_p50 = pd.Series(np.percentile(equity_matrix, 50, axis=1), index=idx)
        mc_results.equity_p75 = pd.Series(np.percentile(equity_matrix, 75, axis=1), index=idx)
        mc_results.equity_p95 = pd.Series(np.percentile(equity_matrix, 95, axis=1), index=idx)

        # Métricas de riesgo
        mc_results.prob_profit = float(np.mean(final_capitals > self.initial_capital) * 100)
        mc_results.prob_ruin   = float(np.mean(final_capitals < self.initial_capital * 0.5) * 100)
        mc_results.expected_return = float(
            (np.mean(final_capitals) / self.initial_capital - 1) * 100
        )

        # Drawdown máximo promedio
        peaks = np.maximum.accumulate(equity_matrix, axis=0)
        dd_matrix = (equity_matrix - peaks) / peaks * 100
        max_dds = dd_matrix.min(axis=0)
        mc_results.avg_max_drawdown = float(np.mean(max_dds))

        logger.info(
            f"[MonteCarlo] Completado | "
            f"P50=${mc_results.p50:,.0f} | "
            f"Prob.ganancia={mc_results.prob_profit:.1f}% | "
            f"DD avg={mc_results.avg_max_drawdown:.1f}%"
        )

        return mc_results
