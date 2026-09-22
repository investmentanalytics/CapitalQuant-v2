"""
portfolio/manager.py
PortfolioManager multi-activo y multi-estrategia.

Arquitectura completamente nueva — el original solo soportaba
un único activo con múltiples estrategias.

Este módulo permite:
- N activos simultáneos (BTCUSD, XAUUSD, SP500, EURUSD...)
- M estrategias por activo con pesos independientes
- Asignación de capital flexible a nivel activo y estrategia
- Métricas institucionales completas (Sharpe, Sortino, Calmar, Ulcer)
- Correlación activo×activo y estrategia×estrategia
- Descomposición de riesgo por componente
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from core.types import BacktestResults
from core.metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio,
    cagr as calc_cagr, compute_drawdown, ulcer_index,
)


# ---------------------------------------------------------------------------
# Tipos de datos del portfolio
# ---------------------------------------------------------------------------

@dataclass
class AssetAllocation:
    """Asignación de estrategias para un activo específico."""
    asset: str
    strategies: Dict[str, float]   # {strategy_name: weight_fraction}

    def normalized_weights(self) -> Dict[str, float]:
        total = sum(self.strategies.values())
        if total <= 0:
            n = len(self.strategies)
            return {k: 1 / n for k in self.strategies}
        return {k: v / total for k, v in self.strategies.items()}


@dataclass
class PortfolioResults:
    """Resultados completos del portafolio consolidado."""

    # Curvas temporales
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)

    # Curvas por componente {asset::strategy: equity_series}
    component_equity: Dict[str, pd.Series] = field(default_factory=dict)
    asset_equity: Dict[str, pd.Series] = field(default_factory=dict)

    # Matrices de correlación
    asset_correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy_correlation: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Descomposición de riesgo (volatilidad anualizada ponderada)
    risk_by_asset: Dict[str, float] = field(default_factory=dict)
    risk_by_strategy: Dict[str, float] = field(default_factory=dict)

    # Pesos finales normalizados
    asset_weights: Dict[str, float] = field(default_factory=dict)
    component_weights: Dict[str, float] = field(default_factory=dict)

    # Métricas
    initial_capital: float = 0.0
    final_capital: float = 0.0
    net_profit: float = 0.0
    net_profit_pct: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown_pct: float = 0.0
    ulcer: float = 0.0

    # Resumen por componente
    component_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    def metrics_dict(self) -> dict:
        return {
            "Initial Capital":  f"${self.initial_capital:,.2f}",
            "Final Capital":    f"${self.final_capital:,.2f}",
            "Net Profit":       f"${self.net_profit:,.2f}",
            "Net Profit %":     f"{self.net_profit_pct:.2f}%",
            "CAGR":             f"{self.cagr:.2f}%",
            "Max Drawdown %":   f"{self.max_drawdown_pct:.2f}%",
            "Sharpe Ratio":     f"{self.sharpe:.2f}",
            "Sortino Ratio":    f"{self.sortino:.2f}",
            "Calmar Ratio":     f"{self.calmar:.2f}",
            "Ulcer Index":      f"{self.ulcer:.2f}",
            "Components":       len(self.component_weights),
        }


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------

class PortfolioManager:
    """
    Gestor de portafolio multi-activo y multi-estrategia.

    Uso:
        pm = PortfolioManager(initial_capital=100_000, timeframe="1d")

        # Configurar asignaciones
        pm.add_allocation(AssetAllocation(
            asset="BTCUSD",
            strategies={"Donchian Breakout": 0.6, "RSI Mean Reversion": 0.4}
        ))
        pm.add_allocation(AssetAllocation(
            asset="XAUUSD",
            strategies={"EMA Trend Following": 1.0}
        ))

        # Registrar resultados de backtest
        pm.register_result("BTCUSD", "Donchian Breakout", btc_don_results)
        pm.register_result("BTCUSD", "RSI Mean Reversion", btc_rsi_results)
        pm.register_result("XAUUSD", "EMA Trend Following", xau_ema_results)

        # Calcular
        results = pm.compute(asset_weights={"BTCUSD": 0.6, "XAUUSD": 0.4})
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        timeframe: str = "1d",
    ):
        self.initial_capital = initial_capital
        self.timeframe = timeframe
        self._allocations: Dict[str, AssetAllocation] = {}
        self._results: Dict[Tuple[str, str], BacktestResults] = {}

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------

    def add_allocation(self, allocation: AssetAllocation) -> None:
        """Registra la configuración de asignación para un activo."""
        self._allocations[allocation.asset] = allocation

    def register_result(
        self,
        asset: str,
        strategy_name: str,
        results: BacktestResults,
    ) -> None:
        """Registra el resultado de backtest para un par activo/estrategia."""
        self._results[(asset, strategy_name)] = results
        logger.debug(f"[Portfolio] Registrado: {asset}::{strategy_name} "
                     f"(sharpe={results.sharpe_ratio:.2f})")

    def clear(self) -> None:
        """Limpia todos los datos registrados."""
        self._allocations.clear()
        self._results.clear()

    # ------------------------------------------------------------------
    # Cálculo del portafolio
    # ------------------------------------------------------------------

    def compute(
        self,
        asset_weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioResults:
        """
        Calcula el portafolio consolidado con todas las métricas.

        Parameters
        ----------
        asset_weights : dict, opcional
            Pesos por activo {asset: weight}. Si None, equal weight entre activos.

        Returns
        -------
        PortfolioResults con métricas, curvas y correlaciones.
        """
        if not self._results:
            raise ValueError("No hay resultados de backtest registrados. "
                             "Llama register_result() primero.")

        assets = list(self._allocations.keys()) or list({k[0] for k in self._results.keys()})

        # Normalizar pesos de activos
        if asset_weights is None:
            asset_weights = {a: 1.0 / len(assets) for a in assets}
        total_aw = sum(asset_weights.values())
        asset_weights = {k: v / total_aw for k, v in asset_weights.items()}

        # Construir retornos ponderados por componente
        all_returns: Dict[str, pd.Series] = {}
        all_weights: Dict[str, float] = {}

        for asset in assets:
            aw = asset_weights.get(asset, 0.0)
            if aw == 0:
                continue

            allocation = self._allocations.get(asset)
            if allocation:
                strat_weights = allocation.normalized_weights()
            else:
                # Sin asignación explícita: equal weight entre estrategias registradas
                strats = [k[1] for k in self._results.keys() if k[0] == asset]
                strat_weights = {s: 1.0 / len(strats) for s in strats}

            for strat_name, sw in strat_weights.items():
                key = (asset, strat_name)
                if key not in self._results:
                    logger.warning(f"[Portfolio] Resultado no encontrado para {asset}::{strat_name}")
                    continue

                res = self._results[key]
                component_key = f"{asset}::{strat_name}"
                ret = res.equity_curve.pct_change().fillna(0)
                all_returns[component_key] = ret
                all_weights[component_key] = aw * sw

        if not all_returns:
            raise ValueError("No se encontraron resultados válidos para calcular el portafolio.")

        # Alinear índices temporales (unión con fill forward)
        returns_df = pd.DataFrame(all_returns).fillna(0)

        # Retorno ponderado del portafolio
        portfolio_returns = pd.Series(0.0, index=returns_df.index)
        for component, w in all_weights.items():
            if component in returns_df.columns:
                portfolio_returns += returns_df[component] * w

        # Normalizar pesos (pueden no sumar exactamente 1 por gaps de datos)
        weight_sum = sum(all_weights.values())
        if weight_sum > 0 and abs(weight_sum - 1.0) > 0.01:
            portfolio_returns /= weight_sum

        # Curva de capital consolidada
        equity_curve = (1 + portfolio_returns).cumprod() * self.initial_capital

        # Drawdown
        dd_series, _, max_dd_pct = compute_drawdown(equity_curve)

        final_capital = float(equity_curve.iloc[-1])
        net_profit = final_capital - self.initial_capital
        n_days = max((equity_curve.index[-1] - equity_curve.index[0]).days, 1)

        pr = PortfolioResults(
            equity_curve=equity_curve,
            drawdown_series=dd_series,
            returns=portfolio_returns,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            net_profit=net_profit,
            net_profit_pct=(net_profit / self.initial_capital) * 100,
            cagr=calc_cagr(self.initial_capital, final_capital, n_days),
            sharpe=sharpe_ratio(portfolio_returns, self.timeframe),
            sortino=sortino_ratio(portfolio_returns, self.timeframe),
            max_drawdown_pct=max_dd_pct,
            ulcer=ulcer_index(equity_curve),
            asset_weights=asset_weights,
            component_weights=all_weights,
        )

        pr.calmar = calmar_ratio(pr.cagr, pr.max_drawdown_pct)

        # Curvas por componente y por activo
        for component, ret in all_returns.items():
            pr.component_equity[component] = (1 + ret).cumprod() * self.initial_capital

        for asset in assets:
            asset_cols = [c for c in returns_df.columns if c.startswith(f"{asset}::")]
            if asset_cols:
                asset_ret = returns_df[asset_cols].mean(axis=1)
                pr.asset_equity[asset] = (1 + asset_ret).cumprod() * self.initial_capital
                pr.risk_by_asset[asset] = float(
                    asset_ret.std() * np.sqrt(252) * asset_weights.get(asset, 0)
                )

        # Correlación activo×activo
        if len(pr.asset_equity) > 1:
            asset_ret_df = pd.DataFrame({
                a: eq.pct_change().fillna(0) for a, eq in pr.asset_equity.items()
            })
            pr.asset_correlation = asset_ret_df.corr()

        # Correlación componente×componente (estrategia×estrategia)
        if returns_df.shape[1] > 1:
            pr.strategy_correlation = returns_df.corr()

        # Descomposición de riesgo por estrategia
        for component, ret in all_returns.items():
            w = all_weights.get(component, 0)
            pr.risk_by_strategy[component] = float(ret.std() * np.sqrt(252) * w)

        # Resumen comparativo por componente
        summary_rows = []
        for component, _ in all_returns.items():
            asset_name, strat_name = component.split("::", 1)
            res = self._results.get((asset_name, strat_name))
            if res:
                summary_rows.append({
                    "component":    component,
                    "asset":        asset_name,
                    "strategy":     strat_name,
                    "weight":       round(all_weights.get(component, 0) * 100, 1),
                    "net_profit_%": round(res.net_profit_pct, 2),
                    "cagr_%":       round(res.cagr, 2),
                    "sharpe":       round(res.sharpe_ratio, 2),
                    "max_dd_%":     round(res.max_drawdown_pct, 2),
                    "win_rate_%":   round(res.win_rate, 1),
                    "trades":       res.total_trades,
                })

        if summary_rows:
            pr.component_summary = pd.DataFrame(summary_rows)

        logger.info(
            f"[Portfolio] Calculado: {len(all_weights)} componentes | "
            f"Sharpe={pr.sharpe:.2f} CAGR={pr.cagr:.2f}% DD={pr.max_drawdown_pct:.2f}%"
        )

        return pr

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def registered_components(self) -> List[str]:
        return [f"{a}::{s}" for a, s in self._results.keys()]

    @property
    def assets(self) -> List[str]:
        return list(self._allocations.keys())
