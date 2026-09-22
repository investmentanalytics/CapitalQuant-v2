"""
portfolio/optimizer.py
Optimización real de pesos de portafolio — Fase 2 del roadmap.

Antes de este módulo, PortfolioManager (portfolio/manager.py) sabía
CALCULAR las métricas de un portafolio dados unos pesos, pero los pesos
en sí siempre eran equal-weight o los que el usuario tipeara a mano — no
existía optimización real. Este módulo agrega dos enfoques
institucionales estándar, ambos operando sobre retornos de COMPONENTES
(activo::estrategia), no sobre precios:

- **Risk Parity (paridad de riesgo)**: cada componente contribuye lo
  mismo a la volatilidad total del portafolio. No requiere estimar
  retornos esperados (que son mucho más ruidosos que la volatilidad) —
  solo la matriz de covarianza. Referencia: Maillard, Roncalli & Teiletche
  (2010), "The Properties of Equally Weighted Risk Contribution
  Portfolios".
- **Mean-Variance (Markowitz)**: máximo Sharpe o mínima varianza sujeta
  a un retorno objetivo, sobre la frontera eficiente clásica
  (Markowitz, 1952). Requiere estimar retornos esperados (aquí: la media
  histórica de cada componente) — más sensible a error de estimación que
  risk parity, por eso se ofrecen ambos y no solo uno.

Ambos devuelven pesos por COMPONENTE (`{"BTCUSD::Donchian Breakout": 0.34,
...}`), listos para descomponerse en (asset_weights, strategies por
activo) y alimentar PortfolioManager.compute() sin tocar esa clase.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger


@dataclass
class OptimizedWeights:
    """Resultado de una optimización de pesos de portafolio."""
    method: str
    weights: Dict[str, float] = field(default_factory=dict)      # {component: weight}, suma 1.0
    expected_return: Optional[float] = None                       # anualizado, si aplica
    expected_vol: Optional[float] = None                          # anualizado
    expected_sharpe: Optional[float] = None
    risk_contributions: Dict[str, float] = field(default_factory=dict)  # fracción de riesgo total por componente
    converged: bool = True
    message: str = ""


def _annualization_factor(timeframe: str) -> float:
    from config.settings import TIMEFRAMES
    cfg = TIMEFRAMES.get(timeframe)
    return float(cfg["ann_factor"]) if cfg else 252.0


def _prep_returns(
    component_returns: Dict[str, pd.Series],
) -> pd.DataFrame:
    """Alinea las series de retornos de todos los componentes en un único
    DataFrame (mismo criterio que PortfolioManager.compute: unión de
    fechas, huecos rellenados con 0 — un componente sin operación ese
    día no gana ni pierde)."""
    if not component_returns:
        raise ValueError("No hay retornos de componentes para optimizar. Registra resultados primero.")
    df = pd.DataFrame(component_returns).fillna(0)
    if df.shape[1] < 2:
        raise ValueError("Se necesitan al menos 2 componentes para optimizar pesos de portafolio.")
    return df


def _risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Contribución de riesgo (varianza) de cada componente al total,
    como fracción del riesgo total del portafolio (suma 1.0)."""
    port_var = float(weights @ cov @ weights)
    if port_var <= 0:
        return np.ones(len(weights)) / len(weights)
    marginal = cov @ weights
    contrib = weights * marginal
    return contrib / port_var


# ---------------------------------------------------------------------------
# Risk Parity
# ---------------------------------------------------------------------------

def risk_parity_weights(
    component_returns: Dict[str, pd.Series],
    timeframe: str = "1d",
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> OptimizedWeights:
    """
    Pesos de paridad de riesgo: cada componente contribuye lo mismo a la
    volatilidad total del portafolio. Minimiza la suma de las diferencias
    al cuadrado entre contribuciones de riesgo, sujeto a sum(w)=1 y
    min_weight <= w_i <= max_weight.
    """
    returns_df = _prep_returns(component_returns)
    names = list(returns_df.columns)
    n = len(names)
    ann = _annualization_factor(timeframe)
    cov = returns_df.cov().values * ann

    x0 = np.ones(n) / n

    def objective(w):
        rc = _risk_contributions(w, cov)
        target = 1.0 / n
        return float(np.sum((rc - target) ** 2))

    bounds = [(min_weight, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    w = result.x if result.success else x0
    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else x0

    port_ret = float(returns_df.mean().values @ w) * ann
    port_vol = float(np.sqrt(w @ cov @ w))
    rc = _risk_contributions(w, cov)

    if not result.success:
        logger.warning(f"[Portfolio] Risk parity no convergió del todo: {result.message}")

    return OptimizedWeights(
        method="risk_parity",
        weights={names[i]: float(w[i]) for i in range(n)},
        expected_return=port_ret,
        expected_vol=port_vol,
        expected_sharpe=(port_ret / port_vol) if port_vol > 0 else 0.0,
        risk_contributions={names[i]: float(rc[i]) for i in range(n)},
        converged=bool(result.success),
        message=str(result.message),
    )


# ---------------------------------------------------------------------------
# Mean-Variance (Markowitz)
# ---------------------------------------------------------------------------

def mean_variance_weights(
    component_returns: Dict[str, pd.Series],
    timeframe: str = "1d",
    method: str = "max_sharpe",       # "max_sharpe" | "min_variance" | "target_return"
    target_return: Optional[float] = None,  # anualizado, solo si method="target_return"
    risk_free: float = 0.0,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> OptimizedWeights:
    """
    Optimización media-varianza clásica (Markowitz, 1952) sobre retornos
    de componentes. `min_weight=0.0` (default) prohíbe posiciones cortas
    entre componentes (no se puede "vender" una estrategia); bajar
    `max_weight` limita concentración en un solo componente.
    """
    if method not in ("max_sharpe", "min_variance", "target_return"):
        raise ValueError(f"method debe ser 'max_sharpe', 'min_variance' o 'target_return', recibido: {method}")

    returns_df = _prep_returns(component_returns)
    names = list(returns_df.columns)
    n = len(names)
    ann = _annualization_factor(timeframe)
    mean_returns = returns_df.mean().values * ann
    cov = returns_df.cov().values * ann

    x0 = np.ones(n) / n
    bounds = [(min_weight, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if method == "min_variance":
        def objective(w):
            return float(w @ cov @ w)

    elif method == "max_sharpe":
        def objective(w):
            port_ret = float(mean_returns @ w)
            port_vol = float(np.sqrt(w @ cov @ w))
            if port_vol <= 1e-12:
                return 1e6
            return -((port_ret - risk_free) / port_vol)  # minimizar el negativo = maximizar Sharpe

    else:  # target_return
        if target_return is None:
            target_return = float(np.median(mean_returns))
        constraints.append({"type": "eq", "fun": lambda w: float(mean_returns @ w) - target_return})

        def objective(w):
            return float(w @ cov @ w)

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    w = result.x if result.success else x0
    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else x0

    port_ret = float(mean_returns @ w)
    port_vol = float(np.sqrt(w @ cov @ w))

    if not result.success:
        logger.warning(f"[Portfolio] Mean-variance ({method}) no convergió del todo: {result.message}")

    return OptimizedWeights(
        method=f"mean_variance_{method}",
        weights={names[i]: float(w[i]) for i in range(n)},
        expected_return=port_ret,
        expected_vol=port_vol,
        expected_sharpe=((port_ret - risk_free) / port_vol) if port_vol > 0 else 0.0,
        risk_contributions={names[i]: float(v) for i, v in enumerate(_risk_contributions(w, cov))},
        converged=bool(result.success),
        message=str(result.message),
    )


# ---------------------------------------------------------------------------
# Utilidad: descomponer pesos planos por componente en (asset_weights,
# strategies por activo) para alimentar PortfolioManager.compute() sin
# modificar esa clase.
# ---------------------------------------------------------------------------

def decompose_component_weights(
    weights: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Convierte pesos planos {"activo::estrategia": w} en:
    - asset_weights: {activo: suma de pesos de sus componentes}
    - strategies_by_asset: {activo: {estrategia: peso normalizado DENTRO del activo}}

    PortfolioManager.compute(asset_weights=...) junto con
    AssetAllocation(asset, strategies=strategies_by_asset[asset]) reconstruye
    EXACTAMENTE los pesos planos originales (asset_weight * peso-normalizado-
    dentro-del-activo = peso plano), sin necesitar cambiar PortfolioManager.
    """
    asset_weights: Dict[str, float] = {}
    raw_by_asset: Dict[str, Dict[str, float]] = {}

    for component, w in weights.items():
        if "::" not in component:
            continue
        asset, strat = component.split("::", 1)
        asset_weights[asset] = asset_weights.get(asset, 0.0) + w
        raw_by_asset.setdefault(asset, {})[strat] = w

    strategies_by_asset: Dict[str, Dict[str, float]] = {}
    for asset, strat_w in raw_by_asset.items():
        total = asset_weights.get(asset, 0.0)
        if total > 0:
            strategies_by_asset[asset] = {s: v / total for s, v in strat_w.items()}
        else:
            n = len(strat_w)
            strategies_by_asset[asset] = {s: 1.0 / n for s in strat_w}

    return asset_weights, strategies_by_asset
