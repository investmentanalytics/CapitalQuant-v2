"""
optimization/objectives.py
Funciones objetivo para optimización de estrategias.

Separadas del optimizer para poder añadir nuevos objetivos
sin modificar el motor de optimización.
"""
from core.types import BacktestResults
from core.metrics import composite_score

MIN_TRADES = 10


def _guard(results: BacktestResults, min_trades: int = MIN_TRADES) -> bool:
    return results.total_trades >= min_trades


def objective_sharpe(results: BacktestResults) -> float:
    if not _guard(results):
        return -999.0
    return results.sharpe_ratio


def objective_sortino(results: BacktestResults) -> float:
    if not _guard(results):
        return -999.0
    return results.sortino_ratio


def objective_calmar(results: BacktestResults) -> float:
    if not _guard(results) or results.max_drawdown_pct == 0:
        return -999.0
    return results.calmar_ratio


def objective_profit_factor(results: BacktestResults) -> float:
    if not _guard(results):
        return -999.0
    return min(results.profit_factor, 10.0)


def objective_expectancy(results: BacktestResults) -> float:
    if not _guard(results):
        return -999.0
    return results.expectancy


def objective_cagr(results: BacktestResults) -> float:
    if not _guard(results):
        return -999.0
    return results.cagr


def objective_composite(results: BacktestResults) -> float:
    """Score compuesto institucional (Sharpe + Calmar + PF + WinRate)."""
    if not _guard(results):
        return -999.0
    return composite_score(
        sharpe=results.sharpe_ratio,
        calmar=results.calmar_ratio,
        profit_factor_val=results.profit_factor,
        win_rate=results.win_rate,
        total_trades=results.total_trades,
    )


# Registro de objetivos disponibles
OBJECTIVES: dict = {
    "sharpe":         objective_sharpe,
    "sortino":        objective_sortino,
    "calmar":         objective_calmar,
    "profit_factor":  objective_profit_factor,
    "expectancy":     objective_expectancy,
    "cagr":           objective_cagr,
    "composite":      objective_composite,
}

OBJECTIVES_LABELS: dict = {
    "sharpe":         "Sharpe Ratio",
    "sortino":        "Sortino Ratio",
    "calmar":         "Calmar Ratio",
    "profit_factor":  "Profit Factor",
    "expectancy":     "Expectancy ($)",
    "cagr":           "CAGR (%)",
    "composite":      "Score Compuesto",
}
