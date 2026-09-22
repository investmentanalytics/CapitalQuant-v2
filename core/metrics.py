"""
core/metrics.py
Cálculo de métricas financieras — funciones puras sin efectos secundarios.

Centraliza toda la matemática financiera del sistema.
Reutilizable en backtester, portfolio, optimizer y API móvil.
"""
from typing import List
import numpy as np
import pandas as pd
from scipy.stats import norm
from config.settings import TIMEFRAMES


# ---------------------------------------------------------------------------
# Factor de anualización
# ---------------------------------------------------------------------------

def annualization_factor(timeframe: str) -> float:
    """Retorna el factor de anualización para un timeframe dado."""
    cfg = TIMEFRAMES.get(timeframe)
    if cfg:
        return float(cfg["ann_factor"])
    return 252.0  # Default: diario


# ---------------------------------------------------------------------------
# Métricas de rentabilidad
# ---------------------------------------------------------------------------

def cagr(initial: float, final: float, n_days: float) -> float:
    """Tasa de crecimiento anual compuesta (%)."""
    if initial <= 0 or final <= 0 or n_days <= 0:
        return -100.0
    n_years = max(n_days / 365.25, 1 / 365.25)
    return ((final / initial) ** (1.0 / n_years) - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Métricas de riesgo ajustadas
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, timeframe: str = "1d") -> float:
    """Sharpe Ratio anualizado (sin tasa libre de riesgo)."""
    ann = annualization_factor(timeframe)
    clean = returns.dropna()
    if len(clean) < 2 or clean.std() == 0:
        return 0.0
    mu = clean.mean() * ann
    sigma = clean.std() * np.sqrt(ann)
    return float(mu / sigma) if sigma > 0 else 0.0


def sortino_ratio(returns: pd.Series, timeframe: str = "1d") -> float:
    """Sortino Ratio anualizado (penaliza solo retornos negativos)."""
    ann = annualization_factor(timeframe)
    clean = returns.dropna()
    mu = clean.mean() * ann
    downside = clean[clean < 0]
    if len(downside) == 0:
        return 0.0
    sigma_down = downside.std() * np.sqrt(ann)
    return float(mu / sigma_down) if sigma_down > 0 else 0.0


def calmar_ratio(cagr_val: float, max_drawdown_pct: float) -> float:
    """Calmar Ratio = CAGR / |Max Drawdown %|."""
    if max_drawdown_pct >= 0:
        return 0.0
    return cagr_val / abs(max_drawdown_pct)


# ---------------------------------------------------------------------------
# Métricas de drawdown
# ---------------------------------------------------------------------------

def compute_drawdown(equity_curve: pd.Series) -> tuple[pd.Series, float, float]:
    """
    Calcula la serie de drawdown y métricas máximas.

    Returns
    -------
    (drawdown_pct_series, max_drawdown_abs, max_drawdown_pct)
    """
    peak = equity_curve.cummax()
    dd_abs = equity_curve - peak
    dd_pct = (dd_abs / peak * 100).fillna(0)
    return dd_pct, float(dd_abs.min()), float(dd_pct.min())


def ulcer_index(equity_curve: pd.Series) -> float:
    """Ulcer Index: raíz cuadrada del promedio de drawdowns al cuadrado."""
    peak = equity_curve.cummax()
    dd_pct = ((equity_curve - peak) / peak * 100).fillna(0)
    return float(np.sqrt(np.mean(np.square(dd_pct))))


# ---------------------------------------------------------------------------
# Métricas de operaciones
# ---------------------------------------------------------------------------

def profit_factor(net_pnls: List[float]) -> float:
    """Profit Factor = suma de ganancias / suma de pérdidas."""
    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_streak(pnls: List[float], win: bool = True) -> int:
    """Racha máxima de victorias (win=True) o derrotas (win=False)."""
    max_s = current = 0
    for p in pnls:
        if (win and p > 0) or (not win and p <= 0):
            current += 1
            max_s = max(max_s, current)
        else:
            current = 0
    return max_s


def compute_exposure_pct(trades_df: pd.DataFrame, price_index: pd.DatetimeIndex) -> float:
    """
    Calcula el porcentaje de tiempo en mercado de forma vectorizada O(n).

    Parameters
    ----------
    trades_df : DataFrame con columnas entry_date, exit_date
    price_index : índice del DataFrame de precios
    """
    if trades_df is None or trades_df.empty or len(price_index) == 0:
        return 0.0
    in_market = pd.Series(False, index=price_index)
    for _, row in trades_df.iterrows():
        try:
            mask = (price_index >= row["entry_date"]) & (price_index <= row["exit_date"])
            in_market |= mask
        except Exception:
            continue
    return float(in_market.mean() * 100)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio — corrección por tests múltiples
# ---------------------------------------------------------------------------
# Referencia: Bailey, D.H. & Lopez de Prado, M. (2014), "The Deflated Sharpe
# Ratio: Correcting for Selection Bias, Backtest Overfitting and
# Non-Normality", Journal of Portfolio Management.
#
# Problema que resuelve: cualquier búsqueda que evalúa N combinaciones y se
# queda con la de mejor Sharpe (como hace un algoritmo genético con miles de
# individuos por corrida) va a encontrar Sharpes altos SOLO por azar, incluso
# si ninguna combinación tiene ventaja real — cuantas más se prueban, más
# alto es el mejor Sharpe esperado bajo pura suerte. El DSR "descuenta" el
# Sharpe observado por ese efecto y devuelve la probabilidad de que siga
# siendo genuino.

_EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe_null(n_trials: int, sharpe_std: float) -> float:
    """Sharpe esperado del MEJOR resultado entre `n_trials` intentos
    independientes sin ventaja real (puro ruido), cada uno con desviación
    estándar `sharpe_std` en su propia estimación de Sharpe. Es el listón
    que un Sharpe observado debe superar para no explicarse solo por haber
    probado muchas combinaciones."""
    if n_trials <= 1 or sharpe_std <= 0:
        return 0.0
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sharpe_std * ((1 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    sharpe_std_across_trials: float,
    n_trials: int,
    n_periods: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilidad (0.0-1.0) de que `observed_sharpe` sea genuinamente
    superior al Sharpe esperado del mejor resultado puramente aleatorio
    entre `n_trials` intentos — corrige el sesgo de selección/tests
    múltiples inherente a cualquier búsqueda que evalúa muchas estrategias y
    se queda con la mejor.

    Todos los Sharpe deben estar en la MISMA escala (no anualizados) porque
    la fórmula usa `n_periods` observaciones para el término de error
    estándar. >= 0.95 se considera convencionalmente "estadísticamente
    significativo" (no explicable solo por el número de intentos).

    Parameters
    ----------
    observed_sharpe : Sharpe (por periodo, NO anualizado) de la estrategia evaluada.
    sharpe_std_across_trials : desviación estándar de los Sharpe (por periodo)
        observados a lo largo de TODOS los intentos de la búsqueda (no solo
        el ganador) — mide cuánto varía el Sharpe entre combinaciones.
    n_trials : número total de combinaciones evaluadas en la búsqueda.
    n_periods : número de observaciones de retorno detrás del Sharpe evaluado.
    skew, kurtosis : asimetría y curtosis (no excedente; normal = 3) de los
        retornos de la estrategia evaluada — la no-normalidad de los
        retornos de trading empeora la estimación del Sharpe y el DSR lo
        penaliza explícitamente por esto.
    """
    if n_periods <= 1:
        return 0.0
    sr0 = expected_max_sharpe_null(n_trials, sharpe_std_across_trials)
    denom = 1 - skew * observed_sharpe + ((kurtosis - 1) / 4.0) * observed_sharpe ** 2
    if denom <= 0:
        denom = 1e-6
    z = (observed_sharpe - sr0) * np.sqrt(n_periods - 1) / np.sqrt(denom)
    return float(norm.cdf(z))


# ---------------------------------------------------------------------------
# Composite Score institucional
# ---------------------------------------------------------------------------

def composite_score(
    sharpe: float,
    calmar: float,
    profit_factor_val: float,
    win_rate: float,
    total_trades: int,
    min_trades: int = 10,
) -> float:
    """
    Score compuesto para ranking institucional de estrategias.
    Pondera las métricas más relevantes con penalización por bajo número de trades.
    """
    if total_trades < min_trades:
        return -999.0

    # Normalizar profit_factor (cap en 10 para no sobreponderar outliers)
    pf_norm = min(profit_factor_val, 10.0) / 10.0

    score = (
        0.35 * max(sharpe, 0) +
        0.25 * max(calmar, 0) +
        0.25 * pf_norm * 10 +
        0.15 * (win_rate / 100)
    )
    return float(score)
