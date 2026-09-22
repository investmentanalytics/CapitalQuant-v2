"""
discovery/generalization.py
Prueba de generalización cruzada de instrumentos + estabilidad de
régimen/estacionalidad (Auditoría sección 2.4 y 3.c, Roadmap Fase 1
puntos 2 y 3).

Motivación: ninguna estrategia que sale del Descubridor Genético (ni
tampoco una escrita a mano en el Constructor) debería considerarse
seria antes de comprobar dos cosas adicionales al Deflated Sharpe Ratio
interno:

1. GENERALIZACIÓN ENTRE SÍMBOLOS
   Si una regla solo es rentable en el símbolo exacto donde el genético
   la encontró, es más probable que haya memorizado ruido de ese activo
   que encontrado una señal real de mercado. Se re-ejecuta la MISMA
   estrategia con los MISMOS parámetros (sin volver a optimizar) sobre
   un conjunto de símbolos relacionados y se reporta cuántos siguen
   siendo rentables / tienen Sharpe positivo.

2. ESTABILIDAD DE RÉGIMEN Y ESTACIONALIDAD
   Se parte el propio histórico del símbolo de origen en (a) meses del
   calendario y (b) régimen de tendencia vs. rango (proxy con ADX), y se
   comprueba que la estrategia no dependa de un solo mes o de un único
   régimen para ser rentable.

Estas pruebas son deliberadamente independientes del motor genético:
trabajan sobre CUALQUIER `BaseStrategy` ya instanciado con sus mejores
parámetros, sea que provenga del Descubridor, del Constructor de
Estrategias o de un archivo escrito a mano.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from core.types import BacktestConfig, BacktestResults
from engine.backtester import BacktestEngine
from strategies.base import BaseStrategy


# ---------------------------------------------------------------------------
# Generalización entre símbolos
# ---------------------------------------------------------------------------

@dataclass
class SymbolGeneralizationResult:
    symbol: str
    ok: bool
    sharpe: float = 0.0
    net_profit_pct: float = 0.0
    total_trades: int = 0
    error: str = ""
    # Desglose por régimen de mercado (Aletheia — mismas 3 etiquetas que el
    # resto de CapitalQuant, no el proxy ADX de 2 categorías usado más abajo
    # en run_regime_and_seasonality_test): regimen -> {net_profit_pct,
    # sharpe, n_trades}. Vacío si el cálculo de régimen falló para ese
    # símbolo (degradación segura, nunca invalida el resultado global).
    regime_breakdown: Dict[str, dict] = field(default_factory=dict)


@dataclass
class GeneralizationReport:
    origin_symbol: str
    results: List[SymbolGeneralizationResult] = field(default_factory=list)

    @property
    def n_tested(self) -> int:
        return len(self.results)

    @property
    def n_profitable(self) -> int:
        return sum(1 for r in self.results if r.ok and r.net_profit_pct > 0)

    @property
    def n_positive_sharpe(self) -> int:
        return sum(1 for r in self.results if r.ok and r.sharpe > 0)

    @property
    def generalization_rate(self) -> float:
        """Fracción de símbolos (excluyendo el de origen) donde la estrategia sigue rentable."""
        others = [r for r in self.results if r.symbol != self.origin_symbol]
        if not others:
            return 0.0
        return sum(1 for r in others if r.ok and r.net_profit_pct > 0) / len(others)

    def verdict(self) -> str:
        rate = self.generalization_rate
        if rate >= 0.6:
            return "GENERALIZA BIEN"
        elif rate >= 0.35:
            return "GENERALIZACIÓN PARCIAL"
        else:
            return "NO GENERALIZA (probable sobreajuste al símbolo de origen)"

    def regime_generalization_summary(self) -> Dict[str, dict]:
        """Para cada régimen de mercado, en cuántos de los símbolos
        probados (excluyendo el de origen) la estrategia fue rentable
        DENTRO de ese régimen — la versión "por régimen" de
        `generalization_rate`. Símbolos sin trades en un régimen dado no
        cuentan ni a favor ni en contra (no hay evidencia)."""
        others = [r for r in self.results if r.ok and r.symbol != self.origin_symbol and r.regime_breakdown]
        summary: Dict[str, dict] = {}
        if not others:
            return summary
        labels = set()
        for r in others:
            labels.update(r.regime_breakdown.keys())
        for label in labels:
            with_trades = [r for r in others if r.regime_breakdown.get(label, {}).get("n_trades", 0) > 0]
            profitable = [r for r in with_trades if r.regime_breakdown[label]["net_profit_pct"] > 0]
            summary[label] = {
                "n_symbols_with_trades": len(with_trades),
                "n_symbols_profitable": len(profitable),
                "rate": (len(profitable) / len(with_trades)) if with_trades else 0.0,
            }
        return summary


def _regime_breakdown_for_trades(df: pd.DataFrame, trades: list, timeframe: str) -> Dict[str, dict]:
    """Etiqueta cada vela de `df` con el régimen Aletheia (mismo motor de
    3 etiquetas que Backtesting/Discovery — no el proxy ADX de abajo) usando
    calibración por defecto (sin búsqueda de calibración por símbolo, para
    no multiplicar el tiempo de la prueba de generalización al correr sobre
    8-15 símbolos), y agrupa cada trade cerrado por el régimen vigente en
    su fecha de entrada. Degrada a `{}` si el cálculo de régimen falla —
    nunca invalida el resultado de generalización del símbolo."""
    try:
        from config.settings import TIMEFRAMES
        from regime.service import compute_regime_summary, REGIME_LABELS
        ann_factor = TIMEFRAMES.get(timeframe, {}).get("ann_factor", 252)
        summary = compute_regime_summary(df, ann_factor=ann_factor)
        regime_at = summary.regime_series
    except Exception as e:
        logger.warning(f"[Generalization] No se pudo calcular régimen para el desglose: {e}")
        return {}

    breakdown: Dict[str, dict] = {}
    for label in REGIME_LABELS:
        label_dates = set(df.index[regime_at == label])
        label_trades = [t for t in trades if t.entry_date in label_dates]
        n = len(label_trades)
        if n == 0:
            breakdown[label] = {"net_profit_pct": 0.0, "sharpe": 0.0, "n_trades": 0}
            continue
        pnls = [t.pnl_pct for t in label_trades]
        sharpe_proxy = (float(np.mean(pnls)) / float(np.std(pnls))) if np.std(pnls) > 0 else 0.0
        breakdown[label] = {"net_profit_pct": float(sum(pnls)), "sharpe": sharpe_proxy, "n_trades": n}
    return breakdown


def run_cross_symbol_generalization(
    strategy: BaseStrategy,
    datasets: Dict[str, pd.DataFrame],
    origin_symbol: str,
    config: Optional[BacktestConfig] = None,
    timeframe: str = "",
) -> GeneralizationReport:
    """
    Re-ejecuta `strategy` (ya instanciada con sus parámetros finales, sin
    volver a optimizar nada) sobre cada símbolo en `datasets`.

    Parameters
    ----------
    strategy : BaseStrategy
        Instancia ya parametrizada (ej. `strategy_class(**best_params)`).
    datasets : dict[str, pd.DataFrame]
        Mapa símbolo -> OHLCV. Debe incluir el símbolo de origen para
        que el reporte pueda excluirlo del cálculo de generalization_rate.
    origin_symbol : str
        Símbolo en el que la estrategia fue descubierta/optimizada.
    """
    cfg = config or BacktestConfig()
    engine = BacktestEngine(cfg)
    report = GeneralizationReport(origin_symbol=origin_symbol)

    def _run_one(symbol: str, df: pd.DataFrame) -> SymbolGeneralizationResult:
        try:
            signals = strategy.generate_signals(df.copy())
            res: BacktestResults = engine.run(
                signals, strategy_name=strategy.name, asset=symbol, timeframe=timeframe
            )
            return SymbolGeneralizationResult(
                symbol=symbol, ok=True,
                sharpe=res.sharpe_ratio,
                net_profit_pct=res.net_profit_pct,
                total_trades=res.total_trades,
                regime_breakdown=_regime_breakdown_for_trades(df, res.trades or [], timeframe),
            )
        except Exception as e:
            logger.warning(f"[Generalization] Falló en {symbol}: {e}")
            return SymbolGeneralizationResult(symbol=symbol, ok=False, error=str(e))

    # backend="threading": cada símbolo hace su propia copia local de `df` y
    # su propio `signals`/`res` — BacktestEngine.run() no muta `self` (solo
    # lee self.config), así que compartir la misma instancia entre hilos es
    # seguro. Se evita el backend por procesos para no tener que serializar
    # el objeto `strategy` (puede traer closures/estado no picklable).
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_run_one)(symbol, df) for symbol, df in datasets.items()
    )
    report.results = list(results)

    logger.info(
        f"[Generalization] {origin_symbol}: {report.n_profitable}/{report.n_tested} símbolos "
        f"rentables ({report.verdict()})"
    )
    return report


# ---------------------------------------------------------------------------
# Estabilidad de régimen y estacionalidad
# ---------------------------------------------------------------------------

@dataclass
class MonthlyStability:
    month: int
    net_profit_pct: float
    total_trades: int


@dataclass
class RegimeStability:
    regime: str            # "trend" | "range"
    net_profit_pct: float
    total_trades: int
    sharpe: float


@dataclass
class StabilityReport:
    monthly: List[MonthlyStability] = field(default_factory=list)
    regimes: List[RegimeStability] = field(default_factory=list)

    @property
    def profitable_months(self) -> int:
        return sum(1 for m in self.monthly if m.net_profit_pct > 0 and m.total_trades > 0)

    @property
    def months_with_trades(self) -> int:
        return sum(1 for m in self.monthly if m.total_trades > 0)

    def seasonality_verdict(self) -> str:
        if self.months_with_trades == 0:
            return "SIN DATOS SUFICIENTES"
        rate = self.profitable_months / self.months_with_trades
        if rate >= 0.75:
            return "ESTABLE (no depende de un mes puntual)"
        elif rate >= 0.5:
            return "MODERADA"
        else:
            return "POSIBLE ANOMALÍA DE CALENDARIO (depende de pocos meses)"

    def regime_verdict(self) -> str:
        profitable_regimes = [r for r in self.regimes if r.net_profit_pct > 0 and r.total_trades >= 3]
        if len(profitable_regimes) >= 2:
            return "FUNCIONA EN AMBOS REGÍMENES (tendencia y rango)"
        elif len(profitable_regimes) == 1:
            return f"SOLO RENTABLE EN RÉGIMEN DE {profitable_regimes[0].regime.upper()}"
        else:
            return "NO RENTABLE EN NINGÚN RÉGIMEN AISLADO"


def _adx_regime_mask(df: pd.DataFrame, period: int = 14, trend_threshold: float = 25.0) -> pd.Series:
    """
    Clasifica cada vela como régimen 'trend' o 'range' usando ADX, un proxy
    estándar y barato de calcular (no requiere reentrenar nada). ADX alto
    = mercado direccional; ADX bajo = mercado lateral.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()

    return (adx >= trend_threshold).fillna(False)


def run_regime_and_seasonality_test(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    asset: str = "",
    timeframe: str = "",
) -> StabilityReport:
    """
    Corre la estrategia UNA vez sobre todo el histórico (sin re-optimizar)
    y luego atribuye cada trade cerrado a (a) el mes calendario de su
    fecha de entrada y (b) el régimen ADX vigente en la fecha de entrada,
    para reportar si el resultado depende de un mes o un régimen puntual.
    """
    cfg = config or BacktestConfig()
    engine = BacktestEngine(cfg)
    signals = strategy.generate_signals(data.copy())
    results = engine.run(signals, strategy_name=strategy.name, asset=asset, timeframe=timeframe)

    report = StabilityReport()

    # --- Estacionalidad mensual ---
    for month in range(1, 13):
        month_trades = [t for t in results.trades if t.entry_date.month == month]
        pnl_pct = (
            sum(t.net_pnl for t in month_trades) / cfg.initial_capital * 100
            if month_trades else 0.0
        )
        report.monthly.append(MonthlyStability(
            month=month, net_profit_pct=pnl_pct, total_trades=len(month_trades)
        ))

    # --- Régimen ADX ---
    try:
        trend_mask = _adx_regime_mask(data)
        for regime_name, mask_value in (("trend", True), ("range", False)):
            regime_dates = set(data.index[trend_mask == mask_value])
            regime_trades = [t for t in results.trades if t.entry_date in regime_dates]
            if regime_trades:
                pnl_pct = sum(t.net_pnl for t in regime_trades) / cfg.initial_capital * 100
                wins = [t.net_pnl for t in regime_trades]
                sharpe_proxy = (np.mean(wins) / np.std(wins)) if np.std(wins) > 0 else 0.0
            else:
                pnl_pct, sharpe_proxy = 0.0, 0.0
            report.regimes.append(RegimeStability(
                regime=regime_name, net_profit_pct=pnl_pct,
                total_trades=len(regime_trades), sharpe=float(sharpe_proxy),
            ))
    except Exception as e:
        logger.warning(f"[Stability] No se pudo calcular régimen ADX: {e}")

    logger.info(
        f"[Stability] Estacionalidad: {report.seasonality_verdict()} | "
        f"Régimen: {report.regime_verdict()}"
    )
    return report
