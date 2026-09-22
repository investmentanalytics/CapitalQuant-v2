"""
ui/components/regime_filter.py
Widget reutilizable: filtro por régimen de mercado para Backtesting,
Optimización y el Descubridor Genético. Permite restringir NUEVAS
entradas de la estrategia a uno o varios regímenes (Tendencia Alcista /
Tendencia Bajista / Consolidación), calculados con el mismo motor
causal usado en la herramienta "Régimen de Mercado" (ver
`regime/service.py`) y, desde aquí, con la MISMA calibración
auto-optimizada por activo (ver `regime/calibration.py`) que usa esa
página -- ninguna parte del sistema que filtra por régimen usa una
calibración distinta a las demás.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from config.settings import TIMEFRAMES
from regime.service import REGIME_LABELS, compute_regime_summary, regime_entry_mask
from regime.calibration import get_cached_calibration


@st.cache_data(ttl=600, show_spinner=False)
def _cached_regime_series(df: pd.DataFrame, ann_factor: float, asset: str, timeframe: str) -> pd.Series:
    calib = get_cached_calibration(df, ann_factor, asset, timeframe)
    return compute_regime_summary(df, ann_factor=ann_factor, overrides=calib.overrides).regime_series


def render_regime_filter(asset: str, timeframe: str, key_prefix: str):
    """
    Renderiza el selector de régimen(es) permitido(s).

    Returns
    -------
    allowed_regimes: list[str]
        Lista vacía = sin filtro (comportamiento de siempre, todas las
        velas habilitadas para entrar).
    """
    enabled = st.toggle(
        "Filtrar por régimen de mercado",
        value=False,
        key=f"{key_prefix}_regime_toggle",
        help=(
            "Restringe las NUEVAS entradas de la estrategia a las velas "
            "clasificadas en el/los régimen(es) elegidos (detector de "
            "régimen causal, sin look-ahead). Las posiciones abiertas "
            "siguen cerrándose por su lógica normal (señal contraria, "
            "stop loss o take profit) aunque el régimen cambie a mitad "
            "del trade."
        ),
    )
    if not enabled:
        return []

    selected = st.multiselect(
        "Régimen(es) permitido(s) para entrar",
        REGIME_LABELS,
        default=[REGIME_LABELS[0]],
        key=f"{key_prefix}_regime_select",
    )
    return selected


def compute_regime_mask(df: pd.DataFrame, timeframe: str, allowed_regimes: list[str],
                         asset: str = "") -> pd.Series | None:
    """
    Serie booleana (mismo índice que `df`) True solo en las velas cuyo
    régimen esté en `allowed_regimes`, usando la calibración auto-optimizada
    para `asset`/`timeframe` (ver `regime/calibration.py`). Devuelve None si
    `allowed_regimes` está vacío (sin filtro) o si el cálculo del régimen falla.
    """
    if not allowed_regimes:
        return None
    ann_factor = TIMEFRAMES.get(timeframe, {}).get("ann_factor", 252)
    try:
        regime_series = _cached_regime_series(df, ann_factor, asset, timeframe)
    except Exception as e:
        st.warning(f"No se pudo calcular el régimen de mercado ({e}); se ignora el filtro.")
        return None
    return regime_entry_mask(regime_series, allowed_regimes)


def apply_regime_filter(df: pd.DataFrame, signals: pd.DataFrame, asset: str,
                         timeframe: str, allowed_regimes: list[str]) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Si `allowed_regimes` no está vacío, pone a 0 la columna 'signal' de
    `signals` en toda vela cuyo régimen NO esté en `allowed_regimes` —
    es decir, bloquea nuevas entradas fuera de esos regímenes sin tocar
    el resto de la lógica de la estrategia ni el motor de backtest.

    Returns
    -------
    (signals_filtrado, regime_series | None) — la serie de régimen se
    devuelve también para poder pintarla en el gráfico de resultados.
    """
    if not allowed_regimes or "signal" not in signals.columns:
        return signals, None

    ann_factor = TIMEFRAMES.get(timeframe, {}).get("ann_factor", 252)
    try:
        regime_series = _cached_regime_series(df, ann_factor, asset, timeframe)
    except Exception as e:
        st.warning(f"No se pudo calcular el régimen de mercado ({e}); se ignora el filtro.")
        return signals, None

    mask = regime_entry_mask(regime_series.reindex(signals.index), allowed_regimes)
    filtered = signals.copy()
    filtered.loc[~mask, "signal"] = 0
    return filtered, regime_series
