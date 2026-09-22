"""
Módulo 1 — Clasificador de Microestados
=========================================

Traduce la lógica del indicador Pine Script original (basado en Williams %R
y una línea de señal "sig") a un clasificador discreto que devuelve un
entero (0-11) por vela, en vez de un color RGB.

IMPORTANTE — supuesto de diseño:
El script Pine original usa las variables `wpr` (Williams %R) y `sig`
(una señal suavizada de `wpr`) sin especificar explícitamente cómo se
calcula `sig`. Aquí se asume la convención más habitual en este tipo de
indicadores: `sig` es una media móvil (por defecto EMA) de `wpr`.
Ambos parámetros (período de %R y período/tipo de la señal) son
configurables mediante `MicrostateConfig`, así que si tu indicador
original usa otros períodos, ajústalos ahí — el resto del pipeline no
depende de estos valores concretos, solo de la columna `state` que
produce este módulo.

Códigos de microestado (idénticos a la tabla original):

    0  Verde          Impulso alcista fuerte
    1  Verde claro     Continuación alcista
    2  Verde oscuro    Alcista agotándose
    3  Amarillo        Corrección alcista
    4  Naranja         Corrección bajista
    5  Rojo            Impulso bajista
    6  Magenta         Cambio de estructura
    7  Morado          Bajista extremo
    8  Cian            Recuperación
    9  Azul claro      Recuperación fuerte
    10 Azul oscuro     Cruce alcista (crossunder wpr, -50)
    11 Fucsia          Cruce bajista (crossover wpr, -50)

Este módulo es intencionalmente independiente y reutilizable: no sabe
nada de cadenas de estados, matrices de transición ni regímenes. En el
futuro puedes sustituir su lógica interna (o añadir microestados nuevos)
sin tocar el resto del sistema, siempre que siga devolviendo una columna
entera `state` alineada con el índice temporal de entrada.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class Microstate(IntEnum):
    """Nombres legibles para cada código de microestado."""

    IMPULSO_ALCISTA_FUERTE = 0        # Verde
    CONTINUACION_ALCISTA = 1          # Verde claro
    ALCISTA_AGOTANDOSE = 2            # Verde oscuro
    CORRECCION_ALCISTA = 3            # Amarillo
    CORRECCION_BAJISTA = 4            # Naranja
    IMPULSO_BAJISTA = 5               # Rojo
    CAMBIO_DE_ESTRUCTURA = 6          # Magenta
    BAJISTA_EXTREMO = 7               # Morado
    RECUPERACION = 8                  # Cian
    RECUPERACION_FUERTE = 9           # Azul claro
    CRUCE_ALCISTA = 10                # Azul oscuro
    CRUCE_BAJISTA = 11                # Fucsia


MICROSTATE_LABELS: dict[int, str] = {m.value: m.name for m in Microstate}

MICROSTATE_COLORS: dict[int, str] = {
    0: "#00C853",   # Verde
    1: "#69F0AE",   # Verde claro
    2: "#1B5E20",   # Verde oscuro
    3: "#FFEB3B",   # Amarillo
    4: "#FF9800",   # Naranja
    5: "#F44336",   # Rojo
    6: "#E91E63",   # Magenta
    7: "#6A1B9A",   # Morado
    8: "#00BCD4",   # Cian
    9: "#40C4FF",   # Azul claro
    10: "#1A237E",  # Azul oscuro
    11: "#FF4081",  # Fucsia
}


@dataclass
class MicrostateConfig:
    """Parámetros del clasificador."""

    wpr_period: int = 21
    """Período de Williams %R."""

    signal_period: int = 13
    """Período de la media móvil usada como línea de señal `sig`."""

    signal_type: str = "ema"
    """Tipo de media para `sig`: 'ema' o 'sma'."""

    threshold_high: float = -27.0
    """Umbral superior usado para separar estados 1 / 2 y 8 / 9."""

    threshold_mid: float = -50.0
    """Umbral central de Williams %R (equivalente al -50 del script)."""

    threshold_low: float = -72.0
    """Umbral inferior usado para separar estados 6 / 7."""


def williams_pct_r(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int) -> pd.Series:
    """Williams %R clásico, en el rango [-100, 0]."""
    highest_high = high.rolling(period, min_periods=period).max()
    lowest_low = low.rolling(period, min_periods=period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    wpr = -100.0 * (highest_high - close) / denom
    return wpr


def _signal_line(wpr: pd.Series, period: int, kind: str) -> pd.Series:
    if kind.lower() == "sma":
        return wpr.rolling(period, min_periods=period).mean()
    # EMA por defecto
    return wpr.ewm(span=period, adjust=False, min_periods=period).mean()


def classify_microstates(df: pd.DataFrame,
                          config: MicrostateConfig | None = None) -> pd.DataFrame:
    """
    Calcula el microestado (0-11) para cada vela de `df`.

    Parameters
    ----------
    df : DataFrame con columnas ['open', 'high', 'low', 'close'] (case-insensitive)
         e índice temporal (DatetimeIndex recomendado, aunque no obligatorio).
    config : MicrostateConfig, opcional.

    Returns
    -------
    DataFrame igual a `df` más las columnas:
        - 'wpr'    : Williams %R
        - 'sig'    : línea de señal
        - 'state'  : microestado entero (0-11), NaN mientras no haya
                     suficiente histórico para calcular wpr/sig.
    """
    config = config or MicrostateConfig()
    cols = {c.lower(): c for c in df.columns}
    for required in ("high", "low", "close"):
        if required not in cols:
            raise ValueError(f"Falta la columna requerida '{required}' en el DataFrame")

    high = df[cols["high"]].astype(float)
    low = df[cols["low"]].astype(float)
    close = df[cols["close"]].astype(float)

    wpr = williams_pct_r(high, low, close, config.wpr_period)
    sig = _signal_line(wpr, config.signal_period, config.signal_type)

    out = df.copy()
    out["wpr"] = wpr
    out["sig"] = sig

    wpr_prev = wpr.shift(1)
    sig_prev = sig.shift(1)

    rising_sig = sig >= sig_prev
    falling_sig = sig <= sig_prev
    rising_wpr = wpr > wpr_prev
    falling_wpr = wpr < wpr_prev

    thr_mid = config.threshold_mid
    thr_high = config.threshold_high
    thr_low = config.threshold_low

    state = pd.Series(np.nan, index=df.index, dtype="float64")

    # Réplica exacta del orden de condiciones del script Pine.
    # (En Pine, cada `if` posterior puede sobrescribir al anterior si
    # también se cumple; replicamos ese comportamiento aplicando las
    # condiciones en el mismo orden con np.where encadenado / máscara.)
    cond0 = (wpr < thr_mid) & rising_sig & rising_wpr
    cond1 = (wpr > thr_mid) & (wpr < thr_high) & rising_sig & rising_wpr
    cond2 = (wpr > thr_high) & rising_sig & rising_wpr
    cond3 = (wpr > thr_mid) & falling_sig & falling_wpr
    cond4 = (wpr < thr_mid) & falling_sig & falling_wpr
    cond5 = (wpr > thr_mid) & rising_sig & falling_wpr
    cond6 = (wpr < thr_mid) & (wpr > thr_low) & rising_sig & falling_wpr
    cond7 = (wpr < thr_low) & rising_sig & falling_wpr
    cond8 = (wpr < thr_mid) & falling_sig & rising_wpr
    cond9 = (wpr > thr_mid) & falling_sig & rising_wpr

    crossunder = (wpr < thr_mid) & (wpr_prev >= thr_mid)
    crossover = (wpr > thr_mid) & (wpr_prev <= thr_mid)

    for value, cond in [
        (0, cond0), (1, cond1), (2, cond2), (3, cond3), (4, cond4),
        (5, cond5), (6, cond6), (7, cond7), (8, cond8), (9, cond9),
        (10, crossunder), (11, crossover),
    ]:
        state = state.where(~cond.fillna(False), value)

    out["state"] = state
    return out


def get_state_labels() -> dict[int, str]:
    """Devuelve el mapeo {codigo: nombre_legible}."""
    return dict(MICROSTATE_LABELS)
