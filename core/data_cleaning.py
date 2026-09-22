"""
core/data_cleaning.py
Limpieza y control de calidad de datos OHLCV para CapitalQuant.

FILOSOFÍA
---------
MT5 (única fuente de mercado de CapitalQuant) YA no genera velas cuando el mercado está cerrado
(fin de semana, feriados, fuera de sesión). El DataFrame que llega aquí no
tiene "huecos" en el sentido de filas vacías -- lo que se ve como espacio
en blanco en un gráfico de NASDAQ/Oro es solo el eje de tiempo real
(continuo) sin velas que dibujar ahí.

Por eso esta limpieza NO inventa velas para rellenar huecos (eso
contaminaría ATR, volatilidad, retornos y cualquier métrica estadística
con datos ficticios). En su lugar:

  1. Garantiza integridad básica de la serie: sin duplicados, ordenada,
     sin filas con OHLC nulo o inconsistente (ej. high < low).
  2. Distingue huecos NORMALES (cierre de fin de semana / feriado) de
     huecos SOSPECHOSOS (posible vela faltante por un problema del feed/
     broker dentro de horario de mercado) -- esto sí importa para el
     análisis estadístico, porque una vela real faltante sesga retornos,
     volatilidad y cualquier métrica basada en conteo de barras.
  3. Expone un reporte de calidad de datos por símbolo/timeframe, para
     que el usuario pueda confiar (o desconfiar, con motivo) de una
     serie antes de usarla en backtesting/optimización/discovery.
  4. Calcula qué fechas del calendario están completamente ausentes de
     la serie (fines de semana, feriados) -- usado por
     `visualization/charts.py` para ocultar esos huecos del EJE del
     gráfico (rangebreaks de Plotly), sin tocar los datos reales.
  5. Permite ELIMINAR los huecos sospechosos de la serie (no solo
     reportarlos): `remove_suspicious_gaps` recorta el DataFrame al tramo
     continuo y verificado más largo (o al más reciente, según se pida),
     para dejar una serie 100% limpia lista para backtesting/optimización/
     discovery -- sin inventar ni interpolar ninguna vela.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Duración de cada timeframe de CapitalQuant en minutos (para detectar huecos).
# Debe reflejar las claves que usa config.settings.TIMEFRAMES / mt5_provider.
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1, "2m": 2, "3m": 3, "4m": 4, "5m": 5, "6m": 6,
    "10m": 10, "12m": 12, "15m": 15, "20m": 20, "30m": 30,
    "1h": 60, "2h": 120, "3h": 180, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "1w": 10080, "1M": 43200,
}

# Multiplicador sobre la duración esperada del timeframe a partir del cual
# un hueco (que no coincide con un cierre de fin de semana) se considera
# sospechoso y se reporta.
MAX_NORMAL_GAP_MULTIPLIER = 3

# Margen de días de calendario adicionales, más allá del fin de semana
# (sábado/domingo) que caiga dentro del hueco, que se toleran como cierre
# normal por FESTIVOS bursátiles pegados al fin de semana (Viernes Santo,
# Navidad, Año Nuevo, etc.). Sin este margen, cualquier festivo que mueve
# el último cierre de viernes a jueves (o el reinicio de lunes a martes)
# se marcaba como "sospechoso" sin serlo -- CapitalQuant no mantiene un
# calendario de festivos por bolsa/símbolo, así que en vez de eso tolera
# un hueco de hasta `fin_de_semana + este margen` días de calendario,
# venga del día de la semana que venga.
MAX_HOLIDAY_EXTRA_DAYS = 2


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza mínima e integridad de un DataFrame OHLCV indexado por
    DatetimeIndex (formato usado en todo CapitalQuant).

    - Quita duplicados de timestamp (se queda con la última vela recibida,
      que suele ser la más reciente/consolidada).
    - Ordena cronológicamente.
    - Descarta filas con OHLC nulo o inconsistente (high < low, o
      valores <= 0 en un activo que no debería tenerlos).

    No rellena ni interpola nada -- ver el docstring del módulo.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if cols:
        df = df.dropna(subset=cols)
        # high siempre >= low; si no, la vela es inválida (error de feed)
        if "high" in df.columns and "low" in df.columns:
            df = df[df["high"] >= df["low"]]
        # Precios negativos o cero no tienen sentido en OHLC de mercado
        df = df[(df[cols] > 0).all(axis=1)]

    return df


def detect_suspicious_gaps(
    df: pd.DataFrame,
    timeframe: str,
    max_gap_multiplier: int = MAX_NORMAL_GAP_MULTIPLIER,
    max_holiday_days: int = MAX_HOLIDAY_EXTRA_DAYS,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con los huecos SOSPECHOSOS de la serie: aquellos
    que no se explican por un cierre normal de fin de semana (+ un pequeño
    margen de festivos pegados a él) y que exceden `max_gap_multiplier`
    veces la duración esperada del timeframe.

    Un hueco se considera cierre NORMAL (no sospechoso) si su duración en
    días de calendario no supera los días de fin de semana (sábado/domingo)
    que contiene más `max_holiday_days` de margen -- esto cubre festivos
    bursátiles habituales (Viernes Santo, Navidad, Año Nuevo, ...) que
    desplazan el último cierre de viernes a jueves, o el reinicio de lunes
    a martes, sin que sea un error real de datos. Un hueco de semanas o
    meses (una vela real faltante del feed) sigue quedando fuera de ese
    margen y se reporta igual.

    Columnas devueltas: gap_start, gap_end, gap_minutes, gap_bars_missing.
    Un resultado vacío no garantiza datos perfectos, pero sí que no hay
    huecos evidentes dentro de horario de mercado.
    """
    if df is None or df.empty or len(df) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_minutes", "gap_bars_missing"])

    tf_min = TIMEFRAME_MINUTES.get(timeframe)
    if not tf_min:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_minutes", "gap_bars_missing"])

    idx = df.index
    prev_ts = pd.Series(idx, index=idx).shift(1)
    delta_min = idx.to_series().diff().dt.total_seconds() / 60.0

    limite = tf_min * max_gap_multiplier
    candidato = (delta_min > limite).values

    es_cierre_normal = np.zeros(len(idx), dtype=bool)
    for i in np.where(candidato)[0]:
        g_start = prev_ts.iloc[i]
        g_end = idx[i]
        calendar_days = (g_end.normalize() - g_start.normalize()).days
        if calendar_days <= 0:
            es_cierre_normal[i] = True
            continue
        dias_intermedios = pd.date_range(
            g_start.normalize() + pd.Timedelta(days=1), g_end.normalize(), freq="D",
        )
        weekend_days = int((dias_intermedios.dayofweek >= 5).sum())
        # El margen de festivos solo se aplica si el hueco realmente
        # atraviesa un fin de semana -- si no, un día laboral aislado
        # genuinamente perdido (sin fin de semana de por medio) debe seguir
        # marcándose como sospechoso, no tolerarse por este margen.
        es_cierre_normal[i] = weekend_days >= 1 and calendar_days <= weekend_days + max_holiday_days

    sospechoso = candidato & (~es_cierre_normal)

    if not sospechoso.any():
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_minutes", "gap_bars_missing"])

    reporte = pd.DataFrame({
        "gap_end": idx[sospechoso],
        "gap_minutes": delta_min[sospechoso].values,
    })
    reporte["gap_start"] = reporte["gap_end"] - pd.to_timedelta(reporte["gap_minutes"], unit="m")
    reporte["gap_bars_missing"] = (reporte["gap_minutes"] / tf_min).round().astype(int) - 1
    return reporte[["gap_start", "gap_end", "gap_minutes", "gap_bars_missing"]].reset_index(drop=True)


def data_quality_summary(df: pd.DataFrame, timeframe: str, symbol: str = "") -> dict:
    """
    Resumen compacto de calidad de datos, pensado para mostrarse en la UI
    (métricas o un expander) antes de usar la serie en análisis estadístico.
    """
    if df is None or df.empty:
        return {
            "symbol": symbol, "timeframe": timeframe, "velas": 0,
            "desde": None, "hasta": None, "huecos_sospechosos": 0,
            "velas_faltantes_estimadas": 0,
        }

    huecos = detect_suspicious_gaps(df, timeframe)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "velas": len(df),
        "desde": df.index.min(),
        "hasta": df.index.max(),
        "huecos_sospechosos": len(huecos),
        "velas_faltantes_estimadas": int(huecos["gap_bars_missing"].sum()) if not huecos.empty else 0,
    }


def remove_suspicious_gaps(
    df: pd.DataFrame,
    timeframe: str,
    max_gap_multiplier: int = MAX_NORMAL_GAP_MULTIPLIER,
    keep: str = "longest",
) -> tuple[pd.DataFrame, dict]:
    """
    Elimina de la serie los huecos SOSPECHOSOS (no solo los reporta): parte
    el DataFrame en los puntos donde hay un hueco sospechoso y devuelve
    solo un tramo continuo y verificado -- sin ningún hueco sospechoso
    dentro de horario de mercado.

    No inventa ni interpola velas (seguiría contaminando ATR/volatilidad/
    retornos con datos ficticios, ver el docstring del módulo). En su
    lugar descarta el/los tramo(s) alrededor del hueco, quedándose con
    datos 100% reales.

    Parameters
    ----------
    keep : "longest" (por defecto) -- conserva el tramo continuo con más
        velas de todo el histórico, maximizando el tamaño de la serie
        limpia resultante.
        "latest" -- conserva únicamente el tramo más reciente (desde el
        último hueco sospechoso hasta el final), útil cuando lo que
        importa es el estado actual del mercado más que el histórico
        completo.

    Returns
    -------
    (df_limpio, info) donde info resume cuántas velas y qué huecos se
    descartaron, para mostrarlo en la UI antes/después de aplicar la
    limpieza.
    """
    if df is None or df.empty:
        return df, {
            "velas_originales": 0, "velas_resultantes": 0,
            "velas_eliminadas": 0, "huecos_eliminados": 0, "tramos": 1,
        }

    gaps = detect_suspicious_gaps(df, timeframe, max_gap_multiplier)
    n_original = len(df)

    if gaps.empty:
        return df, {
            "velas_originales": n_original, "velas_resultantes": n_original,
            "velas_eliminadas": 0, "huecos_eliminados": 0, "tramos": 1,
        }

    # Puntos de corte: cada `gap_end` marca el inicio de un tramo nuevo.
    cut_points = pd.DatetimeIndex(gaps["gap_end"]).sort_values()
    idx = df.index
    segment_id = idx.searchsorted(cut_points, side="left")
    # Vector de segmento por vela: 0 antes del primer corte, 1 entre el
    # primer y segundo corte, etc.
    seg_labels = np.searchsorted(cut_points.values, idx.values, side="left")

    segments = []
    for seg in range(seg_labels.max() + 1):
        segment_df = df[seg_labels == seg]
        if not segment_df.empty:
            segments.append(segment_df)

    if not segments:
        return df, {
            "velas_originales": n_original, "velas_resultantes": n_original,
            "velas_eliminadas": 0, "huecos_eliminados": 0, "tramos": 1,
        }

    if keep == "latest":
        df_clean = segments[-1]
    else:  # "longest"
        df_clean = max(segments, key=len)

    # Recalcular huecos sospechosos dentro del tramo elegido -- debe dar 0.
    huecos_restantes = detect_suspicious_gaps(df_clean, timeframe, max_gap_multiplier)

    info = {
        "velas_originales": n_original,
        "velas_resultantes": len(df_clean),
        "velas_eliminadas": n_original - len(df_clean),
        "huecos_eliminados": len(gaps),
        "huecos_restantes": len(huecos_restantes),
        "tramos": len(segments),
    }
    return df_clean, info


def missing_calendar_dates(index: pd.DatetimeIndex, cap: int = 5000) -> list[str]:
    """
    Fechas completas del calendario (YYYY-MM-DD) que NO tienen ninguna vela
    en la serie -- típicamente fines de semana y feriados de mercado.

    Se usa exclusivamente para el EJE del gráfico (Plotly `rangebreaks`),
    de modo que las velas queden visualmente pegadas sin espacios vacíos,
    sin alterar en absoluto los datos usados en el análisis estadístico.
    """
    if index is None or len(index) < 2:
        return []

    dias_con_datos = pd.DatetimeIndex(index.normalize().unique())
    todos_los_dias = pd.date_range(dias_con_datos.min(), dias_con_datos.max(), freq="D", tz=index.tz)
    faltantes = todos_los_dias.difference(dias_con_datos)

    if len(faltantes) > cap:
        faltantes = faltantes[:cap]  # límite de seguridad para series muy largas

    return [d.strftime("%Y-%m-%d") for d in faltantes]
