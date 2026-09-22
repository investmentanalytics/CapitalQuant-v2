"""
regime/service.py
Capa fina sobre `aletheia.run_pipeline` (motor de régimen, el mismo de
Regime Engine Platform) adaptada a los timeframes/activos de CapitalQuant.

Todo el cálculo es CAUSAL (ver aletheia.regime_detector.classify_regimes_causal
y aletheia.transition_matrix.causal_expected_duration): el régimen asignado a
la vela N solo usa información de las velas 0..N. Esto es lo que permite
usar la serie de régimen para filtrar entradas de un backtest sin
introducir look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aletheia import SegmenterConfig, run_pipeline
from aletheia.regime_detector import RegimeClassification
from aletheia.segmenter import combined_regime_signal

REGIME_LABELS = ["Tendencia Alcista", "Tendencia Bajista", "Consolidación"]

# Parámetros del segmentador por timeframe. En vez de una tabla fija por
# cada clave de config.settings.TIMEFRAMES (17 timeframes distintos), se
# derivan de `ann_factor` (barras/año, ya definido ahí) para que el
# horizonte temporal REAL que ve el segmentador sea consistente sin
# importar la temporalidad elegida:
#   - suavizado / segmento mínimo ≈ 20 días de mercado
#   - ventana de tendencia         ≈ 50 días de mercado
# Esto reproduce a propósito los valores por defecto del segmentador
# (smoothing=20, trend=50, min_segment=20) para el timeframe Diario, y
# escala de forma proporcional para el resto.
_MIN_SMOOTHING = 5
_MIN_TREND = 20
_MAX_BARS_CAP = 1500  # evita ventanas absurdas en timeframes muy intradía


def _segmenter_config_for_timeframe(ann_factor: float, overrides: dict | None = None) -> SegmenterConfig:
    bars_per_day = max(ann_factor, 1) / 252.0
    smoothing = int(min(max(round(bars_per_day * 20), _MIN_SMOOTHING), _MAX_BARS_CAP))
    min_seg = int(min(max(round(bars_per_day * 20), _MIN_SMOOTHING), _MAX_BARS_CAP))
    trend = int(min(max(round(bars_per_day * 50), _MIN_TREND), _MAX_BARS_CAP * 2))
    # La ventana de calibración causal de umbrales (`causal_window`) debe
    # escalar junto con `min_segment_length`: por defecto SegmenterConfig
    # trae 500 velas fijas, calibrado pensando en Diario. En timeframes
    # intradía (más barras/día -> min_segment_length más grande) una
    # ventana fija de 500 puede terminar más pequeña que el propio
    # min_periods derivado de min_segment_length, lo cual además de ser
    # una calibración pobre (muy poco histórico para el percentil) podía
    # antes causar un error de cálculo -- ver la salvaguarda añadida en
    # `_causal_thresholds`/`_causal_thresholds_dynamic` para el caso
    # defensivo; aquí se corrige la causa real escalando la ventana.
    causal_window = int(min(max(round(bars_per_day * 500), min_seg * 3, 500), _MAX_BARS_CAP * 4))

    overrides = overrides or {}
    min_seg = max(1, int(round(min_seg * float(overrides.get("min_segment_mult", 1.0)))))
    causal_window = max(causal_window, min_seg * 3)

    kwargs = dict(
        smoothing_window=smoothing,
        price_trend_window=trend,
        min_segment_length=min_seg,
        causal_window=causal_window,
    )
    if "price_weight" in overrides:
        kwargs["price_weight"] = float(overrides["price_weight"])
    if "enter_percentile" in overrides:
        kwargs["enter_percentile"] = float(overrides["enter_percentile"])
    if "exit_percentile" in overrides:
        ep = float(overrides["exit_percentile"])
        # exit debe ser >= enter para que la histéresis siga teniendo sentido
        ep = max(ep, kwargs.get("enter_percentile", SegmenterConfig().enter_percentile))
        kwargs["exit_percentile"] = ep
    return SegmenterConfig(**kwargs)


@dataclass
class RegimeSummary:
    regime_series: pd.Series          # régimen por vela, alineado al índice del OHLC recibido
    confidence_series: pd.Series      # confianza del segmento vigente en cada vela
    strength_series: pd.Series        # señal continua [-1, 1] (precio+microestados) por vela
    classifications: list[RegimeClassification]
    segments_df: pd.DataFrame


def compute_regime_summary(df: pd.DataFrame, ann_factor: float = 252,
                            overrides: dict | None = None) -> RegimeSummary:
    """
    Corre el motor de régimen sobre un OHLC de CapitalQuant y devuelve,
    entre otras cosas, una Serie con el régimen vigente EN CADA VELA
    (no solo por segmento), lista para alinear con cualquier DataFrame
    de señales vía `.reindex(...)`.

    `overrides` permite afinar manualmente la sensibilidad del segmentador
    para este análisis puntual (ver `_segmenter_config_for_timeframe`):
    claves soportadas: price_weight, enter_percentile, exit_percentile,
    min_segment_mult. Ninguna cambia el carácter causal del pipeline —
    siguen calibrándose vela a vela solo con historia pasada.
    """
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"El OHLC no tiene las columnas requeridas: {missing}")

    ohlc = df.rename(columns={cols[c]: c for c in required})[required]
    seg_config = _segmenter_config_for_timeframe(ann_factor, overrides)
    result = run_pipeline(ohlc, segmenter_config=seg_config)

    chain_index = result.chain.index  # posiciones start_idx/end_idx son relativas a ESTE índice
    regime_series = pd.Series(index=chain_index, dtype=object)
    confidence_series = pd.Series(index=chain_index, dtype=float)
    for classification in result.classifications:
        seg = classification.segment
        idx_slice = chain_index[seg.start_idx:seg.end_idx]
        regime_series.loc[idx_slice] = classification.regime
        confidence_series.loc[idx_slice] = seg.confidence

    # Señal continua vela a vela (misma que decide los cortes de segmento),
    # útil como lectura de "fuerza de régimen" más rápida y granular que la
    # confianza por segmento, que solo se actualiza cuando un segmento se
    # confirma. Sigue siendo 100% causal (ver combined_regime_signal).
    strength_raw = combined_regime_signal(result.chain, seg_config, close=ohlc["close"])
    strength_series = pd.Series(strength_raw, index=chain_index, dtype=float)

    # `chain_index` puede excluir las primeras velas (sin histórico
    # suficiente para calcular wpr/señal); se reindexa al OHLC completo
    # para que el llamador siempre reciba una serie con el mismo índice
    # que le pasó. Esas primeras velas quedan NaN (correcto: aún no hay
    # régimen calculable ahí); las últimas SÍ deben quedar cubiertas,
    # ya que `chain_index` llega hasta la última vela igual que `ohlc`.
    regime_series = regime_series.reindex(ohlc.index)
    confidence_series = confidence_series.reindex(ohlc.index)
    strength_series = strength_series.reindex(ohlc.index)

    segments_df = result.segments_dataframe()
    return RegimeSummary(
        regime_series=regime_series,
        confidence_series=confidence_series,
        strength_series=strength_series,
        classifications=result.classifications,
        segments_df=segments_df,
    )


def compute_regime_series(df: pd.DataFrame, ann_factor: float = 252) -> pd.Series:
    """Atajo cuando solo hace falta la serie de régimen por vela."""
    return compute_regime_summary(df, ann_factor=ann_factor).regime_series


def regime_entry_mask(regime_series: pd.Series, allowed_regimes: list[str]) -> pd.Series:
    """
    Máscara booleana (misma longitud/índice que `regime_series`) que vale
    True solo en las velas cuyo régimen esté en `allowed_regimes`.

    Se usa para permitir NUEVAS entradas de una estrategia solo durante
    los regímenes elegidos (p.ej. "solo abrir largos en Tendencia
    Alcista"). No fuerza el cierre de una posición ya abierta cuando el
    régimen cambia a mitad de un trade: el motor de backtest sigue
    cerrándola por su lógica normal (señal contraria, stop loss o take
    profit), ya que forzar cierres por cambio de régimen es una decisión
    de estrategia distinta, no un simple filtro de entrada.
    """
    if not allowed_regimes:
        return pd.Series(True, index=regime_series.index)
    return regime_series.isin(allowed_regimes).fillna(False)
