"""
regime/calibration.py
Auto-calibración del Detector de Régimen POR ACTIVO.

Objetivo: el usuario no debería tener que mover sliders a mano. Este
módulo prueba un conjunto acotado de configuraciones del segmentador
(`aletheia.SegmenterConfig`) sobre el historial del activo/timeframe
elegido y elige la que mejor separa el comportamiento del PRECIO por
régimen -- sin usar ninguna etiqueta externa, porque no existe un
"régimen verdadero" contra el cual entrenar. La señal que se usa para
elegir es enteramente posterior a la clasificación (retornos futuros,
volatilidad realizada, estabilidad de la clasificación), nunca al revés.

CÓMO SE EVITA EL SOBREAJUSTE (importante, léase antes de confiar en esto):

1. Grid acotado, no búsqueda libre. Se prueban ~27 combinaciones fijas
   de (price_weight, enter_percentile, min_segment_mult); no se optimiza
   sobre un espacio continuo con un optimizador libre, que encontraría
   con facilidad una combinación que ajusta el ruido de esta muestra en
   particular ("hackeo del hiperparámetro").

2. Validación por sub-periodos, NO una sola puntuación agregada. La
   muestra se parte en `N_FOLDS` tramos contiguos con purga (embargo)
   entre ellos. Cada candidato se puntúa en CADA tramo por separado y
   se exige que funcione de forma consistente en todos -- se penaliza
   con la desviación estándar entre tramos, no solo se promedia. Un
   candidato que solo funciona en un tramo de suerte queda descartado
   aunque su promedio simple sea alto.
   (Esto no es un split supervisado train/test clásico -- aquí no hay
   una etiqueta que "entrenar"; es una prueba de estabilidad temporal,
   en el mismo espíritu que el resto de validación walk-forward/CPCV
   que ya usa CapitalQuant en Optimización.)

3. Regularización hacia el default. Se resta una pequeña penalización
   proporcional a qué tan lejos está el candidato de la configuración
   por defecto (la calibración ya razonable y documentada del motor).
   Ante la duda, o si dos candidatos puntúan parecido, gana el más
   cercano al default -- el sistema solo se aleja de él si los datos
   lo justifican con claridad.

4. Candidatos degenerados se descartan. Si una configuración produce
   muy pocos segmentos o un régimen casi vacío en algún tramo, ese
   tramo se marca inválido y penaliza fuertemente al candidato en vez
   de ignorarse.

Ninguna de estas salvaguardas elimina el riesgo de sobreajuste al
100% -- ninguna calibración automática puede, cuando no existe una
etiqueta verdadera contra la cual medir "acierto". Lo que sí garantizan
es que la elección no dependa de un solo tramo de historia ni se aleje
del default sin una razón consistente en el tiempo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from aletheia import SegmenterConfig  # noqa: F401 (re-exported para quien construya overrides a mano)
from regime.service import compute_regime_summary, REGIME_LABELS

# --- Grid acotado ----------------------------------------------------
_PRICE_WEIGHT_GRID = (0.5, 0.7, 0.9)
_ENTER_PCTL_GRID = (0.50, 0.60, 0.70)
_MIN_SEG_MULT_GRID = (0.75, 1.0, 1.5)
_EXIT_OFFSET = 0.20  # exit_percentile = enter + offset (mantiene la histéresis fija)

_DEFAULT_OVERRIDES = dict(price_weight=0.70, enter_percentile=0.60,
                           exit_percentile=0.80, min_segment_mult=1.0)

N_FOLDS = 4
EMBARGO_FRAC = 0.02          # % de la muestra purgado a cada lado de cada corte de fold
MIN_BARS_PER_REGIME = 20     # por debajo de esto, el fold se considera degenerado para ese candidato
REGULARIZATION_WEIGHT = 0.15
CONSISTENCY_PENALTY_WEIGHT = 1.0
WHIPSAW_PENALTY_WEIGHT = 2.0
CONSOLIDACION_WEIGHT = 0.5


@dataclass
class CalibrationResult:
    overrides: dict
    is_default: bool                 # True si ganó el default (o nada superó al default con margen)
    final_score: float
    default_score: float
    fold_scores: list = field(default_factory=list)      # puntuación del candidato ganador, por fold
    n_candidates_evaluated: int = 0
    n_candidates_valid: int = 0


def _param_grid() -> list[dict]:
    grid = []
    for pw, ep, msm in product(_PRICE_WEIGHT_GRID, _ENTER_PCTL_GRID, _MIN_SEG_MULT_GRID):
        grid.append(dict(
            price_weight=pw,
            enter_percentile=ep,
            exit_percentile=min(ep + _EXIT_OFFSET, 0.95),
            min_segment_mult=msm,
        ))
    return grid


def _normalized_distance(overrides: dict, default: dict = _DEFAULT_OVERRIDES) -> float:
    """Distancia euclídea normalizada [0,1]-ish entre un candidato y el default,
    usada solo para regularizar (penalizar) el alejamiento del default."""
    spans = dict(price_weight=0.5, enter_percentile=0.25, exit_percentile=0.25, min_segment_mult=1.5)
    total = 0.0
    for k, span in spans.items():
        total += ((overrides.get(k, default[k]) - default[k]) / span) ** 2
    return float(np.sqrt(total) / np.sqrt(len(spans)))


def _fold_bounds(n: int, n_folds: int, embargo_frac: float) -> list[tuple[int, int]]:
    """Devuelve [(start, end), ...] posiciones de cada fold, ya con el
    embargo recortado en sus bordes (para no puntuar barras justo pegadas
    al corte, donde el suavizado/histéresis del segmentador aún arrastra
    información del tramo vecino)."""
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    embargo = max(1, int(n * embargo_frac))
    bounds = []
    for i in range(n_folds):
        start = edges[i] + (embargo if i > 0 else 0)
        end = edges[i + 1] - (embargo if i < n_folds - 1 else 0)
        if end - start > MIN_BARS_PER_REGIME * 3:
            bounds.append((start, end))
    return bounds


def _score_fold(close: pd.Series, regime_series: pd.Series, start: int, end: int,
                 fwd_k: int) -> float | None:
    """Puntúa un candidato en UN tramo. None = tramo degenerado para este candidato
    (muy pocos segmentos, un régimen casi vacío, etc.) -- se penaliza aparte,
    no se ignora silenciosamente."""
    seg_regime = regime_series.iloc[start:end]
    seg_close = close.iloc[start:end]
    if seg_regime.isna().mean() > 0.5:
        return None

    counts = seg_regime.value_counts()
    if any(counts.get(r, 0) < MIN_BARS_PER_REGIME for r in REGIME_LABELS):
        return None  # algún régimen casi no aparece en este tramo -> no hay nada que separar

    fwd_ret = seg_close.pct_change(fwd_k).shift(-fwd_k)
    std_fwd = fwd_ret.std()
    if not std_fwd or np.isnan(std_fwd) or std_fwd == 0:
        return None

    mean_up = fwd_ret[seg_regime == REGIME_LABELS[0]].mean()
    mean_down = fwd_ret[seg_regime == REGIME_LABELS[1]].mean()
    if np.isnan(mean_up) or np.isnan(mean_down):
        return None
    separation = (mean_up - mean_down) / std_fwd  # tipo Cohen's-d: separación normalizada

    ret_1 = seg_close.pct_change()
    vol_trend = ret_1[seg_regime.isin(REGIME_LABELS[:2])].std()
    vol_flat = ret_1[seg_regime == REGIME_LABELS[2]].std()
    if vol_trend and not np.isnan(vol_trend) and vol_trend > 0:
        consolidacion_quality = float(np.clip(1.0 - (vol_flat / vol_trend), -1.0, 1.0))
    else:
        consolidacion_quality = 0.0

    n_switches = int((seg_regime != seg_regime.shift(1)).sum())
    whipsaw_rate = n_switches / max(len(seg_regime), 1) * 100  # cambios por 100 velas

    return float(separation + CONSOLIDACION_WEIGHT * consolidacion_quality
                  - WHIPSAW_PENALTY_WEIGHT * whipsaw_rate / 100.0)


def _aggregate_fold_scores(fold_scores: list[float | None]) -> tuple[float, list[float]]:
    valid = [s for s in fold_scores if s is not None]
    n_invalid = len(fold_scores) - len(valid)
    if not valid:
        return -999.0, []
    mean_s = float(np.mean(valid))
    std_s = float(np.std(valid)) if len(valid) > 1 else 0.0
    # penaliza tanto la inconsistencia entre folds válidos como los folds
    # que resultaron degenerados para este candidato (ninguno de los dos
    # se puede compensar simplemente con un promedio alto en los demás)
    penalty = CONSISTENCY_PENALTY_WEIGHT * std_s + 0.5 * n_invalid
    return mean_s - penalty, valid


def calibrate_regime_detector(df: pd.DataFrame, ann_factor: float) -> CalibrationResult:
    """
    Corre el grid completo sobre `df` y devuelve la configuración ganadora
    (o el default, si nada lo supera con margen suficiente).

    `df` debe traer como mínimo la columna 'close' (o 'Close'); se usa el
    OHLC completo internamente vía `compute_regime_summary`.
    """
    cols = {c.lower(): c for c in df.columns}
    close = df[cols.get("close", "close")]
    n = len(df)
    bars_per_day = max(ann_factor, 1) / 252.0
    fwd_k = int(np.clip(round(bars_per_day * 5), 3, 60))  # ~5 días de mercado hacia adelante

    fold_bounds = _fold_bounds(n, N_FOLDS, EMBARGO_FRAC)
    if len(fold_bounds) < 2:
        # historial demasiado corto para validar con consistencia -> no hay
        # base razonable para desviarse del default.
        return CalibrationResult(overrides=dict(_DEFAULT_OVERRIDES), is_default=True,
                                  final_score=0.0, default_score=0.0)

    candidates = _param_grid()
    results = []
    for overrides in candidates:
        try:
            summary = compute_regime_summary(df, ann_factor=ann_factor, overrides=overrides)
        except Exception:
            continue
        regime_series = summary.regime_series.reset_index(drop=True)
        close_reset = close.reset_index(drop=True)
        fold_scores = [_score_fold(close_reset, regime_series, s, e, fwd_k) for s, e in fold_bounds]
        agg_score, valid_scores = _aggregate_fold_scores(fold_scores)
        reg_penalty = REGULARIZATION_WEIGHT * _normalized_distance(overrides)
        final = agg_score - reg_penalty
        results.append((final, overrides, valid_scores))

    # Puntúa también el default explícitamente, para poder compararlo
    # directamente contra el ganador del grid (diagnóstico honesto).
    try:
        default_summary = compute_regime_summary(df, ann_factor=ann_factor, overrides=_DEFAULT_OVERRIDES)
        default_regime = default_summary.regime_series.reset_index(drop=True)
        close_reset = close.reset_index(drop=True)
        default_fold_scores = [_score_fold(close_reset, default_regime, s, e, fwd_k) for s, e in fold_bounds]
        default_agg, _ = _aggregate_fold_scores(default_fold_scores)
    except Exception:
        default_agg = -999.0

    if not results:
        return CalibrationResult(overrides=dict(_DEFAULT_OVERRIDES), is_default=True,
                                  final_score=default_agg, default_score=default_agg,
                                  n_candidates_evaluated=len(candidates))

    results.sort(key=lambda r: r[0], reverse=True)
    best_score, best_overrides, best_fold_scores = results[0]
    n_valid = sum(1 for r in results if r[0] > -900)

    # Margen mínimo: el ganador debe superar al default por más que el
    # ruido esperado entre folds (aprox. la penalización de consistencia
    # de un candidato perfectamente estable) -- si no, no vale la pena
    # abandonar el default por una mejora que bien podría ser ruido.
    margin = 0.05
    if best_score <= default_agg + margin:
        return CalibrationResult(
            overrides=dict(_DEFAULT_OVERRIDES), is_default=True,
            final_score=default_agg, default_score=default_agg,
            fold_scores=[s for s in default_fold_scores if s is not None],
            n_candidates_evaluated=len(candidates), n_candidates_valid=n_valid,
        )

    return CalibrationResult(
        overrides=best_overrides, is_default=False,
        final_score=best_score, default_score=default_agg,
        fold_scores=best_fold_scores,
        n_candidates_evaluated=len(candidates), n_candidates_valid=n_valid,
    )


# --- Caché en memoria (independiente de Streamlit, para poder testear
# este módulo sin levantar la app) ------------------------------------
_cache: dict[tuple, CalibrationResult] = {}


def get_cached_calibration(df: pd.DataFrame, ann_factor: float, asset: str, timeframe: str,
                            force_recompute: bool = False) -> CalibrationResult:
    """
    Punto de entrada único: cachea por (activo, timeframe, rango de fechas,
    nº de velas) -- se recalcula solo cuando cambia el activo, el rango
    elegido, o llega historial nuevo. `force_recompute=True` ignora la
    caché (botón "Recalibrar" en la UI).
    """
    key = (asset, timeframe, len(df),
           df.index[0] if len(df) else None, df.index[-1] if len(df) else None)
    if not force_recompute and key in _cache:
        return _cache[key]
    result = calibrate_regime_detector(df, ann_factor)
    _cache[key] = result
    return result


def regime_feature_frame(df: pd.DataFrame, ann_factor: float, asset: str,
                          timeframe: str) -> tuple[pd.DataFrame, dict]:
    """Expone el régimen de mercado (con la MISMA calibración auto-optimizada
    por activo que usa el resto del sistema) como columnas de features, para
    que el Descubridor Genético pueda tratarlo como una condición atómica más
    en vez de solo como un filtro post-hoc que enmascara la señal.

    100% causal (hereda esa propiedad de `compute_regime_summary`): la vela N
    solo usa información de 0..N, así que no hay look-ahead al usarlo como
    condición de entrada.

    Columnas devueltas (alineadas al índice de `df`):
      - regime_bullish / regime_bearish / regime_consolidation: booleanas,
        una por cada REGIME_LABELS.
      - regime_confidence: confianza [0, 1] del segmento vigente en esa vela.
      - regime_strength: señal continua [-1, 1] (misma que decide los cortes
        de segmento) — más granular que la confianza, útil para condiciones
        de "fuerza de régimen" con umbral.

    Si el cálculo falla por cualquier motivo, devuelve un DataFrame con las
    mismas columnas en False/NaN en vez de propagar la excepción — así una
    regla que use estas condiciones simplemente nunca dispara (degradación
    segura) en vez de romper toda la búsqueda genética.

    Returns
    -------
    (features, overrides): además del DataFrame, devuelve el diccionario de
    overrides de calibración efectivamente usado. Congelarlo y guardarlo
    junto a cualquier estrategia descubierta que use la familia "Regime" es
    lo que permite que el código exportado (`discovery/codegen.py`)
    reproduzca EXACTAMENTE el mismo régimen visto durante el descubrimiento,
    en vez de recalibrar (y potencialmente obtener algo distinto) en cada
    ejecución posterior.
    """
    cols = ["regime_bullish", "regime_bearish", "regime_consolidation",
            "regime_confidence", "regime_strength"]
    try:
        calib = get_cached_calibration(df, ann_factor, asset, timeframe)
        summary = compute_regime_summary(df, ann_factor=ann_factor, overrides=calib.overrides)
        out = pd.DataFrame(index=df.index)
        out["regime_bullish"] = (summary.regime_series == REGIME_LABELS[0]).fillna(False)
        out["regime_bearish"] = (summary.regime_series == REGIME_LABELS[1]).fillna(False)
        out["regime_consolidation"] = (summary.regime_series == REGIME_LABELS[2]).fillna(False)
        out["regime_confidence"] = summary.confidence_series.reindex(df.index)
        out["regime_strength"] = summary.strength_series.reindex(df.index)
        return out, dict(calib.overrides)
    except Exception:
        out = pd.DataFrame(index=df.index)
        for c in cols[:3]:
            out[c] = False
        for c in cols[3:]:
            out[c] = float("nan")
        return out, {}
