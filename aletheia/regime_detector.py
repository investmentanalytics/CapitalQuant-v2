"""
Módulo 5 — Detector de Régimen
=================================

Clasifica cada Segmento (producido por el módulo 4) en un régimen de
mercado: Alcista, Bajista o Consolidación, usando la polaridad media y
la entropía del segmento como señales principales.

Diseño extensible: los regímenes están definidos como una lista de
`RegimeRule` evaluadas en orden; para añadir un nuevo régimen (p.ej.
"Alta Volatilidad" o "Ruptura") basta con añadir una regla nueva, sin
tocar el resto del núcleo.

La salida es un régimen POR SEGMENTO, nunca por vela.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .segmenter import Segment


@dataclass
class RegimeRule:
    name: str
    condition: Callable[[Segment], bool]
    priority: int = 0
    """Reglas con prioridad más alta se evalúan primero."""


@dataclass
class RegimeClassification:
    segment: Segment
    regime: str


DEFAULT_RULES: list[RegimeRule] = [
    RegimeRule(
        name="Tendencia Alcista",
        condition=lambda seg: seg.avg_polarity > 0.2,
        priority=10,
    ),
    RegimeRule(
        name="Tendencia Bajista",
        condition=lambda seg: seg.avg_polarity < -0.2,
        priority=10,
    ),
    RegimeRule(
        name="Consolidación",
        condition=lambda seg: True,   # regla por defecto (catch-all)
        priority=0,
    ),
]


def build_adaptive_rules(segments: list[Segment], percentile: float = 0.55,
                          floor: float = 0.08, ceiling: float = 0.5) -> list[RegimeRule]:
    """
    Igual que `DEFAULT_RULES`, pero con el umbral de `avg_polarity`
    calibrado a la propia distribución de los segmentos de ESTE análisis,
    en vez de un ±0.2 fijo.

    ADVERTENCIA -- LOOK-AHEAD BIAS: esta función calibra el umbral con
    TODOS los segmentos pasados como argumento en una sola pasada, así que
    si se le pasan los segmentos de todo el histórico, el umbral usado
    para clasificar un segmento del año pasado ya "sabe" cómo se comportó
    la polaridad en segmentos futuros. Eso es información que no existía
    en tiempo real y sobreestima artificialmente cualquier backtest hecho
    sobre estas clasificaciones.

    Se mantiene por compatibilidad (útil para una foto fija / exploratoria
    de "cómo se vería" con la distribución completa), pero el pipeline por
    defecto usa `classify_regimes_causal`, que jamás mira segmentos
    futuros. Para clasificación pensada en backtest/producción, usa esa.
    """
    if not segments:
        return DEFAULT_RULES

    abs_polarities = np.abs([s.avg_polarity for s in segments])
    if len(segments) < 4:
        thr = floor
    else:
        thr = float(np.clip(np.quantile(abs_polarities, percentile), floor, ceiling))

    return [
        RegimeRule(name="Tendencia Alcista",
                   condition=lambda seg, t=thr: seg.avg_polarity > t, priority=10),
        RegimeRule(name="Tendencia Bajista",
                   condition=lambda seg, t=thr: seg.avg_polarity < -t, priority=10),
        RegimeRule(name="Consolidación", condition=lambda seg: True, priority=0),
    ]


def classify_regimes_causal(segments: list[Segment], percentile: float = 0.55,
                             floor: float = 0.08, ceiling: float = 0.5,
                             min_history: int = 4) -> list[RegimeClassification]:
    """
    Clasifica cada segmento en régimen SIN look-ahead bias: el umbral de
    `avg_polarity` para clasificar el segmento N se calibra usando
    únicamente los segmentos 0..N-1 (los que ya habían ocurrido en tiempo
    real), nunca el segmento actual ni los posteriores.

    Con menos de `min_history` segmentos previos, se usa `floor` de forma
    conservadora (no hay historial suficiente para un percentil fiable) --
    igual que hacía `build_adaptive_rules`, pero ahora respetando el orden
    temporal en vez de usar el conjunto completo de una sola vez.

    Es el reemplazo directo de `build_adaptive_rules` + `classify_regimes`
    para cualquier uso donde importe que la clasificación sea válida para
    backtest o para producción en tiempo real (no solo para visualizar).
    """
    results: list[RegimeClassification] = []
    history_abs_polarity: list[float] = []

    for seg in segments:
        if len(history_abs_polarity) < min_history:
            thr = floor
        else:
            thr = float(np.clip(np.quantile(history_abs_polarity, percentile), floor, ceiling))

        if seg.avg_polarity > thr:
            regime = "Tendencia Alcista"
        elif seg.avg_polarity < -thr:
            regime = "Tendencia Bajista"
        else:
            regime = "Consolidación"

        results.append(RegimeClassification(segment=seg, regime=regime))
        history_abs_polarity.append(abs(seg.avg_polarity))

    return results


def classify_regimes(segments: list[Segment],
                      rules: list[RegimeRule] | None = None) -> list[RegimeClassification]:
    """
    Clasifica cada segmento en un régimen según las reglas dadas
    (o `DEFAULT_RULES` si no se especifican). Las reglas se evalúan por
    prioridad descendente y gana la primera que se cumpla.
    """
    rules = sorted(rules or DEFAULT_RULES, key=lambda r: -r.priority)
    results: list[RegimeClassification] = []
    for seg in segments:
        chosen = "Sin clasificar"
        for rule in rules:
            if rule.condition(seg):
                chosen = rule.name
                break
        results.append(RegimeClassification(segment=seg, regime=chosen))
    return results


def summarize_regimes(classifications: list[RegimeClassification]) -> dict[str, dict]:
    """Estadísticas agregadas por régimen: nº de segmentos, velas totales,
    duración media, confianza media."""
    summary: dict[str, dict] = {}
    for c in classifications:
        s = summary.setdefault(c.regime, {
            "num_segments": 0, "total_bars": 0, "confidences": [],
        })
        s["num_segments"] += 1
        s["total_bars"] += c.segment.duration_bars
        s["confidences"].append(c.segment.confidence)

    for regime, s in summary.items():
        confs = s.pop("confidences")
        s["avg_confidence"] = sum(confs) / len(confs) if confs else 0.0
        s["avg_duration_bars"] = s["total_bars"] / s["num_segments"] if s["num_segments"] else 0.0

    return summary
