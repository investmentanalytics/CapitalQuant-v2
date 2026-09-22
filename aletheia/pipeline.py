"""
Pipeline — Aletheia Regime Engine
====================================

Orquesta la arquitectura completa descrita en el proyecto:

    OHLC
      -> Clasificador de Microestados
      -> Cadena de Estados
      -> Modelo de Transición (Markov)
      -> Segmentador Temporal
      -> Detector de Régimen
      -> (Motor de Activación de Estrategias: fuera de alcance de este
         repo, pero `RegimeEngineResult` expone todo lo necesario para
         que CapitalQuant / Aletheia Discovery lo consuman)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .microstate_classifier import MicrostateConfig, classify_microstates
from .regime_detector import (RegimeClassification, RegimeRule,
                               build_adaptive_rules, classify_regimes,
                               classify_regimes_causal, summarize_regimes)
from .segmenter import Segment, SegmenterConfig, segment_chain
from .state_chain import StateChain, build_state_chain
from .transition_matrix import TransitionModel, build_transition_matrix, causal_expected_duration


@dataclass
class RegimeEngineResult:
    df: pd.DataFrame                       # OHLC + wpr + sig + state
    chain: StateChain
    transition_model: TransitionModel
    segments: list[Segment]
    classifications: list[RegimeClassification]
    summary: dict

    def segments_dataframe(self) -> pd.DataFrame:
        from .segmenter import segments_to_dataframe
        rows = segments_to_dataframe(self.segments)
        rows.insert(0, "regime", [c.regime for c in self.classifications])
        return rows


def run_pipeline(df: pd.DataFrame,
                  microstate_config: MicrostateConfig | None = None,
                  segmenter_config: SegmenterConfig | None = None,
                  regime_rules: list[RegimeRule] | None = None,
                  structure_profile_series: pd.Series | None = None,
                  structure_profile_table: dict[str, dict] | None = None) -> RegimeEngineResult:
    """
    Ejecuta el pipeline completo sobre un DataFrame OHLC.

    Parameters
    ----------
    df : DataFrame con columnas ['open','high','low','close'] (case-insensitive)
         e idealmente un DatetimeIndex.
    structure_profile_series : Serie opcional (índice alineable con `df`)
        con el perfil de estructura de precio VIGENTE en cada vela
        (típicamente `config.rolling_structure_profile`, calculado de
        forma causal). Si se proporciona junto con
        `structure_profile_table`, el segmentador usa la sensibilidad del
        perfil correspondiente a cada vela en vez de un único perfil fijo
        para todo el análisis -- ver `segmenter._causal_thresholds_dynamic`.
    """
    df_states = classify_microstates(df, microstate_config)
    chain = build_state_chain(df_states, state_col="state")

    n_states = int(df_states["state"].dropna().max()) + 1 if chain.states.size else 12
    n_states = max(n_states, 12)  # siempre reservar los 12 microestados definidos

    transition_model = build_transition_matrix(chain, n_states=n_states)

    close_col = next((c for c in df.columns if c.lower() == "close"), None)
    close_series = df[close_col] if close_col else None

    profile_ids = None
    if structure_profile_series is not None and len(chain.index):
        profile_ids = (structure_profile_series
                        .reindex(chain.index)
                        .ffill().bfill()
                        .to_numpy())

    # Duración esperada por microestado, aprendida causalmente de la
    # propia cadena (nunca mira transiciones futuras) -- ver
    # `causal_expected_duration` para el porqué. Esto es lo que finalmente
    # conecta la matriz de Markov con la confianza del régimen.
    expected_duration = causal_expected_duration(chain, n_states=n_states)

    segments = segment_chain(chain, n_states=n_states, config=segmenter_config,
                              close=close_series, profile_ids=profile_ids,
                              profile_table=structure_profile_table,
                              expected_duration=expected_duration)
    # Si el llamador no fija reglas explícitas, se clasifica de forma
    # CAUSAL: el umbral de polaridad para el segmento N se calibra solo con
    # los segmentos 0..N-1 (ver classify_regimes_causal) -- nunca con
    # segmentos futuros, para que esto sea válido para backtest/producción
    # y no solo para visualizar un histórico ya conocido.
    if regime_rules is not None:
        classifications = classify_regimes(segments, rules=regime_rules)
    else:
        classifications = classify_regimes_causal(segments)
    summary = summarize_regimes(classifications)

    return RegimeEngineResult(
        df=df_states,
        chain=chain,
        transition_model=transition_model,
        segments=segments,
        classifications=classifications,
        summary=summary,
    )
