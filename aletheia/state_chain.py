"""
Módulo 2 — Cadena de Microestados
====================================

Toma la columna `state` producida por el clasificador y la convierte en
una secuencia limpia (sin NaNs) lista para el análisis de Markov y la
segmentación temporal. No añade información nueva: su responsabilidad
única es garantizar una cadena de enteros consistente y expuesta con una
API simple.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StateChain:
    """Secuencia de microestados alineada con un índice temporal."""

    index: pd.Index
    states: np.ndarray  # dtype int

    def __len__(self) -> int:
        return len(self.states)

    def as_series(self) -> pd.Series:
        return pd.Series(self.states, index=self.index, name="state")

    def runs(self) -> list[tuple[int, int, int]]:
        """
        Devuelve las "rachas" (runs) de estados consecutivos idénticos
        como lista de tuplas (estado, pos_inicio, pos_fin_exclusivo).
        Útil para medir persistencia de un microestado.
        """
        result: list[tuple[int, int, int]] = []
        if len(self.states) == 0:
            return result
        start = 0
        current = self.states[0]
        for i in range(1, len(self.states)):
            if self.states[i] != current:
                result.append((int(current), start, i))
                start = i
                current = self.states[i]
        result.append((int(current), start, len(self.states)))
        return result


def build_state_chain(df_with_state: pd.DataFrame,
                       state_col: str = "state") -> StateChain:
    """
    Construye una StateChain a partir de un DataFrame que contenga la
    columna de microestados (típicamente la salida de
    `classify_microstates`). Descarta las filas iniciales sin estado
    (NaN) por falta de histórico suficiente.
    """
    clean = df_with_state.dropna(subset=[state_col])
    states = clean[state_col].astype(int).to_numpy()
    return StateChain(index=clean.index, states=states)
