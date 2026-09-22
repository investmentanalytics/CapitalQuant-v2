"""
Módulo 3 — Modelo de Transición (Markov)
===========================================

Construye automáticamente, a partir de la StateChain, todo lo necesario
para "aprender" el comportamiento del mercado en términos de
microestados:

  - Matriz de transición de primer orden (probabilidades P(siguiente | actual)).
  - Persistencia media de cada microestado (cuántas velas dura antes de cambiar).
  - Duración promedio de cada racha.
  - Secuencias (n-gramas) más frecuentes y más raras.
  - Búsqueda de qué suele ocurrir después de una subsecuencia dada, p.ej.
    "0 → 1 → 2 → 3 → 6" y con qué probabilidad.

No contiene reglas manuales: todo se deriva de los datos que se le pasen.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .state_chain import StateChain


@dataclass
class TransitionModel:
    n_states: int
    matrix: np.ndarray                     # shape (n_states, n_states), filas suman 1
    counts: np.ndarray                     # conteos brutos de transiciones
    state_counts: np.ndarray               # nº de apariciones de cada estado
    avg_persistence: dict[int, float] = field(default_factory=dict)
    avg_duration: dict[int, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Matriz de transición como DataFrame legible (filas=actual, cols=siguiente)."""
        return pd.DataFrame(
            self.matrix,
            index=[f"state_{i}" for i in range(self.n_states)],
            columns=[f"state_{i}" for i in range(self.n_states)],
        )

    def next_state_probs(self, current_state: int) -> pd.Series:
        """Distribución de probabilidad del siguiente estado dado el actual."""
        return pd.Series(self.matrix[current_state], name=f"P(next | state={current_state})")

    def probability_of_sequence(self, sequence: list[int]) -> float:
        """
        Probabilidad (bajo el modelo de Markov de 1er orden) de observar
        la secuencia completa dada, multiplicando las transiciones.
        """
        if len(sequence) < 2:
            return 1.0
        prob = 1.0
        for a, b in zip(sequence[:-1], sequence[1:]):
            prob *= self.matrix[a, b]
        return prob


def causal_expected_duration(chain: StateChain, n_states: int | None = None,
                              min_history: int = 100, laplace: float = 1.0) -> np.ndarray:
    """
    Duración esperada (en velas) del microestado vigente en CADA vela,
    estimada de forma CAUSAL: en la vela i se usa únicamente lo aprendido
    de las transiciones 0..i-1 (nunca futuras). Antes de `min_history`
    transiciones observadas para ese estado en particular, se usa un
    valor neutro (2.0 velas, la duración esperada de un estado con
    P(quedarse)=0.5) en vez de un estimador con poca base estadística.

    Por qué existe: antes de esto, `build_transition_matrix` (y toda la
    cadena de Markov -- persistencia, duración media, n-gramas más
    frecuentes) se calculaba en el pipeline pero NO alimentaba ni la
    segmentación ni la confianza del régimen; quedaba como estadística
    descriptiva sin cerrar el loop. Esta función es lo que permite usarla
    de verdad: comparar si un segmento duró lo que históricamente duran
    segmentos dominados por ese microestado, o si es anormalmente
    corto/largo (ver `segmenter.segment_chain`, componente `persistence_fit`
    de la confianza).
    """
    states = chain.states
    n = len(states)
    if n_states is None:
        n_states = int(states.max()) + 1 if n else 0

    stay_counts = np.zeros(n_states)
    total_counts = np.zeros(n_states)
    expected = np.full(n, 2.0)

    for i in range(n):
        s = int(states[i])
        total_seen = total_counts[s]
        if total_seen >= min_history:
            p_stay = float(np.clip((stay_counts[s] + laplace) / (total_seen + 2 * laplace), 0.01, 0.99))
            expected[i] = 1.0 / (1.0 - p_stay)
        # Actualizar contadores DESPUÉS de fijar `expected[i]`, con la
        # transición i -> i+1 que en tiempo real recién se acaba de
        # observar (así la vela i nunca se beneficia de su propia
        # transición saliente).
        if i + 1 < n:
            total_counts[s] += 1
            if int(states[i + 1]) == s:
                stay_counts[s] += 1

    return expected


def build_transition_matrix(chain: StateChain, n_states: int | None = None,
                             laplace_smoothing: float = 1e-6) -> TransitionModel:
    """
    Construye la matriz de transición de Markov de primer orden a partir
    de la cadena de microestados observada.
    """
    states = chain.states
    if n_states is None:
        n_states = int(states.max()) + 1 if len(states) else 0

    counts = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        counts[a, b] += 1

    state_counts = np.bincount(states, minlength=n_states).astype(float)

    smoothed = counts + laplace_smoothing
    row_sums = smoothed.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix = smoothed / row_sums

    # Persistencia: P(quedarse en el mismo estado)
    avg_persistence = {i: float(matrix[i, i]) for i in range(n_states)}

    # Duración media de cada racha (run-length) por estado
    runs = chain.runs()
    durations: dict[int, list[int]] = {i: [] for i in range(n_states)}
    for state, start, end in runs:
        durations.setdefault(state, []).append(end - start)
    avg_duration = {
        s: float(np.mean(d)) if d else 0.0 for s, d in durations.items()
    }

    return TransitionModel(
        n_states=n_states,
        matrix=matrix,
        counts=counts,
        state_counts=state_counts,
        avg_persistence=avg_persistence,
        avg_duration=avg_duration,
    )


def find_frequent_sequences(chain: StateChain, length: int = 4,
                             top_n: int = 15) -> pd.DataFrame:
    """
    Cuenta la frecuencia de todas las subsecuencias (n-gramas) de longitud
    `length` observadas en la cadena, y devuelve las `top_n` más
    frecuentes junto con su frecuencia relativa.
    """
    states = chain.states
    if len(states) < length:
        return pd.DataFrame(columns=["sequence", "count", "frequency"])

    counter: Counter[tuple[int, ...]] = Counter()
    for i in range(len(states) - length + 1):
        counter[tuple(int(s) for s in states[i:i + length])] += 1

    total = sum(counter.values())
    rows = [
        {"sequence": seq, "count": cnt, "frequency": cnt / total}
        for seq, cnt in counter.most_common(top_n)
    ]
    return pd.DataFrame(rows)


def find_rare_sequences(chain: StateChain, length: int = 4,
                         top_n: int = 15, min_count: int = 1) -> pd.DataFrame:
    """Igual que `find_frequent_sequences` pero para las menos frecuentes."""
    states = chain.states
    if len(states) < length:
        return pd.DataFrame(columns=["sequence", "count", "frequency"])

    counter: Counter[tuple[int, ...]] = Counter()
    for i in range(len(states) - length + 1):
        counter[tuple(int(s) for s in states[i:i + length])] += 1

    total = sum(counter.values())
    rare = [item for item in counter.items() if item[1] >= min_count]
    rare.sort(key=lambda kv: kv[1])
    rows = [
        {"sequence": seq, "count": cnt, "frequency": cnt / total}
        for seq, cnt in rare[:top_n]
    ]
    return pd.DataFrame(rows)


def what_follows(chain: StateChain, prefix: list[int],
                  horizon: int = 1, top_n: int = 5) -> pd.DataFrame:
    """
    Dado un prefijo de microestados (p.ej. [0, 1, 2, 3, 6]), busca en el
    histórico todas las veces que ocurrió ese prefijo exacto y calcula
    qué microestado (o secuencia de longitud `horizon`) tendió a seguir,
    con su probabilidad empírica.

    Esto es exactamente lo que permite descubrir patrones como
    "0 → 1 → 2 → 3 → 6 precede una consolidación": aquí devolvemos la
    distribución empírica de continuaciones; la interpretación en
    términos de "régimen" la hace el Detector de Régimen (módulo 5)
    sobre los segmentos que resultan de esas continuaciones.
    """
    states = chain.states
    p = len(prefix)
    if len(states) < p + horizon:
        return pd.DataFrame(columns=["continuation", "count", "probability"])

    counter: Counter[tuple[int, ...]] = Counter()
    matches = 0
    for i in range(len(states) - p - horizon + 1):
        window = states[i:i + p]
        if list(window) == list(prefix):
            matches += 1
            cont = tuple(int(s) for s in states[i + p:i + p + horizon])
            counter[cont] += 1

    if matches == 0:
        return pd.DataFrame(columns=["continuation", "count", "probability"])

    rows = [
        {"continuation": cont, "count": cnt, "probability": cnt / matches}
        for cont, cnt in counter.most_common(top_n)
    ]
    return pd.DataFrame(rows)
