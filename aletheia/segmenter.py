"""
Módulo 4 — Segmentador Temporal
==================================

En vez de clasificar régimen vela a vela, este módulo agrupa la cadena
de microestados en bloques (segmentos) temporalmente coherentes.

Estrategia:
  1. Se asigna a cada microestado una "polaridad" continua en [-1, 1]
     (definida en `STATE_POLARITY`, derivada directamente del
     significado de cada microestado: alcista fuerte = +1,
     bajista extremo = -1, etc.)
  2. Se suaviza esa polaridad con una media móvil (para reducir ruido).
  3. Se detectan puntos de cambio en la polaridad suavizada usando
     histéresis (dos umbrales, uno para entrar en una zona y otro más
     estricto para salir), lo que evita que el ruido cause cambios de
     segmento constantes — exactamente el problema que un score
     tradicional no resuelve.
  4. Cada segmento resultante se describe con estadísticas completas:
     fechas de inicio/fin, duración, distribución de microestados,
     entropía de la secuencia y una "confianza" basada en la consistencia
     interna del segmento.

La salida de este módulo son *segmentos*, no vela por vela — que es
justo lo que luego consume el Detector de Régimen (módulo 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .state_chain import StateChain

# Polaridad de cada microestado: qué tan "alcista" (+1) o "bajista" (-1)
# es conceptualmente, según la tabla de significados original.
STATE_POLARITY: dict[int, float] = {
    0: 1.0,    # Impulso alcista fuerte
    1: 0.6,    # Continuación alcista
    2: 0.2,    # Alcista agotándose
    3: 0.1,    # Corrección alcista (aún dentro de tendencia alcista)
    4: -0.1,   # Corrección bajista (aún dentro de tendencia bajista)
    5: -1.0,   # Impulso bajista
    6: -0.3,   # Cambio de estructura (ambiguo, ligeramente negativo)
    7: -0.8,   # Bajista extremo
    8: 0.3,    # Recuperación
    9: 0.7,    # Recuperación fuerte
    10: -0.4,  # Cruce alcista de wpr hacia abajo del umbral (bajista técnico)
    11: 0.4,   # Cruce bajista de wpr hacia arriba del umbral (alcista técnico)
}


@dataclass
class Segment:
    start_idx: int
    end_idx: int                 # exclusivo
    start_time: object
    end_time: object
    states: list[int]
    dominant_states: list[int]
    state_distribution: dict[int, float]
    entropy: float
    avg_polarity: float
    change_probability: float
    confidence: float

    @property
    def duration_bars(self) -> int:
        return self.end_idx - self.start_idx

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_bars": self.duration_bars,
            "dominant_states": self.dominant_states,
            "entropy": round(self.entropy, 4),
            "avg_polarity": round(self.avg_polarity, 4),
            "change_probability": round(self.change_probability, 4),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class SegmenterConfig:
    smoothing_window: int = 20
    """Ventana de la media móvil aplicada a la polaridad de microestados."""

    enter_threshold: float = 0.20
    """Umbral (en valor absoluto) para considerar que empieza una nueva zona."""

    exit_threshold: float = 0.40
    """
    Umbral más exigente para confirmar el cambio de zona (histéresis).
    Debe ser >= enter_threshold. Cuanto mayor la diferencia entre ambos,
    más resistente es el segmentador al ruido.
    """

    min_segment_length: int = 20
    """Duración mínima (en velas) para que un segmento se mantenga aislado;
    segmentos más cortos se fusionan con el vecino más similar."""

    price_trend_window: int = 50
    """
    Ventana (en velas) usada para medir la fuerza de tendencia directamente
    sobre el precio (pendiente normalizada, similar a un ADX con signo).
    Esta señal es mucho menos ruidosa que la polaridad de microestados por
    sí sola, y es la que principalmente decide si un tramo es tendencial
    o lateral.
    """

    price_weight: float = 0.7
    """
    Peso (0-1) de la señal de tendencia basada en precio frente a la
    polaridad basada en microestados en la señal combinada que alimenta
    al segmentador. 1.0 = solo precio, 0.0 = solo microestados.
    """

    adaptive_thresholds: bool = True
    """
    Si es True (recomendado), `enter_threshold`/`exit_threshold` dejan de
    usarse como valores absolutos fijos y pasan a ser un PISO MÍNIMO: los
    umbrales reales se calibran a partir de la distribución empírica de la
    señal combinada (`combined_regime_signal`) de la propia serie que se
    está analizando — no de un valor fijo pensado para datos sintéticos.

    Por qué: los datos sintéticos originales usan bloques de drift muy
    limpios, así que la señal combinada cruza fácilmente ±0.20 y todo
    funciona bien con un umbral fijo. Los datos reales (ETH, US500, XAU,
    forex, etc.) son mucho más ruidosos, cada uno con su propia escala
    de volatilidad y asimetría (rachas de tendencia fuerte + tramos
    planos largos), así que un umbral fijo casi nunca se detectan cortes
    de segmento en algunos activos y en otros produce ruido excesivo.
    """

    threshold_method: str = "percentile"
    """
    Cómo se calibra el umbral cuando `adaptive_thresholds=True`:

    - 'percentile' (por defecto, más robusto): usa percentiles empíricos
      de |señal| de la propia serie. No asume ninguna forma de
      distribución, así que funciona igual de bien con señales de cola
      gorda o asimétricas (típico en mercados reales) que con señales
      limpias tipo sintético.
    - 'std': usa un múltiplo de la desviación estándar (método anterior).
      Más simple, pero asume una distribución razonablemente simétrica;
      puede des-calibrarse en activos con rachas de tendencia muy
      extremas y poco frecuentes (colas gordas), que inflan la σ y hacen
      el umbral demasiado alto para el resto de la serie.
    """

    enter_percentile: float = 0.60
    """Percentil de |señal combinada| usado como umbral de entrada cuando
    `threshold_method='percentile'`. 0.60 = el 40% de las velas con señal
    más extrema (en valor absoluto) se consideran candidatas a tendencia."""

    exit_percentile: float = 0.80
    """Percentil de |señal combinada| usado como umbral de salida
    (histéresis) cuando `threshold_method='percentile'`."""

    enter_threshold_std: float = 0.5
    """Múltiplo de la desviación estándar de la señal usado como umbral
    de entrada cuando `threshold_method='std'`."""

    exit_threshold_std: float = 1.0
    """Múltiplo de la desviación estándar de la señal usado como umbral
    de salida (histéresis) cuando `adaptive_thresholds=True`."""

    max_threshold: float = 0.6
    """Techo absoluto para los umbrales adaptativos, para que un activo
    extremadamente volátil no termine con un umbral tan alto que nunca se
    detecten cambios de régimen."""

    causal_window: int | None = 500
    """
    Ventana MÓVIL (en velas) usada para calibrar los umbrales de forma
    causal: en cada vela, el umbral se recalcula usando solo las últimas
    `causal_window` velas anteriores (nunca datos futuros). `None` o un
    valor >= al tamaño de la serie equivale a ventana expansiva (usa todo
    el histórico disponible HASTA esa vela). Ver `_causal_thresholds` para
    el porqué esto es necesario -- la versión anterior calibraba con la
    serie completa (pasado y futuro), lo cual es look-ahead bias y invalida
    cualquier backtest hecho sobre esos umbrales.
    """

    causal_min_periods: int | None = None
    """
    Nº mínimo de velas de histórico antes de confiar en el percentil móvil;
    con menos que esto se usa el piso (`enter_threshold`/`exit_threshold`)
    de forma conservadora. Por defecto: max(30, min_segment_length * 2).
    """


def _polarity_series(chain: StateChain) -> np.ndarray:
    return np.array([STATE_POLARITY.get(int(s), 0.0) for s in chain.states])


def price_trend_strength(close: pd.Series, window: int = 50) -> np.ndarray:
    """
    Mide la fuerza y dirección de la tendencia directamente sobre el
    precio: para cada punto, ajusta una recta a los últimos `window`
    valores de log(precio) y normaliza la pendiente por la volatilidad
    de los retornos en esa misma ventana (un t-stat de la pendiente,
    análogo en espíritu a un ADX con signo). El resultado se comprime a
    [-1, 1] con tanh, así que es directamente comparable a la polaridad
    de microestados.

    +1  -> tendencia alcista fuerte y limpia (poco ruido relativo al movimiento)
     0  -> sin tendencia clara / movimiento lateral
    -1  -> tendencia bajista fuerte y limpia

    Implementación vectorizada (convolución + rolling de pandas) en vez
    de un bucle Python por vela: procesar varios millones de velas es
    cuestión de segundos en vez de minutos/horas, algo imprescindible si
    luego se entrena el modelo de transición con histórico masivo como
    se planteó en el diseño original.
    """
    log_p = np.log(close.to_numpy(dtype=float))
    n = len(log_p)
    strength = np.zeros(n)
    if n < window:
        return strength

    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    weights = x - x_mean               # pesos centrados, sum(weights)=0
    denom = (weights ** 2).sum()

    # sum_i weights[i] * y[t-window+1+i]  para cada t, vía convolución
    # (equivale exactamente a la covarianza no normalizada de la
    # regresión lineal en cada ventana, mismo resultado que un
    # polyfit(x, y, 1) por ventana pero con complejidad efectiva O(n)).
    numerator = np.convolve(log_p, weights[::-1], mode="valid")
    slopes = numerator / denom          # longitud n - window + 1

    returns = pd.Series(np.diff(log_p, prepend=log_p[0]))
    vol = returns.rolling(window, min_periods=window).std().to_numpy()
    vol_aligned = vol[window - 1:]      # alinear con `slopes`

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(
            vol_aligned > 1e-12,
            (slopes * window) / (vol_aligned * np.sqrt(window)),
            0.0,
        )

    strength[window - 1:] = np.tanh(t_stat / 3.0)
    strength[:window - 1] = strength[window - 1]
    return strength


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    # 'same' con relleno por los bordes usando el propio valor
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _entropy(states: list[int], n_states: int) -> float:
    if not states:
        return 0.0
    counts = np.bincount(states, minlength=n_states).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _causal_thresholds(smoothed_polarity: np.ndarray,
                        config: SegmenterConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Versión CAUSAL de `_resolve_thresholds` -- corrige un look-ahead bias
    real que tenía la versión anterior.

    La versión anterior calibraba enter/exit con `np.quantile()` sobre TODA
    la serie de una sola vez (pasado Y futuro respecto a cualquier vela).
    Eso significa que el umbral usado para decidir el régimen "de hoy" ya
    incorporaba información de cómo se movió el precio en los meses
    siguientes -- información que en tiempo real, en el momento de esa
    vela, no podía existir. Un backtest sobre esos umbrales sería
    optimista de forma artificial (el problema clásico que hace que una
    estrategia se vea genial en backtest y falle en vivo).

    Aquí cada vela recibe su propio umbral, calculado con una ventana
    MÓVIL de las últimas `causal_window` velas (nunca datos posteriores).
    Antes de acumular `causal_min_periods` velas de histórico, se usa el
    piso (`enter_threshold`/`exit_threshold`) de forma conservadora, en vez
    de arriesgarse a un percentil poco fiable con pocos datos.
    """
    n = len(smoothed_polarity)
    if not config.adaptive_thresholds or n == 0:
        return (np.full(n, config.enter_threshold), np.full(n, config.exit_threshold))

    abs_signal = pd.Series(np.abs(smoothed_polarity))
    min_periods = config.causal_min_periods or max(30, config.min_segment_length * 2)
    window = config.causal_window if (config.causal_window and config.causal_window < n) else n
    # min_periods no puede exceder window (pandas lo rechaza) -- puede pasar
    # cuando min_segment_length crece (p.ej. timeframes intradía con muchas
    # barras/día) pero causal_window se dejó en su valor por defecto sin
    # escalar junto con él. Recortar aquí es la salvaguarda genérica: sea
    # cual sea el origen de la config (UI, calibración por timeframe,
    # llamada directa), el cálculo del umbral nunca debe romperse por esto.
    min_periods = min(min_periods, window)
    roll_abs = abs_signal.rolling(window=window, min_periods=min_periods)

    if config.threshold_method == "percentile":
        enter = roll_abs.quantile(config.enter_percentile)
        exit_ = roll_abs.quantile(config.exit_percentile)
    else:
        roll_std = pd.Series(smoothed_polarity).rolling(window=window, min_periods=min_periods).std()
        enter = config.enter_threshold_std * roll_std
        exit_ = config.exit_threshold_std * roll_std

    enter = enter.fillna(config.enter_threshold).clip(lower=config.enter_threshold, upper=config.max_threshold)
    exit_ = exit_.fillna(config.exit_threshold).clip(lower=config.exit_threshold, upper=config.max_threshold)

    exit_arr = exit_.to_numpy(copy=True)
    enter_arr = enter.to_numpy()
    too_close = exit_arr <= enter_arr
    exit_arr[too_close] = np.minimum(config.max_threshold, enter_arr[too_close] * 1.5)
    return enter_arr, exit_arr


def _causal_thresholds_dynamic(smoothed_polarity: np.ndarray, config: SegmenterConfig,
                                profile_ids: np.ndarray,
                                profile_table: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Igual que `_causal_thresholds`, pero el percentil/ventana usados en
    CADA vela dependen del perfil de estructura de precio vigente en esa
    vela (`profile_ids[i]`), no de un único perfil fijo para todo el
    análisis.

    Por qué existe: `_causal_thresholds` ya resolvió el look-ahead bias
    temporal (nunca mira datos futuros), pero seguía asumiendo un solo
    `enter_percentile`/`exit_percentile`/`causal_window` constante para
    toda la serie -- el que correspondía al perfil de estructura medido
    una vez sobre las últimas velas. Si el activo cambió de estructura a
    mitad del histórico analizado (p.ej. años de rango seguidos de una
    tendencia volátil reciente), esa mezcla castigaba a una de las dos
    mitades con la sensibilidad equivocada. Aquí cada vela usa los
    parámetros de SU PROPIO perfil vigente (medido causalmente, ver
    `config.rolling_structure_profile`).

    `profile_table`: dict {profile_id: {enter_percentile, exit_percentile,
    causal_window}}, ya resuelto (con los defaults del `config` base para
    cualquier clave no sobreescrita por el perfil).
    """
    n = len(smoothed_polarity)
    if n == 0:
        return np.array([]), np.array([])

    abs_signal = pd.Series(np.abs(smoothed_polarity))
    min_periods = config.causal_min_periods or max(30, config.min_segment_length * 2)

    # Solo hay un puñado de ventanas/percentiles distintos entre los
    # perfiles definidos (4 perfiles) -- se calcula la serie rolling UNA
    # vez por cada combinación (window, percentile) distinta que aparezca,
    # y luego se selecciona el valor correspondiente vela a vela. Mucho
    # más barato que recalcular un rolling distinto por vela.
    distinct_windows = sorted({
        int(p.get("causal_window") or config.causal_window or n)
        for p in profile_table.values()
    } | {int(config.causal_window or n)})
    distinct_percentiles = sorted({
        p.get("enter_percentile", config.enter_percentile) for p in profile_table.values()
    } | {
        p.get("exit_percentile", config.exit_percentile) for p in profile_table.values()
    } | {config.enter_percentile, config.exit_percentile})

    quantile_cache: dict[tuple[int, float], np.ndarray] = {}
    for w in distinct_windows:
        win = w if (w and w < n) else n
        # Igual salvaguarda que en `_causal_thresholds`: min_periods no
        # puede exceder la ventana de ESTE perfil en particular.
        win_min_periods = min(min_periods, win)
        roll = abs_signal.rolling(window=win, min_periods=win_min_periods)
        for q in distinct_percentiles:
            quantile_cache[(w, q)] = roll.quantile(q).to_numpy()

    default_params = {
        "enter_percentile": config.enter_percentile,
        "exit_percentile": config.exit_percentile,
        "causal_window": config.causal_window or n,
    }

    enter_arr = np.full(n, config.enter_threshold)
    exit_arr = np.full(n, config.exit_threshold)
    for i in range(n):
        params = profile_table.get(str(profile_ids[i]), default_params)
        w = int(params.get("causal_window") or config.causal_window or n)
        ep = params.get("enter_percentile", config.enter_percentile)
        xp = params.get("exit_percentile", config.exit_percentile)
        e_val = quantile_cache[(w, ep)][i]
        x_val = quantile_cache[(w, xp)][i]
        enter_arr[i] = e_val if not np.isnan(e_val) else config.enter_threshold
        exit_arr[i] = x_val if not np.isnan(x_val) else config.exit_threshold

    enter_arr = np.clip(enter_arr, config.enter_threshold, config.max_threshold)
    exit_arr = np.clip(exit_arr, config.exit_threshold, config.max_threshold)

    too_close = exit_arr <= enter_arr
    exit_arr[too_close] = np.minimum(config.max_threshold, enter_arr[too_close] * 1.5)
    return enter_arr, exit_arr


def _raw_breakpoints(smoothed_polarity: np.ndarray, enter_threshold: np.ndarray,
                      exit_threshold: np.ndarray) -> list[int]:
    """
    Detecta puntos de corte usando histéresis de tres zonas:
    alcista (> enter_threshold tras confirmar con exit_threshold),
    bajista (< -enter_threshold ...), neutral (en medio).

    `enter_threshold`/`exit_threshold` son ahora ARRAYS (un valor por vela,
    calibrado causalmente por `_causal_thresholds`), no escalares fijos
    para toda la serie -- ver esa función para el porqué.
    """
    def zone_of(v: float, current_zone: int, enter_t: float, exit_t: float) -> int:
        # current_zone: -1 bajista, 0 neutral, 1 alcista
        if current_zone == 1:
            return 1 if v > -exit_t + enter_t else (
                0 if v > -exit_t else -1
            )
        if current_zone == -1:
            return -1 if v < exit_t - enter_t else (
                0 if v < exit_t else 1
            )
        # neutral -> requiere superar enter_threshold para salir
        if v > enter_t:
            return 1
        if v < -enter_t:
            return -1
        return 0

    zones = []
    current = 0
    for i, v in enumerate(smoothed_polarity):
        current = zone_of(float(v), current, float(enter_threshold[i]), float(exit_threshold[i]))
        zones.append(current)

    breakpoints = [0]
    for i in range(1, len(zones)):
        if zones[i] != zones[i - 1]:
            breakpoints.append(i)
    breakpoints.append(len(zones))
    return sorted(set(breakpoints))


def combined_regime_signal(chain: StateChain, config: SegmenterConfig,
                            close: pd.Series | None = None) -> np.ndarray:
    """
    Señal continua en [-1, 1] usada por el segmentador para decidir los
    cortes de régimen (precio + microestados combinados). Se expone
    como función pública porque es útil por sí misma más allá de la
    segmentación en bloques discretos:

      - Para el Motor de Activación de Estrategias: se puede usar como
        una "confianza gradual" de régimen (p.ej. mezclar el peso de una
        estrategia de tendencia y una de reversión proporcionalmente al
        valor de esta señal, en vez de un interruptor binario).
      - Para calibrar parámetros (ver examples/calibrate_params.py).
    """
    micro_polarity = _polarity_series(chain)
    micro_smoothed = _smooth(micro_polarity, config.smoothing_window)

    if close is not None:
        close_aligned = close.reindex(chain.index).ffill().bfill()
        price_signal = price_trend_strength(close_aligned, window=config.price_trend_window)
        w = np.clip(config.price_weight, 0.0, 1.0)
        return w * price_signal + (1 - w) * micro_smoothed
    return micro_smoothed


def segment_chain(chain: StateChain, n_states: int,
                   config: SegmenterConfig | None = None,
                   close: pd.Series | None = None,
                   profile_ids: np.ndarray | None = None,
                   profile_table: dict[str, dict] | None = None,
                   expected_duration: np.ndarray | None = None) -> list[Segment]:
    """
    Segmenta la cadena de microestados en bloques temporalmente
    coherentes, aplicando suavizado + histéresis para evitar que el
    ruido produzca cambios de segmento constantes, y fusionando
    segmentos demasiado cortos con su vecino más afín.

    Si se proporciona `close` (precio de cierre alineado con la cadena),
    la señal que decide los cortes de segmento combina la polaridad de
    microestados con la fuerza de tendencia medida directamente sobre el
    precio (ver `price_trend_strength`). El componente de precio aporta
    la señal de "tendencia real" con mucho menos ruido, y domina por
    defecto (`price_weight=0.7`).

    Si además se proporciona `profile_ids` (un perfil de estructura de
    precio por vela, típicamente `config.rolling_structure_profile`,
    alineado con `chain.index`) junto con `profile_table`, los umbrales
    enter/exit se calibran con el perfil VIGENTE en cada vela en vez de
    un único perfil fijo para todo el análisis -- ver
    `_causal_thresholds_dynamic` para el razonamiento completo.

    Si se proporciona `expected_duration` (típicamente
    `transition_matrix.causal_expected_duration`, alineado con
    `chain.states`), la confianza de cada segmento incorpora un cuarto
    componente (`persistence_fit`): qué tan cerca está la duración REAL
    del segmento de lo que la cadena de Markov de microestados esperaba
    para su estado dominante, aprendido de forma causal. Esto es lo que
    finalmente le da uso a la matriz de transición dentro de la
    clasificación de régimen, en vez de quedar como estadística
    descriptiva aislada.
    """
    config = config or SegmenterConfig()
    if len(chain.states) == 0:
        return []

    smoothed = combined_regime_signal(chain, config, close)
    if profile_ids is not None and profile_table:
        enter_thr_arr, exit_thr_arr = _causal_thresholds_dynamic(
            smoothed, config, profile_ids, profile_table)
    else:
        enter_thr_arr, exit_thr_arr = _causal_thresholds(smoothed, config)
    breakpoints = _raw_breakpoints(smoothed, enter_thr_arr, exit_thr_arr)

    # Construir segmentos iniciales
    raw_segments: list[tuple[int, int]] = list(zip(breakpoints[:-1], breakpoints[1:]))

    # Fusionar segmentos más cortos que min_segment_length con el vecino
    # cuya polaridad media sea más cercana.
    merged: list[list[int]] = [list(seg) for seg in raw_segments]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, (s, e) in enumerate(merged):
            if e - s < config.min_segment_length:
                seg_mean = smoothed[s:e].mean() if e > s else 0.0
                neighbors = []
                if i > 0:
                    ps, pe = merged[i - 1]
                    neighbors.append((abs(smoothed[ps:pe].mean() - seg_mean), i - 1))
                if i < len(merged) - 1:
                    ns, ne = merged[i + 1]
                    neighbors.append((abs(smoothed[ns:ne].mean() - seg_mean), i + 1))
                if not neighbors:
                    continue
                neighbors.sort(key=lambda t: t[0])
                target = neighbors[0][1]
                if target < i:
                    merged[target][1] = e
                else:
                    merged[target][0] = s
                del merged[i]
                changed = True
                break

    segments: list[Segment] = []
    idx = chain.index
    for s, e in merged:
        if e <= s:
            continue
        seg_states = [int(x) for x in chain.states[s:e]]
        vals, counts = np.unique(seg_states, return_counts=True)
        order = np.argsort(-counts)
        dominant = [int(vals[i]) for i in order[:3]]
        distribution = {int(v): float(c) / len(seg_states) for v, c in zip(vals, counts)}
        ent = _entropy(seg_states, n_states)
        # avg_polarity usa la señal COMBINADA (precio + microestados) que
        # realmente decidió los cortes de segmento, no solo la polaridad
        # cruda de microestados -- así el régimen asignado es coherente
        # con la señal que originó el segmento.
        avg_pol = float(np.mean(smoothed[s:e]))

        # Probabilidad empírica de cambio de estado dentro del segmento
        # (proporción de velas donde el estado difiere del anterior)
        changes = sum(1 for i in range(1, len(seg_states)) if seg_states[i] != seg_states[i - 1])
        change_prob = changes / max(1, len(seg_states) - 1)

        # Confianza — REDISEÑADA (v2: ahora con la cadena de Markov).
        #
        # La versión anterior basaba el 60% de la confianza en la "pureza"
        # (1 - entropía normalizada) de la distribución de MICROESTADOS
        # dentro del segmento. Es un mal proxy de qué tan claro es el
        # régimen: incluso una tendencia alcista limpia visita rutinariamente
        # varios de los 12 microestados (impulso, continuación, corrección,
        # recuperación...), así que esa entropía es casi siempre "media"
        # tanto en tendencias limpias como en consolidaciones reales.
        #
        # La versión siguiente medía directamente qué tan clara era la
        # señal que decidió el segmento (precio + microestados), pero
        # ignoraba por completo la cadena de Markov de microestados que el
        # pipeline sí calculaba (persistencia, duración esperada por
        # estado) -- quedaba como estadística descriptiva sin uso real.
        #
        # Ahora la confianza combina CUATRO componentes:
        #   1. strength:         qué tan lejos está la polaridad media del
        #                        segmento respecto al umbral que lo confirmó.
        #   2. consistency:      proporción de velas cuya señal tiene el
        #                        mismo signo que la polaridad media.
        #   3. size_factor:      más historial (hasta 2x min_segment_length)
        #                        = más confianza.
        #   4. persistence_fit:  qué tan cerca está la duración REAL del
        #                        segmento de la duración que la cadena de
        #                        Markov esperaba (de forma causal) para su
        #                        microestado dominante -- un segmento que
        #                        dura mucho MENOS de lo típico para ese
        #                        estado es sospechoso de ser ruido que
        #                        alcanzó el umbral por casualidad; uno que
        #                        dura lo esperado o más es más confiable.
        seg_signal = smoothed[s:e]
        seg_sign_ref = avg_pol if avg_pol != 0 else 1e-9
        # Umbral causal vigente al FINAL del segmento (el que estaba
        # activo en el momento en que, en tiempo real, se habría podido
        # confirmar este régimen) -- no un escalar global de toda la serie.
        seg_exit_thr = float(exit_thr_arr[e - 1])
        strength = float(np.clip(abs(avg_pol) / max(seg_exit_thr, 1e-6), 0.0, 1.0))
        consistency = float(np.mean(np.sign(seg_signal) == np.sign(seg_sign_ref))) if len(seg_signal) else 0.5
        size_factor = min(1.0, len(seg_states) / (config.min_segment_length * 2))

        if expected_duration is not None and len(seg_states):
            dom_state = dominant[0]
            dom_mask = np.array(seg_states) == dom_state
            exp_for_dom = np.asarray(expected_duration[s:e])[dom_mask]
            avg_expected = float(np.mean(exp_for_dom)) if len(exp_for_dom) else 2.0
            actual = float(len(seg_states))
            # log-ratio acotado: 1.0 si duración real == esperada, decae
            # simétricamente si es mucho más corta o mucho más larga
            # (una racha 5x más larga de lo típico también es rara, no
            # solo una demasiado corta).
            log_ratio = abs(np.log((actual + 1e-9) / (avg_expected + 1e-9)))
            persistence_fit = float(np.clip(1.0 - log_ratio / np.log(5.0), 0.0, 1.0))
        else:
            persistence_fit = 0.5  # neutro si no se proporcionó la señal causal

        confidence = float(np.clip(
            0.35 * strength + 0.25 * consistency + 0.15 * size_factor + 0.25 * persistence_fit,
            0.0, 1.0
        ))

        segments.append(Segment(
            start_idx=s, end_idx=e,
            start_time=idx[s], end_time=idx[e - 1],
            states=seg_states,
            dominant_states=dominant,
            state_distribution=distribution,
            entropy=ent,
            avg_polarity=avg_pol,
            change_probability=change_prob,
            confidence=confidence,
        ))

    return segments


def segments_to_dataframe(segments: list[Segment]) -> pd.DataFrame:
    return pd.DataFrame([s.to_dict() for s in segments])
