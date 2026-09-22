"""
core/multiple_testing.py
Corrección por tests múltiples — False Discovery Rate (Benjamini-Hochberg).

Complementa (no reemplaza) al Deflated Sharpe Ratio (core/metrics.py):
- El DSR corrige el Sharpe de UNA estrategia por cuántas combinaciones
  probó el buscador (genético/optimizador) para llegar a ELLA. Es una
  corrección "por estrategia", mirando hacia atrás en su propio proceso
  de búsqueda.
- El FDR (Benjamini-Hochberg, 1995) resuelve un problema distinto y
  complementario: cuando se mira TODO el Hall of Fame a la vez —
  decenas de estrategias, cada una con su propio p-valor de
  significancia — y se pregunta "¿cuántas de estas docenas de
  candidatas son en realidad ruido?", evaluar cada una por separado con
  un umbral fijo (ej. "DSR >= 95%") deja pasar demasiados falsos
  positivos: con 60 candidatas, un 5% de nivel de significancia
  individual implica ~3 falsos positivos esperados solo por azar. BH
  ajusta el umbral de decisión para controlar la fracción esperada de
  falsos descubrimientos sobre el conjunto completo.

Referencia: Benjamini, Y. & Hochberg, Y. (1995), "Controlling the False
Discovery Rate: A Practical and Powerful Approach to Multiple Testing",
Journal of the Royal Statistical Society B.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class FDRResult:
    """Resultado de aplicar Benjamini-Hochberg a un conjunto de p-valores."""
    p_values: List[float]
    adjusted_p_values: List[float]     # p-valores ajustados (BH), mismo orden que la entrada
    rejected: List[bool]               # True = sigue siendo significativa tras la corrección
    alpha: float
    n_tested: int
    n_significant: int

    def summary(self) -> str:
        return (f"{self.n_significant}/{self.n_tested} estrategias siguen siendo "
                f"significativas tras controlar FDR al {self.alpha*100:.0f}% "
                f"(Benjamini-Hochberg).")


def benjamini_hochberg(p_values: List[Optional[float]], alpha: float = 0.10) -> FDRResult:
    """
    Aplica la corrección de Benjamini-Hochberg a una lista de p-valores.

    Parameters
    ----------
    p_values : lista de p-valores (uno por estrategia/hipótesis evaluada).
        Puede contener None (estrategia sin p-valor calculable, p.ej. no
        verificada con el motor real) — esas entradas se excluyen del
        cálculo y se devuelven como no-rechazadas (rejected=False) sin
        afectar el ajuste de las demás.
    alpha : float
        Nivel de FDR objetivo (fracción esperada de falsos descubrimientos
        tolerada sobre el total de estrategias declaradas significativas).
        0.10 (10%) es un valor convencional para descubrimiento exploratorio
        de estrategias; 0.05 es más conservador.

    Returns
    -------
    FDRResult con p-valores ajustados y la máscara de cuáles siguen siendo
    significativas al nivel `alpha`, en el MISMO orden que `p_values`.
    """
    n = len(p_values)
    if n == 0:
        return FDRResult([], [], [], alpha, 0, 0)

    # Separar los que sí tienen p-valor de los None
    valid_idx = [i for i, p in enumerate(p_values) if p is not None]
    valid_p = [float(p_values[i]) for i in valid_idx]

    adjusted = [1.0] * n
    rejected = [False] * n

    if valid_p:
        m = len(valid_p)
        order = np.argsort(valid_p)
        sorted_p = np.array(valid_p)[order]

        # p-valor ajustado BH: p_(i) * m / i, con monotonía garantizada
        # (cada ajustado no puede ser menor que el siguiente en el orden
        # descendente — "step-up" procedure).
        raw_adjusted = sorted_p * m / (np.arange(m) + 1)
        # Asegurar monotonía no-decreciente al recorrer de mayor a menor rank
        adjusted_sorted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
        adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

        # Máscara de rechazo: el mayor i tal que p_(i) <= (i/m)*alpha, y
        # todos los de rank menor también se declaran significativos.
        thresholds = (np.arange(m) + 1) / m * alpha
        below = sorted_p <= thresholds
        if below.any():
            max_i = np.max(np.where(below)[0])
            reject_sorted = np.zeros(m, dtype=bool)
            reject_sorted[: max_i + 1] = True
        else:
            reject_sorted = np.zeros(m, dtype=bool)

        # Volver al orden original
        for rank, orig_i in enumerate(order):
            i = valid_idx[orig_i]
            adjusted[i] = float(adjusted_sorted[rank])
            rejected[i] = bool(reject_sorted[rank])

    return FDRResult(
        p_values=list(p_values),
        adjusted_p_values=adjusted,
        rejected=rejected,
        alpha=alpha,
        n_tested=len(valid_p),
        n_significant=int(sum(rejected)),
    )


def dsr_to_pvalue(dsr: Optional[float]) -> Optional[float]:
    """
    Convierte un Deflated Sharpe Ratio (probabilidad 0-1 de que el Sharpe
    sea genuino, ver core/metrics.deflated_sharpe_ratio) a un p-valor
    utilizable por Benjamini-Hochberg: p = 1 - DSR.

    DSR ya es, por construcción, 1 - p_valor de la hipótesis nula "esta
    estrategia no tiene ventaja real, su Sharpe se explica solo por el
    número de combinaciones probadas" (Bailey & López de Prado, 2014) —
    esta función solo hace explícita esa relación para poder alimentar
    el paso de FDR sin recalcular nada.
    """
    if dsr is None:
        return None
    return float(np.clip(1.0 - dsr, 0.0, 1.0))
