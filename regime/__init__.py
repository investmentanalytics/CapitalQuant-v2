"""
regime/
Integración del detector de régimen (motor Aletheia, el mismo usado en
Regime Engine Platform) dentro de CapitalQuant: expone una serie de
régimen por vela, alineada al OHLC de cualquier activo/timeframe de
CapitalQuant, para poder:

  1. Mostrarla como herramienta propia ("Régimen de Mercado" en la
     navegación superior).
  2. Filtrar backtests y optimizaciones para que solo operen en los
     periodos que coincidan con el/los régimen(es) elegidos.
"""
from .service import (
    REGIME_LABELS,
    RegimeSummary,
    compute_regime_series,
    compute_regime_summary,
    regime_entry_mask,
)

__all__ = [
    "REGIME_LABELS",
    "RegimeSummary",
    "compute_regime_series",
    "compute_regime_summary",
    "regime_entry_mask",
]
