"""
core/costs.py
Modelo de costos de ejecución realista para el backtester.

Motivación (ver Auditoría 2.1 / Roadmap Fase 1, punto 4):
El motor original modela comisión y slippage como % fijo del notional.
Es razonable como aproximación, pero no es lo que realmente cobra un
bróker de FX/CFDs, que carga:
  - un SPREAD (diferencia bid/ask) expresado en puntos del símbolo,
  - opcionalmente una comisión fija por lote (ej. $7 por lote redondo),
  - y para holding overnight, swap/rollover (no cubierto aquí todavía,
    ver TODO en roadmap para lógica de sesiones/triple swap miércoles-viernes).

Este módulo NO reemplaza el modelo de % fijo (que sigue siendo el
default para no romper compatibilidad ni resultados existentes) — añade
un modelo alternativo activable explícitamente vía
`BacktestConfig.use_realistic_costs = True`, calibrado con datos reales
del símbolo (`MT5Provider.get_symbol_info`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InstrumentCostProfile:
    """
    Perfil de costos de un instrumento, derivable de `symbol_info` de MT5.

    Parameters
    ----------
    point : float
        Tamaño de un "punto" de precio (ej. 0.00001 en EURUSD 5 dígitos).
    spread_points : float
        Spread típico del símbolo, en puntos (de `symbol_info['spread']`).
    contract_size : float
        Tamaño de un lote estándar en unidades del activo base
        (100_000 para FX mayoritariamente; 1 para CFDs de acciones, etc.).
    commission_per_lot : float
        Comisión monetaria fija por lote *redondo* (ida + vuelta),
        cobrada por brokers ECN/RAW típicamente ($3-$7). 0.0 en brokers
        que solo cobran vía spread ("standard").
    lot_step : float
        Incremento mínimo de tamaño de posición permitido por el bróker.
    min_lot : float
        Tamaño mínimo de posición permitido.
    """
    point: float = 0.0001
    spread_points: float = 1.0
    contract_size: float = 100_000.0
    commission_per_lot: float = 0.0
    lot_step: float = 0.01
    min_lot: float = 0.01

    @classmethod
    def from_symbol_info(cls, info: dict, commission_per_lot: float = 0.0) -> "InstrumentCostProfile":
        """Construye el perfil a partir del dict que devuelve MT5Provider.get_symbol_info()."""
        return cls(
            point=float(info.get("point", 0.0001)),
            spread_points=float(info.get("spread", 1.0)),
            contract_size=float(info.get("contract_size", 100_000.0)),
            commission_per_lot=commission_per_lot,
            lot_step=float(info.get("lot_step", 0.01)),
            min_lot=float(info.get("min_lot", 0.01)),
        )

    def spread_cost_per_unit(self) -> float:
        """
        Costo del spread por unidad del activo base (no por lote), en la
        moneda de cotización — el costo total de cruzar bid/ask una vez.
        `entry_cost`/`exit_cost` cobran cada uno la mitad (mismo total que
        cobrarlo completo en la entrada; se reparte para que cada trade
        cerrado a mitad de camino ya refleje el costo proporcional que
        pagó hasta ese punto).
        """
        return self.spread_points * self.point

    def round_trip_commission(self, size_in_units: float) -> float:
        """Comisión fija por el tamaño de la operación, convertido a lotes."""
        if self.contract_size <= 0:
            return 0.0
        lots = size_in_units / self.contract_size
        return lots * self.commission_per_lot

    def snap_size_to_lot_step(self, size_in_units: float) -> float:
        """Redondea el tamaño de posición al step de lote permitido por el bróker."""
        if self.contract_size <= 0 or self.lot_step <= 0:
            return size_in_units
        lots = size_in_units / self.contract_size
        min_lots = self.min_lot
        if lots < min_lots:
            return 0.0
        snapped_lots = (lots // self.lot_step) * self.lot_step
        return max(snapped_lots, min_lots) * self.contract_size


def entry_cost(profile: InstrumentCostProfile, size_in_units: float, price: float) -> float:
    """
    Costo total de ENTRAR una posición bajo el modelo realista:
    mitad del spread (cruzar bid/ask) + comisión de entrada (mitad del
    round-trip, la otra mitad se cobra al salir).
    """
    spread_cost = 0.5 * profile.spread_cost_per_unit() * size_in_units
    commission = 0.5 * profile.round_trip_commission(size_in_units)
    return spread_cost + commission


def exit_cost(profile: InstrumentCostProfile, size_in_units: float, price: float) -> float:
    """Costo total de SALIR una posición (la otra mitad del spread + comisión)."""
    spread_cost = 0.5 * profile.spread_cost_per_unit() * size_in_units
    commission = 0.5 * profile.round_trip_commission(size_in_units)
    return spread_cost + commission
