"""
core/types.py
Tipos de datos centralizados para todo el sistema CapitalQuant.

Sin dependencias de UI ni de lógica de negocio.
Reutilizable en API REST y futura app móvil sin cambios.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuración del backtester
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Parámetros de configuración del motor de backtesting."""
    initial_capital: float = 100_000.0
    commission: float = 0.001          # 0.1% por operación
    slippage: float = 0.0005           # 0.05% por operación
    risk_per_trade: float = 0.02       # 2% del capital en riesgo
    allow_long: bool = True            # Permite entradas en largo (signal == 1)
    allow_short: bool = True           # Permite entradas en corto (signal == -1)
    use_position_sizing: bool = True

    # --- Modelo de costos realista (opcional, ver core/costs.py) ---
    # Por defecto queda desactivado para no alterar resultados existentes
    # ni requerir symbol_info. Al activarlo, `commission`/`slippage` (%)
    # dejan de usarse y se reemplazan por spread real (en puntos) +
    # comisión fija por lote, tal como cobra un bróker de verdad.
    use_realistic_costs: bool = False
    cost_profile: Optional["object"] = None  # InstrumentCostProfile, evita import circular aquí

    # Nota: el motor solo soporta UNA posición abierta a la vez (sin pirámide).
    # Anteriormente existían `max_positions` y `trade_on_close` como campos de
    # configuración, pero el motor nunca los leía — quedaban ahí sin efecto
    # alguno, lo cual podía hacer creer al usuario que los estaba controlando.
    # Se retiraron para no inducir a error; si se implementa pirámide de
    # posiciones en el futuro, debe reintroducirse junto con la lógica real
    # en engine/backtester.py.


# ---------------------------------------------------------------------------
# Operación individual
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Registro completo de una operación cerrada."""
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    direction: int             # 1=Long, -1=Short
    entry_price: float
    exit_price: float = 0.0
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""      # "signal" | "stop_loss" | "take_profit" | "end"
    capital_at_entry: float = 0.0
    stop_loss: float = np.nan
    take_profit: float = np.nan
    trade_number: int = 0
    duration_bars: int = 0     # nº de velas que la posición estuvo abierta (lo fija el engine al cerrar)

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    def to_dict(self) -> dict:
        def _fmt(v, decimals=4):
            return round(float(v), decimals) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else None

        return {
            "trade_number":  self.trade_number,
            "entry_date":    self.entry_date,
            "exit_date":     self.exit_date,
            "direction":     "Long" if self.direction == 1 else "Short",
            "entry_price":   _fmt(self.entry_price),
            "exit_price":    _fmt(self.exit_price),
            "stop_loss":     _fmt(self.stop_loss),
            "take_profit":   _fmt(self.take_profit),
            "size":          _fmt(self.size),
            "pnl":           _fmt(self.pnl, 2),
            "net_pnl":       _fmt(self.net_pnl, 2),
            "pnl_pct":       _fmt(self.pnl_pct, 2),
            "commission":    _fmt(self.commission, 2),
            "duration_bars": self.duration_bars,
            "exit_reason":   self.exit_reason,
        }


# ---------------------------------------------------------------------------
# Resultados del backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestResults:
    """Resultados completos del backtest. Solo datos — sin lógica."""

    # Identificación
    config: Optional[BacktestConfig] = None
    strategy_name: str = ""
    asset: str = ""
    timeframe: str = ""
    start_date: str = ""
    end_date: str = ""

    # Series temporales
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)

    # Operaciones
    trades: List[Trade] = field(default_factory=list)
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Métricas de capital
    initial_capital: float = 0.0
    final_capital: float = 0.0
    net_profit: float = 0.0
    net_profit_pct: float = 0.0

    # Métricas de operaciones
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Métricas de riesgo
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    ulcer_index: float = 0.0
    exposure_pct: float = 0.0

    # Métricas de rendimiento ajustadas
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    def to_dict(self) -> dict:
        """Convierte métricas principales a diccionario para display."""
        return {
            "Net Profit":       f"${self.net_profit:,.2f}",
            "Net Profit %":     f"{self.net_profit_pct:.2f}%",
            "Total Trades":     self.total_trades,
            "Win Rate":         f"{self.win_rate:.1f}%",
            "Profit Factor":    f"{self.profit_factor:.2f}",
            "Max Drawdown %":   f"{self.max_drawdown_pct:.2f}%",
            "CAGR":             f"{self.cagr:.2f}%",
            "Sharpe Ratio":     f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio":    f"{self.sortino_ratio:.2f}",
            "Calmar Ratio":     f"{self.calmar_ratio:.2f}",
            "Ulcer Index":      f"{self.ulcer_index:.2f}",
            "Expectancy":       f"${self.expectancy:,.2f}",
            "Avg Win":          f"${self.avg_win:,.2f}",
            "Avg Loss":         f"${self.avg_loss:,.2f}",
            "Max Consec. Wins": self.max_consecutive_wins,
            "Max Consec. Loss": self.max_consecutive_losses,
            "Exposure %":       f"{self.exposure_pct:.1f}%",
        }

    def to_summary(self) -> dict:
        """Resumen compacto para comparación de portafolio."""
        return {
            "net_profit_pct": self.net_profit_pct,
            "cagr":           self.cagr,
            "sharpe":         self.sharpe_ratio,
            "sortino":        self.sortino_ratio,
            "calmar":         self.calmar_ratio,
            "max_dd_pct":     self.max_drawdown_pct,
            "win_rate":       self.win_rate,
            "profit_factor":  self.profit_factor,
            "total_trades":   self.total_trades,
            "expectancy":     self.expectancy,
        }
