"""
execution/risk_overlay.py
Capa de riesgo obligatoria entre CUALQUIER señal/orden y el envío real a
MT5 — Fase 3 del roadmap ("una estrategia con risk overlay").

Esta capa NUNCA envía órdenes por sí misma: solo decide si una orden
propuesta está permitida, y calcula el tamaño de posición. El envío
real siempre pasa por execution/mt5_execution.py.

Dos modos de autorización humana usan esta misma capa, sin relajar
ninguno de sus límites:
  - "Mercado en Vivo" (ui/views/live.py): confirmación explícita por
    CADA orden — el modo original.
  - Trading Algorítmico (execution/live_trader.py, AlgoTradingEngine):
    confirmación de UNA VEZ al iniciar la sesión ("un clic para iniciar,
    no un clic por orden" — cambio de modelo pedido explícitamente por
    el usuario). Por eso estas guardas se revalidan en TODOS los ciclos
    del motor, no solo al momento de enviar una orden puntual.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List
from loguru import logger


@dataclass
class RiskOverlayConfig:
    """Límites de riesgo — se configuran una vez por sesión de trading."""
    risk_per_trade_pct: float = 0.02       # 2% del equity por operación (mismo default que el backtester)
    max_daily_loss_pct: float = 0.05       # kill switch: -5% del equity del día detiene todo envío
    max_open_positions: int = 1            # el roadmap pide "una estrategia" -> 1 posición a la vez
    max_position_pct_of_equity: float = 0.95  # tope duro, igual que _calculate_size en el backtester


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str = ""
    suggested_volume: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


class RiskOverlay:
    """
    Guardia de riesgo con estado (recuerda el equity de inicio del día
    para el kill switch). Se instancia UNA vez por sesión de la página
    "Mercado en Vivo" y se reutiliza en cada intento de envío de orden.
    """

    def __init__(self, config: Optional[RiskOverlayConfig] = None):
        self.config = config or RiskOverlayConfig()
        self._day_start_equity: Optional[float] = None
        self._day_start_date: Optional[date] = None
        self.kill_switch_triggered = False
        self.kill_switch_reason = ""

    def _refresh_day_anchor(self, current_equity: float) -> None:
        today = date.today()
        if self._day_start_date != today:
            self._day_start_date = today
            self._day_start_equity = current_equity
            self.kill_switch_triggered = False
            self.kill_switch_reason = ""
            logger.info(f"[RiskOverlay] Nuevo día — ancla de equity: {current_equity:.2f}")

    def check_daily_loss(self, current_equity: float) -> RiskCheckResult:
        """Actualiza el ancla del día si corresponde y evalúa el kill switch."""
        self._refresh_day_anchor(current_equity)
        if self._day_start_equity and self._day_start_equity > 0:
            daily_pnl_pct = (current_equity - self._day_start_equity) / self._day_start_equity
            if daily_pnl_pct <= -self.config.max_daily_loss_pct:
                self.kill_switch_triggered = True
                self.kill_switch_reason = (
                    f"Pérdida del día {daily_pnl_pct*100:.1f}% alcanzó el límite de "
                    f"-{self.config.max_daily_loss_pct*100:.0f}%."
                )
        if self.kill_switch_triggered:
            return RiskCheckResult(allowed=False, reason=self.kill_switch_reason)
        return RiskCheckResult(allowed=True)

    def check_order(
        self,
        account_equity: float,
        account_is_demo: bool,
        n_open_positions: int,
        entry_price: float,
        stop_loss_price: Optional[float],
        contract_size: float,
        volume_min: float,
        volume_max: float,
        volume_step: float,
        allow_real_account: bool = False,
    ) -> RiskCheckResult:
        """
        Valida y dimensiona una orden propuesta. Devuelve allowed=False
        si CUALQUIER condición de seguridad no se cumple — el llamador
        (UI) nunca debe enviar la orden si allowed es False, sin
        excepciones.

        `allow_real_account`: por defecto en False. Una cuenta REAL solo
        puede operar si el llamador pasa explícitamente True — en la UI de
        Trading Algorítmico esto exige que el usuario marque una casilla
        de confirmación explícita ("entiendo que esto opera con dinero
        real") antes de iniciar la sesión. Todas las DEMÁS guardas (SL
        obligatorio, kill switch diario, límite de posiciones,
        dimensionamiento acotado) se aplican exactamente igual en cuenta
        real que en demo — no se relajan por operar con dinero real.
        """
        warnings: List[str] = []

        # 1) Cuenta real -> requiere confirmación explícita del llamador.
        if not account_is_demo and not allow_real_account:
            return RiskCheckResult(
                allowed=False,
                reason="Cuenta REAL detectada, pero no se confirmó explícitamente operar en "
                       "cuenta real. Activa la casilla de confirmación en Trading Algorítmico, "
                       "o conecta una cuenta DEMO para probar primero.",
            )

        # 2) Kill switch de pérdida diaria
        daily_check = self.check_daily_loss(account_equity)
        if not daily_check.allowed:
            return daily_check

        # 3) Límite de posiciones abiertas simultáneas
        if n_open_positions >= self.config.max_open_positions:
            return RiskCheckResult(
                allowed=False,
                reason=f"Ya hay {n_open_positions} posición(es) abierta(s); el límite configurado "
                       f"es {self.config.max_open_positions} (\"una estrategia a la vez\").",
            )

        # 4) Dimensionamiento por riesgo (misma filosofía que
        #    engine/backtester._calculate_size: arriesgar un % fijo del
        #    equity, tamaño = riesgo / distancia al stop).
        if stop_loss_price is None or stop_loss_price <= 0 or entry_price <= 0:
            return RiskCheckResult(
                allowed=False,
                reason="No se puede dimensionar la orden sin un stop-loss válido — esta capa de "
                       "riesgo exige SL obligatorio en todas las órdenes.",
            )

        risk_amount = account_equity * self.config.risk_per_trade_pct
        risk_per_unit = abs(entry_price - stop_loss_price) * contract_size
        if risk_per_unit <= 0:
            return RiskCheckResult(allowed=False, reason="Distancia al stop-loss inválida (0).")

        raw_volume = risk_amount / risk_per_unit
        # Tope duro: no comprometer más del X% del equity en margen nocional
        max_notional_volume = (account_equity * self.config.max_position_pct_of_equity) / (entry_price * contract_size)
        volume = min(raw_volume, max_notional_volume, volume_max)
        # Redondear al step del símbolo (hacia abajo, nunca sobre-dimensionar)
        if volume_step > 0:
            steps = int(volume / volume_step)
            volume = steps * volume_step
        volume = round(volume, 8)

        if volume < volume_min:
            return RiskCheckResult(
                allowed=False,
                reason=f"El tamaño calculado ({volume}) es menor al mínimo del símbolo "
                       f"({volume_min}) — el riesgo configurado ({self.config.risk_per_trade_pct*100:.1f}%) "
                       f"es demasiado bajo para este stop-loss. Sube el riesgo por operación o ensancha el SL.",
            )

        if raw_volume > max_notional_volume:
            warnings.append(
                f"El tamaño por riesgo puro ({raw_volume:.4f}) excedía el tope de "
                f"{self.config.max_position_pct_of_equity*100:.0f}% del equity — se recortó a {volume:.4f}."
            )

        return RiskCheckResult(allowed=True, suggested_volume=volume, warnings=warnings)
