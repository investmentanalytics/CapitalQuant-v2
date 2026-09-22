"""
execution/mt5_execution.py
Capa de ejecución hacia MetaTrader 5 — Fase 3 del roadmap, evolucionada a
Trading Algorítmico (ver execution/live_trader.py): en vez de requerir un
clic humano por cada orden, el usuario autoriza UNA sesión algorítmica
completa (botón "Iniciar") y desde ahí el motor abre/cierra posiciones por
sí mismo, siguiendo la señal de la estrategia en cada ciclo.

Guardas de seguridad NO opcionales, en este orden, en TODA orden:
  1. Debe pasar por RiskOverlay.check_order() (kill switch diario,
     límite de posiciones abiertas, SL obligatorio, tamaño acotado) —
     se revalida en CADA ciclo del motor, no solo al enviar.
  2. Cuenta REAL requiere `allow_real_account=True` explícito (por
     defecto False) — en la UI, esto exige que el usuario marque una
     casilla de confirmación ("opero con dinero real") antes de poder
     iniciar la sesión sobre esa cuenta. Cuenta DEMO no requiere nada
     adicional.
  3. Requiere que el llamador haya pasado confirm=True explícitamente.
     Para el ticket manual de la UI esto sigue siendo un clic humano por
     orden; para el motor algorítmico, `confirm=True` lo pasa el propio
     motor una vez que el usuario ya autorizó la sesión — el punto de
     autorización humana se movió de "por orden" a "por sesión", a
     petición explícita del usuario.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from loguru import logger

from market_data.mt5_provider import get_mt5_provider
from execution.risk_overlay import RiskOverlay, RiskCheckResult


@dataclass
class OrderResult:
    success: bool
    message: str
    ticket: Optional[int] = None
    volume: Optional[float] = None
    price: Optional[float] = None


@dataclass
class Position:
    ticket: int
    symbol: str
    direction: str      # "buy" | "sell"
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float


class DemoAccountRequiredError(Exception):
    """Se lanza si algo intenta operar en una cuenta REAL sin haber pasado
    `allow_real_account=True` explícitamente. Nunca debe capturarse para
    'reintentar' con el flag forzado desde dentro del propio código — el
    flag solo debe originarse en una casilla de confirmación que el
    usuario marcó a mano en la UI."""
    pass


class MT5ExecutionClient:
    """
    Envoltorio delgado sobre MetaTrader5 para envío de órdenes,
    consulta de posiciones y cierre — SIEMPRE detrás del RiskOverlay.
    Cuenta DEMO: sin restricciones adicionales. Cuenta REAL: requiere
    `allow_real_account=True` en el constructor, que en la UI solo se
    activa si el usuario marcó la casilla de confirmación explícita.
    """

    def __init__(self, risk_overlay: Optional[RiskOverlay] = None, allow_real_account: bool = False):
        self.risk_overlay = risk_overlay or RiskOverlay()
        self.allow_real_account = allow_real_account

    # ------------------------------------------------------------------
    # Estado de cuenta / guardas
    # ------------------------------------------------------------------

    def account_summary(self) -> Optional[dict]:
        return get_mt5_provider().account_info()

    def is_demo(self) -> bool:
        info = self.account_summary()
        return bool(info and info.get("is_demo"))

    def _verify_account_allowed(self) -> dict:
        info = self.account_summary()
        if not info:
            raise DemoAccountRequiredError("No hay conexión con MT5 — no se puede verificar la cuenta.")
        if not info.get("is_demo") and not self.allow_real_account:
            raise DemoAccountRequiredError(
                f"La cuenta conectada ({info.get('login')} @ {info.get('server')}) es una cuenta "
                "REAL y no se confirmó explícitamente operar con ella (allow_real_account=False)."
            )
        return info

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        mt5 = get_mt5_provider().get_raw_mt5()
        if mt5 is None:
            return []
        try:
            raw = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            if not raw:
                return []
            out = []
            for p in raw:
                out.append(Position(
                    ticket=p.ticket, symbol=p.symbol,
                    direction="buy" if p.type == 0 else "sell",
                    volume=p.volume, price_open=p.price_open,
                    price_current=p.price_current, sl=p.sl, tp=p.tp,
                    profit=p.profit,
                ))
            return out
        except Exception as e:
            logger.error(f"[Execution] Error obteniendo posiciones: {e}")
            return []

    def get_symbol_tick(self, symbol: str) -> Optional[dict]:
        mt5 = get_mt5_provider().get_raw_mt5()
        if mt5 is None:
            return None
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}
        except Exception as e:
            logger.error(f"[Execution] Error obteniendo tick de {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # Dimensionamiento (delegado al RiskOverlay, expuesto aquí para la UI)
    # ------------------------------------------------------------------

    def check_proposed_order(
        self, symbol: str, direction: str, stop_loss_price: float,
    ) -> RiskCheckResult:
        """Valida y dimensiona una orden ANTES de mostrar el botón de
        confirmación al usuario — no envía nada."""
        try:
            account = self._verify_account_allowed()
        except DemoAccountRequiredError as e:
            return RiskCheckResult(allowed=False, reason=str(e))

        provider = get_mt5_provider()
        sym_info = provider.get_symbol_info(symbol)
        if sym_info is None:
            return RiskCheckResult(allowed=False, reason=f"No se pudo obtener información del símbolo {symbol}.")

        tick = self.get_symbol_tick(symbol)
        if tick is None:
            return RiskCheckResult(allowed=False, reason=f"No se pudo obtener el precio actual de {symbol}.")

        entry_price = tick["ask"] if direction == "buy" else tick["bid"]
        n_open = len(self.get_positions())

        return self.risk_overlay.check_order(
            account_equity=account["equity"],
            account_is_demo=account["is_demo"],
            n_open_positions=n_open,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            contract_size=sym_info["contract_size"],
            volume_min=sym_info["min_lot"],
            volume_max=sym_info["max_lot"],
            volume_step=sym_info["lot_step"],
            allow_real_account=self.allow_real_account,
        )

    # ------------------------------------------------------------------
    # Envío de órdenes — SIEMPRE requiere confirm=True explícito
    # ------------------------------------------------------------------

    def send_market_order(
        self,
        symbol: str,
        direction: str,          # "buy" | "sell"
        volume: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        comment: str = "CapitalQuant",
        confirm: bool = False,
    ) -> OrderResult:
        """
        Envía una orden de mercado. `confirm` DEBE ser True — este
        parámetro existe a propósito para que el código de la UI no
        pueda llamar esto "por accidente" en un re-render de Streamlit;
        el valor True solo debe originarse en el manejador de un botón
        que el usuario acaba de presionar.
        """
        if not confirm:
            return OrderResult(success=False, message="Orden no enviada: falta confirmación explícita del usuario.")

        try:
            account = self._verify_account_allowed()
        except DemoAccountRequiredError as e:
            return OrderResult(success=False, message=str(e))

        mt5 = get_mt5_provider().get_raw_mt5()
        if mt5 is None:
            return OrderResult(success=False, message="Sin conexión activa a MT5.")

        # Revalidar con el risk overlay justo antes de enviar (el precio
        # y el equity pudieron cambiar desde que se mostró el botón).
        check = self.check_proposed_order(symbol, direction, stop_loss_price) if stop_loss_price else None
        if check is not None and not check.allowed:
            return OrderResult(success=False, message=f"Rechazado por el risk overlay: {check.reason}")
        if check is not None and check.suggested_volume is not None:
            volume = check.suggested_volume

        tick = self.get_symbol_tick(symbol)
        if tick is None:
            return OrderResult(success=False, message=f"No se pudo obtener precio de {symbol}.")

        order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
        price = tick["ask"] if direction == "buy" else tick["bid"]

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": 20,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if stop_loss_price:
            request["sl"] = float(stop_loss_price)
        if take_profit_price:
            request["tp"] = float(take_profit_price)

        try:
            result = mt5.order_send(request)
        except Exception as e:
            logger.error(f"[Execution] Excepción enviando orden: {e}")
            return OrderResult(success=False, message=f"Excepción enviando orden: {e}")

        if result is None:
            return OrderResult(success=False, message=f"order_send devolvió None: {mt5.last_error()}")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(success=False, message=f"MT5 rechazó la orden (retcode={result.retcode}): {result.comment}")

        logger.info(f"[Execution][DEMO] Orden enviada: {symbol} {direction} vol={volume} ticket={result.order}")
        return OrderResult(success=True, message="Orden ejecutada.", ticket=result.order,
                            volume=volume, price=price)

    def close_position(self, position: Position, confirm: bool = False) -> OrderResult:
        """Cierra una posición abierta con una orden de mercado opuesta."""
        if not confirm:
            return OrderResult(success=False, message="Cierre no ejecutado: falta confirmación explícita.")

        try:
            self._verify_account_allowed()
        except DemoAccountRequiredError as e:
            return OrderResult(success=False, message=str(e))

        mt5 = get_mt5_provider().get_raw_mt5()
        if mt5 is None:
            return OrderResult(success=False, message="Sin conexión activa a MT5.")

        tick = self.get_symbol_tick(position.symbol)
        if tick is None:
            return OrderResult(success=False, message=f"No se pudo obtener precio de {position.symbol}.")

        close_type = mt5.ORDER_TYPE_SELL if position.direction == "buy" else mt5.ORDER_TYPE_BUY
        price = tick["bid"] if position.direction == "buy" else tick["ask"]

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "comment": "CapitalQuant close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        try:
            result = mt5.order_send(request)
        except Exception as e:
            return OrderResult(success=False, message=f"Excepción cerrando posición: {e}")

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = mt5.last_error() if result is None else f"retcode={result.retcode}: {result.comment}"
            return OrderResult(success=False, message=f"No se pudo cerrar la posición: {msg}")

        logger.info(f"[Execution][DEMO] Posición {position.ticket} cerrada.")
        return OrderResult(success=True, message="Posición cerrada.", ticket=position.ticket)

    def close_all_positions(self, confirm: bool = False) -> List[OrderResult]:
        """Cierra TODAS las posiciones abiertas — el botón de pánico de
        la UI. Requiere confirm=True explícito, igual que una orden individual."""
        if not confirm:
            return [OrderResult(success=False, message="Cierre masivo no ejecutado: falta confirmación explícita.")]
        return [self.close_position(p, confirm=True) for p in self.get_positions()]
