"""
execution/live_trader.py
Motor de Trading Algorítmico — reemplaza el antiguo panel de "Ejecución
(Demo)" (orden manual + confirmación por clic) por ejecución automática:
el usuario elige estrategia + parámetros + riesgo, presiona "Iniciar" UNA
vez, y desde ahí el motor sondea MT5 en segundo plano, recalcula la señal
de la estrategia con el MISMO motor de backtest de siempre
(`engine/backtester.py` vía `strategy.generate_signals`), y abre/cierra
posiciones por sí mismo sin requerir un clic humano por operación.

Cambio de modelo de seguridad, pedido explícitamente por el usuario: la
autorización humana pasa de "un clic por orden" a "un clic para iniciar
la sesión algorítmica completa". Por eso mismo las guardas de
`execution/risk_overlay.py` (SL obligatorio, kill switch diario, límite de
posiciones, dimensionamiento acotado, y ahora también el opt-in explícito
para cuenta REAL) se revalidan en TODOS los ciclos del motor, no solo al
momento de enviar una orden puntual.

Arquitectura: el motor corre en un `threading.Thread` daemon dentro del
mismo proceso de Streamlit — funciona mientras la app quede abierta en tu
máquina (uso previsto: un usuario, una sesión local). Si cierras la
terminal de Streamlit o el proceso se reinicia, el motor se detiene junto
con él (no hay ejecución fuera del proceso de la app). El estado
(`status()`) se protege con un lock y la UI solo LEE ese estado en cada
rerun — nunca toca MT5 directamente mientras el motor está corriendo.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
from loguru import logger

from market_data.mt5_provider import get_mt5_provider
from execution.mt5_execution import MT5ExecutionClient
from execution.risk_overlay import RiskOverlay, RiskOverlayConfig
from strategies import STRATEGY_REGISTRY


@dataclass
class LiveTradeLogEntry:
    timestamp: datetime
    action: str          # "start"|"stop"|"skip"|"open_long"|"open_short"|"close"|"error"|"kill_switch"
    detail: str
    ticket: Optional[int] = None


@dataclass
class AlgoTradingConfig:
    """Configuración de una sesión algorítmica — todo lo que el usuario
    puede ajustar desde la UI antes de presionar 'Iniciar'."""
    strategy_name: str
    strategy_params: dict          # incluye los periodos de indicadores de la estrategia
    asset: str
    timeframe: str
    risk_per_trade_pct: float = 0.02
    max_daily_loss_pct: float = 0.05
    max_open_positions: int = 1
    poll_seconds: int = 20
    n_bars: int = 500
    allow_real_account: bool = False   # requiere casilla de confirmación explícita en la UI


class AlgoTradingEngine:
    """
    Motor de UNA estrategia en ejecución automática sobre un activo y
    temporalidad. Uso:

        engine = AlgoTradingEngine(config)
        engine.start()      # dispara el hilo de fondo
        ...
        st = engine.status()   # la UI solo lee esto en cada rerun
        ...
        engine.stop()        # detiene el hilo; NO cierra posiciones abiertas
    """

    def __init__(self, config: AlgoTradingConfig):
        self.config = config
        overlay_cfg = RiskOverlayConfig(
            risk_per_trade_pct=config.risk_per_trade_pct,
            max_daily_loss_pct=config.max_daily_loss_pct,
            max_open_positions=config.max_open_positions,
        )
        self.risk_overlay = RiskOverlay(overlay_cfg)
        self.client = MT5ExecutionClient(self.risk_overlay, allow_real_account=config.allow_real_account)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._running = False
        self._status: Dict[str, Any] = {
            "running": False,
            "last_check": None,
            "last_signal": None,
            "last_error": None,
            "n_checks": 0,
            "log": [],
        }

    # ------------------------------------------------------------------
    # Control — llamado desde la UI (nunca desde el propio hilo)
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._status["running"] = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._log("start", f"Sesión algorítmica iniciada: {self.config.strategy_name} · "
                            f"{self.config.asset} {self.config.timeframe} · "
                            f"riesgo {self.config.risk_per_trade_pct*100:.1f}%/operación · "
                            f"sondeo cada {self.config.poll_seconds}s.")

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._status["running"] = False
        self._log("stop", "Sesión algorítmica detenida. Las posiciones abiertas NO se cierran "
                           "automáticamente al detener — hazlo manualmente desde el panel si "
                           "quieres salir de todo ahora mismo.")

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def status(self) -> dict:
        """Copia superficial del estado — segura de llamar desde la UI en
        cada rerun de Streamlit mientras el hilo sigue corriendo."""
        with self._lock:
            return dict(self._status)

    # ------------------------------------------------------------------
    # Ciclo del motor
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self.is_running():
            try:
                self._check_once()
            except Exception as e:
                logger.error(f"[AlgoTrading] Error inesperado en el ciclo: {e}\n{traceback.format_exc()}")
                with self._lock:
                    self._status["last_error"] = str(e)
                self._log("error", f"Error inesperado en el ciclo (ver logs): {e}")
            time.sleep(max(5, self.config.poll_seconds))

    def _check_once(self) -> None:
        cfg = self.config
        mt5 = get_mt5_provider()

        with self._lock:
            self._status["n_checks"] += 1
            self._status["last_check"] = datetime.now()

        account = self.client.account_summary()
        if account is None:
            self._log("skip", "Sin conexión MT5 en este ciclo — se reintentará en el próximo.")
            return

        # Kill switch: se revalida en TODOS los ciclos, no solo al enviar
        # una orden puntual — si la cuenta cruzó el límite de pérdida
        # diaria entre ciclos, el motor deja de operar de inmediato.
        daily = self.risk_overlay.check_daily_loss(account["equity"])
        if not daily.allowed:
            self._log("kill_switch", daily.reason)
            return

        if not account["is_demo"] and not cfg.allow_real_account:
            self._log("error", "Cuenta REAL sin confirmación explícita — deteniendo la sesión.")
            self.stop()
            return

        df = mt5.get_data(cfg.asset, cfg.timeframe, n_bars=cfg.n_bars)
        if df is None or df.empty:
            self._log("skip", f"MT5 no devolvió velas para {cfg.asset} {cfg.timeframe}.")
            return

        strategy_cls = STRATEGY_REGISTRY.get(cfg.strategy_name)
        if strategy_cls is None:
            self._log("error", f"La estrategia '{cfg.strategy_name}' ya no está registrada — deteniendo la sesión.")
            self.stop()
            return

        try:
            strategy = strategy_cls(**cfg.strategy_params)
            sig_df = strategy.generate_signals(df)
        except Exception as e:
            self._log("error", f"La estrategia falló al generar señales: {e}")
            return

        if sig_df is None or sig_df.empty or "signal" not in sig_df.columns:
            self._log("error", "La estrategia no produjo una columna 'signal' válida.")
            return

        last = sig_df.iloc[-1]
        desired_dir = int(last.get("signal", 0) or 0)
        with self._lock:
            self._status["last_signal"] = desired_dir

        positions = self.client.get_positions(cfg.asset)
        current_dir = 0
        if positions:
            current_dir = 1 if positions[0].direction == "buy" else -1

        if desired_dir == current_dir:
            self._log("skip", f"Señal sin cambios (dir={desired_dir}) — no se envía ninguna orden.")
            return

        # La señal cambió de lado (o pasó a plano) respecto de la posición
        # actual: cerrar lo que hay abierto primero.
        for p in positions:
            result = self.client.close_position(p, confirm=True)
            self._log("close" if result.success else "error", result.message, ticket=p.ticket)

        if desired_dir == 0:
            return  # la señal pasó a plano — ya se cerró arriba, nada más que hacer

        direction = "buy" if desired_dir == 1 else "sell"
        sl = None
        if "stop_loss" in sig_df.columns and pd.notna(last.get("stop_loss")):
            sl = float(last["stop_loss"])
        if sl is None:
            self._log("skip", "La estrategia no calculó un stop-loss para esta señal — el risk "
                               "overlay exige SL obligatorio, no se envía la orden.")
            return

        tp = None
        if "take_profit" in sig_df.columns and pd.notna(last.get("take_profit")):
            tp = float(last["take_profit"])

        check = self.client.check_proposed_order(cfg.asset, direction, sl)
        if not check.allowed:
            self._log("skip", f"Orden rechazada por el risk overlay: {check.reason}")
            return
        for w in check.warnings:
            self._log("skip", f"Advertencia del risk overlay: {w}")

        result = self.client.send_market_order(
            cfg.asset, direction, check.suggested_volume,
            stop_loss_price=sl, take_profit_price=tp, confirm=True,
        )
        action = "open_long" if direction == "buy" else "open_short"
        self._log(action if result.success else "error", result.message, ticket=result.ticket)

    def _log(self, action: str, detail: str, ticket: Optional[int] = None) -> None:
        entry = LiveTradeLogEntry(timestamp=datetime.now(), action=action, detail=detail, ticket=ticket)
        with self._lock:
            log = self._status["log"]
            log.append(entry)
            self._status["log"] = log[-200:]  # tope: no crecer sin límite en sesiones largas
        logger.info(f"[AlgoTrading][{action}] {detail}")
