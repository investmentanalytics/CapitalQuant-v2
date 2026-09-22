"""
engine/backtester.py
Motor de backtesting — refactorizado y optimizado.

Cambios respecto al original:
- Tipos movidos a core/types.py
- Métricas movidas a core/metrics.py
- Arrays NumPy para el loop caliente (10-50x más rápido que iloc)
- SL/TP como campos limpios en Trade (no __dict__ dinámico)
- Validación de entrada con core/validators.py
- Cero código debug en producción
"""
from __future__ import annotations
from typing import Optional, List

import numpy as np
import pandas as pd
from loguru import logger

from core.types import BacktestConfig, Trade, BacktestResults
from core.metrics import (
    cagr, sharpe_ratio, sortino_ratio, calmar_ratio,
    compute_drawdown, ulcer_index, profit_factor,
    max_streak, compute_exposure_pct,
)
from core.validators import validate_ohlcv
from core.costs import InstrumentCostProfile, entry_cost, exit_cost


class BacktestEngine:
    """
    Motor de backtesting con simulación realista de ejecución.

    Soporta:
    - Stop loss y take profit dinámicos con check intrabar
    - Position sizing basado en riesgo (ATR o fixed)
    - Comisiones y slippage
    - Long-only, short-only o ambos (allow_long / allow_short)
    - Vectorizado con NumPy para máximo rendimiento
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        if self.config.use_realistic_costs and self.config.cost_profile is None:
            # Sin symbol_info disponible, usar un perfil genérico razonable
            # en vez de fallar silenciosamente con costo cero.
            logger.warning(
                "use_realistic_costs=True pero no se proveyó cost_profile; "
                "usando InstrumentCostProfile por defecto (spread=1pt genérico)."
            )
            self.config.cost_profile = InstrumentCostProfile()

    def _entry_transaction_cost(self, price: float, size: float) -> float:
        """Costo monetario total de abrir la posición (comisión + spread/slippage)."""
        cfg = self.config
        if cfg.use_realistic_costs:
            return entry_cost(cfg.cost_profile, size, price)
        return price * size * cfg.commission + price * size * cfg.slippage

    def _exit_transaction_cost(self, price: float, size: float) -> float:
        """Costo monetario total de cerrar la posición (comisión + spread/slippage)."""
        cfg = self.config
        if cfg.use_realistic_costs:
            return exit_cost(cfg.cost_profile, size, price)
        return price * size * cfg.commission + price * size * cfg.slippage

    def run(
        self,
        signals_df: pd.DataFrame,
        strategy_name: str = "",
        asset: str = "",
        timeframe: str = "",
        progress_callback=None,
        progress_every: int = 50,
    ) -> BacktestResults:
        """
        Ejecuta el backtest sobre el DataFrame con señales generadas.

        Parameters
        ----------
        signals_df : pd.DataFrame
            DataFrame OHLCV + columna 'signal' (1=Long, -1=Short, 0=Sin posición)
            Opcionalmente: stop_loss, take_profit

        Notas de ejecución
        ------------------
        La señal en la fila `i` se calcula con el `close` de la fila `i`
        (indicadores basados en cierre), por lo que solo se conoce una vez
        cerrada esa vela. El motor la ejecuta —tanto para entradas como para
        salidas— en el `open` de la fila `i+1`, nunca en el `open` de la
        misma fila que la generó. Esto evita look-ahead bias (no se puede
        operar en un open con información que todavía no existía a esa hora).
        """
        # Validar y preparar datos
        validate_ohlcv(signals_df, name=f"{asset} {timeframe}")
        df = signals_df.copy().dropna(subset=["close", "signal"])

        cfg = self.config

        # Extraer arrays NumPy para máximo rendimiento en el loop
        opens   = df["open"].values.astype(float)
        highs   = df["high"].values.astype(float)
        lows    = df["low"].values.astype(float)
        closes  = df["close"].values.astype(float)
        sigs    = df["signal"].values.astype(int)

        has_sl = "stop_loss" in df.columns
        has_tp = "take_profit" in df.columns
        sl_arr = df["stop_loss"].values.astype(float) if has_sl else np.full(len(df), np.nan)
        tp_arr = df["take_profit"].values.astype(float) if has_tp else np.full(len(df), np.nan)

        # Estado
        capital = cfg.initial_capital
        equity_arr = np.full(len(df), np.nan)
        equity_arr[0] = capital

        trades: List[Trade] = []
        trade_counter = 0

        in_position = False
        pos_direction = 0
        pos_entry_price = 0.0
        pos_size = 0.0
        pos_sl = np.nan
        pos_tp = np.nan
        pos_entry_idx = 0
        pos_commission = 0.0
        pos_slippage = 0.0
        pos_capital_at_entry = 0.0
        progress_every = max(1, int(progress_every))
        last_progress_bar = -progress_every

        def _emit_progress(bar_index: int, force: bool = False) -> None:
            nonlocal last_progress_bar
            if progress_callback is None:
                return
            if not force and (bar_index - last_progress_bar) < progress_every:
                return
            last_progress_bar = bar_index
            partial = pd.Series(equity_arr[:bar_index + 1], index=df.index[:bar_index + 1]).ffill().fillna(cfg.initial_capital)
            progress_callback({
                "bar_index": int(bar_index),
                "total_bars": int(len(df)),
                "progress": float((bar_index + 1) / max(len(df), 1)),
                "current_date": df.index[bar_index],
                "capital": float(capital),
                "equity_curve": partial,
                "trades": list(trades),
                "open_position": bool(in_position),
                "position_direction": int(pos_direction),
                "entry_price": float(pos_entry_price) if in_position else None,
            })

        _emit_progress(0, force=True)

        for i in range(1, len(df)):
            exec_price = opens[i]
            block_reentry_this_bar = False

            # La señal de la vela i solo se conoce con el cierre de la vela i
            # (los indicadores usan close[i]). Por lo tanto es ejecutable
            # recién en la apertura de la vela SIGUIENTE, usando la señal
            # generada en la vela anterior (i-1). Usar sigs[i] junto con
            # opens[i] en la misma iteración sería look-ahead bias: se
            # estaría reaccionando en el open a información que solo existe
            # tras el close de esa misma vela.
            sig_prev = sigs[i - 1]
            sl_prev = sl_arr[i - 1]
            tp_prev = tp_arr[i - 1]

            # ---- Gestión de posición abierta ----
            if in_position:
                exit_reason = None
                exit_price = exec_price

                if pos_direction == 1:  # Long
                    if not np.isnan(pos_sl) and lows[i] <= pos_sl:
                        exit_price = pos_sl
                        exit_reason = "stop_loss"
                    elif not np.isnan(pos_tp) and highs[i] >= pos_tp:
                        exit_price = pos_tp
                        exit_reason = "take_profit"
                    elif sig_prev == -1:
                        exit_reason = "signal"
                else:  # Short
                    if not np.isnan(pos_sl) and highs[i] >= pos_sl:
                        exit_price = pos_sl
                        exit_reason = "stop_loss"
                    elif not np.isnan(pos_tp) and lows[i] <= pos_tp:
                        exit_price = pos_tp
                        exit_reason = "take_profit"
                    elif sig_prev == 1:
                        exit_reason = "signal"

                if exit_reason:
                    trade_counter += 1
                    if cfg.use_realistic_costs:
                        comm_exit, slip_exit = self._exit_transaction_cost(exit_price, pos_size), 0.0
                    else:
                        comm_exit = exit_price * pos_size * cfg.commission
                        slip_exit = exit_price * pos_size * cfg.slippage
                    raw_pnl = (exit_price - pos_entry_price) * pos_size * pos_direction
                    net_pnl = raw_pnl - pos_commission - comm_exit - pos_slippage - slip_exit

                    t = Trade(
                        trade_number=trade_counter,
                        entry_date=df.index[pos_entry_idx],
                        exit_date=df.index[i],
                        direction=pos_direction,
                        entry_price=pos_entry_price,
                        exit_price=exit_price,
                        size=pos_size,
                        pnl=raw_pnl,
                        pnl_pct=(raw_pnl / (pos_entry_price * pos_size)) * 100 if pos_size > 0 else 0,
                        commission=pos_commission + comm_exit,
                        slippage_cost=pos_slippage + slip_exit,
                        net_pnl=net_pnl,
                        exit_reason=exit_reason,
                        capital_at_entry=pos_capital_at_entry,
                        stop_loss=pos_sl,
                        take_profit=pos_tp,
                        duration_bars=i - pos_entry_idx,
                    )
                    capital += net_pnl
                    trades.append(t)
                    in_position = False

                    # Un cierre por stop_loss/take_profit ocurre intravela
                    # (en algún punto entre open[i] y el high/low que lo
                    # disparó): no sabemos si fue antes o después del
                    # open[i]. Por eso NO se permite reabrir una posición
                    # en esta misma vela — eso equivaldría a abrir una
                    # entrada nueva "antes" de que la anterior se cerrara
                    # realmente. La reentrada más temprana posible es en la
                    # vela siguiente (i+1), con la señal que corresponda.
                    # Un cierre por "signal" (reversión) no tiene esta
                    # ambigüedad, ya que se ejecuta limpio en el open con
                    # información ya conocida del cierre de la vela previa,
                    # así que sí puede reabrir en la misma vela (flip).
                    if exit_reason in ("stop_loss", "take_profit"):
                        block_reentry_this_bar = True

            # ---- Nueva señal de entrada ----
            if not in_position and sig_prev != 0 and not block_reentry_this_bar:
                direction = int(sig_prev)
                if direction == -1 and not cfg.allow_short:
                    equity_arr[i] = capital
                    continue
                if direction == 1 and not cfg.allow_long:
                    equity_arr[i] = capital
                    continue

                sl_price = sl_prev
                tp_price = tp_prev
                size = self._calculate_size(capital, exec_price, sl_price, direction, cfg)

                if size <= 0:
                    equity_arr[i] = capital
                    continue

                if cfg.use_realistic_costs:
                    comm, slip = self._entry_transaction_cost(exec_price, size), 0.0
                else:
                    comm = exec_price * size * cfg.commission
                    slip = exec_price * size * cfg.slippage
                capital -= (comm + slip)

                in_position = True
                pos_direction = direction
                pos_entry_price = exec_price
                pos_size = size
                pos_sl = sl_price
                pos_tp = tp_price
                pos_entry_idx = i
                pos_commission = comm
                pos_slippage = slip
                pos_capital_at_entry = capital

            # Equity con unrealized PnL
            if in_position:
                unrealized = (closes[i] - pos_entry_price) * pos_size * pos_direction
                equity_arr[i] = capital + unrealized
            else:
                equity_arr[i] = capital

            _emit_progress(i)

        # Cerrar posición al final si queda abierta
        if in_position:
            trade_counter += 1
            last_price = closes[-1]
            if cfg.use_realistic_costs:
                comm_exit, slip_exit = self._exit_transaction_cost(last_price, pos_size), 0.0
            else:
                comm_exit = last_price * pos_size * cfg.commission
                slip_exit = last_price * pos_size * cfg.slippage
            raw_pnl = (last_price - pos_entry_price) * pos_size * pos_direction
            net_pnl = raw_pnl - pos_commission - comm_exit - pos_slippage - slip_exit

            t = Trade(
                trade_number=trade_counter,
                entry_date=df.index[pos_entry_idx],
                exit_date=df.index[-1],
                direction=pos_direction,
                entry_price=pos_entry_price,
                exit_price=last_price,
                size=pos_size,
                pnl=raw_pnl,
                pnl_pct=(raw_pnl / (pos_entry_price * pos_size)) * 100 if pos_size > 0 else 0,
                commission=pos_commission + comm_exit,
                slippage_cost=pos_slippage + slip_exit,
                net_pnl=net_pnl,
                exit_reason="end",
                capital_at_entry=pos_capital_at_entry,
                stop_loss=pos_sl,
                take_profit=pos_tp,
                duration_bars=(len(df) - 1) - pos_entry_idx,
            )
            capital += net_pnl
            trades.append(t)
            equity_arr[-1] = capital

        _emit_progress(len(df) - 1, force=True)

        # Rellenar NaN en equity (períodos sin cambio)
        equity_series = pd.Series(equity_arr, index=df.index)
        equity_series = equity_series.ffill().fillna(cfg.initial_capital)

        return self._compute_metrics(equity_series, trades, cfg, df, strategy_name, asset, timeframe)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_size(
        capital: float,
        price: float,
        sl_price: float,
        direction: int,
        cfg: BacktestConfig,
    ) -> float:
        """Position sizing basado en riesgo por operación."""
        if not cfg.use_position_sizing or np.isnan(sl_price) or sl_price <= 0:
            return (capital * 0.95) / price

        risk_amount = capital * cfg.risk_per_trade
        risk_per_unit = abs(price - sl_price)
        if risk_per_unit <= 0:
            return 0.0

        size = risk_amount / risk_per_unit
        max_size = (capital * 0.95) / price
        return min(size, max_size)

    def _compute_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Trade],
        cfg: BacktestConfig,
        df: pd.DataFrame,
        strategy_name: str,
        asset: str,
        timeframe: str,
    ) -> BacktestResults:
        """Calcula todas las métricas de rendimiento."""
        r = BacktestResults()
        r.config = cfg
        r.strategy_name = strategy_name
        r.asset = asset
        r.timeframe = timeframe
        r.initial_capital = cfg.initial_capital
        r.final_capital = float(equity_curve.iloc[-1])
        r.equity_curve = equity_curve
        r.start_date = str(df.index[0].date())
        r.end_date = str(df.index[-1].date())

        # Drawdown
        r.drawdown_series, r.max_drawdown, r.max_drawdown_pct = compute_drawdown(equity_curve)
        r.ulcer_index = ulcer_index(equity_curve)

        # P&L
        r.net_profit = r.final_capital - cfg.initial_capital
        r.net_profit_pct = (r.net_profit / cfg.initial_capital) * 100

        # CAGR
        n_days = max((df.index[-1] - df.index[0]).days, 1)
        r.cagr = cagr(cfg.initial_capital, r.final_capital, n_days)

        # Sharpe / Sortino
        returns = equity_curve.pct_change().dropna()
        r.sharpe_ratio = sharpe_ratio(returns, timeframe)
        r.sortino_ratio = sortino_ratio(returns, timeframe)
        r.calmar_ratio = calmar_ratio(r.cagr, r.max_drawdown_pct)

        # Métricas de trades
        r.trades = trades
        r.total_trades = len(trades)

        if trades:
            net_pnls = [t.net_pnl for t in trades]
            wins  = [p for p in net_pnls if p > 0]
            losses = [p for p in net_pnls if p <= 0]

            r.winning_trades = len(wins)
            r.losing_trades  = len(losses)
            r.win_rate       = (len(wins) / len(trades)) * 100
            r.avg_win        = float(np.mean(wins)) if wins else 0.0
            r.avg_loss       = float(np.mean(losses)) if losses else 0.0
            r.avg_trade      = float(np.mean(net_pnls))
            r.expectancy     = r.avg_trade
            r.profit_factor  = profit_factor(net_pnls)
            r.max_consecutive_wins   = max_streak(net_pnls, win=True)
            r.max_consecutive_losses = max_streak(net_pnls, win=False)

            r.trades_df = pd.DataFrame([t.to_dict() for t in trades])
            r.exposure_pct = compute_exposure_pct(r.trades_df, df.index)

        logger.info(
            f"Backtest [{strategy_name}|{asset}|{timeframe}] "
            f"trades={r.total_trades} sharpe={r.sharpe_ratio:.2f} "
            f"cagr={r.cagr:.2f}% dd={r.max_drawdown_pct:.2f}%"
        )

        return r
