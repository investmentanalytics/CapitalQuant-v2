"""
discovery/backtest_engine.py
Backtester interno y rápido usado ÚNICAMENTE por el motor genético
(genetic_discovery.py) para puntuar miles de individuos por corrida
evolutiva. NO es una funcionalidad expuesta al usuario: el backtesting
"real" que ve el usuario siempre corre en engine/backtester.py de
CapitalQuant, una vez la estrategia descubierta se exporta.

Bar-by-bar (NumPy) con stop-loss/take-profit basados en ATR, comisiones y
slippage en dólares y métricas — deliberadamente reescrito para replicar
BIT A BIT la semántica de ejecución de engine/backtester.py:

- Rezago de 1 vela: la señal calculada en la vela i-1 (con el close de esa
  vela) se ejecuta en el `open` de la vela i, nunca en el close/high/low de
  la misma vela que la generó. Antes este simulador entraba/salía en el
  close de la MISMA vela y comprobaba SL/TP contra el high/low de esa misma
  vela — eso es look-ahead bias puro, y era la causa principal de que el
  Hall of Fame luciera excelente durante la evolución (p.ej. Sharpe~1.6) y
  se desplomara al verificarlo con el motor real (Sharpe negativo): el GA
  llevaba 70 generaciones optimizando un artefacto que no existe en
  ejecución real, no una ventaja genuina.
- Position sizing y comisión/slippage en dólares sobre precio×tamaño,
  idénticos a `BacktestEngine._calculate_size` del motor real (antes se
  usaban bps sobre el retorno, con otra escala de costos).
- ATR/SL/TP de una posición se calculan con el ATR de la vela i-1 (la
  misma vela que generó la señal), igual que hace `build_execution_frame`
  para el motor real — no con el ATR de la vela de entrada.

Con esto, el ranking de fitness que guía la búsqueda evolutiva ya es
consistente con lo que confirmará después el motor real, en vez de premiar
reglas que solo "funcionan" por mirar información que en producción no
existe todavía. Aun así, sigue siendo un simulador simplificado (sin
niveles de comisión configurables por bróker, etc.); el paso de
verificación con engine/backtester.py en discovery_engine.py sigue siendo
la cifra final y definitiva que se muestra al usuario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .indicators import atr as atr_fn


TRADING_DAYS_PER_YEAR = 252


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int          # 1 long, -1 short
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    exit_reason: str        # "tp", "sl", "signal_flip", "end_of_data"
    pnl_pct: float
    bars_held: int


@dataclass
class BacktestResult:
    strategy_id: str
    equity_curve: pd.Series
    returns: pd.Series
    trades: list[Trade]
    metrics: dict


def _annualization_factor(timeframe: str) -> float:
    bars_per_day = {
        "M1": 1440, "M5": 288, "M15": 96, "M30": 48,
        "H1": 24, "H4": 6, "D1": 1, "W1": 1 / 7, "MN1": 1 / 30,
    }.get(timeframe, 24)
    return bars_per_day * TRADING_DAYS_PER_YEAR


def run_backtest_signal(strategy_id: str, signal: pd.Series, df: pd.DataFrame,
                         stop_loss_atr: float = 2.0, take_profit_atr: float = 3.0,
                         atr_period: int = 14, trailing_stop_atr: Optional[float] = None,
                         warmup: int = 30, timeframe: str = "H1",
                         initial_capital: float = 100_000.0, commission: float = 0.001,
                         slippage: float = 0.0005, risk_per_trade: float = 0.02) -> BacktestResult:
    """Event-driven backtest from an arbitrary pre-computed {-1,0,1} signal
    series (e.g. produced by rule_engine.evaluate_rule-derived long/short
    signals via genetic_discovery). Mirrors engine/backtester.py's execution
    semantics (1-bar lag, dollar-based commission/slippage/sizing) so that
    fitness computed here is consistent with what the real engine will later
    confirm. `commission`/`slippage` are fractions of price×size per leg
    (0.001 = 0.10%), matching core.types.BacktestConfig defaults — NOT bps."""
    return _simulate(
        strategy_id=strategy_id, signal=signal, df=df,
        stop_loss_atr=stop_loss_atr, take_profit_atr=take_profit_atr,
        atr_period=atr_period, trailing_stop_atr=trailing_stop_atr,
        warmup=warmup, timeframe=timeframe, initial_capital=initial_capital,
        commission=commission, slippage=slippage, risk_per_trade=risk_per_trade,
    )


def _simulate(strategy_id: str, signal: pd.Series, df: pd.DataFrame,
              stop_loss_atr: float, take_profit_atr: float, atr_period: int,
              trailing_stop_atr: Optional[float], warmup: int, timeframe: str,
              initial_capital: float, commission: float, slippage: float,
              risk_per_trade: float) -> BacktestResult:
    opens = df["open"].to_numpy()
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr_series = atr_fn(df["high"], df["low"], df["close"], atr_period).to_numpy()
    signal_arr = signal.reindex(df.index).fillna(0).to_numpy()
    n = len(df)
    idx = df.index
    warmup = min(max(warmup, 1), n - 1) if n > 1 else 0

    equity = np.empty(n)
    equity[:] = initial_capital
    position = 0            # -1, 0, 1
    entry_price = 0.0
    entry_i = -1
    entry_size = 0.0
    entry_commission = 0.0
    entry_slippage = 0.0
    sl_price = 0.0
    tp_price = 0.0
    trail_price = 0.0
    trades: list[Trade] = []

    cash = initial_capital
    equity[0] = cash

    for i in range(1, n):
        if i < warmup or np.isnan(atr_series[i - 1]):
            equity[i] = cash
            continue

        # Rezago de 1 vela: la señal y el ATR usados aquí son los de la vela
        # i-1 (conocidos recién al cerrar esa vela); se ejecutan en el open
        # de la vela i — igual que engine/backtester.py. bar_high/bar_low de
        # la vela i solo se usan para comprobar si el SL/TP fue tocado
        # DURANTE la vela i, nunca para decidir la señal de esa misma vela.
        exec_price = opens[i]
        bar_high, bar_low = high[i], low[i]
        sig_prev = int(signal_arr[i - 1])
        atr_prev = atr_series[i - 1]

        block_reentry_this_bar = False

        if position != 0:
            exit_reason = None
            exit_price = exec_price

            if position == 1:
                if bar_low <= sl_price:
                    exit_price, exit_reason = sl_price, "sl"
                elif bar_high >= tp_price:
                    exit_price, exit_reason = tp_price, "tp"
                elif sig_prev == -1:
                    exit_reason = "signal_flip"
            else:
                if bar_high >= sl_price:
                    exit_price, exit_reason = sl_price, "sl"
                elif bar_low <= tp_price:
                    exit_price, exit_reason = tp_price, "tp"
                elif sig_prev == 1:
                    exit_reason = "signal_flip"

            if exit_reason is not None:
                comm_exit = exit_price * entry_size * commission
                slip_exit = exit_price * entry_size * slippage
                raw_pnl = (exit_price - entry_price) * entry_size * position
                net_pnl = raw_pnl - entry_commission - comm_exit - entry_slippage - slip_exit
                cash = max(cash + net_pnl, 1e-6)
                net_ret = net_pnl / (entry_price * entry_size) if entry_size > 0 else 0.0
                trades.append(Trade(
                    entry_time=idx[entry_i], exit_time=idx[i], direction=position,
                    entry_price=entry_price, exit_price=exit_price, sl_price=sl_price,
                    tp_price=tp_price, exit_reason=exit_reason, pnl_pct=net_ret,
                    bars_held=i - entry_i,
                ))
                position = 0

                # Un cierre por sl/tp ocurre intravela (en algun punto entre
                # open[i] y el high/low que lo disparo): no sabemos si fue
                # antes o despues del open[i]. Por eso NO se permite reabrir
                # en esta misma vela -- igual que engine/backtester.py. Un
                # cierre por "signal_flip" si puede reabrir en la misma vela,
                # porque ocurre limpio en el open con informacion ya conocida
                # del cierre de la vela previa (sin ambiguedad de timing).
                if exit_reason in ("sl", "tp"):
                    block_reentry_this_bar = True

        if position == 0 and sig_prev != 0 and not block_reentry_this_bar:
            a = atr_prev if atr_prev > 0 else exec_price * 0.001
            candidate_sl = (exec_price - stop_loss_atr * a) if sig_prev == 1 else (exec_price + stop_loss_atr * a)
            size = _position_size(cash, exec_price, candidate_sl, risk_per_trade)
            if size > 0:
                position = sig_prev
                entry_price = exec_price
                entry_i = i
                entry_size = size
                entry_commission = exec_price * size * commission
                entry_slippage = exec_price * size * slippage
                cash -= (entry_commission + entry_slippage)
                if position == 1:
                    sl_price = entry_price - stop_loss_atr * a
                    tp_price = entry_price + take_profit_atr * a
                else:
                    sl_price = entry_price + stop_loss_atr * a
                    tp_price = entry_price - take_profit_atr * a
                trail_price = sl_price

        if position != 0:
            unrealized = (close[i] - entry_price) * entry_size * position
            equity[i] = cash + unrealized
        else:
            equity[i] = cash

    if position != 0:
        exit_price = close[-1]
        comm_exit = exit_price * entry_size * commission
        slip_exit = exit_price * entry_size * slippage
        raw_pnl = (exit_price - entry_price) * entry_size * position
        net_pnl = raw_pnl - entry_commission - comm_exit - entry_slippage - slip_exit
        cash = max(cash + net_pnl, 1e-6)
        net_ret = net_pnl / (entry_price * entry_size) if entry_size > 0 else 0.0
        equity[-1] = cash
        trades.append(Trade(
            entry_time=idx[entry_i], exit_time=idx[-1], direction=position,
            entry_price=entry_price, exit_price=exit_price, sl_price=sl_price,
            tp_price=tp_price, exit_reason="end_of_data", pnl_pct=net_ret,
            bars_held=n - 1 - entry_i,
        ))

    equity_series = pd.Series(equity, index=idx, name="equity")
    returns_series = equity_series.pct_change().fillna(0.0)
    metrics = compute_metrics(equity_series, returns_series, trades, timeframe)

    return BacktestResult(
        strategy_id=strategy_id, equity_curve=equity_series,
        returns=returns_series, trades=trades, metrics=metrics,
    )


def _position_size(capital: float, price: float, sl_price: float, risk_per_trade: float) -> float:
    """Tamaño de posición por riesgo, idéntico a
    engine.backtester.BacktestEngine._calculate_size: nunca apalanca (tope
    duro de 0.95x capital/precio), y dimensiona por distancia al stop en
    dólares, no como fracción de retorno. Antes este simulador rápido
    escalaba el retorno bruto por operación hasta 5x vía un cap de
    apalancamiento fantasma — eso, sumado al look-ahead de timing, explicaba
    la mayor parte de la brecha entre las cifras del Descubridor y las del
    Backtesting real para la misma estrategia."""
    if sl_price is None or np.isnan(sl_price) or sl_price <= 0 or price <= 0:
        return (capital * 0.95) / price if price > 0 else 0.0
    risk_per_unit = abs(price - sl_price)
    if risk_per_unit <= 0:
        return 0.0
    risk_amount = capital * risk_per_trade
    size = risk_amount / risk_per_unit
    max_size = (capital * 0.95) / price
    return min(size, max_size)


def compute_metrics(equity: pd.Series, returns: pd.Series, trades: list[Trade],
                     timeframe: str = "H1") -> dict:
    ann_factor = _annualization_factor(timeframe)
    n_bars = len(equity)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 0 else 0.0

    years = n_bars / ann_factor if ann_factor > 0 else np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years and years > 0 and equity.iloc[0] > 0 else 0.0

    ret_std = returns.std()
    sharpe = (returns.mean() / ret_std * np.sqrt(ann_factor)) if ret_std and ret_std > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std()
    sortino = (returns.mean() / downside_std * np.sqrt(ann_factor)) if downside_std and downside_std > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min() if len(drawdown) else 0.0
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

    n_trades = len(trades)
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    gross_profit = sum(t.pnl_pct for t in wins)
    gross_loss = abs(sum(t.pnl_pct for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    avg_bars_held = np.mean([t.bars_held for t in trades]) if trades else 0.0

    if n_trades > 1:
        pnl_series = np.array([t.pnl_pct for t in trades])
        streaks = []
        cur = 0
        cur_sign = 0
        for p in pnl_series:
            sign = 1 if p > 0 else -1
            if sign == cur_sign:
                cur += 1
            else:
                cur = 1
                cur_sign = sign
            streaks.append(cur * sign)
        max_win_streak = max((s for s in streaks if s > 0), default=0)
        max_loss_streak = abs(min((s for s in streaks if s < 0), default=0))
    else:
        max_win_streak = 0
        max_loss_streak = 0

    exposure = n_trades and sum(t.bars_held for t in trades) / max(n_bars, 1) or 0.0

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(max_drawdown),
        "n_trades": int(n_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else 999.0,
        "expectancy": float(expectancy),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "avg_bars_held": float(avg_bars_held),
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_loss_streak),
        "exposure": float(exposure),
        "final_equity": float(equity.iloc[-1]) if len(equity) else 0.0,
    }


def trades_to_records(trades: list[Trade]) -> list[dict]:
    return [{
        "entry_time": t.entry_time.isoformat(), "exit_time": t.exit_time.isoformat(),
        "direction": "long" if t.direction == 1 else "short",
        "entry_price": t.entry_price, "exit_price": t.exit_price,
        "sl_price": t.sl_price, "tp_price": t.tp_price, "exit_reason": t.exit_reason,
        "pnl_pct": t.pnl_pct, "bars_held": t.bars_held,
    } for t in trades]
