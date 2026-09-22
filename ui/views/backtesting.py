"""
ui/views/backtesting.py
Backtesting — selector de fechas y datos vía MT5.
"""
import streamlit as st
import pandas as pd
from datetime import date, timedelta

from strategies import STRATEGY_REGISTRY, list_strategies
from core.types import BacktestConfig
from engine.backtester import BacktestEngine
from market_data import get_data
from market_data.mt5_provider import get_mt5_provider, history_gap_warning
from ui.state import AppState
from ui.components.metrics_grid import render_metrics_grid
from ui.components.cost_toggle import render_cost_toggle
from ui.components.data_source import render_data_source_selector, load_historical_data
from ui.components.regime_filter import render_regime_filter, apply_regime_filter
from visualization.charts import (
    candlestick_chart, equity_curve_chart, trades_histogram, indicators_chart,
)


def render(asset: str, timeframe: str) -> None:
    st.markdown("## Backtesting")
    data_source = render_data_source_selector(timeframe, key_prefix="bt_source", compact=True)
    st.caption(f"Fuente histórica activa: **{data_source}**")

    # ── Configuracion ──
    with st.expander("Configuración", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        strategy_name = col1.selectbox(
            "Estrategia", list_strategies(), key="bt_strategy",
        )
        initial_capital = col2.number_input(
            "Capital ($)", 1000, 100_000_000, 100_000, 5000, key="bt_capital",
        )
        commission = col3.number_input(
            "Comisión (%)", 0.0, 2.0, 0.10, 0.01, key="bt_commission",
        ) / 100
        risk_per_trade = col4.slider(
            "Riesgo/trade (%)", 0.5, 10.0, 2.0, 0.5, key="bt_risk",
        ) / 100
        direction_mode = col4.selectbox(
            "Dirección", ["Largos y cortos", "Solo largos", "Solo cortos"],
            key="bt_direction",
        )
        allow_long  = direction_mode in ("Largos y cortos", "Solo largos")
        allow_short = direction_mode in ("Largos y cortos", "Solo cortos")

        st.markdown("###### Modelo de costos")
        use_realistic_costs, cost_profile = render_cost_toggle(asset, key_prefix="bt")

        st.markdown("###### Régimen de mercado")
        allowed_regimes = render_regime_filter(asset, timeframe, key_prefix="bt")

    # ── Rango de fechas ──
    with st.expander("Rango de fechas del backtest", expanded=True):
        col1, col2, col3 = st.columns([2, 2, 3])

        date_mode = col1.radio(
            "Modo", ["Rango personalizado", "Máxima historia disponible"],
            key="bt_date_mode", horizontal=False,
        )

        use_full_history = (date_mode == "Máxima historia disponible")

        if not use_full_history:
            default_end   = date.today()
            default_start = default_end - timedelta(days=365 * 3)
            bt_start = col2.date_input("Fecha inicio", value=default_start, key="bt_start")
            bt_end   = col2.date_input("Fecha fin",    value=default_end,   key="bt_end")
        else:
            bt_start = bt_end = None
            col2.info("Se usará toda la historia disponible en MT5 (o caché local).")

        if col3.button("ℹ Ver datos disponibles", key="bt_info_btn"):
            df_info = _load_data_range(asset, timeframe, None, None)
            if df_info is not None and not df_info.empty:
                col3.success(
                    f"**{len(df_info):,} velas disponibles**  \n"
                    f"Desde: {df_info.index[0].date()}  \n"
                    f"Hasta: {df_info.index[-1].date()}"
                )
            else:
                col3.warning("Sin datos. Verifica el activo o pulsa Conectar en la barra superior.")

    # ── Parametros de estrategia ──
    strategy_class = STRATEGY_REGISTRY.get(strategy_name)
    params = {}
    if strategy_class:
        default_instance = strategy_class()
        param_space = default_instance.get_param_space()
        if param_space:
            with st.expander("Parámetros de estrategia", expanded=True):
                cols = st.columns(min(len(param_space), 5))
                for i, (pname, spec) in enumerate(param_space.items()):
                    col = cols[i % len(cols)]
                    ptype = spec[0]
                    # Valor real con el que la estrategia fue encontrada/guardada
                    # (no el mínimo del rango), acotado al rango por seguridad.
                    real_default = getattr(default_instance, pname, spec[1])
                    if ptype == "int":
                        lo, hi = int(spec[1]), int(spec[2])
                        default_val = min(max(int(real_default), lo), hi)
                        params[pname] = col.number_input(
                            pname, lo, hi, default_val, 1, key=f"bt_p_{pname}",
                        )
                    elif ptype == "float":
                        lo, hi = float(spec[1]), float(spec[2])
                        default_val = min(max(float(real_default), lo), hi)
                        params[pname] = col.number_input(
                            pname, lo, hi, default_val, 0.1, key=f"bt_p_{pname}",
                        )
                    elif ptype in ("bool", "categorical"):
                        opts = [True, False] if ptype == "bool" else spec[1]
                        default_val = real_default if real_default in opts else opts[0]
                        params[pname] = col.selectbox(
                            pname, opts, index=opts.index(default_val), key=f"bt_p_{pname}",
                        )

    # ── Ejecutar ──
    cost_tag = "realcost" if use_realistic_costs and cost_profile else "pctcost"
    regime_tag = ",".join(sorted(allowed_regimes)) if allowed_regimes else "sin_filtro"
    bt_key = f"{strategy_name}_{asset}_{timeframe}_{initial_capital}_{bt_start}_{bt_end}_{cost_tag}_{regime_tag}"
    if st.button("Ejecutar Backtest", type="primary"):
        _run_backtest(
            bt_key, strategy_name, strategy_class, params,
            asset, timeframe, initial_capital, commission, risk_per_trade,
            allow_long, allow_short, bt_start, bt_end, use_full_history,
            use_realistic_costs, cost_profile, allowed_regimes,
        )

    # ── Resultados progresivos/persistentes ──
    results = AppState.get_backtest(bt_key)
    result_progress = st.progress(
        1.0 if results is not None else 0.0,
        text=(f"Backtest completado · {results.total_trades:,} operaciones" if results is not None
              else "Esperando ejecución del backtest...")
    )
    if results is None:
        st.markdown("### Resultados")
        mc = st.columns(4)
        mc[0].metric("Operaciones", "—")
        mc[1].metric("Profit Factor", "—")
        mc[2].metric("Win Rate", "—")
        mc[3].metric("Max Drawdown", "—")
        st.caption("Los resultados se llenarán vela por vela mientras se ejecuta el backtest.")
        st.caption("Gráfico de precio: esperando datos…")
        st.caption("Curva de equity: esperando datos…")
        st.dataframe(pd.DataFrame(columns=["Entrada","Salida","Dirección","Precio entrada","Precio salida","P&L"]), width="stretch", hide_index=True, height=120)
        return

    _show_results(results, strategy_name, asset, timeframe, strategy_class, params)


# ---------------------------------------------------------------------------

def _run_backtest(bt_key, strategy_name, strategy_class, params,
                   asset, timeframe, initial_capital, commission, risk_per_trade,
                   allow_long, allow_short, bt_start, bt_end, use_full_history,
                   use_realistic_costs=False, cost_profile=None, allowed_regimes=None):
    with st.spinner(f"Ejecutando {strategy_name} en {asset} {timeframe}..."):
        df = _load_data_range(asset, timeframe, bt_start, bt_end)
        if df is None or df.empty:
            st.error(f"Sin datos para {asset} {timeframe} en el rango seleccionado.")
            return

        if not use_full_history and bt_start and bt_end:
            df = df.loc[str(bt_start):str(bt_end)]
            if df.empty:
                st.error("Sin datos en el rango de fechas seleccionado.")
                return
            gap_msg = history_gap_warning(df, bt_start, asset, timeframe)
            if gap_msg:
                st.warning(gap_msg)

        config = BacktestConfig(
            initial_capital=float(initial_capital),
            commission=commission,
            risk_per_trade=risk_per_trade,
            allow_long=allow_long,
            allow_short=allow_short,
            use_realistic_costs=bool(use_realistic_costs and cost_profile is not None),
            cost_profile=cost_profile,
        )
        try:
            strategy = strategy_class(**params)
            signals  = strategy.generate_signals(df)
            if allowed_regimes:
                signals, regime_series = apply_regime_filter(
                    df, signals, asset, timeframe, allowed_regimes,
                )
                if regime_series is not None:
                    n_allowed = int(regime_series.reindex(signals.index).isin(allowed_regimes).sum())
                    st.caption(
                        f"Filtro de régimen activo: {n_allowed}/{len(signals)} velas habilitadas "
                        f"para nuevas entradas ({', '.join(allowed_regimes)})."
                    )
            engine = BacktestEngine(config)

            # Resultados progresivos: el usuario ve el backtest construirse
            # mientras el motor recorre las velas. No se altera la lógica de
            # ejecución; solo se exponen snapshots periódicos del mismo estado
            # que terminará formando BacktestResults.
            st.markdown("### Resultados en vivo")
            live_progress = st.progress(0.0, text="Iniciando backtest...")
            live_status = st.empty()
            live_metrics = st.empty()
            live_price = st.empty()
            live_equity = st.empty()
            live_trades = st.empty()

            def _live_cb(snapshot):
                idx = int(snapshot["bar_index"])
                total = int(snapshot["total_bars"])
                pct = float(snapshot["progress"]) * 100.0
                current_date = snapshot.get("current_date")
                trades_live = snapshot.get("trades", []) or []
                equity_live = snapshot.get("equity_curve")
                live_progress.progress(min(max(pct / 100.0, 0.0), 1.0), text=f"Procesando {asset} {timeframe} · vela {idx + 1:,}/{total:,} ({pct:.1f}%)")
                live_status.info(
                    f"**Procesando {asset} {timeframe}** · vela {idx + 1:,}/{total:,} "
                    f"({pct:.1f}%) · {len(trades_live):,} operaciones cerradas"
                    + (" · posición abierta" if snapshot.get("open_position") else "")
                )

                if equity_live is not None and len(equity_live):
                    initial = float(config.initial_capital)
                    final = float(equity_live.iloc[-1])
                    peak = equity_live.cummax()
                    dd_pct = float(((equity_live / peak) - 1.0).min() * 100.0) if len(peak) else 0.0
                    live_metrics.metric(
                        "Capital / operaciones / DD",
                        f"${final:,.2f} / {len(trades_live):,} / {dd_pct:.2f}%",
                        delta=f"${final - initial:,.2f}",
                    )

                    eq_curves = [{
                        "name": f"{strategy_name} — {asset}",
                        "equity": equity_live,
                        "drawdown": (equity_live / peak - 1.0) * 100.0,
                        "best": True,
                    }]
                    live_equity.plotly_chart(
                        equity_curve_chart(eq_curves, height=360), width="stretch",
                    )

                partial_df = signals.iloc[:idx + 1].copy()
                trades_df_live = pd.DataFrame([t.to_dict() for t in trades_live]) if trades_live else pd.DataFrame()
                try:
                    live_price.plotly_chart(
                        candlestick_chart(
                            df=partial_df,
                            title=f"{asset} {timeframe} — {strategy_name} · progreso",
                            signals=partial_df.get("signal"),
                            trades_df=trades_df_live if not trades_df_live.empty else None,
                            height=430,
                        ), width="stretch",
                    )
                except Exception as exc:
                    live_price.caption(f"Gráfico de precio en actualización: {exc}")

                if not trades_df_live.empty:
                    live_trades.dataframe(
                        trades_df_live, width="stretch", hide_index=True, height=260,
                    )
                else:
                    live_trades.caption("Las operaciones cerradas aparecerán aquí en cuanto se produzcan.")

            results = engine.run(
                signals, strategy_name=strategy_name, asset=asset, timeframe=timeframe,
                progress_callback=_live_cb, progress_every=50,
            )
            AppState.save_backtest(bt_key, results)
            AppState.save_live_config(
                strategy_name, asset, timeframe,
                params=params,
                config=dict(
                    initial_capital=float(initial_capital), commission=commission,
                    risk_per_trade=risk_per_trade, allow_long=allow_long, allow_short=allow_short,
                ),
            )
            cost_note = "costos realistas (spread+comisión)" if config.use_realistic_costs else "costos en % fijo"
            live_progress.progress(1.0, text=f"Backtest completado · {results.total_trades:,} operaciones")
            st.success(
                f"{results.total_trades} operaciones | "
                f"Sharpe: {results.sharpe_ratio:.2f} | "
                f"CAGR: {results.cagr:.2f}% | "
                f"Max DD: {results.max_drawdown_pct:.2f}% | "
                f"Modelo: {cost_note}"
            )
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())


def _show_results(results, strategy_name, asset, timeframe, strategy_class, params):
    st.markdown("### Métricas")
    render_metrics_grid(results.to_dict(), n_cols=4)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Precio", "Equity", "Operaciones", "Distribución"])

    with tab1:
        df_viz = _load_data_range(asset, timeframe, None, None)
        if df_viz is not None and strategy_class:
            df_viz = df_viz.loc[results.start_date:results.end_date]
            strategy_viz = strategy_class(**params)
            sig_df = strategy_viz.generate_signals(df_viz)
            indicators = {c: sig_df[c] for c in strategy_viz.get_indicator_columns() if c in sig_df.columns}
            fig = candlestick_chart(
                df=sig_df, title=f"{asset} {timeframe} — {strategy_name}",
                signals=sig_df.get("signal"),
                trades_df=results.trades_df if not results.trades_df.empty else None,
                height=520,
            )
            st.plotly_chart(fig, width='stretch')
            if indicators:
                st.plotly_chart(indicators_chart(indicators, height=280), width='stretch')

    with tab2:
        curves = [{"name": f"{strategy_name} — {asset}", "equity": results.equity_curve,
                   "drawdown": results.drawdown_series, "best": True}]
        st.plotly_chart(equity_curve_chart(curves, height=500), width='stretch')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital inicial", f"${results.initial_capital:,.0f}")
        c2.metric("Capital final",   f"${results.final_capital:,.0f}")
        c3.metric("CAGR",            f"{results.cagr:.2f}%")
        c4.metric("Max DD",          f"{results.max_drawdown_pct:.2f}%")

    with tab3:
        if results.trades_df is not None and not results.trades_df.empty:
            st.markdown(f"**{results.total_trades} operaciones** | Win Rate: {results.win_rate:.1f}% | PF: {results.profit_factor:.2f}")
            st.dataframe(results.trades_df, width='stretch', hide_index=True, height=400)
            st.download_button("CSV", results.trades_df.to_csv(index=False),
                               f"trades_{strategy_name}_{asset}.csv", "text/csv")
        else:
            st.info("Sin operaciones.")

    with tab4:
        if results.trades_df is not None and not results.trades_df.empty:
            st.plotly_chart(trades_histogram(results.trades_df, height=400), width='stretch')


def _load_data_range(asset: str, timeframe: str, date_from, date_to):
    return load_historical_data(asset, timeframe, date_from=date_from, date_to=date_to)
