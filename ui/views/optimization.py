"""
ui/views/optimization.py
Página de Optimización Avanzada.

Incluye: Optimización estándar, Walk-Forward, Monte Carlo,
Análisis de Sensibilidad y Ranking institucional.
"""
import streamlit as st
import numpy as np
import pandas as pd

from strategies import STRATEGY_REGISTRY, list_strategies
from core.types import BacktestConfig
from engine.backtester import BacktestEngine
from optimization.optimizer import StrategyOptimizer
from optimization.objectives import OBJECTIVES, OBJECTIVES_LABELS
from optimization.walk_forward import WalkForwardOptimizer
from optimization.cpcv import CPCVEngine
from optimization.monte_carlo import MonteCarloAnalyzer
from optimization.sensitivity import SensitivityAnalyzer
from market_data import get_data
from ui.state import AppState
from ui.components.metrics_grid import render_metrics_grid
from ui.components.cost_toggle import render_cost_toggle
from ui.components.data_source import render_data_source_selector, load_historical_data
from ui.components.regime_filter import render_regime_filter, compute_regime_mask
from visualization.charts import (
    equity_curve_chart, optimization_history_chart,
)
from visualization.optimization_charts import (
    wfo_results_chart, cpcv_results_chart,
    monte_carlo_fan_chart, monte_carlo_histogram,
    sensitivity_1d_chart, sensitivity_2d_surface, sensitivity_2d_heatmap,
    robustness_scatter,
)
from config.settings import TIMEFRAMES, ASSETS


def render(asset: str, timeframe: str) -> None:
    """Renderiza la página completa de Optimización."""

    st.markdown("## Optimización Avanzada")
    data_source = render_data_source_selector(timeframe, key_prefix="opt_source", compact=True)
    st.caption(f"Fuente histórica activa: **{data_source}**")

    tab_std, tab_wfo, tab_cpcv, tab_mc, tab_sens = st.tabs([
        "Optimización Estándar",
        "Walk-Forward",
        "CPCV (Sobreajuste)",
        "Monte Carlo",
        "Sensibilidad",
    ])

    with tab_std:
        _render_standard_optimization(asset, timeframe)

    with tab_wfo:
        _render_walk_forward(asset, timeframe)

    with tab_cpcv:
        _render_cpcv(asset, timeframe)

    with tab_mc:
        _render_monte_carlo()

    with tab_sens:
        _render_sensitivity(asset, timeframe)


# ===========================================================================
# OPTIMIZACIÓN ESTÁNDAR
# ===========================================================================

def _render_standard_optimization(asset: str, timeframe: str) -> None:
    st.markdown("### Optimización Bayesiana")
    st.caption("Búsqueda inteligente de parámetros usando Optuna TPE.")

    col1, col2, col3, col4 = st.columns(4)
    strategy_name   = col1.selectbox("Estrategia", list_strategies(), key="opt_strat")
    objective       = col2.selectbox(
        "Objetivo", list(OBJECTIVES.keys()),
        format_func=lambda x: OBJECTIVES_LABELS[x], key="opt_obj",
    )
    n_trials        = col3.slider("Número de trials", 50, 500, 100, 25, key="opt_trials")
    initial_capital = col4.number_input(
        "Capital ($)", 1000, 10_000_000, 100_000, 5000, key="opt_cap",
    )

    col5, col6 = st.columns(2)
    commission     = col5.number_input("Comisión (%)", 0.0, 1.0, 0.10, 0.01, key="opt_comm") / 100
    risk_per_trade = col6.slider("Riesgo por trade (%)", 0.5, 10.0, 2.0, 0.5, key="opt_risk") / 100

    st.markdown("###### Modelo de costos")
    use_realistic_costs, cost_profile = render_cost_toggle(asset, key_prefix="opt")

    st.markdown("###### Régimen de mercado")
    allowed_regimes = render_regime_filter(asset, timeframe, key_prefix="opt")

    # ── Rango de fechas de optimización ──
    with st.expander("Rango de fechas de optimización", expanded=True):
        from datetime import date as _date, timedelta as _td
        c1, c2, c3 = st.columns(3)
        opt_date_mode = c1.radio(
            "Período", ["Rango personalizado", "Máxima historia"],
            key="opt_date_mode", horizontal=False,
        )
        use_full = (opt_date_mode == "Máxima historia")
        if not use_full:
            default_end   = _date.today()
            default_start = default_end - _td(days=365 * 2)
            opt_start = c2.date_input("Inicio", value=default_start, key="opt_start")
            opt_end   = c2.date_input("Fin",    value=default_end,   key="opt_end")
        else:
            opt_start = opt_end = None
            c2.info("Se usará toda la historia disponible.")
        c3.info(
            "**Consejo:** Para Walk-Forward usa al menos 3 años de datos. "
            "Para optimización estándar, usa el 70% más reciente del histórico disponible."
        )

    if st.button("▶ Iniciar Optimización", type="primary", key="opt_run_btn"):
        _run_standard_opt(
            strategy_name, asset, timeframe, objective, n_trials,
            initial_capital, commission, risk_per_trade,
            opt_start, opt_end, use_full,
            use_realistic_costs, cost_profile, allowed_regimes,
        )

    # Mostrar resultados
    opt_data = AppState.get_optimization()
    if opt_data["history"] is None:
        st.info("Inicia la optimización para ver resultados.")
        return

    _show_opt_results(
        opt_data, strategy_name, asset, timeframe, initial_capital, commission, risk_per_trade,
        use_realistic_costs, cost_profile,
    )


def _run_standard_opt(
    strategy_name, asset, timeframe, objective, n_trials,
    initial_capital, commission, risk_per_trade,
    opt_start=None, opt_end=None, use_full=True,
    use_realistic_costs=False, cost_profile=None, allowed_regimes=None,
):
    from market_data.mt5_provider import history_gap_warning
    df = _load_data(asset, timeframe, opt_start if not use_full else None, opt_end if not use_full else None)
    if df is not None and not use_full and opt_start and opt_end:
        df = df.loc[str(opt_start):str(opt_end)]
    if df is None:
        st.error(f"Sin datos para {asset} {timeframe}")
        return
    if not use_full:
        gap_msg = history_gap_warning(df, opt_start, asset, timeframe)
        if gap_msg:
            st.warning(gap_msg)

    regime_mask = None
    if allowed_regimes:
        regime_mask = compute_regime_mask(df, timeframe, allowed_regimes, asset=asset)
        if regime_mask is not None:
            st.caption(
                f"Filtro de régimen activo: {int(regime_mask.sum())}/{len(df)} velas "
                f"habilitadas para nuevas entradas ({', '.join(allowed_regimes)})."
            )

    config = BacktestConfig(
        initial_capital=float(initial_capital),
        commission=commission,
        risk_per_trade=risk_per_trade,
        use_realistic_costs=bool(use_realistic_costs and cost_profile is not None),
        cost_profile=cost_profile,
    )
    strategy_class = STRATEGY_REGISTRY[strategy_name]

    progress_bar = st.progress(0, text="Iniciando optimización...")
    status_text  = st.empty()
    live_chart = st.empty()
    live_table = st.empty()

    def _cb(trial_num, total, best_val):
        p = trial_num / total
        progress_bar.progress(p, text=f"Trial {trial_num}/{total} | Mejor: {best_val:.4f}")
        status_text.caption(f"Probando combinación {trial_num} de {total}...")
        # Optuna conserva todos los trials ya completados. Los mostramos
        # mientras avanza la búsqueda, no únicamente después de study.optimize.
        try:
            hist = optimizer.get_optimization_history() if 'optimizer' in locals() else None
            if hist is not None and not hist.empty:
                top_live = hist.nlargest(min(10, len(hist)), "value").reset_index(drop=True)
                live_table.dataframe(top_live, width="stretch", hide_index=True)
                live_chart.plotly_chart(optimization_history_chart(hist, height=300), width="stretch")
        except Exception:
            pass

    optimizer = StrategyOptimizer(
        strategy_class=strategy_class,
        data=df, config=config,
        asset=asset, timeframe=timeframe,
        regime_mask=regime_mask,
    )

    best_params = optimizer.optimize(
        n_trials=n_trials,
        objective=objective,
        progress_callback=_cb,
    )

    progress_bar.progress(1.0, text=f"Optimización completada · {n_trials}/{n_trials} trials")
    status_text.success("Optimización terminada. El historial permanece visible debajo.")

    history_df = optimizer.get_optimization_history()
    AppState.save_optimization(history_df, best_params, strategy_name, asset, timeframe, allowed_regimes)

    # Nunca sustituimos la estrategia original: publicamos una variante independiente.
    optimized_variant_name = None
    try:
        from strategies import save_optimized_variant, refresh_registry
        optimized_variant_name = save_optimized_variant(
            strategy_name,
            best_params,
            objective=objective,
            asset=asset,
            timeframe=timeframe,
        )
        refresh_registry()
        st.success(
            f"Variante creada: **{optimized_variant_name}**. "
            "La original permanece intacta y ambas aparecen por separado en Backtesting."
        )
    except Exception as exc:
        st.warning(f"La optimización terminó, pero no se pudo publicar la variante optimizada: {exc}")

    st.success(
        f"Optimización completada — Mejor {objective}: "
        f"{optimizer._study.best_value:.4f}"
    )


def _show_opt_results(opt_data, strategy_name, asset, timeframe, initial_capital, commission, risk_per_trade,
                       use_realistic_costs=False, cost_profile=None):
    best_params = opt_data["best_params"]
    history_df  = opt_data["history"]

    st.markdown("#### Mejores parámetros")
    if best_params:
        param_cols = st.columns(min(len(best_params), 5))
        for i, (k, v) in enumerate(best_params.items()):
            param_cols[i % len(param_cols)].metric(k, str(v))

    optimized_name = f"{strategy_name} Optimizada"
    st.info(
        f"Catálogo: **{strategy_name}** = original · **{optimized_name}** = variante con los mejores parámetros. "
        "Puedes seleccionar cualquiera de las dos en Backtesting."
    )

    # Backtest con los mejores parámetros
    if best_params and st.button("▶ Backtest con mejores parámetros", key="opt_backtest_btn"):
        df = _load_data(asset, timeframe)
        if df is not None:
            config = BacktestConfig(
                initial_capital=float(initial_capital),
                commission=commission,
                risk_per_trade=risk_per_trade,
                use_realistic_costs=bool(use_realistic_costs and cost_profile is not None),
                cost_profile=cost_profile,
            )
            strategy_class = STRATEGY_REGISTRY.get(
                opt_data.get("strategy", strategy_name)
            )
            if strategy_class:
                strategy = strategy_class(**best_params)
                signals = strategy.generate_signals(df)
                allowed_regimes = opt_data.get("allowed_regimes") or []
                if allowed_regimes:
                    mask = compute_regime_mask(df, timeframe, allowed_regimes, asset=asset)
                    if mask is not None:
                        signals = signals.copy()
                        signals.loc[~mask.reindex(signals.index).fillna(False), "signal"] = 0
                engine = BacktestEngine(config)
                results = engine.run(signals, strategy_name=strategy_name, asset=asset, timeframe=timeframe)
                st.markdown("#### Métricas del mejor backtest")
                render_metrics_grid(results.to_dict(), n_cols=4)
                curves = [{"name": f"{strategy_name} (opt)", "equity": results.equity_curve,
                            "drawdown": results.drawdown_series, "best": True}]
                fig_eq = equity_curve_chart(curves, height=400)
                st.plotly_chart(fig_eq, width='stretch')

    # Historial de optimización
    if history_df is not None and not history_df.empty:
        st.markdown("#### Evolución de la optimización")
        fig_hist = optimization_history_chart(history_df, height=300)
        st.plotly_chart(fig_hist, width='stretch')

        # Robustness scatter
        st.markdown("#### Dispersión de resultados")
        fig_scatter = robustness_scatter(history_df, height=350)
        st.plotly_chart(fig_scatter, width='stretch')

        # TOP 10 tabla
        st.markdown("#### TOP 10 configuraciones")
        top10 = history_df.nlargest(10, "value")
        st.dataframe(top10.reset_index(drop=True), width='stretch', hide_index=True)


# ===========================================================================
# WALK-FORWARD
# ===========================================================================

def _render_walk_forward(asset: str, timeframe: str) -> None:
    st.markdown("### Walk-Forward Optimization")
    st.caption(
        "Valida la estrategia en períodos out-of-sample para detectar overfitting. "
        "Un **Efficiency Ratio > 0.7** indica estrategia robusta."
    )

    col1, col2, col3, col4 = st.columns(4)
    strategy_name = col1.selectbox("Estrategia", list_strategies(), key="wfo_strat")
    n_windows     = col2.slider("Ventanas", 3, 10, 5, key="wfo_windows")
    train_pct     = col3.slider("% Entrenamiento", 50, 85, 70, key="wfo_train") / 100
    n_trials      = col4.slider("Trials por ventana", 20, 150, 50, key="wfo_trials")

    objective = st.selectbox(
        "Objetivo de optimización",
        list(OBJECTIVES.keys()),
        format_func=lambda x: OBJECTIVES_LABELS[x],
        key="wfo_obj",
    )

    st.markdown("###### Modelo de costos")
    wfo_use_realistic_costs, wfo_cost_profile = render_cost_toggle(asset, key_prefix="wfo")

    if st.button("▶ Ejecutar Walk-Forward", type="primary", key="wfo_run_btn"):
        df = _load_data(asset, timeframe)
        if df is None:
            st.error(f"Sin datos para {asset} {timeframe}")
            return

        strategy_class = STRATEGY_REGISTRY[strategy_name]
        config = BacktestConfig(
            initial_capital=100_000.0,
            use_realistic_costs=bool(wfo_use_realistic_costs and wfo_cost_profile is not None),
            cost_profile=wfo_cost_profile,
        )

        progress_bar = st.progress(0, text="Iniciando Walk-Forward...")
        live_wfo_table = st.empty()
        live_wfo_chart = st.empty()

        def _wfo_cb(window_id, total, window=None, partial_results=None):
            progress_bar.progress(window_id / total, text=f"Ventana {window_id}/{total}...")
            if partial_results is not None and partial_results.windows:
                rows = []
                curves = []
                for w in partial_results.windows:
                    rows.append({"Ventana": w.window_id, "Estado": "Completada" if w.test_results is not None else "Omitida/Error", "Train": getattr(w, "train_score", None), "OOS": getattr(w, "test_score", None)})
                    if getattr(w, "test_results", None) is not None:
                        curves.append({"name": f"W{w.window_id} OOS", "equity": w.test_results.equity_curve, "drawdown": w.test_results.drawdown_series, "best": False})
                live_wfo_table.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                if curves:
                    live_wfo_chart.plotly_chart(equity_curve_chart(curves, height=330), width="stretch")

        wfo = WalkForwardOptimizer(
            strategy_class=strategy_class,
            data=df, config=config,
            asset=asset, timeframe=timeframe,
            train_pct=train_pct,
            n_windows=n_windows,
            n_trials=n_trials,
        )
        wfo_results = wfo.run(objective=objective, progress_callback=_wfo_cb)

        progress_bar.progress(1.0, text=f"Walk-Forward completado · {n_windows}/{n_windows} ventanas")
        AppState.save_wfo(wfo_results)
        st.success(
            f"Walk-Forward completado | "
            f"Efficiency Ratio: {wfo_results.efficiency_ratio:.2f} | "
            f"{wfo_results.robustness_label()}"
        )
        st.rerun()

    wfo_results = AppState.get_wfo()
    if wfo_results is None:
        st.info("Ejecuta el Walk-Forward para ver resultados.")
        return

    # Métricas de robustez
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Efficiency Ratio", f"{wfo_results.efficiency_ratio:.2f}")
    col2.metric("Avg Train Score",  f"{wfo_results.avg_train_score:.2f}")
    col3.metric("Avg Test Score",   f"{wfo_results.avg_test_score:.2f}")
    col4.metric("Robustez", wfo_results.robustness_label())

    # Gráfico WFO
    fig_wfo = wfo_results_chart(wfo_results, height=520)
    st.plotly_chart(fig_wfo, width='stretch')

    # Estabilidad de parámetros
    if wfo_results.parameter_stability:
        st.markdown("#### Estabilidad de Parámetros (CV — menor = más estable)")
        stability_df = pd.DataFrame([
            {"Parámetro": k, "Coef. Variación": f"{v:.4f}",
             "Estabilidad": "Alta" if v < 0.2 else ("Media" if v < 0.5 else "Baja")}
            for k, v in wfo_results.parameter_stability.items()
        ])
        st.dataframe(stability_df, width='stretch', hide_index=True)

    # Detalles por ventana
    if wfo_results.windows:
        st.markdown("#### Detalle por Ventana")
        rows = []
        for w in wfo_results.windows:
            row = {
                "Ventana": w.window_id,
                "Train inicio": str(w.train_start.date()),
                "Train fin":    str(w.train_end.date()),
                "Test inicio":  str(w.test_start.date()),
                "Test fin":     str(w.test_end.date()),
                "Train Score":  f"{w.train_score:.3f}",
                "Test Score":   f"{w.test_score:.3f}",
            }
            row.update({f"Param: {k}": v for k, v in w.best_params.items()})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

# ===========================================================================
# CPCV — Combinatorial Purged Cross-Validation + Probabilidad de Sobreajuste
# ===========================================================================

def _render_cpcv(asset: str, timeframe: str) -> None:
    st.markdown("### CPCV — Validación Cruzada Purgada Combinatoria")
    st.caption(
        "Complementa al Walk-Forward: evalúa todas las combinaciones posibles de grupos de "
        "test (con purga + embargo) sobre un mismo pool de candidatos, y calcula la "
        "**Probabilidad de Sobreajuste de Backtest (PBO)** — la fracción de particiones donde "
        "el mejor candidato in-sample quedó por debajo de la mediana out-of-sample."
    )

    col1, col2, col3, col4 = st.columns(4)
    strategy_name = col1.selectbox("Estrategia", list_strategies(), key="cpcv_strat")
    n_groups = col2.slider("Grupos (N)", 4, 10, 6, key="cpcv_groups",
                             help="Cuántos bloques contiguos se arman con el histórico.")
    n_test_groups = col3.slider("Grupos de test por partición (k)", 1, 3, 2, key="cpcv_test_groups")
    n_candidates = col4.slider("Candidatos evaluados", 5, 50, 20, key="cpcv_candidates",
                                 help="Pool FIJO de configuraciones de parámetros, muestreadas al "
                                      "azar del espacio de la estrategia — el MISMO pool se evalúa "
                                      "en todas las particiones, para que el PBO sea comparable.")

    from math import comb
    n_splits_preview = comb(n_groups, n_test_groups) if n_test_groups <= n_groups else 0
    st.caption(f"→ {n_splits_preview} particiones combinatorias × {n_candidates} candidatos = "
               f"{n_splits_preview * n_candidates * 2} backtests a correr.")

    objective = st.selectbox(
        "Objetivo de optimización", list(OBJECTIVES.keys()),
        format_func=lambda x: OBJECTIVES_LABELS[x], key="cpcv_obj",
    )

    st.markdown("###### Modelo de costos")
    cpcv_use_realistic_costs, cpcv_cost_profile = render_cost_toggle(asset, key_prefix="cpcv")

    if st.button("▶ Ejecutar CPCV", type="primary", key="cpcv_run_btn"):
        if n_splits_preview > 45:
            st.warning(
                f"{n_splits_preview} particiones es bastante — puede tardar varios minutos. "
                "Considera bajar N o k si esto es solo una prueba rápida."
            )
        df = _load_data(asset, timeframe)
        if df is None:
            st.error(f"Sin datos para {asset} {timeframe}")
            return

        strategy_class = STRATEGY_REGISTRY[strategy_name]
        config = BacktestConfig(
            initial_capital=100_000.0,
            use_realistic_costs=bool(cpcv_use_realistic_costs and cpcv_cost_profile is not None),
            cost_profile=cpcv_cost_profile,
        )

        progress_bar = st.progress(0, text="Iniciando CPCV...")
        live_cpcv_table = st.empty()

        def _cpcv_cb(split_id, total, partial_results=None):
            progress_bar.progress(split_id / total, text=f"Partición {split_id}/{total}...")
            if partial_results is not None and partial_results.per_result:
                rows = [{"Partición": r.split_id, "Candidato": r.candidate_id, "IS": r.is_score, "OOS": r.oos_score} for r in partial_results.per_result]
                live_cpcv_table.dataframe(pd.DataFrame(rows).tail(150), width="stretch", hide_index=True, height=320)

        engine = CPCVEngine(
            strategy_class=strategy_class, data=df, config=config,
            asset=asset, timeframe=timeframe,
            n_groups=n_groups, n_test_groups=n_test_groups,
        )
        try:
            cpcv_results = engine.run(
                objective=objective, n_candidates=n_candidates,
                progress_callback=_cpcv_cb,
            )
        except ValueError as e:
            progress_bar.progress(1.0, text="CPCV detenido por error")
            st.error(str(e))
            return

        progress_bar.progress(1.0, text="CPCV completado")
        AppState.save_cpcv(cpcv_results)
        st.success(f"CPCV completado | {cpcv_results.summary()}")
        st.rerun()

    cpcv_results = AppState.get_cpcv()
    if cpcv_results is None:
        st.info("Ejecuta CPCV para ver resultados.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PBO", f"{cpcv_results.pbo*100:.1f}%")
    col2.metric("Veredicto", cpcv_results.verdict())
    col3.metric("Particiones", cpcv_results.n_splits)
    col4.metric("Candidatos", cpcv_results.n_candidates)

    fig_cpcv = cpcv_results_chart(cpcv_results, height=460)
    st.plotly_chart(fig_cpcv, width='stretch')

    if cpcv_results.best_candidate_by_is and cpcv_results.best_candidate_by_oos:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Mejor candidato por score IN-SAMPLE")
            bis = cpcv_results.best_candidate_by_is
            st.json(bis["params"])
            st.caption(f"IS promedio: {bis['mean_is_score']:.3f} · "
                       f"OOS promedio: {bis['mean_oos_score']:.3f}" if bis["mean_oos_score"] is not None
                       else f"IS promedio: {bis['mean_is_score']:.3f}")
        with c2:
            st.markdown("#### Mejor candidato por score OUT-OF-SAMPLE")
            bos = cpcv_results.best_candidate_by_oos
            st.json(bos["params"])
            st.caption(f"OOS promedio: {bos['mean_oos_score']:.3f}" +
                       (f" · IS promedio: {bos['mean_is_score']:.3f}" if bos["mean_is_score"] is not None else ""))
        if bis["params"] != bos["params"]:
            st.warning(
                "El candidato que mejor rindió in-sample **no** es el que mejor generaliza "
                "out-of-sample — señal de que optimizar solo con el histórico completo (como hace "
                "la pestaña 'Optimización Estándar') puede estar sobreajustando. Considera usar los "
                "parámetros de la columna OOS para operar."
            )
        else:
            st.success("El candidato que mejor rindió in-sample también es el mejor out-of-sample — "
                       "buena señal de que no hay sobreajuste severo en este espacio de búsqueda.")



def _render_monte_carlo() -> None:
    st.markdown("### Análisis Monte Carlo")
    st.caption(
        "Simula miles de trayectorias posibles remuestreando la distribución "
        "empírica de retornos del backtest. Visualiza el rango de resultados posibles."
    )

    # Verificar que haya un backtest disponible
    bt_keys = AppState.list_backtests()
    if not bt_keys:
        st.warning("Necesitas ejecutar al menos un backtest antes del análisis Monte Carlo. "
                   "Ve a la página de Backtesting.")
        return

    col1, col2 = st.columns(2)
    selected_bt = col1.selectbox("Backtest base", bt_keys, key="mc_bt_select")
    n_simulations = col2.select_slider(
        "Número de simulaciones",
        options=[500, 1000, 2000, 5000, 10000],
        value=5000,
        key="mc_n_sims",
    )

    if st.button("▶ Ejecutar Monte Carlo", type="primary", key="mc_run_btn"):
        results = AppState.get_backtest(selected_bt)
        if results is None:
            st.error("No se encontró el backtest seleccionado.")
            return

        st.markdown("### Resultados Monte Carlo en vivo")
        mc_progress = st.progress(0, text="Iniciando simulaciones...")
        mc_live_metrics = st.empty()
        mc_live_fan = st.empty()
        mc_live_hist = st.empty()

        def _mc_cb(done, total, partial):
            mc_progress.progress(done / max(total, 1), text=f"Simulaciones {done:,}/{total:,}")
            mc_live_metrics.metric("P50 / Prob. ganancia / DD medio", f"${partial.p50:,.0f} / {partial.prob_profit:.1f}% / {partial.avg_max_drawdown:.2f}%")
            mc_live_fan.plotly_chart(monte_carlo_fan_chart(partial, height=360), width="stretch")
            mc_live_hist.plotly_chart(monte_carlo_histogram(partial, height=280), width="stretch")

        mc = MonteCarloAnalyzer(results, n_simulations=int(n_simulations))
        mc_results = mc.run(progress_callback=_mc_cb, chunk_size=500)
        mc_progress.progress(1.0, text=f"Monte Carlo completado · {int(n_simulations):,}/{int(n_simulations):,} simulaciones")

        AppState.save_mc(mc_results)
        st.success(
            f"Monte Carlo completado | "
            f"Prob. ganancia: {mc_results.prob_profit:.1f}% | "
            f"Mediana: ${mc_results.p50:,.0f}"
        )
        st.rerun()

    mc_results = AppState.get_mc()
    if mc_results is None:
        st.info("Ejecuta el análisis Monte Carlo para ver resultados.")
        return

    # Resumen
    render_metrics_grid(mc_results.summary(), n_cols=4)

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_fan = monte_carlo_fan_chart(mc_results, height=480)
        st.plotly_chart(fig_fan, width='stretch')
    with col2:
        fig_hist = monte_carlo_histogram(mc_results, height=280)
        st.plotly_chart(fig_hist, width='stretch')

        st.markdown("**Percentiles de capital final:**")
        for label, val in [("P5", mc_results.p5), ("P25", mc_results.p25),
                            ("P50", mc_results.p50), ("P75", mc_results.p75),
                            ("P95", mc_results.p95)]:
            delta = (val / mc_results.initial_capital - 1) * 100
            st.metric(label, f"${val:,.0f}", f"{delta:+.1f}%")


# ===========================================================================
# SENSIBILIDAD
# ===========================================================================

def _render_sensitivity(asset: str, timeframe: str) -> None:
    st.markdown("### Análisis de Sensibilidad")
    st.caption(
        "Mapas 2D y 3D que muestran cómo varía el rendimiento al cambiar los parámetros. "
        "Identifica zonas de robustez vs alta sensibilidad."
    )

    col1, col2 = st.columns(2)
    strategy_name = col1.selectbox("Estrategia", list_strategies(), key="sens_strat")
    objective     = col2.selectbox(
        "Objetivo",
        list(OBJECTIVES.keys()),
        format_func=lambda x: OBJECTIVES_LABELS[x],
        key="sens_obj",
    )

    # Verificar si hay parámetros optimizados disponibles
    opt_data = AppState.get_optimization()
    base_params = opt_data.get("best_params", {}) or {}
    strategy_class = STRATEGY_REGISTRY.get(strategy_name)

    if strategy_class and not base_params:
        # Usar parámetros por defecto
        instance = strategy_class()
        space = instance.get_param_space()
        base_params = {k: (spec[1] + spec[2]) // 2 if spec[0] == "int"
                       else (spec[1] + spec[2]) / 2
                       for k, spec in space.items()}

    if not strategy_class:
        st.warning("Estrategia no encontrada.")
        return

    param_space = strategy_class().get_param_space()
    if not param_space:
        st.warning("Esta estrategia no tiene parámetros optimizables.")
        return

    param_names = list(param_space.keys())

    st.markdown("#### Parámetros a analizar")
    mode = st.radio("Modo", ["1D (un parámetro)", "2D (dos parámetros)"],
                    horizontal=True, key="sens_mode")

    col1, col2 = st.columns(2)
    param1 = col1.selectbox("Parámetro 1", param_names, key="sens_p1")
    param2 = col2.selectbox("Parámetro 2", param_names,
                             index=min(1, len(param_names) - 1),
                             key="sens_p2") if mode == "2D (dos parámetros)" else None

    # Rango del parámetro 1
    spec1 = param_space[param1]
    n_points = st.slider("Puntos de muestreo", 5, 20, 10, key="sens_npts")
    if spec1[0] == "int":
        values1 = list(range(spec1[1], spec1[2] + 1, max(1, (spec1[2] - spec1[1]) // n_points)))
    else:
        values1 = list(np.linspace(spec1[1], spec1[2], n_points))

    values2 = None
    if param2 and mode == "2D (dos parámetros)":
        spec2 = param_space[param2]
        if spec2[0] == "int":
            values2 = list(range(spec2[1], spec2[2] + 1, max(1, (spec2[2] - spec2[1]) // n_points)))
        else:
            values2 = list(np.linspace(spec2[1], spec2[2], n_points))

    total_runs = len(values1) * (len(values2) if values2 else 1)
    st.caption(f"Se ejecutarán {total_runs} backtests para el análisis.")

    st.markdown("###### Modelo de costos")
    sens_use_realistic_costs, sens_cost_profile = render_cost_toggle(asset, key_prefix="sens")

    if st.button("▶ Analizar Sensibilidad", type="primary", key="sens_run_btn"):
        df = _load_data(asset, timeframe)
        if df is None:
            st.error(f"Sin datos para {asset} {timeframe}")
            return

        config = BacktestConfig(
            initial_capital=100_000.0,
            use_realistic_costs=bool(sens_use_realistic_costs and sens_cost_profile is not None),
            cost_profile=sens_cost_profile,
        )
        sa = SensitivityAnalyzer(
            strategy_class=strategy_class,
            data=df, config=config,
            asset=asset, timeframe=timeframe,
        )

        sens_progress = st.progress(0, text="Iniciando sensibilidad...")
        sens_live = st.empty()
        sens_live_chart = st.empty()

        def _sens_cb(done, total, snapshot):
            sens_progress.progress(done / max(total, 1), text=f"Combinaciones {done}/{total}")
            if mode == "1D (un parámetro)":
                vals = list(values1[:len(snapshot.get("scores", []))])
                sdf = pd.DataFrame({param1: vals, "score": snapshot.get("scores", [])})
                sens_live.dataframe(sdf, width="stretch", hide_index=True)
                if len(sdf) >= 2:
                    from types import SimpleNamespace
                    live_result = SimpleNamespace(scores=np.asarray(sdf["score"]), param1_values=vals, param1_name=param1, objective=objective)
                    sens_live_chart.plotly_chart(sensitivity_1d_chart(live_result, height=300), width="stretch")
            else:
                matrix = snapshot.get("scores")
                if matrix is not None:
                    sens_live.dataframe(pd.DataFrame(matrix, index=values1, columns=values2), width="stretch")

        if mode == "1D (un parámetro)":
            result = sa.analyze_1d(base_params=base_params, param=param1, values=values1, objective=objective, progress_callback=_sens_cb)
        else:
            result = sa.analyze_2d(base_params=base_params, param1=param1, values1=values1, param2=param2, values2=values2, objective=objective, progress_callback=_sens_cb)
        sens_progress.progress(1.0, text=f"Sensibilidad completada · {total_runs}/{total_runs} combinaciones")

        AppState.save_sensitivity(result)
        st.success(f"Análisis completado | Mejor score: {result.best_score:.4f}")
        st.rerun()

    sens_result = AppState.get_sensitivity()
    if sens_result is None:
        st.info("Ejecuta el análisis para ver resultados.")
        return

    # Mostrar gráficos
    if sens_result.scores.ndim == 1:
        fig = sensitivity_1d_chart(sens_result, height=400)
        st.plotly_chart(fig, width='stretch')
    else:
        tab_3d, tab_heat = st.tabs(["Superficie 3D", "Heatmap 2D"])
        with tab_3d:
            fig_3d = sensitivity_2d_surface(sens_result, height=550)
            st.plotly_chart(fig_3d, width='stretch')
        with tab_heat:
            fig_heat = sensitivity_2d_heatmap(sens_result, height=450)
            st.plotly_chart(fig_heat, width='stretch')

    st.info(
        f"**Mejores parámetros encontrados:** "
        f"{sens_result.best_params}  {objective}: {sens_result.best_score:.4f}"
    )


def _load_data(asset: str, timeframe: str, date_from=None, date_to=None):
    return load_historical_data(asset, timeframe, date_from=date_from, date_to=date_to)
