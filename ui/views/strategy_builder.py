"""
ui/views/strategy_builder.py
Constructor de Estrategias — editor de código dentro de la app.

Flujo: escribir/editar código -> Validar -> Backtest rápido (con control
de solo largos / solo cortos / ambos) -> Optimizar parámetros -> Guardar
como archivo real en strategies/, con lo que queda disponible al instante
en Backtesting, Optimización, Walk-Forward, Monte Carlo y Sensibilidad.

Todo corre sobre el mismo BacktestEngine / StrategyOptimizer que usa el
resto de la plataforma — no hay un motor paralelo para las estrategias
creadas aquí.
"""
import streamlit as st
import pandas as pd

from strategies.code_compiler import compile_strategy_code, validate_strategy_class, suggest_class_name
from strategies import save_user_strategy_file, refresh_registry, list_user_strategy_files, STRATEGY_REGISTRY
from ui.views.builder_templates import TEMPLATES
from core.types import BacktestConfig
from engine.backtester import BacktestEngine
from optimization.optimizer import StrategyOptimizer
from optimization.objectives import OBJECTIVES, OBJECTIVES_LABELS
from market_data import get_data
from market_data.mt5_provider import get_mt5_provider
from ui.state import AppState
from ui.components.data_source import render_data_source_selector, load_historical_data
from ui.components.metrics_grid import render_metrics_grid
from visualization.charts import candlestick_chart, equity_curve_chart, indicators_chart

try:
    from streamlit_ace import st_ace
    _HAS_ACE = True
except ImportError:
    _HAS_ACE = False


def render(asset: str, timeframe: str) -> None:
    st.markdown("## Constructor de Estrategias")
    data_source = render_data_source_selector(timeframe, key_prefix="builder_source", compact=True)
    st.caption(f"Fuente histórica activa: **{data_source}**")

    mode = st.radio(
        "Modo",
        ["Visual (combinar indicadores)", "Código (avanzado)"],
        horizontal=True, key="sb_mode",
    )

    if mode == "Visual (combinar indicadores)":
        from ui.views.strategy_builder_visual import render as render_visual
        render_visual(asset, timeframe)
        return

    st.caption(
        "Escribe tu propia estrategia en Python: cualquier combinación de indicadores, "
        "solo largos, solo cortos, o ambos — con SL/TP a tu gusto. Valídala, backtestéala "
        "y optimízala aquí mismo, y guárdala para que aparezca en el resto de la plataforma."
    )

    # ── Plantillas ──
    col_t1, col_t2 = st.columns([3, 1])
    template_name = col_t1.selectbox("Punto de partida", list(TEMPLATES.keys()), key="tpl_select")
    load_template = col_t2.button("Cargar plantilla", width='stretch')

    if load_template or "builder_code" not in st.session_state:
        st.session_state["builder_code"] = TEMPLATES[template_name]

    with st.expander("Cómo funciona el editor", expanded=False):
        st.markdown(
            "- Tu clase debe heredar de `BaseStrategy` (ya está disponible, no la importes).\n"
            "- `pd`, `np` e `Indicators` ya están disponibles en el editor.\n"
            "- `generate_signals(self, data)` debe devolver un DataFrame con una columna "
            "`signal`: `1` = largo, `-1` = corto, `0` = sin posición.\n"
            "- Para **solo largos**, nunca asignes `-1`. Para **solo cortos**, nunca asignes `1`.\n"
            "- `stop_loss` / `take_profit` son opcionales — fíjalos **en la misma fila** que la señal.\n"
            "- Por seguridad, el editor no permite `import`, `open`, `eval`/`exec` ni acceso a `os`/`sys`."
        )
        st.markdown("**Indicadores disponibles vía `Indicators.<nombre>(...)`:**")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(
                "*Tendencia*\n"
                "- sma, ema, wma\n- dema, tema, hma\n- kama, vwma\n- adx, aroon\n"
                "- parabolic_sar\n- supertrend\n- ichimoku\n- trix, vortex\n"
                "- mass_index, dpo\n- linreg_slope/line\n- choppiness_index\n- coppock_curve"
            )
        with c2:
            st.markdown(
                "*Momentum*\n"
                "- rsi, stochastic\n- stoch_rsi\n- macd, ppo\n- cci, williams_r\n"
                "- roc, momentum\n- awesome_oscillator\n- ultimate_oscillator\n"
                "- fisher_transform\n- tsi, kst"
            )
        with c3:
            st.markdown(
                "*Volatilidad*\n"
                "- atr, natr\n- bollinger\n- bollinger_percent_b\n- bollinger_bandwidth\n"
                "- keltner_channels\n- donchian\n- std_dev\n- historical_volatility"
            )
        with c4:
            st.markdown(
                "*Volumen*\n"
                "- volume_sma\n- vwap\n- obv, mfi\n- cmf, ad_line\n"
                "- chaikin_oscillator\n- force_index\n- eom, pvt\n- volume_oscillator"
            )
        with c5:
            st.markdown(
                "*Otros*\n"
                "- pivot_points\n- zscore\n- rolling_correlation"
            )
        st.caption(
            "Ejemplo: `df[\"adx_plus\"], df[\"adx_minus\"], df[\"adx\"] = Indicators.adx(df, 14)` — "
            "los que devuelven varias series (tuplas) se asignan así; los que devuelven una sola, "
            "directo: `df[\"rsi\"] = Indicators.rsi(df[\"close\"], 14)`."
        )

    # ── Editor ──
    if _HAS_ACE:
        code = st_ace(
            value=st.session_state["builder_code"],
            language="python",
            theme="gruvbox",
            key="builder_ace",
            height=420,
            font_size=14,
            tab_size=4,
            show_gutter=True,
            wrap=False,
            auto_update=True,
        )
    else:
        st.info("Instala `streamlit-ace` (ver requirements.txt) para resaltado de sintaxis. Usando editor simple mientras tanto.")
        code = st.text_area(
            "Código de la estrategia", value=st.session_state["builder_code"],
            height=420, key="builder_textarea",
        )
    st.session_state["builder_code"] = code

    # ── Configuración del backtest rápido ──
    with st.expander("Configuración del backtest", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        b_capital = c1.number_input("Capital ($)", 1000, 100_000_000, 100_000, 5000, key="b_capital")
        b_commission = c2.number_input("Comisión (%)", 0.0, 2.0, 0.10, 0.01, key="b_commission") / 100
        b_risk = c3.slider("Riesgo/trade (%)", 0.5, 10.0, 2.0, 0.5, key="b_risk") / 100
        b_direction = c4.selectbox(
            "Dirección", ["Largos y cortos", "Solo largos", "Solo cortos"], key="b_direction",
        )
        allow_long = b_direction in ("Largos y cortos", "Solo largos")
        allow_short = b_direction in ("Largos y cortos", "Solo cortos")
        st.caption(f"Datos de prueba: **{asset} — {timeframe}** (se puede cambiar en la barra lateral).")

    # ── Acciones ──
    col_v, col_b, col_o = st.columns(3)
    validate_clicked = col_v.button("Validar código", width='stretch')
    backtest_clicked = col_b.button("Backtest rápido", width='stretch', type="primary")
    optimize_clicked = col_o.button("Optimizar parámetros", width='stretch')

    df = _load_data(asset, timeframe)

    if validate_clicked:
        _run_validation(code, df)

    if backtest_clicked:
        _run_quick_backtest(code, df, asset, timeframe, b_capital, b_commission, b_risk, allow_long, allow_short)

    if optimize_clicked:
        _run_quick_optimization(code, df, asset, timeframe, b_capital, b_commission, b_risk)

    st.divider()

    tab_bt, tab_opt, tab_save, tab_mine = st.tabs([
        "Resultado del Backtest", "Resultado de la Optimización",
        "Guardar Estrategia", "Mis Estrategias",
    ])

    with tab_bt:
        _show_backtest_tab(asset, timeframe)
    with tab_opt:
        _show_optimization_tab()
    with tab_save:
        _show_save_tab(code)
    with tab_mine:
        _show_saved_strategies_tab()


# ---------------------------------------------------------------------------

def _run_validation(code: str, df: pd.DataFrame) -> None:
    res = compile_strategy_code(code)
    if not res.ok:
        st.error(res.error)
        return
    val = validate_strategy_class(res.strategy_class, df if df is not None else _synthetic_data())
    if not val.ok:
        st.error(val.error)
        return
    space = res.strategy_class().get_param_space()
    st.success(
        f"'{res.strategy_class.name}' es válida. "
        f"{len(space)} parámetro(s) optimizable(s): {', '.join(space.keys()) if space else 'ninguno'}."
    )
    AppState.save_builder_compiled(res.strategy_class)


def _run_quick_backtest(code, df, asset, timeframe, capital, commission, risk, allow_long, allow_short) -> None:
    if df is None or df.empty:
        st.error(f"Sin datos para {asset} {timeframe}. Pulsa Conectar en la barra superior o revisa el activo seleccionado.")
        return

    res = compile_strategy_code(code)
    if not res.ok:
        st.error(res.error)
        return
    val = validate_strategy_class(res.strategy_class, df)
    if not val.ok:
        st.error(val.error)
        return

    AppState.save_builder_compiled(res.strategy_class)

    try:
        with st.spinner("Ejecutando backtest..."):
            strategy = res.strategy_class()
            signals = strategy.generate_signals(df)
            config = BacktestConfig(
                initial_capital=float(capital), commission=commission,
                risk_per_trade=risk, allow_long=allow_long, allow_short=allow_short,
            )
            engine = BacktestEngine(config)
            live_progress = st.progress(0.0, text="Ejecutando backtest del Constructor...")
            live_equity = st.empty()
            live_trades = st.empty()
            def _builder_bt_cb(snapshot):
                live_progress.progress(float(snapshot.get("progress", 0.0)), text=f"Vela {int(snapshot.get('bar_index', 0))+1:,}/{int(snapshot.get('total_bars', 1)):,}")
                eq = snapshot.get("equity_curve")
                if eq is not None and len(eq):
                    from visualization.charts import equity_curve_chart
                    peak = eq.cummax()
                    live_equity.plotly_chart(equity_curve_chart([{
                        "name": res.strategy_class.name, "equity": eq, "drawdown": (eq/peak-1)*100, "best": True
                    }], height=280), width="stretch")
                tr = snapshot.get("trades", []) or []
                if tr:
                    live_trades.dataframe(pd.DataFrame([t.to_dict() for t in tr]), width="stretch", hide_index=True, height=220)
            results = engine.run(signals, strategy_name=res.strategy_class.name, asset=asset, timeframe=timeframe, progress_callback=_builder_bt_cb, progress_every=50)
            live_progress.progress(1.0, text=f"Backtest completado · {results.total_trades:,} operaciones")
        AppState.save_builder_backtest(results)
        st.success(
            f"{results.total_trades} operaciones | Sharpe: {results.sharpe_ratio:.2f} | "
            f"CAGR: {results.cagr:.2f}% | Max DD: {results.max_drawdown_pct:.2f}%"
        )
    except Exception as e:
        import traceback
        st.error(f"Error al ejecutar el backtest: {e}")
        st.code(traceback.format_exc())


def _run_quick_optimization(code, df, asset, timeframe, capital, commission, risk) -> None:
    if df is None or df.empty:
        st.error(f"Sin datos para {asset} {timeframe}.")
        return

    res = compile_strategy_code(code)
    if not res.ok:
        st.error(res.error)
        return
    val = validate_strategy_class(res.strategy_class, df)
    if not val.ok:
        st.error(val.error)
        return

    space = res.strategy_class().get_param_space()
    if not space:
        st.warning(
            "Esta estrategia no define `get_param_space()`, así que no hay nada que optimizar. "
            "Añade el método a tu clase con los rangos de los parámetros que quieras explorar."
        )
        return

    AppState.save_builder_compiled(res.strategy_class)

    col_obj, col_trials = st.columns(2)
    objective = col_obj.selectbox(
        "Objetivo", list(OBJECTIVES.keys()), format_func=lambda x: OBJECTIVES_LABELS[x], key="b_opt_obj",
    )
    n_trials = col_trials.slider("Trials", 20, 300, 60, 10, key="b_opt_trials")

    config = BacktestConfig(initial_capital=float(capital), commission=commission, risk_per_trade=risk)
    optimizer = StrategyOptimizer(
        strategy_class=res.strategy_class, data=df, config=config, asset=asset, timeframe=timeframe,
    )
    progress = st.progress(0, text="Optimizando...")

    def _cb(trial_num, total, best_val):
        progress.progress(trial_num / total, text=f"Trial {trial_num}/{total} — mejor {objective}: {best_val:.4f}")

    with st.spinner("Buscando los mejores parámetros..."):
        best_params = optimizer.optimize(n_trials=n_trials, objective=objective, progress_callback=_cb)
    progress.progress(1.0, text=f"Optimización completada · {n_trials}/{n_trials} trials")

    AppState.save_builder_optimization(optimizer.get_optimization_history(), best_params)
    st.success(f"Optimización completada — mejor {objective}: {optimizer._study.best_value:.4f}")


def _show_backtest_tab(asset, timeframe) -> None:
    results = AppState.get_builder_backtest()
    if results is None:
        st.info("Ejecuta 'Backtest rápido' para ver resultados aquí.")
        return
    render_metrics_grid(results.to_dict(), n_cols=4)
    st.divider()
    strategy_class = AppState.get_builder_compiled()
    df = _load_data(asset, timeframe)
    if df is not None and strategy_class is not None:
        df_viz = df.loc[results.start_date:results.end_date]
        sig_df = strategy_class().generate_signals(df_viz)
        indicators = {c: sig_df[c] for c in strategy_class().get_indicator_columns() if c in sig_df.columns}
        fig = candlestick_chart(
            df=sig_df, title=f"{asset} {timeframe} — {results.strategy_name}",
            signals=sig_df.get("signal"),
            trades_df=results.trades_df if not results.trades_df.empty else None,
            height=520,
        )
        st.plotly_chart(fig, width='stretch')
        if indicators:
            st.plotly_chart(indicators_chart(indicators, height=280), width='stretch')
    curves = [{"name": results.strategy_name, "equity": results.equity_curve,
               "drawdown": results.drawdown_series, "best": True}]
    st.plotly_chart(equity_curve_chart(curves, height=400), width='stretch')
    if results.trades_df is not None and not results.trades_df.empty:
        st.dataframe(results.trades_df, width='stretch', hide_index=True, height=300)


def _show_optimization_tab() -> None:
    opt = AppState.get_builder_optimization()
    if opt["history"] is None:
        st.info("Ejecuta 'Optimizar parámetros' para ver resultados aquí.")
        return
    st.markdown("#### Mejores parámetros encontrados")
    best = opt["best_params"]
    if best:
        cols = st.columns(min(len(best), 5))
        for i, (k, v) in enumerate(best.items()):
            cols[i % len(cols)].metric(k, str(v))
    st.markdown("#### Historial de trials")
    st.dataframe(opt["history"], width='stretch', hide_index=True, height=350)
    st.caption(
        "Copia estos valores como los `default` de tu `__init__` en el editor, "
        "y guarda la estrategia para dejarla fijada con los mejores parámetros."
    )


def _show_save_tab(code: str) -> None:
    strategy_class = AppState.get_builder_compiled()
    if strategy_class is None:
        st.info("Valida el código (botón 'Validar código' o 'Backtest rápido') antes de guardar.")
        return

    st.markdown(f"Estrategia validada: **{strategy_class.name}**")
    default_stem = suggest_class_name(strategy_class.name).lower()
    filename_stem = st.text_input(
        "Nombre de archivo (sin espacios, se guarda en strategies/)",
        value=default_stem, key="b_save_filename",
    )
    if st.button("Guardar estrategia", type="primary", key="b_save_btn"):
        existing = filename_stem.strip().lower().replace(" ", "_")
        target_path = None
        try:
            target_path = save_user_strategy_file(existing, code)
            refresh_registry()
        except Exception as e:
            st.error(f"No se pudo guardar: {e}")
            return
        if strategy_class.name in STRATEGY_REGISTRY:
            st.success(
                f"Guardada en `{target_path.name}`. '{strategy_class.name}' ya está disponible "
                "en Backtesting, Optimización, Walk-Forward, Monte Carlo y Sensibilidad."
            )
        else:
            st.warning(
                f"Se guardó `{target_path.name}`, pero no apareció en el listado de estrategias. "
                "Revisa que el nombre de la clase (`name = ...`) no choque con otra estrategia existente."
            )


def _show_saved_strategies_tab() -> None:
    files = list_user_strategy_files()
    if not files:
        st.info("Todavía no has guardado ninguna estrategia desde el Constructor.")
        return
    st.markdown(f"**{len(files)} estrategia(s) guardada(s) desde el Constructor:**")
    for f in files:
        st.markdown(f"- `strategies/{f}.py`")
    st.caption("Bórralas manualmente del directorio `strategies/` si ya no las necesitas.")


def _load_data(asset: str, timeframe: str):
    return load_historical_data(asset, timeframe)


def _synthetic_data():
    import numpy as np
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.4)
    return pd.DataFrame({
        "open": close, "high": close + 0.2, "low": close - 0.2,
        "close": close, "volume": 100,
    }, index=dates)
