"""
ui/views/strategy_builder_visual.py
Constructor Visual de Estrategias — combina indicadores por clic, sin
escribir código.

Cada estrategia es una regla en Forma Normal Disyuntiva (DNF):
    Señal = (condición Y condición Y ...)  O  (condición Y condición Y ...)  O ...

Las condiciones disponibles cubren ~140 combinaciones de indicador+nivel
repartidas en más de una decena de familias (RSI, MACD, Medias Móviles,
Bollinger, ADX, Estocástico, SuperTrend, Ichimoku, Donchian, Keltner,
CCI, Williams %R, Volumen, Volatilidad, Patrones de vela...), así que
cualquier combinación entre familias distintas es posible.

Reutiliza exactamente el mismo generador de código que usa el Descubridor
Genético (discovery/codegen.py) para producir una clase de estrategia real
compatible con el resto de la plataforma — la regla que arma el usuario
aquí y la que descubre el algoritmo genético terminan en el mismo tipo de
archivo en strategies/.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from discovery.rule_engine import CONDITION_LIBRARY, rule_to_text
from discovery.codegen import generate_strategy_code, compile_definition_to_class
from strategies import save_user_strategy_file, refresh_registry, STRATEGY_REGISTRY, list_user_strategy_files
from strategies.code_compiler import suggest_class_name
from core.types import BacktestConfig
from engine.backtester import BacktestEngine
from ui.state import AppState
from ui.components.data_source import render_data_source_selector, load_historical_data
from ui.components.metrics_grid import render_metrics_grid
from visualization.charts import candlestick_chart, equity_curve_chart, indicators_chart

# La familia "Regime" requiere calibración por activo (ver regime/calibration.py)
# y ya tiene su propio filtro dedicado en Backtesting/Portfolio — se excluye
# aquí para no ofrecer una condición que nunca dispara sin esa calibración.
_EXCLUDED_FAMILIES = {"Regime"}

_FAMILIES = sorted({c.family for c in CONDITION_LIBRARY.values() if c.family not in _EXCLUDED_FAMILIES})

_FAMILY_LABELS = {
    "MA": "Medias Móviles", "RSI": "RSI", "MACD": "MACD", "Bollinger": "Bandas de Bollinger",
    "ADX": "ADX / Fuerza de Tendencia", "Stochastic": "Estocástico", "CCI": "CCI",
    "WilliamsR": "Williams %R", "Donchian": "Canales de Donchian", "Keltner": "Canales de Keltner",
    "SuperTrend": "SuperTrend", "Ichimoku": "Ichimoku", "ParabolicSAR": "Parabolic SAR",
    "Momentum": "Momentum / ROC", "ZScore": "Z-Score", "VolRegime": "Régimen de Volatilidad",
    "Volume": "Volumen", "Candlestick": "Patrones de Vela",
}


def _init_state():
    st.session_state.setdefault("vb_long_clauses", [])
    st.session_state.setdefault("vb_short_clauses", [])


def _clause_editor(direction_key: str, label: str) -> None:
    st.markdown(f"**Condiciones para {label}**")
    c1, c2 = st.columns([1.3, 3])
    family = c1.selectbox(
        "Familia de indicador", _FAMILIES,
        format_func=lambda f: _FAMILY_LABELS.get(f, f),
        key=f"{direction_key}_family",
    )
    options = [cid for cid, spec in CONDITION_LIBRARY.items() if spec.family == family]
    chosen = c2.multiselect(
        "Condiciones (se combinan con Y dentro de este bloque)",
        options,
        format_func=lambda cid: CONDITION_LIBRARY[cid].description,
        key=f"{direction_key}_multiselect_{family}",
    )
    if st.button(f"+ Añadir bloque ({label})", key=f"{direction_key}_add_btn"):
        if chosen:
            st.session_state[f"vb_{direction_key}_clauses"].append(list(chosen))
            st.rerun()
        else:
            st.warning("Elige al menos una condición antes de añadir el bloque.")

    clauses = st.session_state[f"vb_{direction_key}_clauses"]
    if clauses:
        for i, clause in enumerate(clauses):
            row = st.columns([10, 1])
            desc = " Y ".join(CONDITION_LIBRARY[cid].description for cid in clause)
            row[0].markdown(f"**Bloque {i+1}:** {desc}")
            if row[1].button("✕", key=f"{direction_key}_remove_{i}"):
                clauses.pop(i)
                st.rerun()
        st.caption("La señal se activa cuando se cumple **cualquiera** de los bloques anteriores (O).")
    else:
        st.caption("Sin bloques todavía — añade al menos uno para que esta dirección opere.")


def render(asset: str, timeframe: str) -> None:
    _init_state()
    data_source = render_data_source_selector(timeframe, key_prefix="builder_visual_source", compact=True)
    st.caption(f"Fuente histórica activa: **{data_source}**")
    st.markdown("### Combinar indicadores")
    st.caption(
        "Arma tu estrategia combinando cualquier indicador con cualquier otro: cada **bloque** "
        "combina condiciones con Y (todas deben cumplirse a la vez); varios bloques se combinan "
        "entre sí con O (basta con que se cumpla uno). Define bloques para largos, para cortos, "
        "o ambos."
    )

    tab_long, tab_short = st.tabs(["📈 Condiciones de Largo", "📉 Condiciones de Corto"])
    with tab_long:
        _clause_editor("long", "Largo")
    with tab_short:
        _clause_editor("short", "Corto")

    long_rule = st.session_state["vb_long_clauses"]
    short_rule = st.session_state["vb_short_clauses"]

    if long_rule or short_rule:
        with st.expander("Regla completa en texto", expanded=False):
            if long_rule:
                st.markdown(f"**Largo:** {rule_to_text(long_rule)}")
            if short_rule:
                st.markdown(f"**Corto:** {rule_to_text(short_rule)}")

    with st.expander("Gestión de riesgo (Stop Loss / Take Profit por ATR)", expanded=True):
        rc1, rc2, rc3 = st.columns(3)
        sl_atr = rc1.slider("Stop Loss (múltiplo de ATR)", 0.5, 6.0, 2.0, 0.25, key="vb_sl_atr")
        tp_atr = rc2.slider("Take Profit (múltiplo de ATR)", 0.5, 10.0, 3.0, 0.25, key="vb_tp_atr")
        atr_period = rc3.number_input("Periodo de ATR", 5, 50, 14, 1, key="vb_atr_period")

    with st.expander("Configuración del backtest", expanded=True):
        c1, c2, c3 = st.columns(3)
        capital = c1.number_input("Capital ($)", 1000, 100_000_000, 100_000, 5000, key="vb_capital")
        commission = c2.number_input("Comisión (%)", 0.0, 2.0, 0.10, 0.01, key="vb_commission") / 100
        risk = c3.slider("Riesgo por operación (%)", 0.5, 10.0, 2.0, 0.5, key="vb_risk") / 100
        st.caption(f"Datos de prueba: **{asset} — {timeframe}** (se puede cambiar en la barra superior).")

    if not long_rule and not short_rule:
        st.info("Añade al menos un bloque de condiciones (largo o corto) para poder backtestear.")
        return

    definition = {
        "long_rule": long_rule,
        "short_rule": short_rule,
        "long_rule_text": rule_to_text(long_rule) if long_rule else "(sin regla de largo)",
        "short_rule_text": rule_to_text(short_rule) if short_rule else "(sin regla de corto)",
        "families_used": sorted({CONDITION_LIBRARY[cid].family
                                  for clause in (long_rule + short_rule) for cid in clause}),
        "stop_loss_atr": float(sl_atr),
        "take_profit_atr": float(tp_atr),
        "atr_period": int(atr_period),
    }

    col_bt, col_save = st.columns(2)
    backtest_clicked = col_bt.button("▶ Backtest rápido", type="primary", width='stretch', key="vb_backtest_btn")
    class_name_hint = st.session_state.get("vb_class_name", "MiEstrategiaVisual")

    if backtest_clicked:
        df = _load_data(asset, timeframe)
        if df is None or df.empty:
            st.error(f"Sin datos para {asset} {timeframe}. Verifica el activo/temporalidad seleccionados arriba.")
        else:
            try:
                with st.spinner("Ejecutando backtest..."):
                    strategy_class = compile_definition_to_class(
                        definition, class_name="EstrategiaVisualPreview", symbol=asset, timeframe=timeframe,
                    )
                    strategy = strategy_class()
                    signals = strategy.generate_signals(df)
                    config = BacktestConfig(
                        initial_capital=float(capital), commission=commission, risk_per_trade=risk,
                        allow_long=bool(long_rule), allow_short=bool(short_rule),
                    )
                    engine = BacktestEngine(config)
                    results = engine.run(signals, strategy_name="Estrategia Visual", asset=asset, timeframe=timeframe)
                AppState.save_builder_compiled(strategy_class)
                st.session_state["vb_last_results"] = results
                st.session_state["vb_last_df"] = df
                st.session_state["vb_last_sig_df"] = signals
                st.session_state["vb_last_indicator_cols"] = strategy.get_indicator_columns()
                st.success(
                    f"{results.total_trades} operaciones | Sharpe: {results.sharpe_ratio:.2f} | "
                    f"CAGR: {results.cagr:.2f}% | Max DD: {results.max_drawdown_pct:.2f}%"
                )
            except Exception as e:
                import traceback
                st.error(f"Error al ejecutar el backtest: {e}")
                st.code(traceback.format_exc())

    results = st.session_state.get("vb_last_results")
    if results is not None:
        st.divider()
        st.markdown("#### Resultado")
        render_metrics_grid(results.to_dict(), n_cols=4)
        df_prev = st.session_state.get("vb_last_df")
        if df_prev is not None:
            df_viz = df_prev.loc[results.start_date:results.end_date]
            fig = candlestick_chart(
                df=df_viz, title=f"{asset} {timeframe} — Estrategia Visual",
                signals=None, trades_df=results.trades_df if not results.trades_df.empty else None,
                height=460,
            )
            st.plotly_chart(fig, width='stretch')

            sig_df = st.session_state.get("vb_last_sig_df")
            ind_cols = st.session_state.get("vb_last_indicator_cols") or []
            if sig_df is not None:
                indicators = {c: sig_df[c] for c in ind_cols if c in sig_df.columns}
                if indicators:
                    st.plotly_chart(indicators_chart(indicators, height=260), width='stretch')
        curves = [{"name": "Estrategia Visual", "equity": results.equity_curve,
                   "drawdown": results.drawdown_series, "best": True}]
        st.plotly_chart(equity_curve_chart(curves, height=350), width='stretch')
        if results.trades_df is not None and not results.trades_df.empty:
            st.dataframe(results.trades_df, width='stretch', hide_index=True, height=300)

    st.divider()
    st.markdown("#### Guardar como estrategia real")
    st.caption(
        "Al guardar, esta combinación queda disponible al instante en Backtesting, Optimización, "
        "Walk-Forward, Monte Carlo, Sensibilidad y Portfolio — igual que cualquier otra estrategia."
    )
    sc1, sc2 = st.columns([2, 1])
    class_name = sc1.text_input(
        "Nombre de la estrategia", value=class_name_hint, key="vb_class_name",
        help="Solo letras, números y guion bajo — se usa como nombre de clase Python.",
    )
    if sc2.button("Guardar estrategia", type="primary", key="vb_save_btn"):
        safe_name = "".join(ch for ch in class_name if ch.isalnum() or ch == "_") or "MiEstrategiaVisual"
        if not safe_name[0].isalpha():
            safe_name = f"Estrategia_{safe_name}"
        code = generate_strategy_code(definition, class_name=safe_name, symbol=asset, timeframe=timeframe)
        stem = suggest_class_name(safe_name).lower()
        try:
            target_path = save_user_strategy_file(stem, code)
            refresh_registry()
            if safe_name in STRATEGY_REGISTRY:
                st.success(f"Guardada en `{target_path.name}`. '{safe_name}' ya está disponible en toda la plataforma.")
            else:
                st.warning(f"Se guardó `{target_path.name}`, pero no apareció registrada. Prueba con otro nombre.")
        except Exception as e:
            st.error(f"No se pudo guardar: {e}")

    saved = list_user_strategy_files()
    if saved:
        with st.expander(f"Estrategias guardadas desde el Constructor ({len(saved)})", expanded=False):
            for f in saved:
                st.markdown(f"- `strategies/{f}.py`")


def _load_data(asset: str, timeframe: str):
    return load_historical_data(asset, timeframe)
