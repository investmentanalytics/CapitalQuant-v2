"""
ui/views/portfolio.py
Portfolio — visor de graficos multi-activo y multi-estrategia.

Permite agregar paneles (activo + estrategia) de forma independiente.
Cada panel muestra el grafico de precios con las operaciones de la
estrategia seleccionada superpuestas.
"""
import streamlit as st
import uuid
import pandas as pd

from strategies import STRATEGY_REGISTRY, list_strategies
from core.types import BacktestConfig
from engine.backtester import BacktestEngine
from market_data import get_data
from market_data.mt5_provider import get_mt5_provider
from config.settings import ASSETS, TIMEFRAMES
from visualization.charts import candlestick_chart, equity_curve_chart, indicators_chart
from portfolio.manager import PortfolioManager, AssetAllocation
from portfolio.optimizer import (
    risk_parity_weights, mean_variance_weights, decompose_component_weights,
)
from ui.components.regime_filter import apply_regime_filter
from regime.service import REGIME_LABELS
from visualization.portfolio_charts import (
    portfolio_equity_chart, capital_allocation_pie, correlation_heatmap,
    risk_decomposition_chart,
)
from ui.state import AppState
from ui.components.metrics_grid import render_metrics_grid
from ui.components.data_source import render_data_source_selector, load_historical_data, SOURCE_CLEAN


# ──────────────────────────────────────────────────────────────────────────────
# Estado de paneles
# ──────────────────────────────────────────────────────────────────────────────

def _init_panels():
    if "pf_panels" not in st.session_state:
        st.session_state.pf_panels = []


def _add_panel():
    panel_id = str(uuid.uuid4())[:8]
    st.session_state.pf_panels.append({
        "id": panel_id,
        "asset": None,
        "timeframe": "1d",
        "strategy": None,
        "result": None,
        "df": None,
        "error": None,
    })


def _remove_panel(panel_id: str):
    st.session_state.pf_panels = [
        p for p in st.session_state.pf_panels if p["id"] != panel_id
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Datos
# ──────────────────────────────────────────────────────────────────────────────

def _get_asset_list():
    from research.csv_registry import list_csv_datasets
    from research.data_registry import list_clean_datasets
    csv_assets = {m.get("asset") for m in list_csv_datasets() if m.get("asset")}
    clean_assets = {m.get("asset") for m in list_clean_datasets() if m.get("asset")}
    return sorted(csv_assets | clean_assets)


def _load_data(asset: str, timeframe: str):
    return load_historical_data(asset, timeframe)


def _run_backtest_for_panel(asset, timeframe, strategy_name):
    df = _load_data(asset, timeframe)
    if df is None or df.empty:
        return None, df, f"Sin datos para {asset} {timeframe}"

    strategy_class = STRATEGY_REGISTRY.get(strategy_name)
    if strategy_class is None:
        return None, df, f"Estrategia no encontrada: {strategy_name}"

    config = BacktestConfig(
        initial_capital=100_000.0,
        commission=0.001,
        risk_per_trade=0.02,
        allow_short=True,
    )
    try:
        strategy = strategy_class()
        signals = strategy.generate_signals(df)
        engine = BacktestEngine(config)
        results = engine.run(signals, strategy_name=strategy_name, asset=asset, timeframe=timeframe)
        return results, signals, None
    except Exception as e:
        return None, df, str(e)


# ──────────────────────────────────────────────────────────────────────────────
# Renderizado de un panel
# ──────────────────────────────────────────────────────────────────────────────

def _render_panel(panel: dict, idx: int, asset_list: list, strategy_list: list):
    pid = panel["id"]

    with st.container():
        # Barra de cabecera del panel
        header_col, close_col = st.columns([10, 1])
        with header_col:
            st.markdown(f"#### Panel {idx + 1}")
        with close_col:
            if st.button("", key=f"pf_close_{pid}", help="Eliminar este panel"):
                _remove_panel(pid)
                st.rerun()

        # Controles de seleccion
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])

        # Activo
        current_asset = panel.get("asset") or (asset_list[0] if asset_list else None)
        if current_asset not in asset_list and asset_list:
            current_asset = asset_list[0]

        selected_asset = c1.selectbox(
            "Activo", asset_list,
            index=asset_list.index(current_asset) if current_asset in asset_list else 0,
            key=f"pf_asset_{pid}",
        )

        # Temporalidad
        tf_list = list(TIMEFRAMES.keys())
        tf_labels = {k: v["label"] for k, v in TIMEFRAMES.items()}
        current_tf = panel.get("timeframe", "1d")
        if current_tf not in tf_list:
            current_tf = "1d"

        selected_tf = c2.selectbox(
            "Temporalidad", tf_list,
            format_func=lambda x: tf_labels.get(x, x),
            index=tf_list.index(current_tf),
            key=f"pf_tf_{pid}",
        )

        # Estrategia
        current_strat = panel.get("strategy") or (strategy_list[0] if strategy_list else None)
        selected_strat = c3.selectbox(
            "Estrategia", strategy_list,
            index=strategy_list.index(current_strat) if current_strat in strategy_list else 0,
            key=f"pf_strat_{pid}",
        )

        # Boton cargar
        load_pressed = c4.button("▶ Cargar", key=f"pf_load_{pid}", type="primary", width='stretch')

        # Actualizar panel state
        panel["asset"]     = selected_asset
        panel["timeframe"] = selected_tf
        panel["strategy"]  = selected_strat

        if load_pressed:
            with st.spinner(f"Cargando {selected_asset} {selected_tf} — {selected_strat}..."):
                result, df_or_signals, err = _run_backtest_for_panel(
                    selected_asset, selected_tf, selected_strat
                )
                panel["result"] = result
                panel["df"]     = df_or_signals
                panel["error"]  = err

        # Mostrar contenido del panel
        if panel.get("error"):
            st.error(f" {panel['error']}")
        elif panel.get("result") is not None:
            result = panel["result"]
            df     = panel["df"]

            # Metricas rapidas
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Operaciones", result.total_trades)
            m2.metric("Win Rate", f"{result.win_rate:.1f}%")
            m3.metric("CAGR", f"{result.cagr:.2f}%")
            m4.metric("Sharpe", f"{result.sharpe_ratio:.2f}")
            m5.metric("Max DD", f"{result.max_drawdown_pct:.2f}%")

            # Tabs: Grafico de precio / Equity
            t1, t2 = st.tabs(["Precio con Operaciones", "Equity Curve"])

            with t1:
                strategy_class = STRATEGY_REGISTRY.get(selected_strat)
                indicators = {}
                if strategy_class and df is not None:
                    try:
                        ind_cols = strategy_class().get_indicator_columns()
                        for col in ind_cols:
                            if col in df.columns:
                                indicators[col] = df[col]
                    except Exception:
                        pass

                trades_df = result.trades_df if (result.trades_df is not None and not result.trades_df.empty) else None
                fig = candlestick_chart(
                    df=df,
                    title=f"{selected_asset} {selected_tf} — {selected_strat}",
                    signals=df.get("signal") if hasattr(df, "get") else None,
                    trades_df=trades_df,
                    height=460,
                )
                st.plotly_chart(fig, width='stretch')
                if indicators:
                    st.plotly_chart(indicators_chart(indicators, height=260), width='stretch')

            with t2:
                curves = [{
                    "name": f"{selected_strat} — {selected_asset}",
                    "equity": result.equity_curve,
                    "drawdown": result.drawdown_series,
                    "best": True,
                }]
                st.plotly_chart(equity_curve_chart(curves, height=400), width='stretch')

        elif panel.get("df") is None and not panel.get("error"):
            st.info("Selecciona un activo, temporalidad y estrategia, luego pulsa **▶ Cargar**.")

        st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Render principal
# ──────────────────────────────────────────────────────────────────────────────

def _render_visor() -> None:
    _init_panels()

    st.markdown("### Visor Multi-Activo")
    st.caption(
        "Agrega graficos independientes: cada panel muestra un activo y una estrategia. "
        "Puedes abrir tantos paneles como quieras y eliminarlos individualmente."
    )

    asset_list    = _get_asset_list()
    strategy_list = list_strategies()

    # Boton para agregar panel
    col_btn, col_info = st.columns([2, 8])
    with col_btn:
        if st.button("Agregar panel", type="primary", width='stretch'):
            _add_panel()
            st.rerun()

    if not st.session_state.pf_panels:
        st.markdown(
            """
            <div style="text-align:center; padding:60px 0; color:#8f8c82;">
                <div style="font-size:48px"></div>
                <div style="font-size:16px; margin-top:12px">No hay paneles abiertos.</div>
                <div style="font-size:13px; color:#5c594e; margin-top:6px">
                    Pulsa <b>Agregar panel</b> para comenzar.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Renderizar todos los paneles
    for idx, panel in enumerate(st.session_state.pf_panels):
        _render_panel(panel, idx, asset_list, strategy_list)


# ──────────────────────────────────────────────────────────────────────────────
# Constructor de Portafolio — Fase 2 del roadmap
# ──────────────────────────────────────────────────────────────────────────────

def _init_builder():
    if "pf_builder_components" not in st.session_state:
        st.session_state.pf_builder_components = []   # [{asset, timeframe, strategy, result}]


def _run_component_backtest(asset: str, timeframe: str, strategy_name: str, regimes: list | None = None,
                             direction: str = "Ambos", risk_pct: float = 2.0):
    df = load_historical_data(asset, timeframe)
    if df is None or df.empty:
        return None, f"Sin datos para {asset} {timeframe}"
    strategy_class = STRATEGY_REGISTRY.get(strategy_name)
    if strategy_class is None:
        return None, f"Estrategia no encontrada: {strategy_name}"
    config = BacktestConfig(
        initial_capital=100_000.0, commission=0.001,
        risk_per_trade=risk_pct / 100.0,
        allow_long=direction in ("Solo largos", "Ambos"),
        allow_short=direction in ("Solo cortos", "Ambos"),
    )
    try:
        strategy = strategy_class()
        signals = strategy.generate_signals(df)
        if regimes:
            # Bloquea NUEVAS entradas fuera del/los régimen(es) elegido(s)
            # para este componente — así un mismo activo+estrategia puede
            # aportar al portafolio solo cuando el mercado está, por
            # ejemplo, en Consolidación, y otro componente distinto
            # (mismo activo+estrategia, otro régimen o sin filtro) puede
            # cubrir el resto del tiempo con otra ponderación.
            signals, _ = apply_regime_filter(df, signals, asset, timeframe, regimes)
        engine = BacktestEngine(config)
        results = engine.run(signals, strategy_name=strategy_name, asset=asset, timeframe=timeframe)
        return results, None
    except Exception as e:
        return None, str(e)


def _component_label(comp: dict) -> str:
    """Etiqueta única del componente para las claves de pesos/registro.
    Incluye el régimen elegido (si hay uno) para que el mismo activo+
    estrategia pueda aparecer dos veces como componentes DISTINTOS —
    cada uno con su propio peso — cuando cada uno está restringido a un
    régimen de mercado distinto."""
    regimes = comp.get("regimes") or []
    return f"{comp['strategy']} [{'+'.join(regimes)}]" if regimes else comp["strategy"]


def _render_builder() -> None:
    _init_builder()

    st.markdown("### Constructor de Portafolio")
    st.caption(
        "Arma un portafolio con varias combinaciones activo+estrategia (por ejemplo, distintas "
        "estrategias del Hall of Fame del Descubridor Genético) y compara pesos **igual peso** contra "
        "**paridad de riesgo** y **media-varianza (Markowitz)** — con panel de correlación entre "
        "componentes para ver cuánto diversifican realmente entre sí."
    )

    asset_list = _get_asset_list()
    strategy_list = list_strategies()

    # ------------------------------------------------------------------
    # Agregar componentes
    # ------------------------------------------------------------------
    with st.container():
        st.markdown("#### 1. Componentes del portafolio")
        c1, c2, c3 = st.columns([2, 2, 3])
        add_asset = c1.selectbox("Activo", asset_list, key="pfb_add_asset")
        tf_list = list(TIMEFRAMES.keys())
        tf_labels = {k: v["label"] for k, v in TIMEFRAMES.items()}
        add_tf = c2.selectbox("Temporalidad", tf_list, format_func=lambda x: tf_labels.get(x, x),
                                index=tf_list.index("1d") if "1d" in tf_list else 0, key="pfb_add_tf")
        add_strat = c3.selectbox("Estrategia", strategy_list, key="pfb_add_strat")

        c4, c5, c6 = st.columns([2, 2, 1])
        add_direction = c4.radio(
            "Dirección", ["Ambos", "Solo largos", "Solo cortos"],
            horizontal=True, key="pfb_add_direction",
        )
        add_risk = c5.slider(
            "Riesgo por operación (%)", min_value=0.25, max_value=5.0, value=2.0, step=0.25,
            key="pfb_add_risk",
            help="Porcentaje del capital arriesgado en cada operación de este componente. "
                 "Un valor más alto aumenta tanto el retorno esperado como el drawdown.",
        )
        add_regimes = st.multiselect(
            "Aplicar solo en régimen(es) de mercado (opcional)",
            REGIME_LABELS, default=[], key="pfb_add_regimes",
            help="Deja vacío para operar en todos los regímenes. Si eliges uno o más, este "
                 "componente solo abrirá posiciones nuevas cuando el mercado esté clasificado "
                 "en ese régimen — útil para asignar un peso propio a la misma estrategia según "
                 "el régimen vigente.",
        )
        if c6.button("+ Agregar", type="primary", width='stretch', key="pfb_add_btn"):
            key = (add_asset, add_tf, add_strat, tuple(sorted(add_regimes)), add_direction, add_risk)
            existing = {
                (c["asset"], c["timeframe"], c["strategy"], tuple(sorted(c.get("regimes") or [])),
                 c.get("direction", "Ambos"), c.get("risk_pct", 2.0))
                for c in st.session_state.pf_builder_components
            }
            if key not in existing:
                st.session_state.pf_builder_components.append({
                    "asset": add_asset, "timeframe": add_tf, "strategy": add_strat,
                    "regimes": list(add_regimes), "direction": add_direction,
                    "risk_pct": add_risk, "result": None,
                })
            st.rerun()

    if not st.session_state.pf_builder_components:
        st.info("Agrega al menos 2 componentes para poder optimizar pesos entre ellos.")
        return

    # Tabla de componentes agregados, con opción de quitar
    hcols = st.columns([2.5, 1.5, 2, 2.5, 1.5, 1.2, 2, 1])
    for h, label in zip(hcols, ["Activo", "Temp.", "Estrategia", "Régimen", "Dirección", "Riesgo", "Estado", ""]):
        h.markdown(f"**{label}**")
    for i, comp in enumerate(st.session_state.pf_builder_components):
        row = st.columns([2.5, 1.5, 2, 2.5, 1.5, 1.2, 2, 1])
        row[0].write(f"**{comp['asset']}**")
        row[1].write(tf_labels.get(comp["timeframe"], comp["timeframe"]))
        row[2].write(comp["strategy"])
        row[3].write(", ".join(comp.get("regimes") or []) or "Todos los regímenes")
        row[4].write(comp.get("direction", "Ambos"))
        row[5].write(f"{comp.get('risk_pct', 2.0):.2f}%")
        has_result = comp.get("result") is not None
        row[6].write("✅ backtest OK" if has_result else "⏳ pendiente")
        if row[7].button("✕", key=f"pfb_remove_{i}", help="Quitar componente"):
            st.session_state.pf_builder_components.pop(i)
            st.rerun()

    if st.button("▶ Correr backtests de todos los componentes", type="primary", key="pfb_run_all"):
        progress = st.progress(0, text="Corriendo backtests...")
        live_pf_table = st.empty()
        n = len(st.session_state.pf_builder_components)
        for i, comp in enumerate(st.session_state.pf_builder_components):
            result, err = _run_component_backtest(
                comp["asset"], comp["timeframe"], comp["strategy"],
                regimes=comp.get("regimes"),
                direction=comp.get("direction", "Ambos"),
                risk_pct=comp.get("risk_pct", 2.0),
            )
            comp["result"] = result
            comp["error"] = err
            progress.progress((i + 1) / n, text=f"{comp['asset']} — {comp['strategy']} ({i+1}/{n})")
            rows_live = []
            for c in st.session_state.pf_builder_components:
                r = c.get("result")
                rows_live.append({"Activo": c.get("asset"), "Estrategia": c.get("strategy"), "Estado": "✅ listo" if r is not None else ("❌ error" if c.get("error") else "⏳ pendiente"), "Trades": getattr(r, "total_trades", "—") if r is not None else "—", "Sharpe": getattr(r, "sharpe_ratio", "—") if r is not None else "—", "DD": getattr(r, "max_drawdown_pct", "—") if r is not None else "—"})
            live_pf_table.dataframe(pd.DataFrame(rows_live), width="stretch", hide_index=True)
        progress.progress(1.0, text=f"Backtests de componentes completados · {n}/{n}")
        st.rerun()

    components_ready = [c for c in st.session_state.pf_builder_components if c.get("result") is not None]
    if len(components_ready) < 2:
        st.info("Se necesitan al menos 2 componentes con backtest exitoso para optimizar pesos.")
        return

    # ------------------------------------------------------------------
    # Método de ponderación
    # ------------------------------------------------------------------
    st.markdown("#### 2. Método de ponderación")
    method = st.radio(
        "¿Cómo se reparten los pesos entre componentes?",
        ["Igual peso", "Paridad de Riesgo", "Media-Varianza (Máx. Sharpe)", "Media-Varianza (Mín. Varianza)", "Manual"],
        horizontal=True, key="pfb_method",
    )

    manual_weights = {}
    if method == "Manual":
        st.caption("Asigná el peso relativo de cada componente (se normalizan automáticamente a 100%).")
        cols = st.columns(len(components_ready))
        for col, comp in zip(cols, components_ready):
            key = f"{comp['asset']}::{_component_label(comp)}"
            manual_weights[key] = col.number_input(
                key, min_value=0.0, max_value=1.0, value=1.0 / len(components_ready),
                step=0.05, key=f"pfb_manual_{key}",
            )

    if st.button("📊 Calcular portafolio", type="primary", key="pfb_compute"):
        component_returns = {
            f"{c['asset']}::{_component_label(c)}": c["result"].equity_curve.pct_change().fillna(0)
            for c in components_ready
        }

        try:
            if method == "Igual peso":
                n = len(component_returns)
                flat_weights = {k: 1.0 / n for k in component_returns}
                opt_info = None
            elif method == "Paridad de Riesgo":
                opt_info = risk_parity_weights(component_returns, timeframe="1d")
                flat_weights = opt_info.weights
            elif method == "Media-Varianza (Máx. Sharpe)":
                opt_info = mean_variance_weights(component_returns, timeframe="1d", method="max_sharpe")
                flat_weights = opt_info.weights
            elif method == "Media-Varianza (Mín. Varianza)":
                opt_info = mean_variance_weights(component_returns, timeframe="1d", method="min_variance")
                flat_weights = opt_info.weights
            else:  # Manual
                total = sum(manual_weights.values()) or 1.0
                flat_weights = {k: v / total for k, v in manual_weights.items()}
                opt_info = None
        except ValueError as e:
            st.error(str(e))
            return

        asset_weights, strategies_by_asset = decompose_component_weights(flat_weights)

        pm = PortfolioManager(initial_capital=100_000.0, timeframe="1d")
        for c in components_ready:
            label = _component_label(c)
            pm.add_allocation(AssetAllocation(asset=c["asset"], strategies=strategies_by_asset.get(c["asset"], {label: 1.0})))
            pm.register_result(c["asset"], label, c["result"])

        pr = pm.compute(asset_weights=asset_weights)
        AppState.save_portfolio(pr)
        AppState.save_portfolio_config(strategies_by_asset, asset_weights)
        st.session_state["pfb_last_opt_info"] = opt_info
        st.session_state["pfb_last_method"] = method
        st.success("Portafolio calculado.")
        st.rerun()

    # ------------------------------------------------------------------
    # Resultados
    # ------------------------------------------------------------------
    pr = AppState.get_portfolio()
    if pr is None:
        st.info("Pulsa **Calcular portafolio** para ver resultados.")
        return

    st.markdown("#### 3. Resultado consolidado")
    opt_info = st.session_state.get("pfb_last_opt_info")
    if opt_info is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Retorno esperado (anual.)", f"{opt_info.expected_return*100:.1f}%" if opt_info.expected_return is not None else "—")
        c2.metric("Volatilidad esperada (anual.)", f"{opt_info.expected_vol*100:.1f}%" if opt_info.expected_vol is not None else "—")
        c3.metric("Sharpe esperado", f"{opt_info.expected_sharpe:.2f}" if opt_info.expected_sharpe is not None else "—")
        if not opt_info.converged:
            st.warning(f"El optimizador no convergió del todo: {opt_info.message}")

    render_metrics_grid(pr.metrics_dict())

    st.plotly_chart(portfolio_equity_chart(pr, height=480), width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(capital_allocation_pie(pr, group_by="component", height=380), width='stretch')
    with c2:
        st.markdown("##### Panel de correlación entre componentes")
        st.caption(
            "Correlación de retornos diarios entre las estrategias del portafolio (incluye las que "
            "vengan del Hall of Fame del Descubridor Genético). Correlaciones bajas o negativas entre "
            "componentes con peso alto son la señal de que realmente están diversificando el riesgo, "
            "no solo repartiendo capital."
        )
        st.plotly_chart(correlation_heatmap(pr.strategy_correlation, title="Correlación entre Componentes", height=380),
                         width='stretch')

    st.plotly_chart(risk_decomposition_chart(pr.risk_by_asset, pr.risk_by_strategy, height=350), width='stretch')

    if pr.component_summary is not None and not pr.component_summary.empty:
        st.markdown("##### Resumen por componente")
        st.dataframe(pr.component_summary, width='stretch', hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Render principal
# ──────────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## Portfolio")
    # Fuente histórica común: permite usar exclusivamente datasets publicados desde Gráficos.
    render_data_source_selector(None, key_prefix="portfolio_source", compact=True)
    st.caption("La fuente seleccionada se aplica a los backtests históricos de este módulo. Mercado en Vivo sigue usando MT5 para cotización/ejecución.")

    tab_visor, tab_builder = st.tabs(["Visor Multi-Activo", "Constructor de Portafolio"])

    with tab_visor:
        _render_visor()

    with tab_builder:
        _render_builder()
