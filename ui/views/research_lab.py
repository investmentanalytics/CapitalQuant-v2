from __future__ import annotations

from datetime import date, timedelta, datetime
import json
import re

import pandas as pd
import streamlit as st

from market_data.mt5_provider import MT5_LABELS
from research.data_registry import list_clean_datasets, available_clean_assets, load_clean_dataset
from research.raw_data_registry import list_raw_datasets, load_raw_dataset
from research.config import ResearchConfig, DEFAULT_OBJECTIVES, REGIMES
from research.orchestrator.research_orchestrator import ResearchOrchestrator
from research.results.result_store import load_batch, load_latest_batch, save_batch
from research.objectives.objective_registry import OBJECTIVES
from ui.state import AppState


def _safe_class_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", text).strip("_") or "ResearchStrategy"
    if name[0].isdigit():
        name = "E_" + name
    return name


def _save_candidate(chosen: dict, asset: str, timeframe: str, regime: str | None) -> str:
    from discovery.codegen import generate_strategy_code
    from strategies import save_user_strategy_file, refresh_registry, record_discovery_manifest

    definition = chosen["definition"]
    # Validación previa: una estrategia solo se publica si el mismo código
    # generado que llegará a Backtesting/Live/Portfolio puede compilarse.
    # Esto evita guardar archivos .py corruptos en el catálogo global.
    from discovery.codegen import compile_definition_to_class
    compile_definition_to_class(
        definition, class_name="ResearchSaveValidation", symbol=asset, timeframe=timeframe
    )
    base = _safe_class_name(
        f"Research_{asset}_{timeframe}_{chosen.get('strategy_id', 'strategy')}"
    )
    stem = base.lower()
    code = generate_strategy_code(
        definition, class_name=base, symbol=asset, timeframe=timeframe
    )
    save_user_strategy_file(stem, code)
    manifest = {
        "class_name": base,
        "asset": asset,
        "timeframe": timeframe,
        "regime": regime,
        "source": "Research Lab",
        "objective": chosen.get("objective"),
        "strategy_id": chosen.get("strategy_id"),
        "definition": definition,
        "families": chosen.get("families_used", []),
        "long_rule_text": chosen.get("long_rule_text", ""),
        "short_rule_text": chosen.get("short_rule_text", ""),
        "metrics": chosen.get("metrics", {}),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    record_discovery_manifest(stem, manifest)
    refresh_registry()
    return base


def _run_preview(
    chosen: dict,
    asset: str,
    timeframe: str,
    initial_capital: float,
    commission: float,
    slippage: float,
    risk: float,
):
    """Backtest puntual con el motor real. Se ejecuta solo bajo demanda."""
    from discovery.codegen import compile_definition_to_class
    from core.types import BacktestConfig
    from engine.backtester import BacktestEngine

    # El preview usa exactamente la fuente autorizada del Research Lab:
    # el dataset limpio registrado desde Gráficos. Nunca descarga desde MT5.
    df = load_clean_dataset(asset, timeframe)
    if df is None or df.empty:
        raise ValueError(
            f"No existe un dataset limpio registrado para {asset} {timeframe}. "
            "Envíalo primero desde Gráficos."
        )

    strategy_class = compile_definition_to_class(
        chosen["definition"],
        class_name="ResearchPreview",
        symbol=asset,
        timeframe=timeframe,
    )
    strategy = strategy_class()
    sig_df = strategy.generate_signals(df.copy())

    # Si la estrategia nació dentro de un régimen concreto, el preview usa
    # exactamente el mismo filtro de entradas del descubrimiento. Las velas
    # siguen completas para no alterar indicadores/warmup.
    if chosen.get("regime"):
        from ui.components.regime_filter import compute_regime_mask
        regime_mask = compute_regime_mask(
            df, timeframe, [chosen["regime"]], asset=asset
        )
        if regime_mask is not None:
            sig_df = sig_df.copy()
            sig_df.loc[~regime_mask.reindex(sig_df.index).fillna(False), "signal"] = 0

    definition = chosen["definition"]
    cfg = BacktestConfig(
        initial_capital=float(initial_capital),
        commission=float(commission),
        slippage=float(slippage),
        risk_per_trade=float(risk),
        allow_long=bool(definition.get("long_rule")),
        allow_short=bool(definition.get("short_rule")),
    )
    results = BacktestEngine(cfg).run(
        sig_df,
        strategy_name=chosen.get("name", "Research Strategy"),
        asset=asset,
        timeframe=timeframe,
    )
    return results, sig_df, strategy.get_indicator_columns(), df


def _render_preview(chosen: dict, asset: str, timeframe: str, state: dict):
    from visualization.charts import candlestick_chart, indicators_chart, equity_curve_chart
    from ui.components.metrics_grid import render_metrics_grid

    fr = state["results"]
    render_metrics_grid(fr.to_dict(), n_cols=4)

    m = fr.to_dict()
    # Tabla completa y legible del backtest real. La dejamos antes de los
    # gráficos para que el usuario pueda revisar cifras sin depender de
    # inspeccionar visualmente la curva.
    summary_rows = [{
        "Métrica": "Capital inicial", "Valor": f"${fr.initial_capital:,.2f}"
    }, {
        "Métrica": "Capital final", "Valor": f"${fr.final_capital:,.2f}"
    }, {
        "Métrica": "Net Profit", "Valor": f"${fr.net_profit:,.2f}"
    }, {
        "Métrica": "Retorno", "Valor": f"{fr.net_profit_pct:.2f}%"
    }, {
        "Métrica": "CAGR", "Valor": f"{fr.cagr:.2f}%"
    }, {
        "Métrica": "Sharpe", "Valor": f"{fr.sharpe_ratio:.3f}"
    }, {
        "Métrica": "Sortino", "Valor": f"{fr.sortino_ratio:.3f}"
    }, {
        "Métrica": "Calmar", "Valor": f"{fr.calmar_ratio:.3f}"
    }, {
        "Métrica": "Profit Factor", "Valor": f"{fr.profit_factor:.3f}"
    }, {
        "Métrica": "Win Rate", "Valor": f"{fr.win_rate:.2f}%"
    }, {
        "Métrica": "Expectancy", "Valor": f"${fr.expectancy:,.4f}"
    }, {
        "Métrica": "Max Drawdown", "Valor": f"{fr.max_drawdown_pct:.2f}%"
    }, {
        "Métrica": "Ulcer Index", "Valor": f"{fr.ulcer_index:.3f}"
    }, {
        "Métrica": "Exposure", "Valor": f"{fr.exposure_pct:.2f}%"
    }, {
        "Métrica": "Operaciones", "Valor": str(fr.total_trades)}]
    with st.expander("Tabla estadística completa", expanded=False):
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    c = st.columns(5)
    c[0].metric("Operaciones", int(m.get("n_trades", 0)))
    c[1].metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")
    c[2].metric("Win Rate", f"{m.get('win_rate', 0)*100:.1f}%")
    c[3].metric("Expectancy", f"{m.get('expectancy', 0):.4f}")
    c[4].metric("Drawdown", f"{m.get('max_drawdown', 0)*100:.1f}%")

    df = state["df"]
    sig_df = state["sig_df"]
    df_viz = df.loc[fr.start_date:fr.end_date]

    st.plotly_chart(
        candlestick_chart(
            df=df_viz,
            title=f"{asset} {timeframe} — {chosen.get('name', 'Strategy')}",
            trades_df=fr.trades_df if not fr.trades_df.empty else None,
            height=500,
        ),
        width="stretch",
    )

    indicators = {
        c: sig_df[c]
        for c in state["indicator_cols"]
        if c in sig_df.columns
    }
    if indicators:
        st.plotly_chart(indicators_chart(indicators, height=280), width="stretch")

    curves = [{
        "name": chosen.get("name", "Research Strategy"),
        "equity": fr.equity_curve,
        "drawdown": fr.drawdown_series,
        "best": True,
    }]
    st.plotly_chart(equity_curve_chart(curves, height=380), width="stretch")

    if fr.trades_df is not None and not fr.trades_df.empty:
        st.markdown("#### Operaciones")
        st.dataframe(fr.trades_df, width="stretch", hide_index=True, height=350)


def _flatten_candidates(batch_data: dict) -> list[dict]:
    rows = []
    for job in batch_data.get("jobs", []):
        for c in (job.get("result", {}) or {}).get("candidates", []) or []:
            m = c.get("metrics", {}) or {}
            rows.append({
                "asset": job.get("asset"), "timeframe": job.get("timeframe"),
                "objective": job.get("objective"), "regime": job.get("regime"),
                "strategy_id": c.get("strategy_id"), "fitness": c.get("fitness", 0.0),
                "calmar": m.get("calmar", 0.0), "sharpe": m.get("sharpe", 0.0),
                "profit_factor": m.get("profit_factor", 0.0), "win_rate": m.get("win_rate", 0.0),
                "cagr": m.get("cagr", 0.0), "max_drawdown": m.get("max_drawdown", 0.0),
                "n_trades": m.get("n_trades", 0), "expectancy": m.get("expectancy", 0.0),
                "candidate": c,
            })
    return rows


def _render_strategy_selection(batch_data: dict) -> None:
    rows = _flatten_candidates(batch_data)
    if not rows:
        return
    st.markdown("### Selección automática y descarga")
    st.caption(
        "Filtra candidatos por actividad y riesgo y ordénalos por la métrica que te interese. "
        "La selección no elimina ninguna estrategia: solo propone candidatos para revisión."
    )
    c1,c2,c3,c4 = st.columns(4)
    min_trades = c1.number_input("Mínimo de trades", 0, 10000, 30, 5, key="research_sel_min_trades")
    min_wr = c2.slider("Win Rate mínimo (%)", 0.0, 100.0, 0.0, 1.0, key="research_sel_wr")
    min_pf = c3.number_input("Profit Factor mínimo", 0.0, 20.0, 1.0, 0.05, key="research_sel_pf")
    max_dd = c4.slider("Drawdown máximo (%)", 0.0, 100.0, 50.0, 1.0, key="research_sel_dd")
    metric_labels = {
        "fitness":"Fitness","calmar":"Calmar","sharpe":"Sharpe",
        "profit_factor":"Profit Factor","win_rate":"Win Rate","cagr":"CAGR","expectancy":"Expectancy"
    }
    sort_metric = st.selectbox("Ordenar candidatos por", list(metric_labels), format_func=lambda x: metric_labels[x], key="research_sel_metric")
    top_n = st.slider("Número de estrategias propuestas", 1, min(20, max(1, len(rows))), min(3, max(1, len(rows))), key="research_sel_topn")
    df = pd.DataFrame(rows)
    df["win_rate_pct"] = df["win_rate"] * 100.0
    df["dd_pct"] = df["max_drawdown"].abs() * 100.0
    filtered = df[
        (df["n_trades"] >= int(min_trades)) &
        (df["win_rate_pct"] >= float(min_wr)) &
        (df["profit_factor"] >= float(min_pf)) &
        (df["dd_pct"] <= float(max_dd))
    ].copy()
    filtered = filtered.sort_values(sort_metric, ascending=False).head(top_n)
    display_cols = ["asset","objective","strategy_id","fitness","calmar","sharpe","profit_factor","win_rate_pct","cagr","dd_pct","n_trades"]
    if filtered.empty:
        st.warning("Ningún candidato cumple los filtros actuales.")
        return
    out = filtered[display_cols].rename(columns={"asset":"Activo","objective":"Objetivo","strategy_id":"ID",
        "fitness":"Fitness","calmar":"Calmar","sharpe":"Sharpe","profit_factor":"PF",
        "win_rate_pct":"Win Rate %","cagr":"CAGR","dd_pct":"DD %","n_trades":"Trades"})
    st.dataframe(out, width="stretch", hide_index=True)
    selected_ids = filtered["strategy_id"].tolist()
    st.caption("Propuesta automática actual: " + ", ".join(selected_ids))
    if st.button("💾 Guardar automáticamente estas estrategias", key="research_auto_save"):
        saved = []
        for _, row in filtered.iterrows():
            c = row["candidate"]
            try:
                name = _save_candidate(c, row["asset"], row["timeframe"], row["regime"])
                c["saved"] = True
                c["saved_name"] = name
                saved.append(name)
            except Exception as exc:
                st.warning(f"No se pudo guardar {row['strategy_id']}: {exc}")
        if saved:
            st.success("Guardadas: " + ", ".join(saved))
            st.rerun()

    # Exportar exactamente las seleccionadas, incluyendo definición y código Pine.
    import io, zipfile
    from discovery.codegen import generate_strategy_code
    from pine_generator import generate_pine_script
    payload = {"batch_id": batch_data.get("batch_id"), "strategies": []}
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for _, row in filtered.iterrows():
            c = row["candidate"]; definition = c.get("definition") or {}
            sid = str(row["strategy_id"])
            payload["strategies"].append({
                "asset": row["asset"], "timeframe": row["timeframe"], "objective": row["objective"],
                "regime": row["regime"], "metrics": c.get("metrics", {}), "definition": definition,
            })
            class_name = _safe_class_name(f"Research_{row['asset']}_{row['timeframe']}_{sid}")
            try:
                py_code = generate_strategy_code(definition, class_name=class_name, symbol=row["asset"], timeframe=row["timeframe"])
                z.writestr(f"{sid}/{class_name}.py", py_code)
            except Exception as exc:
                z.writestr(f"{sid}/PYTHON_GENERATION_ERROR.txt", str(exc))
            try:
                pine_code = generate_pine_script(definition, strategy_name=class_name, asset=row["asset"], timeframe=row["timeframe"])
                z.writestr(f"{sid}/{class_name}.pine", pine_code)
            except Exception as exc:
                z.writestr(f"{sid}/PINE_GENERATION_ERROR.txt", str(exc))
        z.writestr("selected_strategies.json", json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    st.download_button(
        "⬇️ Descargar estrategias seleccionadas (Python + Pine v6 + JSON)",
        data=mem.getvalue(),
        file_name=f"CapitalQuant_selected_strategies_{batch_data.get('batch_id','research')}.zip",
        mime="application/zip",
        width="stretch",
    )

def _render_results(batch_data: dict) -> None:
    """Tabla persistente del lote: una fila por investigación y su mejor candidato."""
    jobs = batch_data.get("jobs", [])
    rows = []
    for job in jobs:
        candidates = job.get("result", {}).get("candidates", []) if job.get("result") else []
        best = candidates[0] if candidates else {}
        metrics = best.get("metrics", {}) or {}
        obj = OBJECTIVES.get(job.get("objective"))
        rows.append({
            "Activo": job.get("asset"),
            "Régimen": job.get("regime") or "Todo el histórico",
            "Objetivo": obj.label if obj else job.get("objective"),
            "Estado": job.get("status"),
            "Verificada": "Sí" if best.get("verified") else "Exploración",
            "Fitness": best.get("fitness"),
            "Calmar": metrics.get("calmar"),
            "Sharpe": metrics.get("sharpe"),
            "Profit Factor": metrics.get("profit_factor"),
            "Win Rate": metrics.get("win_rate"),
            "CAGR": metrics.get("cagr"),
            "Expectancy": metrics.get("expectancy"),
            "Drawdown": metrics.get("max_drawdown"),
            "Estabilidad": metrics.get("stability"),
            "Robustez": metrics.get("robustness"),
            "Estrategia": best.get("strategy_id"),
            "Guardada": "Sí" if best.get("saved") else "No",
            "Error": job.get("error") or "",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=360)
        st.download_button(
            "Descargar resultados JSON",
            data=json.dumps(batch_data, ensure_ascii=False, indent=2, default=str),
            file_name=f"{batch_data.get('batch_id', 'research')}.json",
            mime="application/json",
            width="stretch",
        )


def render(asset: str, timeframe: str) -> None:
    st.markdown("## Research Lab")
    st.caption(
        "Descubrimiento multi-activo. El activo y la temporalidad de la barra global son la "
        "referencia de la sesión; aquí solo seleccionas los activos adicionales que quieres investigar."
    )

    clean_rows = list_clean_datasets(timeframe)
    raw_rows = list_raw_datasets(timeframe)
    clean_assets = sorted({x.get("asset") for x in clean_rows if x.get("asset")})
    raw_assets = sorted({x.get("asset") for x in raw_rows if x.get("asset")})
    global_label = MT5_LABELS.get(timeframe, timeframe)

    # 01 Universo: el usuario decide si investiga datos limpios o la copia original.
    with st.container(border=True):
        st.markdown("### 01 · Universo de datos")
        st.caption(
            f"Temporalidad heredada del selector global: **{global_label} ({timeframe})**. "
            "Research Lab no descarga MT5 por su cuenta: trabaja con datasets publicados "
            "desde Gráficos para que la procedencia sea reproducible."
        )
        data_mode = st.radio(
            "Fuente de investigación",
            ["Datos limpios / verificados", "Datos originales publicados"],
            horizontal=True,
            key="research_dataset_source",
        )
        use_raw = data_mode == "Datos originales publicados"
        available_assets = raw_assets if use_raw else clean_assets
        if not available_assets:
            st.warning(
                f"No hay datasets publicados para **{global_label}** en la fuente seleccionada. "
                "Ve a Gráficos y publica la fuente original o la fuente limpia."
            )
            if st.button("↗ IR A GRÁFICOS PARA PREPARAR DATOS", key="research_go_charts", type="primary"):
                st.session_state["cq_active_page"] = "Gráficos"
                st.rerun()
            return
        active_rows = raw_rows if use_raw else clean_rows
        meta_by_asset = {x["asset"]: x for x in active_rows}
        previous = [x for x in st.session_state.get("research_assets", []) if x in available_assets]
        if not previous and asset in available_assets:
            previous = [asset]
        selected_assets = st.multiselect(
            "Activos disponibles en Research Lab",
            available_assets,
            default=previous[:10],
            max_selections=10,
            key="research_assets",
        )
        st.caption(
            f"{len(selected_assets)}/10 activos seleccionados · fuente: "
            f"Gráficos → {'datos originales publicados' if use_raw else 'datos limpios'}"
        )
        if selected_assets:
            cols = st.columns(min(5, len(selected_assets)))
            for i, a in enumerate(selected_assets):
                meta = meta_by_asset.get(a, {})
                with cols[i % len(cols)]:
                    st.markdown(
                        f"**● {a}**  \n"
                        f"{int(meta.get('bars', 0)):,} barras  \n"
                        f"{str(meta.get('start', ''))[:10]} → {str(meta.get('end', ''))[:10]}"
                    )

    # ───────────────────────────────────────────────────────────────────
    # 02 Datos — solo ventana; temporalidad viene de la barra global.
    # ───────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 02 · Datos")
        c1, c2, c3 = st.columns(3)
        mode = c1.radio(
            "Ventana",
            ["Todo el histórico", "Rango de fechas"],
            horizontal=True,
            key="research_data_mode",
        )
        if mode == "Rango de fechas":
            # Preset práctico para el experimento nocturno recomendado:
            # cinco años de desarrollo terminando en 2025, dejando 2026 para OOS.
            end_default = date(2025, 12, 31)
            start_default = date(2020, 1, 1)
            start = c2.date_input("Desde", start_default, key="research_date_start")
            end = c3.date_input("Hasta", end_default, key="research_end")
            if start > end:
                st.error("La fecha inicial no puede ser posterior a la final.")
        else:
            start = end = None
            c2.info(f"Se utilizará todo el histórico disponible para cada activo en {global_label}.")

    if selected_assets:
        st.info(
            f"Fuente bloqueada: **{data_mode}**. "
            "El rango de fechas solo recorta la serie publicada; no inicia ninguna descarga."
        )

    # ───────────────────────────────────────────────────────────────────
    # 03 Régimen
    # ───────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 03 · Régimen de mercado")
        regime_mode = st.radio(
            "Cómo definir la muestra",
            ["Todo el histórico", "Separar por régimen"],
            horizontal=True,
            key="research_regime_mode",
        )
        selected_regimes = []
        if regime_mode == "Separar por régimen":
            selected_regimes = st.multiselect(
                "Regímenes que se investigarán por separado",
                REGIMES,
                default=[REGIMES[0]],
                key="research_regimes",
            )

    # ───────────────────────────────────────────────────────────────────
    # 04 Objetivos
    # ───────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 04 · Objetivos de descubrimiento")
        labels = [OBJECTIVES[k].label for k in DEFAULT_OBJECTIVES]
        selected_labels = st.multiselect(
            "Cada objetivo ejecuta una búsqueda genética independiente",
            labels,
            default=labels,
            key="research_objectives",
        )
        label_to_key = {OBJECTIVES[k].label: k for k in DEFAULT_OBJECTIVES}
        selected_objectives = [label_to_key[x] for x in selected_labels]

    # ───────────────────────────────────────────────────────────────────
    # 05 Motor genético — mismo nivel de configuración que Descubridor.
    # ───────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 05 · Motor genético")
        st.caption(
            "Misma configuración genética del Descubridor Genético. Research Lab no reduce "
            "población, generaciones, diversidad ni espacio de reglas."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Población y generaciones**")
            population = st.number_input("Tamaño de población", 20, 5000, 300, step=10,
                                         key="research_population")
            generations = st.number_input("Número de generaciones", 3, 500, 50, step=1,
                                          key="research_generations")
            tournament_size = st.number_input("Tamaño del torneo", 2, 20, 4, step=1,
                                               key="research_tournament")
            elite_fraction = st.slider("Fracción élite", 0.0, 0.50, 0.10, 0.01,
                                       key="research_elite_fraction")
            crossover_rate = st.slider("Tasa de crossover", 0.0, 1.0, 0.65, 0.05,
                                       key="research_crossover")
            mutation_rate = st.slider("Tasa de mutación", 0.0, 1.0, 0.30, 0.05,
                                      key="research_mutation")
            immigrant_fraction = st.slider("Fracción de inmigrantes", 0.0, 0.50, 0.10, 0.01,
                                           key="research_immigrants")

        with c2:
            st.markdown("**Estructura de reglas**")
            min_clauses = st.slider("Mínimo de cláusulas OR", 1, 10, 1, key="research_min_clauses")
            max_clauses = st.slider("Máximo de cláusulas OR", min_clauses, 10, 2,
                                    key="research_max_clauses")
            min_conditions = st.slider("Mínimo de condiciones AND", 1, 6, 1,
                                       key="research_min_conditions")
            max_conditions = st.slider("Máximo de condiciones AND", min_conditions, 6, 3,
                                       key="research_max_conditions")
            min_trades = st.number_input("Mínimo de trades para ser válida", 5, 5000, 30, step=5,
                                         key="research_min_trades")
            target_trades = st.number_input("Trades objetivo", 20, 10000, 150, step=10,
                                            key="research_target_trades")
            direction = st.radio(
                "Dirección",
                ["Ambos (long y short)", "Solo long", "Solo short"],
                horizontal=False,
                key="research_direction",
            )
            allow_long = direction != "Solo short"
            allow_short = direction != "Solo long"

        with c3:
            st.markdown("**Riesgo, validación y diversidad**")
            evolve_risk = st.checkbox(
                "Evolucionar Stop Loss / Take Profit",
                value=True,
                key="research_evolve_risk",
            )
            sl_range = st.slider(
                "Rango Stop Loss (× ATR)", 0.3, 6.0, (0.8, 4.0), 0.1,
                disabled=not evolve_risk, key="research_sl_range",
            )
            tp_range = st.slider(
                "Rango Take Profit (× ATR)", 0.5, 12.0, (1.0, 8.0), 0.1,
                disabled=not evolve_risk, key="research_tp_range",
            )
            risk_sigma = st.slider(
                "Sigma de mutación del riesgo", 0.05, 0.50, 0.20, 0.01,
                key="research_risk_sigma",
                help="Tamaño de la perturbación de SL/TP como fracción del rango de cada gen.",
            )
            hof_size = st.number_input("Tamaño Hall of Fame", 10, 500, 60, step=10,
                                       key="research_hof")
            n_folds = st.number_input(
                "Ventanas walk-forward",
                1, 12, 3, step=1,
                key="research_folds",
                help="1 = exploración sin validación OOS; >1 = fitness consistente entre ventanas.",
            )
            auto_min_fold = max(5, int(min_trades) // max(1, int(n_folds)))
            override_min_fold = st.checkbox(
                f"Fijar mínimo por ventana manualmente (automático: {auto_min_fold})",
                value=False,
                key="research_override_fold",
            )
            min_trades_holdout = (
                st.number_input("Mínimo trades por ventana", 1, 1000, auto_min_fold, step=1,
                                key="research_min_fold")
                if override_min_fold else None
            )
            niche_similarity_threshold = st.slider(
                "Umbral similitud Jaccard", 0.3, 0.9, 0.6, 0.05,
                key="research_niche_threshold",
            )
            niche_penalty_weight = st.slider(
                "Penalización por vecino", 0.0, 0.30, 0.06, 0.01,
                key="research_niche_penalty",
            )
            hof_diversity_threshold = st.slider(
                "Similitud máxima en Hall of Fame", 0.3, 0.9, 0.6, 0.05,
                key="research_hof_diversity",
            )

        with st.expander("⚙️ Avanzado — semilla, costos y verificación", expanded=False):
            a1, a2, a3 = st.columns(3)
            seed = a1.number_input("Semilla aleatoria", 0, 999_999, 42, step=1,
                                   key="research_seed")
            initial_capital = a1.number_input("Capital inicial", 100.0, 10_000_000.0, 10_000.0,
                                              step=100.0, key="research_initial_capital")
            commission = a2.number_input("Comisión", 0.0, 0.02, 0.001, 0.0001,
                                         format="%.4f", key="research_commission")
            slippage = a2.number_input("Slippage", 0.0, 0.02, 0.0005, 0.0001,
                                       format="%.4f", key="research_slippage")
            risk = a3.number_input("Riesgo por trade", 0.001, 0.10, 0.02, 0.001,
                                   format="%.3f", key="research_risk")
            n_jobs = a3.number_input(
                "Procesos por búsqueda", 1, 32, 1, step=1, key="research_n_jobs",
                help="No se fuerza a usar todos los núcleos. Aumentarlo puede saturar CPU/RAM.",
            )
            verify_real = st.checkbox(
                "Verificar candidatos finales con motor real",
                value=True,
                key="research_verify_real",
                help="La evolución sigue usando el motor genético. Al terminar, solo los mejores candidatos del Hall of Fame se comprueban con el mismo BacktestEngine de CapitalQuant.",
            )
            verify_top_n = st.number_input(
                "Candidatos finales a verificar", 1, int(hof_size), min(10, int(hof_size)), 1,
                key="research_verify_top_n",
                disabled=not verify_real,
                help="Limitar la verificación real mantiene el coste controlado. El backtest completo sigue disponible para cualquier estrategia seleccionada.",
            )

    job_count = (
        len(selected_assets)
        * len(selected_objectives)
        * max(1, len(selected_regimes) if regime_mode == "Separar por régimen" else 1)
    )
    st.markdown(
        f"**Investigaciones a ejecutar: {job_count}**  \\n"
        f"{len(selected_assets)} activos × {len(selected_objectives)} objetivos × "
        f"{max(1, len(selected_regimes) if regime_mode == 'Separar por régimen' else 1)} muestras."
    )

    can_start = bool(
        selected_assets
        and selected_objectives
        and (mode == "Todo el histórico" or start <= end)
        and (regime_mode != "Separar por régimen" or selected_regimes)
    )

    st.markdown("### 🔬 Investigación en vivo")
    research_progress = st.progress(
        float(st.session_state.get("research_progress", 0.0)),
        text=st.session_state.get("research_progress_text", "Esperando inicio de investigación..."),
    )
    live_status = st.empty()
    live_overview = st.empty()
    live_chart = st.empty()
    live_table = st.empty()
    live_history = {}
    if not st.session_state.get("last_research_batch"):
        live_status.caption("Sin investigación iniciada. Los resultados se llenarán generación por generación.")
        live_overview.dataframe(pd.DataFrame(columns=["Activo","Régimen","Objetivo","Estado","Progreso","Detalle"]), width="stretch", hide_index=True, height=120)
        live_table.dataframe(pd.DataFrame(columns=["Activo","Objetivo","ID","Fitness","Sharpe","Calmar","PF","Win Rate","CAGR","DD","Trades"]), width="stretch", hide_index=True, height=120)

    if st.button(
        "🧬 INICIAR INVESTIGACIÓN",
        type="primary",
        width="stretch",
        disabled=not can_start,
        key="research_start_button",
    ):
        # Nuevo lote: la interfaz progresiva empieza desde cero, sin borrar
        # los resultados históricos persistidos en disco.
        st.session_state["research_progress"] = 0.0
        st.session_state["research_progress_text"] = "Preparando nueva investigación..."
        # 1) Leer exclusivamente los datasets registrados como LIMPIOS.
        # En lugar de una barra que no muestra qué está pasando, mostramos una
        # tabla viva por activo: estado, velas, rango y mensaje.
        st.markdown("### 📦 Preparación de datos en vivo")
        load_table = st.empty()
        load_rows = []
        preloaded = {}
        load_errors = []

        def _paint_load():
            load_table.dataframe(
                pd.DataFrame(load_rows) if load_rows else pd.DataFrame(
                    columns=["Activo", "Estado", "Barras", "Desde", "Hasta", "Detalle"]
                ),
                width="stretch", hide_index=True
            )

        for a in selected_assets:
            row = {"Activo": a, "Estado": "⏳ cargando", "Barras": "—", "Desde": "—", "Hasta": "—", "Detalle": "Leyendo dataset limpio de Gráficos"}
            load_rows.append(row)
            _paint_load()
            try:
                df_a = (
                    load_raw_dataset(a, timeframe, start, end)
                    if use_raw else
                    load_clean_dataset(a, timeframe, start, end)
                )
                if df_a is None or df_a.empty:
                    raise ValueError(
                        "El dataset publicado no contiene velas en el rango seleccionado."
                    )
                preloaded[a] = df_a
                row.update({
                    "Estado": "✅ listo",
                    "Barras": f"{len(df_a):,}",
                    "Desde": str(df_a.index.min())[:10],
                    "Hasta": str(df_a.index.max())[:10],
                    "Detalle": "Dataset original publicado preparado para la búsqueda" if use_raw else "Dataset limpio preparado para la búsqueda",
                })
            except Exception as exc:
                row.update({"Estado": "❌ error", "Detalle": str(exc)})
                load_errors.append(f"{a}: {exc}")
            _paint_load()

        if load_errors:
            st.error("No se pudo preparar la investigación porque faltan datos limpios:")
            for err in load_errors:
                st.write(f"- {err}")
        else:
            st.success("Todos los datasets limpios están preparados. La tabla de investigación se actualizará con cada generación.")

            cfg = ResearchConfig(
                assets=selected_assets,
                timeframe=timeframe,
                objectives=selected_objectives,
                regime_filters=selected_regimes if regime_mode == "Separar por régimen" else [],
                date_from=start,
                date_to=end,
                population_size=int(population),
                generations=int(generations),
                tournament_size=int(tournament_size),
                elite_fraction=float(elite_fraction),
                crossover_rate=float(crossover_rate),
                mutation_rate=float(mutation_rate),
                random_immigrant_fraction=float(immigrant_fraction),
                min_clauses=int(min_clauses),
                max_clauses=int(max_clauses),
                min_conditions=int(min_conditions),
                max_conditions=int(max_conditions),
                min_trades=int(min_trades),
                target_trades=int(target_trades),
                n_jobs_per_search=int(n_jobs),
                seed=int(seed),
                initial_capital=float(initial_capital),
                hall_of_fame_size=int(hof_size),
                evolve_risk=bool(evolve_risk),
                sl_atr_min=float(sl_range[0]),
                sl_atr_max=float(sl_range[1]),
                tp_atr_min=float(tp_range[0]),
                tp_atr_max=float(tp_range[1]),
                risk_mutation_sigma_frac=float(risk_sigma),
                commission=float(commission),
                slippage=float(slippage),
                risk_per_trade=float(risk),
                use_oos_validation=(int(n_folds) > 1),
                n_folds=int(n_folds),
                min_trades_holdout=(
                    int(min_trades_holdout) if min_trades_holdout is not None else None
                ),
                niche_similarity_threshold=float(niche_similarity_threshold),
                niche_penalty_weight=float(niche_penalty_weight),
                hof_diversity_threshold=float(hof_diversity_threshold),
                verify_with_real_engine=bool(verify_real),
                verify_top_n=int(verify_top_n),
                allow_long=bool(allow_long),
                allow_short=bool(allow_short),
            )


            def _progress(batch_obj, job_obj):
                done_units = sum(
                    1.0 if j.status in ("completed", "failed") else
                    float(getattr(j, "progress", 0.0) or 0.0)
                    for j in batch_obj.jobs
                )
                current = done_units / max(batch_obj.total, 1)
                research_progress.progress(min(max(current, 0.0), 1.0), text=f"Investigación {sum(1 for j in batch_obj.jobs if j.status in ('completed','failed'))}/{batch_obj.total} · {job_obj.asset} · {float(getattr(job_obj, 'progress', 0.0) or 0.0)*100:.1f}%")
                st.session_state["research_progress"] = float(current)
                st.session_state["research_progress_text"] = f"{job_obj.asset} · {job_obj.objective} · {float(getattr(job_obj, 'progress', 0.0) or 0.0)*100:.1f}%"
                obj_spec = OBJECTIVES.get(job_obj.objective)
                obj_label = obj_spec.label if obj_spec else job_obj.objective
                live_status.info(
                    f"**{job_obj.asset} · {obj_label}** · {getattr(job_obj, 'message', job_obj.status)} · "
                    f"investigación {sum(1 for j in batch_obj.jobs if j.status in ('completed','failed'))}/{batch_obj.total}"
                )
                candidates = getattr(job_obj, "live_candidates", None) or []
                job_key = job_obj.job_id
                best_live = max(candidates, key=lambda c: float(c.get("fitness", -999.0))) if candidates else None
                if best_live is not None:
                    live_history.setdefault(job_key, []).append({
                        "Generación": int(round(float(job_obj.progress) * max(int(generations), 1))),
                        "Fitness": float(best_live.get("fitness", -999.0)),
                        "Sharpe": float(best_live.get("sharpe", 0.0) or 0.0),
                    })
                    try:
                        import plotly.graph_objects as go
                        hdf = pd.DataFrame(live_history[job_key]).drop_duplicates("Generación")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=hdf["Generación"], y=hdf["Fitness"], mode="lines+markers", name="Fitness"))
                        fig.add_trace(go.Scatter(x=hdf["Generación"], y=hdf["Sharpe"], mode="lines", name="Sharpe", yaxis="y2"))
                        fig.update_layout(
                            height=300, margin=dict(l=20, r=20, t=40, b=20),
                            title=f"Evolución · {job_obj.asset} · {obj_label}",
                            xaxis_title="Generación", yaxis_title="Fitness",
                            yaxis2=dict(title="Sharpe", overlaying="y", side="right"),
                        )
                        live_chart.plotly_chart(fig, width="stretch")
                    except Exception:
                        pass
                # Estado de TODOS los trabajos: aunque el orquestador los
                # ejecute secuencialmente, el usuario puede ver qué ya terminó,
                # cuál está activo y cuáles siguen en cola.
                overview_rows = []
                for j in batch_obj.jobs:
                    spec = OBJECTIVES.get(j.objective)
                    overview_rows.append({
                        "Activo": j.asset, "Régimen": j.regime or "Todo",
                        "Objetivo": spec.label if spec else j.objective,
                        "Estado": j.status, "Progreso": f"{float(j.progress) * 100:.1f}%",
                        "Detalle": j.message,
                    })
                live_overview.dataframe(pd.DataFrame(overview_rows), width="stretch", hide_index=True, height=min(420, 38 * len(overview_rows) + 40))

                if candidates:
                    rows = []
                    for c in candidates:
                        rows.append({
                            "Activo": job_obj.asset,
                            "Objetivo": obj_label,
                            "ID": c.get("strategy_id"),
                            "Fitness": round(c.get("fitness", 0.0), 4),
                            "Sharpe": round(c.get("sharpe", 0.0), 3),
                            "Calmar": round(c.get("calmar", 0.0), 3),
                            "PF": round(c.get("profit_factor", 0.0), 3),
                            "Win Rate": f"{c.get('win_rate', 0.0)*100:.1f}%",
                            "CAGR": f"{c.get('cagr', 0.0)*100:.1f}%",
                            "DD": f"{c.get('max_drawdown', 0.0)*100:.1f}%",
                            "Trades": c.get("n_trades", 0),
                        })
                    live_table.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                elif job_obj.status == "running":
                    live_table.info("Generando la primera generación válida… los candidatos aparecerán aquí en cuanto cumplan el mínimo de trades.")

            try:
                batch = ResearchOrchestrator(cfg, preloaded_data=preloaded).run(progress=_progress)
                st.session_state["last_research_batch"] = batch.batch_id
                research_progress.progress(1.0, text=f"Research Lab completado · {batch.completed}/{batch.total}")
                st.session_state["research_progress"] = 1.0
                st.session_state["research_progress_text"] = f"Research Lab completado · {batch.completed}/{batch.total}"
                live_status.success(f"Research Lab completado: {batch.completed} de {batch.total} investigaciones.")
                st.success(
                    f"Research Lab {batch.batch_id} finalizado: "
                    f"{batch.completed} completadas, {batch.failed} con error."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Research Lab se detuvo por un error: {exc}")

    batch_id = st.session_state.get("last_research_batch")
    data = load_batch(batch_id) if batch_id else None
    if not data:
        # Recuperación persistente: al volver a Research Lab, o después de
        # recargar la página, se muestra el último lote sin repetir la búsqueda.
        data = load_latest_batch()
        if data:
            batch_id = data.get("batch_id")
            st.session_state["last_research_batch"] = batch_id
    if not batch_id or not data:
        return

    st.markdown(f"## Resultados · {batch_id}")
    _render_results(data)
    _render_strategy_selection(data)

    all_candidates = []
    for job in data.get("jobs", []):
        for cand in (job.get("result", {}) or {}).get("candidates", []):
            if "definition" in cand:
                all_candidates.append({
                    **cand,
                    "asset": job.get("asset"),
                    "timeframe": job.get(
                        "timeframe",
                        data.get("config", {}).get("timeframe", timeframe),
                    ),
                    "regime": job.get("regime"),
                    "objective": job.get("objective"),
                    "research_date_from": data.get("config", {}).get("date_from"),
                    "research_date_to": data.get("config", {}).get("date_to"),
                    "data_source": "Gráficos → datos limpios",
                })

    if not all_candidates:
        st.info(
            "Este lote no contiene definiciones completas de estrategias. "
            "Ejecuta una nueva investigación para habilitar el backtest y guardado."
        )
        return

    st.markdown("## Estrategia seleccionada")
    options = {
        f"{c['asset']} · {c['regime'] or 'Todo'} · "
        f"{OBJECTIVES.get(c['objective'], c['objective']).label if c.get('objective') in OBJECTIVES else c.get('objective')} · "
        f"{c['strategy_id']}": c
        for c in all_candidates
    }
    chosen_key = st.selectbox(
        "Selecciona una estrategia para inspeccionarla",
        list(options.keys()),
        key=f"research_strategy_{batch_id}",
    )
    chosen = options[chosen_key]

    with st.container(border=True):
        st.markdown(f"### {chosen.get('name', chosen['strategy_id'])}")
        st.caption(
            f"{chosen['asset']} · {chosen['timeframe']} · "
            f"{chosen['regime'] or 'Todo el histórico'} · "
            f"objetivo {OBJECTIVES.get(chosen.get('objective'), chosen.get('objective')).label if chosen.get('objective') in OBJECTIVES else chosen.get('objective')} · "
            f"fuente: {chosen.get('data_source', 'Gráficos → datos limpios')}"
        )
        st.markdown(f"**LONG:** {chosen.get('long_rule_text', '')}")
        st.markdown(f"**SHORT:** {chosen.get('short_rule_text', '')}")

        metrics = chosen.get("metrics", {})
        cols = st.columns(9)
        metric_defs = [
            ("Sharpe", "sharpe", 1),
            ("Calmar", "calmar", 1),
            ("PF", "profit_factor", 1),
            ("Win Rate", "win_rate", 100),
            ("CAGR", "cagr", 100),
            ("Expectancy", "expectancy", 1),
            ("Drawdown", "max_drawdown", 100),
            ("Stability", "stability", 1),
            ("Robustness", "robustness", 1),
        ]
        for col, (label, key, mult) in zip(cols, metric_defs):
            v = metrics.get(key, 0)
            col.metric(label, f"{v * mult:.3f}" if isinstance(v, (int, float)) else str(v))
        st.caption(
            f"Trades: **{int(metrics.get('n_trades', 0))}** · "
            f"SL: **{chosen.get('risk', {}).get('stop_loss_atr', chosen.get('definition', {}).get('stop_loss_atr', 2.0)):.2f}× ATR** · "
            f"TP: **{chosen.get('risk', {}).get('take_profit_atr', chosen.get('definition', {}).get('take_profit_atr', 3.0)):.2f}× ATR**"
        )

        st.markdown("#### Acciones")
        a, b = st.columns(2)
        with a:
            if st.button(
                "📊 BACKTEST RÁPIDO — VER ESTADÍSTICAS",
                type="primary",
                key=f"research_bt_{batch_id}_{chosen['strategy_id']}",
            ):
                with st.spinner("Ejecutando backtest real sobre los datos ya cargados..."):
                    try:
                        fr, sig_df, indicator_cols, df_bt = _run_preview(
                            chosen,
                            chosen["asset"],
                            chosen["timeframe"],
                            float(initial_capital) if "initial_capital" in locals() else 10_000.0,
                            float(commission) if "commission" in locals() else 0.001,
                            float(slippage) if "slippage" in locals() else 0.0005,
                            float(risk) if "risk" in locals() else 0.02,
                        )
                        st.session_state[
                            f"research_preview_{batch_id}_{chosen['strategy_id']}"
                        ] = {
                            "results": fr,
                            "sig_df": sig_df,
                            "indicator_cols": indicator_cols,
                            "df": df_bt,
                        }
                    except Exception as exc:
                        st.error(f"No se pudo ejecutar el backtest: {exc}")

        with b:
            if st.button(
                "💾 GUARDAR ESTRATEGIA",
                key=f"research_save_{batch_id}_{chosen['strategy_id']}",
            ):
                try:
                    saved_name = _save_candidate(
                        chosen,
                        chosen["asset"],
                        chosen["timeframe"],
                        chosen.get("regime"),
                    )
                    # Actualizar el candidato dentro del lote persistente para
                    # que la tabla refleje inmediatamente que ya fue publicado.
                    for job in data.get("jobs", []):
                        if job.get("asset") == chosen.get("asset") and job.get("objective") == chosen.get("objective"):
                            for cand in (job.get("result", {}) or {}).get("candidates", []):
                                if cand.get("strategy_id") == chosen.get("strategy_id"):
                                    cand["saved"] = True
                                    cand["saved_name"] = saved_name
                    save_batch(batch_id, data)
                    chosen["saved"] = True
                    st.success(
                        f"Guardada como **{saved_name}**. Quedó registrada en el catálogo de estrategias "
                        "y estará disponible para Backtesting, Optimización, Portfolio y Mercado en Vivo."
                    )
                    # Atajos: dejan preseleccionada la estrategia guardada en
                    # los módulos que tienen selector directo.
                    q1, q2, q3 = st.columns(3)
                    with q1:
                        if st.button("↗ Abrir Backtesting", key=f"research_open_bt_{batch_id}_{chosen['strategy_id']}"):
                            st.session_state["bt_strategy"] = saved_name
                            st.session_state["cq_active_page"] = "Backtesting"
                            st.rerun()
                    with q2:
                        if st.button("↗ Abrir Optimización", key=f"research_open_opt_{batch_id}_{chosen['strategy_id']}"):
                            st.session_state["opt_strat"] = saved_name
                            st.session_state["cq_active_page"] = "Optimización"
                            st.rerun()
                    with q3:
                        if st.button("↗ Abrir Mercado en Vivo", key=f"research_open_live_{batch_id}_{chosen['strategy_id']}"):
                            st.session_state["live_selected_strategy"] = saved_name
                            st.session_state["cq_active_page"] = "Mercado en Vivo"
                            st.rerun()
                except Exception as exc:
                    st.error(f"No se pudo guardar la estrategia: {exc}")

    preview_key = f"research_preview_{batch_id}_{chosen['strategy_id']}"
    preview = st.session_state.get(preview_key)
    if preview:
        st.markdown("## Resultados del backtest")
        _render_preview(chosen, chosen["asset"], chosen["timeframe"], preview)
        st.caption(
            "Este backtest se ejecuta bajo demanda con el motor real de CapitalQuant. "
            "Las métricas, operaciones y curva de equity son independientes de la puntuación "
            "genética."
        )
