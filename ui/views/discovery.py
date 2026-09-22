"""
ui/views/discovery.py
Descubridor Genético — evoluciona combinaciones de indicadores de familias
distintas (RSI, MACD, ADX, Bollinger, Donchian, Estocástico, CCI, Williams %R,
Momentum, Z-Score, Volumen, Medias Móviles) en reglas long/short, en vez de
barrer periodos de una plantilla fija.

Solo hace UNA cosa (descubrir), a propósito: el resto del pipeline
(backtesting real, walk-forward, Monte Carlo, portfolio, optimización de
periodos) ya existe en CapitalQuant y se reutiliza tal cual — por eso, al
encontrar una estrategia, este módulo la traduce a código de una clase
BaseStrategy real (misma exportación que usa el Constructor) y la registra
como una estrategia más, disponible en Backtesting/Optimización/Portfolio.
"""
import time
from datetime import date, timedelta, datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from loguru import logger

from market_data.mt5_provider import get_mt5_provider, history_gap_warning

from discovery.rule_engine import CONDITION_LIBRARY, FAMILY_TAGS
from discovery.indicators import compute_indicator_frame
from discovery.discovery_engine import DiscoveryEngine, DiscoveryConfig, DiscoveryProgress
from discovery.codegen import generate_strategy_code, compile_definition_to_class
from visualization.theme import base_layout, ACCENT, GREEN
from visualization.charts import candlestick_chart, indicators_chart, equity_curve_chart
from ui.components.regime_filter import render_regime_filter, compute_regime_mask
from ui.components.data_source import render_data_source_selector, load_historical_data
from ui.components.metrics_grid import render_metrics_grid
from core.types import BacktestConfig
from engine.backtester import BacktestEngine

from strategies import (
    save_user_strategy_file, refresh_registry, record_discovery_manifest,
    list_discovered_strategies, delete_user_strategy_file, STRATEGY_REGISTRY,
)

from ui.state import AppState


def _show_saved_discoveries_panel() -> None:
    """
    Panel persistente: lee directamente de disco (archivo .py real +
    manifiesto de metadatos), no del estado de sesión. Por eso sigue
    mostrando las estrategias enviadas aunque hayas reiniciado el programa
    por completo — es la prueba de que quedaron guardadas como código,
    no solo en memoria de esta sesión.
    """
    saved = list_discovered_strategies()
    if not saved:
        return
    with st.expander(f"📚 Estrategias del Descubridor ya guardadas ({len(saved)}) — persisten al reiniciar", expanded=False):
        st.caption(
            "Cada una vive como un archivo real en `strategies/`. Por eso aparecen aquí y en los "
            "selectores de Backtesting/Optimización/Portfolio cada vez que arrancas el programa, "
            "sin necesidad de volver a descubrirlas."
        )
        for item in saved:
            c1, c2, c3 = st.columns([3, 2, 1])
            status = "✅ activa" if item["active"] else ("⚠️ archivo no cargó" if item["file_exists"] else "❌ archivo borrado")
            c1.markdown(
                f"**{item.get('class_name', item['stem'])}** — {item.get('asset','?')} · "
                f"{item.get('timeframe','?')} · {status}"
            )
            if item.get("generalization_validated"):
                c1.caption(f"🧪 Generalización validada — {item.get('generalization_verdict', '')}")
            else:
                c1.caption("🧪 Generalización no validada")
            c2.caption(
                f"Sharpe {item.get('sharpe', 0):.2f} · CAGR {item.get('cagr', 0)*100:.1f}% · "
                f"guardada {item.get('saved_at', '—')[:16].replace('T',' ')}"
            )
            if c3.button("Eliminar", key=f"del_disc_{item['stem']}", width='stretch'):
                delete_user_strategy_file(item["stem"])
                st.rerun()


def render(asset: str, timeframe: str) -> None:
    st.markdown("## Descubridor Genético")
    data_source = render_data_source_selector(timeframe, key_prefix="disc_source", compact=True)
    st.caption(f"Fuente histórica activa: **{data_source}**")
    st.caption(
        "Evoluciona reglas que combinan indicadores de distintas familias con AND/OR, seleccionando "
        "por rentabilidad y robustez a través de generaciones. Ajusta los periodos por activo luego, "
        "en **Optimización**."
    )

    _show_saved_discoveries_panel()

    with st.expander("Rango de datos para la búsqueda", expanded=True):
        col1, col2, col3 = st.columns([2, 2, 3])
        date_mode = col1.radio(
            "Modo", ["Rango personalizado", "Máxima historia disponible"],
            key="disc_date_mode",
        )
        use_full_history = (date_mode == "Máxima historia disponible")
        if not use_full_history:
            default_end = date.today()
            default_start = default_end - timedelta(days=365)
            disc_start = col2.date_input("Fecha inicio", value=default_start, key="disc_start")
            disc_end = col2.date_input("Fecha fin", value=default_end, key="disc_end")
        else:
            disc_start = disc_end = None
            col2.info("Se usará toda la historia disponible en MT5 (o caché local).")
        col3.caption(
            "Limitar el rango evita que el genético entrene sobre todo el histórico (útil para "
            "aislar un régimen de mercado, dejar datos fuera de muestra, o simplemente ir más rápido)."
        )

    df = _load_data_range(asset, timeframe, disc_start, disc_end)
    if df is None or df.empty:
        st.warning("No hay datos disponibles para el activo/temporalidad/rango seleccionados.")
        return
    if not use_full_history and disc_start and disc_end:
        df = df.loc[str(disc_start):str(disc_end)]
    if df.empty:
        st.warning("El rango de fechas elegido no tiene velas disponibles.")
        return

    # MT5 solo entrega por API las velas que el terminal ya tiene sincronizadas
    # localmente: si el rango pedido es más antiguo de lo cacheado, se reciben
    # pocas/0 velas aunque el bróker sí tenga ese histórico (ver mt5_provider).
    if not use_full_history:
        gap_msg = history_gap_warning(df, disc_start, asset, timeframe)
        if gap_msg:
            st.warning(gap_msg)

    regime_calib_overrides: dict = {}
    with st.spinner("Calculando banco de indicadores..."):
        feats = compute_indicator_frame(df)
        try:
            from config.settings import TIMEFRAMES
            from regime.calibration import regime_feature_frame
            ann_factor = TIMEFRAMES.get(timeframe, {}).get("ann_factor", 252)
            regime_feats, regime_calib_overrides = regime_feature_frame(df, ann_factor, asset, timeframe)
            feats = pd.concat([feats, regime_feats], axis=1)
        except Exception:
            # Degradación segura: si el cálculo de régimen falla por
            # cualquier motivo, el banco de indicadores sigue funcionando
            # sin las condiciones de la familia "Regime" (rule_engine ya
            # trata esas columnas como opcionales — ver _col_or_false).
            pass

    from discovery.genetic_discovery import WARMUP_BARS
    usable_bars = max(len(df) - WARMUP_BARS, 0)

    st.caption(
        f"Dataset activo: **{asset} {timeframe}** ({len(df)} barras) — "
        f"{feats.shape[1]} indicadores calculados, {len(CONDITION_LIBRARY)} condiciones atómicas "
        f"en {len(FAMILY_TAGS)} familias: {', '.join(FAMILY_TAGS)}."
    )
    st.caption(
        f"Barras utilizables tras el warmup del banco de indicadores (SMA-200 y similares "
        f"necesitan {WARMUP_BARS} velas antes de poder generar señal): **{usable_bars}**."
    )
    if usable_bars <= 0:
        st.error(
            f"El rango elegido no tiene suficientes velas: se necesitan más de {WARMUP_BARS} "
            f"solo para el warmup de los indicadores. Amplía el rango de fechas o usa una "
            f"temporalidad más fina."
        )

    with st.expander("Descubrir por régimen de mercado", expanded=False):
        st.caption(
            "Restringe la búsqueda a un régimen específico (mismo detector causal que "
            "**Régimen de Mercado**): el genético solo cuenta como trade — para efectos de "
            "fitness Y de la verificación con el motor real — las entradas que caen dentro "
            "del/los régimen(es) elegido(s). Esto empuja al genético a encontrar reglas "
            "afinadas específicamente para ese contexto (p.ej. reversión a la media que solo "
            "necesita funcionar en Consolidación), en vez de una regla genérica promediada "
            "sobre todos los regímenes mezclados. El histórico completo se sigue usando para "
            "calcular indicadores (ningún warmup se recorta); solo se descarta la señal fuera "
            "del régimen elegido."
        )
        allowed_regimes = render_regime_filter(asset, timeframe, key_prefix="disc")
        regime_mask = compute_regime_mask(df, timeframe, allowed_regimes, asset=asset) if allowed_regimes else None
        if allowed_regimes:
            if regime_mask is not None:
                mask_aligned = regime_mask.reindex(df.index).fillna(False)
                pct_in_regime = float(mask_aligned.mean()) * 100
                st.caption(
                    f"**{pct_in_regime:.1f}%** de las velas del rango elegido caen en "
                    f"{'/'.join(allowed_regimes)}. Si el número de trades exigido "
                    f"('Mínimo de trades para ser válida') es alto respecto a este porcentaje, "
                    f"pocas o ninguna regla lo alcanzará — considera bajarlo o ampliar el rango "
                    f"de fechas."
                )
                preview_folds = 3
                from discovery.genetic_discovery import _fold_boundaries as _preview_fold_boundaries
                bounds_preview = _preview_fold_boundaries(
                    len(df), preview_folds, 0, regime_active=mask_aligned.to_numpy(),
                )
                fold_counts = [int(mask_aligned.to_numpy()[s:e].sum()) for _, s, e in bounds_preview]
                min_fold_count = min(fold_counts) if fold_counts else 0
                st.caption(
                    f"Reparto de referencia con {preview_folds} ventanas walk-forward (por "
                    f"ocurrencias de régimen, no por calendario): **{fold_counts}** velas en régimen "
                    f"por ventana. Si vas a usar {preview_folds} ventanas y pides más de "
                    f"**{min_fold_count}** trades por ventana, ninguna regla podrá pasar la ventana "
                    f"más escasa — ajusta 'Ventanas walk-forward' o el mínimo de trades por ventana "
                    f"más abajo en consecuencia."
                )
            else:
                st.warning("No se pudo calcular el régimen; la búsqueda continuará sin filtro.")

    with st.form("discovery_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Población y generaciones**")
            pop_size = st.number_input("Tamaño de población", 20, 500, 100, step=10)
            n_gen = st.number_input("Número de generaciones", 3, 100, 25)
            objective_options = [
                ("composite", "Composite — balance general"),
                ("high_winrate_frequency", "High Win Rate + Frequency"),
                ("high_profitability", "High Profitability"),
                ("sharpe", "Sharpe"),
                ("calmar", "Calmar"),
                ("profit_factor", "Profit Factor"),
                ("win_rate", "Win Rate"),
                ("cagr", "CAGR"),
                ("expectancy", "Expectancy"),
                ("drawdown", "Drawdown"),
                ("stability", "Stability"),
                ("robustness", "Robustness"),
            ]
            objective_labels = [label for _, label in objective_options]
            objective_by_label = {label: key for key, label in objective_options}
            fitness_label = st.selectbox(
                "Objetivo de fitness", objective_labels, index=0,
                key="disc_fitness_objective",
                help=(
                    "El objetivo elegido es la función que guía la selección genética. "
                    "Ahora están disponibles todos los objetivos del Research Lab: Calmar, Sharpe, "
                    "Profit Factor, Win Rate, CAGR, Expectancy, Drawdown, Stability y Robustness, "
                    "además de los objetivos históricos Composite, High Win Rate + Frequency y "
                    "High Profitability."
                ),
            )
            fitness_obj = objective_by_label[fitness_label]
            direction = st.radio(
                "Dirección a buscar", ["Ambos (long y short)", "Solo long", "Solo short"],
                index=0, horizontal=True,
                help="Restringe qué lado evoluciona el genético. 'Solo long' o 'Solo short' no es solo "
                     "un filtro posterior: la población entera deja de generar/mutar reglas del lado "
                     "descartado, así que toda la búsqueda se concentra en el lado que sí interesa.",
            )
            allow_long = direction != "Solo short"
            allow_short = direction != "Solo long"
            evolve_risk = st.checkbox(
                "Evolucionar Stop Loss / Take Profit junto con la regla", value=True,
                help="Cada individuo trae su propio SL/TP (en múltiplos de ATR) como parte de lo "
                     "que el genético optimiza, en vez de forzar el mismo valor fijo a todas las reglas.",
            )
            if evolve_risk:
                sl_range = st.slider("Rango de Stop Loss (múltiplos ATR)", 0.3, 6.0, (0.8, 4.0), 0.1)
                tp_range = st.slider("Rango de Take Profit (múltiplos ATR)", 0.5, 12.0, (1.0, 8.0), 0.1)
            else:
                sl_range, tp_range = (2.0, 2.0), (3.0, 3.0)
        with col2:
            st.markdown("**Estructura de reglas**")
            max_clauses = st.slider("Máximo de cláusulas OR", 1, 10, 2,
                                     help="Más cláusulas OR = más caminos de entrada distintos.")
            max_conditions = st.slider("Máximo de condiciones AND por cláusula", 1, 4, 3)
            min_trades = st.number_input("Mínimo de trades para ser válida", 5, 2000, 30, step=5)
            target_trades = st.number_input("Trades objetivo (para puntuar frecuencia)", 20, 5000, 150, step=10)
            if usable_bars > 0:
                implied_ratio = usable_bars / min_trades
                if implied_ratio < 4:
                    st.warning(
                        f"⚠️ Con {usable_bars} velas utilizables y min_trades={min_trades}, le estás "
                        f"pidiendo a las reglas disparar en promedio cada {implied_ratio:.1f} velas — "
                        f"muy exigente para reglas con varias condiciones AND. Es probable que ningún "
                        f"individuo sea válido (fitness -999 en todas las generaciones). Amplía el rango "
                        f"de fechas, usa una temporalidad más fina, o baja este mínimo."
                    )
        with col3:
            st.markdown("**Reproducibilidad**")
            hof_size = st.number_input("Tamaño del Hall of Fame", 10, 200, 60, step=10)
            seed = st.number_input("Semilla aleatoria", 0, 999_999, 42)
            n_folds = st.number_input(
                "Ventanas walk-forward", 1, 8, 3,
                help="El dataset se parte en N ventanas; una regla solo llega al Hall of Fame si es "
                     "rentable en CADA ventana, no solo en promedio (estándar de facto en validación "
                     "de estrategias — Pardo 1992/2008, López de Prado 2018). Si hay un filtro de "
                     "régimen activo, las ventanas se reparten por OCURRENCIAS del régimen (no por "
                     "calendario), para que ninguna ventana se quede casi sin velas válidas solo "
                     "porque ese régimen se concentró en un tramo del historial. Con 1 ventana se "
                     "evalúa sobre todo el rango sin garantía fuera de muestra: útil solo para "
                     "exploración rápida, no para el Hall of Fame final.",
            )
            auto_min_fold = max(5, int(min_trades) // max(1, int(n_folds)))
            override_min_fold = st.checkbox(
                f"Fijar mínimo de trades por ventana manualmente (automático ahora: {auto_min_fold})",
                value=False,
                help="Por defecto, cada ventana walk-forward exige max(5, mínimo_total // n_ventanas) "
                     "trades — con 'Mínimo de trades para ser válida'=5 y 3 ventanas, eso es 5 trades "
                     "POR VENTANA (15 en total), no 5 en total. Actívalo para fijar tú ese número "
                     "directamente, por ejemplo si quieres relajarlo a 3 por ventana.",
            )
            min_trades_holdout = (
                st.number_input("Mínimo de trades por ventana", 1, 500, auto_min_fold, step=1)
                if override_min_fold else None
            )

        with st.expander("⚙️ Avanzado — selección y diversidad", expanded=False):
            st.caption(
                "El Hall of Fame ya evita por diseño llenarse de variantes casi idénticas: cada "
                "generación penaliza el fitness de selección de un individuo según cuántos vecinos "
                "parecidos (mismas condiciones, Jaccard) tiene esa misma generación, y al armar el "
                "Hall of Fame final se descarta un candidato si es demasiado parecido a uno ya "
                "admitido de mejor fitness. Estos controles ajustan qué tan agresivo es eso — no "
                "hace falta tocarlos para una corrida normal."
            )
            adv1, adv2 = st.columns(2)
            tournament_size = adv1.number_input(
                "Tamaño del torneo de selección", 2, 10, 4,
                help="Cuántos individuos compiten por cada 'padre' elegido para cruce/mutación. Más "
                     "alto = más presión hacia los mejores (converge más rápido, pero puede perder "
                     "diversidad antes). Más bajo = selección más aleatoria, explora más.",
            )
            niche_similarity_threshold = adv1.slider(
                "Umbral de similitud (Jaccard) para considerar 'vecinos'", 0.3, 0.9, 0.6, step=0.05,
                help="Dos individuos se consideran de la 'misma familia de reglas' si comparten esta "
                     "fracción o más de sus condiciones. Más bajo = el sistema es más estricto "
                     "considerando parecidas incluso a reglas con pocas condiciones en común.",
            )
            niche_penalty_weight = adv2.slider(
                "Penalización por vecino similar", 0.0, 0.3, 0.06, step=0.01,
                help="Cuánto fitness se resta (solo para el torneo de selección, no para el ranking "
                     "final) por cada vecino parecido en la misma generación. 0 = desactiva el "
                     "niching por completo (vuelve al comportamiento de convergencia rápida sin "
                     "penalizar aglomeración).",
            )
            hof_diversity_threshold = adv2.slider(
                "Similitud máxima permitida en el Hall of Fame final", 0.3, 0.9, 0.6, step=0.05,
                help="Al armar el Hall of Fame final, un candidato se descarta si su Jaccard de "
                     "condiciones con alguno ya admitido (de mejor fitness) supera este valor. Más "
                     "bajo = Hall of Fame más diverso pero potencialmente con estrategias de menor "
                     "fitness individual; más alto = prioriza fitness puro sobre diversidad.",
            )
            verify_real = st.checkbox(
                "Verificar Hall of Fame con el motor real al terminar", value=True,
                help="El backtest que puntúa cada individuo durante la evolución es un simulador "
                     "rápido, usado solo para la búsqueda (miles de evaluaciones). Con esta opción "
                     "activa, al terminar la evolución cada individuo del Hall of Fame se vuelve a "
                     "correr con el motor real de CapitalQuant (engine/backtester.py) sobre el mismo "
                     "rango de fechas, y las métricas que ves abajo son las del motor real — las "
                     "mismas que obtendrás al enviar la estrategia a Backtesting.",
            )
            st.caption(
                "⚠️ Probar miles de combinaciones (como hace este motor) encuentra Sharpes altos por "
                "puro azar aunque ninguna tenga ventaja real — cuantas más se prueban, más alto es el "
                "mejor Sharpe esperable sin ninguna habilidad. La columna **DSR** (Deflated Sharpe "
                "Ratio, Bailey & López de Prado 2014) corrige esto: es la probabilidad de que el Sharpe "
                "verificado sea genuino y no solo el resultado de haber probado muchas combinaciones. "
                "≥95% se considera estadísticamente significativo."
            )

        submitted = st.form_submit_button("🧬 Ejecutar descubrimiento evolutivo", width='stretch')

    st.markdown("### 🔬 Evolución genética en vivo")
    discovery_progress_bar = st.progress(float(st.session_state.get("discovery_progress", 0.0)), text=st.session_state.get("discovery_progress_text", "Esperando inicio de descubrimiento..."))
    discovery_live_status = st.empty()
    discovery_live_chart = st.empty()
    discovery_live_table = st.empty()
    discovery_live_best = st.empty()
    if not st.session_state.get("discovery_result"):
        discovery_live_status.caption("Sin descubrimiento iniciado. La tabla y la evolución se llenarán durante las generaciones.")
        discovery_live_table.dataframe(pd.DataFrame(columns=["Generación","ID","Fitness","Sharpe","Calmar","PF","Win Rate","CAGR","DD","Trades"]), width="stretch", hide_index=True, height=120)

    if submitted:
        # Un nuevo descubrimiento empieza limpio, pero conserva la interfaz
        # progresiva: la tabla y el gráfico existentes se reemplazan por los
        # nuevos snapshots desde la primera generación.
        st.session_state["discovery_result"] = None
        st.session_state["discovery_running"] = True
        discovery_progress_bar.progress(0.0, text="Preparando evolución genética...")
        cfg = DiscoveryConfig(
            population_size=int(pop_size), n_generations=int(n_gen),
            fitness_objective=fitness_obj, allow_short=allow_short, allow_long=allow_long,
            max_clauses=int(max_clauses), max_conditions=int(max_conditions),
            min_trades=int(min_trades), target_trades=int(target_trades),
            hall_of_fame_size=int(hof_size), seed=int(seed), n_jobs=-1,
            evolve_risk=bool(evolve_risk),
            sl_atr_min=float(sl_range[0]), sl_atr_max=float(sl_range[1]),
            tp_atr_min=float(tp_range[0]), tp_atr_max=float(tp_range[1]),
            n_folds=int(n_folds), use_oos_validation=(int(n_folds) > 1),
            min_trades_holdout=(int(min_trades_holdout) if min_trades_holdout is not None else None),
            tournament_size=int(tournament_size),
            niche_similarity_threshold=float(niche_similarity_threshold),
            niche_penalty_weight=float(niche_penalty_weight),
            hof_diversity_threshold=float(hof_diversity_threshold),
            verify_with_real_engine=bool(verify_real),
        )
        live_status = discovery_live_status
        live_chart = discovery_live_chart
        live_table = discovery_live_table
        live_best = discovery_live_best
        live_history = []

        def _cb(p: DiscoveryProgress):
            pct = float(p.current) / max(float(p.total), 1.0)
            discovery_progress_bar.progress(min(max(pct, 0.0), 1.0), text=f"{p.stage.upper()} · {p.current}/{p.total} · {p.message}")
            st.session_state["discovery_progress"] = min(max(pct, 0.0), 1.0)
            st.session_state["discovery_progress_text"] = f"{p.stage.upper()} · {p.current}/{p.total} · {p.message}"
            live_status.info(f"**{p.stage.upper()}** · {p.message}")
            candidates = getattr(p, "top_candidates", None) or []
            if p.stage == "evolve":
                best = max(candidates, key=lambda c: float(c.get("fitness", -999.0))) if candidates else None
                if best is not None:
                    live_history.append({
                        "Generación": int(p.current),
                        "Mejor fitness": float(best.get("fitness", -999.0)),
                        "Sharpe": float(best.get("sharpe", 0.0) or 0.0),
                        "Calmar": float(best.get("calmar", 0.0) or 0.0),
                        "PF": float(best.get("profit_factor", 0.0) or 0.0),
                    })
                    live_best.metric(
                        "Mejor estrategia encontrada hasta ahora",
                        str(best.get("strategy_id", "—")),
                        delta=f"Fitness {float(best.get('fitness', 0.0)):.4f}",
                    )
                    try:
                        import plotly.graph_objects as go
                        hdf = pd.DataFrame(live_history).drop_duplicates("Generación")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=hdf["Generación"], y=hdf["Mejor fitness"],
                            mode="lines+markers", name="Mejor fitness",
                        ))
                        fig.add_trace(go.Scatter(
                            x=hdf["Generación"], y=hdf["Sharpe"],
                            mode="lines", name="Sharpe", yaxis="y2",
                        ))
                        fig.update_layout(
                            height=320, margin=dict(l=20, r=20, t=45, b=20),
                            title="Evolución en tiempo real",
                            xaxis_title="Generación",
                            yaxis_title="Fitness",
                            yaxis2=dict(title="Sharpe", overlaying="y", side="right"),
                        )
                        live_chart.plotly_chart(fig, width="stretch")
                    except Exception:
                        pass

                if candidates:
                    rows = []
                    for c in candidates:
                        rows.append({
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
                elif p.current == 1:
                    live_table.info("Generación 1 completada sin candidatos válidos; la evolución continúa con la siguiente generación.")

        t0 = time.time()
        engine_disc = DiscoveryEngine(cfg)
        result = engine_disc.run(df, feats, asset, timeframe, progress_callback=_cb,
                                  regime_mask=regime_mask)
        elapsed = time.time() - t0
        result["discovered_regimes"] = allowed_regimes or []
        if regime_calib_overrides:
            # Cualquier estrategia del Hall of Fame que haya usado la familia
            # "Regime" (condición atómica, no el filtro manual) necesita esta
            # calibración congelada para que el código exportado reproduzca
            # el MISMO régimen visto aquí, en vez de recalibrar por su cuenta.
            for entry in result.get("leaderboard", []):
                if "Regime" in entry.get("families_used", []):
                    entry["definition"]["regime_calib_overrides"] = regime_calib_overrides
        st.session_state["discovery_result"] = result
        st.session_state["discovery_running"] = False
        st.session_state["discovery_progress"] = 1.0
        st.session_state["discovery_progress_text"] = f"Descubrimiento completado · {result['n_survivors']} estrategias"
        discovery_progress_bar.progress(1.0, text=st.session_state["discovery_progress_text"])
        live_status.success(f"Investigación terminada en {elapsed:.1f}s. Hall of Fame: {result['n_survivors']} estrategias.")
        regime_note = (
            f" — optimizada y verificada solo dentro de: {'/'.join(allowed_regimes)}"
            if allowed_regimes else ""
        )
        st.success(
            f"Descubrimiento completado en {elapsed:.1f}s: {result['n_generated']} evaluaciones "
            f"genéticas → {result['n_survivors']} estrategias distintas en el hall of fame"
            f"{regime_note}."
        )

    result = st.session_state.get("discovery_result")
    if not result:
        return

    if result.get("generation_stats"):
        st.subheader("📈 Evolución del fitness por generación")
        gs = pd.DataFrame([{
            "generacion": s.generation, "mejor_fitness": s.best_fitness,
            "fitness_promedio": s.mean_fitness, "hall_of_fame": s.hall_of_fame_size,
        } for s in result["generation_stats"]])
        fig = px.line(gs, x="generacion", y=["mejor_fitness", "fitness_promedio"],
                      markers=True, color_discrete_sequence=[ACCENT, GREEN])
        fig.update_layout(base_layout(title="Convergencia de la búsqueda evolutiva", height=350))
        fig.update_traces(mode="lines+markers", marker=dict(size=5))
        st.plotly_chart(fig, width='stretch')

    st.subheader("🏆 Hall of Fame — mejores combinaciones descubiertas")
    n_verified = sum(1 for r in result["leaderboard"] if r.get("verified"))
    n_total = len(result["leaderboard"])
    if n_verified == n_total:
        st.caption(f"✅ Las {n_total} estrategias fueron verificadas con el motor real de Backtesting "
                    f"sobre el mismo rango de fechas. Las métricas de abajo son las del motor real.")
    elif n_verified > 0:
        st.warning(f"{n_verified}/{n_total} estrategias verificadas con el motor real. Las "
                   f"{n_total - n_verified} restantes muestran la estimación rápida interna "
                   f"(columna 'verificado' = No) porque la verificación falló para ellas — revisa "
                   f"esas reglas con cuidado antes de operarlas.")
    else:
        st.info("Verificación con motor real desactivada: las métricas de abajo son la estimación "
                "rápida interna y pueden diferir de lo que veas en Backtesting.")

    if result.get("n_trials"):
        st.caption(
            f"🔬 Esta corrida evaluó **{result['n_trials']:,} combinaciones** en total "
            f"({result.get('n_folds', 1)} ventanas walk-forward cada una) — la columna **DSR** de "
            f"abajo ya descuenta el Sharpe de cada estrategia por ese número de intentos. Reportar "
            f"cuántas combinaciones se probaron es tan importante como el resultado en sí (Bailey & "
            f"López de Prado, 2014) — un Sharpe alto encontrado tras miles de intentos es mucho menos "
            f"sorprendente que el mismo Sharpe encontrado tras unos pocos."
        )

    if result.get("fdr_summary"):
        st.caption(
            f"🧮 {result['fdr_summary']} A diferencia del DSR (que corrige cada estrategia por "
            f"cuántas combinaciones probó el genético para llegar a ella), esta corrección de "
            f"**Benjamini-Hochberg** mira el Hall of Fame **completo** a la vez y ajusta el listón "
            f"de significancia para que, entre todas las marcadas 'FDR ✅', no más de un "
            f"{result['fdr_alpha']*100:.0f}% se espere que sean falsos descubrimientos por puro azar."
        )

    def _fdr_label(r: dict) -> str:
        sig = r.get("fdr_significant")
        if sig is None:
            return "—"
        return "✅" if sig else "❌"

    def _dsr_label(r: dict) -> str:
        dsr = r.get("dsr")
        if dsr is None:
            return "—"
        pct = dsr * 100
        if dsr >= 0.95:
            return f"✅ {pct:.1f}%"
        if dsr >= 0.5:
            return f"⚠️ {pct:.1f}%"
        return f"❌ {pct:.1f}%"

    def _folds_label(r: dict) -> str:
        ff = r.get("fold_fitnesses")
        if not ff:
            return "—"
        n_pos = sum(1 for f in ff if f is not None and f > -999.0 and f > 0)
        return f"{n_pos}/{len(ff)} rentables"

    lb = pd.DataFrame([{
        "name": r["name"],
        "familias": " + ".join(r["families_used"]),
        "pareto": "🏆 óptima" if r.get("pareto_optimal") else "",
        "verificado": "Sí" if r.get("verified") else "No (estimación rápida)",
        "dsr": _dsr_label(r),
        "fdr": _fdr_label(r),
        "folds": _folds_label(r),
        "sharpe": round(r["metrics"]["sharpe"], 3),
        "sortino": round(r["metrics"]["sortino"], 3),
        "calmar": round(r["metrics"]["calmar"], 3),
        "profit_factor": round(r["metrics"]["profit_factor"], 3),
        "win_rate_%": round(r["metrics"]["win_rate"] * 100, 1),
        "n_trades": r["metrics"]["n_trades"],
        "cagr_%": round(r["metrics"]["cagr"] * 100, 1),
        "max_dd_%": round(r["metrics"]["max_drawdown"] * 100, 1),
        "sl_atr": round(r["definition"].get("stop_loss_atr", 2.0), 2),
        "tp_atr": round(r["definition"].get("take_profit_atr", 3.0), 2),
        "fitness": round(r["composite_score"], 3),
        "strategy_id": r["strategy_id"],
    } for r in result["leaderboard"]])
    st.dataframe(lb, width='stretch', hide_index=True, height=350)
    st.caption(
        "🏆 **pareto**: nadie en esta tabla la supera al mismo tiempo en Sharpe Y en drawdown — no "
        "siempre es la fila de mayor Sharpe absoluto, sino la de mejor trade-off riesgo/retorno."
    )

    st.subheader("📤 Enviar una estrategia a Backtesting")
    st.caption(
        "Compila la regla descubierta en una estrategia real de CapitalQuant (misma clase "
        "BaseStrategy que usa el Constructor). Quedará disponible al instante en Backtesting, "
        "Optimización y Portfolio, con sus indicadores, señales y niveles de SL/TP graficados."
    )
    if result.get("discovered_regimes"):
        st.warning(
            f"⚠️ Este Hall of Fame se buscó y verificó **solo dentro de: "
            f"{'/'.join(result['discovered_regimes'])}**. El código exportado NO trae ese "
            f"filtro incorporado por sí solo — para que se comporte igual en producción, "
            f"activa el mismo filtro de régimen (\"Filtrar por régimen de mercado\") en "
            f"Backtesting/Optimización con {'/'.join(result['discovered_regimes'])} "
            f"seleccionado(s) antes de operarla."
        )
    options = {r["strategy_id"]: r for r in result["leaderboard"]}
    chosen_id = st.selectbox(
        "Estrategia descubierta", list(options.keys()),
        format_func=lambda sid: options[sid]["name"],
    )
    chosen = options[chosen_id]

    with st.expander(f"Detalle — {chosen['name']}", expanded=True):
        st.markdown(f"**Regla LONG:** {chosen['long_rule_text']}")
        st.markdown(f"**Regla SHORT:** {chosen['short_rule_text']}")
        d = chosen["definition"]
        st.caption(
            f"Familias combinadas: {', '.join(chosen['families_used'])} — "
            f"SL evolucionado: {d.get('stop_loss_atr', 2.0):.2f}× ATR, "
            f"TP evolucionado: {d.get('take_profit_atr', 3.0):.2f}× ATR."
        )
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        fast = chosen.get("metrics_fast") or {}
        c1.metric("Sharpe", f"{chosen['metrics']['sharpe']:.2f}",
                   delta=(f"{chosen['metrics']['sharpe'] - fast.get('sharpe', 0):.2f} vs. rápido" if fast else None))
        c2.metric("Sortino", f"{chosen['metrics']['sortino']:.2f}")
        c3.metric("CAGR", f"{chosen['metrics']['cagr']*100:.1f}%",
                   delta=(f"{(chosen['metrics']['cagr'] - fast.get('cagr', 0))*100:.1f}% vs. rápido" if fast else None))
        c4.metric("Max Drawdown", f"{chosen['metrics']['max_drawdown']*100:.1f}%")
        dsr_val = chosen.get("dsr")
        c5.metric("DSR (no-azar)", f"{dsr_val*100:.1f}%" if dsr_val is not None else "—",
                   help="Probabilidad de que el Sharpe verificado sea genuino y no el resultado de "
                        "haber probado muchas combinaciones (Deflated Sharpe Ratio). ≥95% = significativo.")
        fdr_sig = chosen.get("fdr_significant")
        fdr_p = chosen.get("fdr_adjusted_p")
        c6.metric("FDR (Hall of Fame)",
                   "✅ Significativa" if fdr_sig else ("❌ No significativa" if fdr_sig is not None else "—"),
                   delta=(f"p-adj={fdr_p:.3f}" if fdr_p is not None else None),
                   help="Corrección de Benjamini-Hochberg aplicada sobre TODO el Hall of Fame a la "
                        "vez (no solo esta estrategia) — controla qué fracción de las estrategias "
                        "marcadas como significativas se espera que sean falsos descubrimientos.")

        fold_fitnesses = chosen.get("fold_fitnesses")
        if fold_fitnesses:
            folds_str = "  ·  ".join(
                f"Ventana {i+1}: {'✅' if f > 0 else '❌'} {f:.2f}" for i, f in enumerate(fold_fitnesses)
            )
            st.caption(f"**Consistencia walk-forward** ({len(fold_fitnesses)} ventanas secuenciales): {folds_str}")

        if chosen.get("verified"):
            st.caption(
                f"✅ Verificado con el motor real (engine/backtester.py). Estimación rápida durante "
                f"la evolución: Sharpe={fast.get('sharpe', 0):.2f}, CAGR={fast.get('cagr', 0)*100:.1f}%, "
                f"trades={fast.get('n_trades', 0)} — la diferencia frente a las cifras de arriba es "
                f"esperable e ilustra por qué esta verificación existe."
            )
        else:
            st.warning(
                "⚠️ No se pudo verificar esta estrategia con el motor real (revisa los logs). Las "
                "cifras de arriba son la estimación rápida interna, no lo que verás en Backtesting."
            )

        st.divider()
        st.markdown("##### Resultados completos del backtest")
        st.caption(
            "Gráfico de precio con las operaciones marcadas, indicadores en un gráfico aparte (para "
            "no tapar las entradas/salidas), todas las métricas y la curva de equity — el mismo "
            "detalle que verías en Backtesting."
        )
        if st.button("📊 Generar resultados completos", key=f"full_result_btn_{chosen_id}"):
            with st.spinner("Backtesteando la estrategia completa..."):
                try:
                    preview_class = compile_definition_to_class(
                        chosen["definition"], class_name="PreviewGenetico", symbol=asset, timeframe=timeframe,
                    )
                    preview_strategy = preview_class()
                    sig_df = preview_strategy.generate_signals(df)
                    bt_config = BacktestConfig(
                        initial_capital=100_000.0, commission=0.001, risk_per_trade=0.02,
                        allow_long=bool(chosen["definition"].get("long_rule")),
                        allow_short=bool(chosen["definition"].get("short_rule")),
                    )
                    full_results = BacktestEngine(bt_config).run(
                        sig_df, strategy_name=chosen["name"], asset=asset, timeframe=timeframe,
                    )
                    st.session_state[f"full_result_{chosen_id}"] = {
                        "results": full_results, "sig_df": sig_df,
                        "indicator_cols": preview_strategy.get_indicator_columns(),
                    }
                except Exception as e:
                    st.error(f"No se pudo generar el detalle completo: {e}")

        full_state = st.session_state.get(f"full_result_{chosen_id}")
        if full_state:
            fr = full_state["results"]
            render_metrics_grid(fr.to_dict(), n_cols=4)

            sig_df = full_state["sig_df"]
            df_viz = df.loc[fr.start_date:fr.end_date]
            fig_price = candlestick_chart(
                df=df_viz, title=f"{asset} {timeframe} — {chosen['name']}",
                trades_df=fr.trades_df if not fr.trades_df.empty else None,
                height=480,
            )
            st.plotly_chart(fig_price, width='stretch')

            indicators = {c: sig_df[c] for c in full_state["indicator_cols"] if c in sig_df.columns}
            if indicators:
                st.plotly_chart(indicators_chart(indicators, height=260), width='stretch')

            curves = [{"name": chosen["name"], "equity": fr.equity_curve,
                       "drawdown": fr.drawdown_series, "best": True}]
            st.plotly_chart(equity_curve_chart(curves, height=350), width='stretch')

            if fr.trades_df is not None and not fr.trades_df.empty:
                st.dataframe(fr.trades_df, width='stretch', hide_index=True, height=300)

    st.subheader("🧪 Validar generalización (recomendado antes de enviar)")
    st.caption(
        "Compila esta regla en una estrategia real y la reejecuta, con los MISMOS parámetros "
        "(sin volver a optimizar), sobre otros símbolos y sobre distintos meses/regímenes del "
        "propio histórico del símbolo de origen — para detectar si la regla memorizó ruido del "
        "símbolo/periodo donde se descubrió en vez de encontrar una señal real de mercado."
    )
    universe = [s for s in _symbol_universe(timeframe) if s != asset]
    gen_symbols = st.multiselect(
        "Símbolos para la prueba de generalización cruzada (recomendado: 8-15)",
        universe, default=universe[: min(12, len(universe))],
        key=f"gen_symbols_{chosen_id}",
    )
    if st.button("🧪 Validar generalización", key=f"gen_btn_{chosen_id}"):
        _run_generalization_validation(chosen, asset, timeframe, gen_symbols)

    gen_state = st.session_state.get(f"gen_report_{chosen_id}")
    if gen_state:
        _show_generalization_report(gen_state)

    default_name = f"Genetica_{asset}_{timeframe}_{chosen_id}".replace(" ", "_").replace("/", "_")
    class_name = st.text_input("Nombre de la clase / estrategia", value=default_name)

    if st.button("📤 Enviar a Backtesting", type="primary", width='stretch'):
        try:
            safe_class = "".join(ch for ch in class_name if ch.isalnum() or ch == "_") or "EstrategiaGenetica"
            if safe_class[0].isdigit():
                safe_class = "E_" + safe_class
            code = generate_strategy_code(chosen["definition"], class_name=safe_class,
                                           symbol=asset, timeframe=timeframe)
            file_stem = safe_class.lower()
            path = save_user_strategy_file(file_stem, code)
            gen_state = st.session_state.get(f"gen_report_{chosen_id}")
            manifest = {
                "class_name": safe_class,
                "asset": asset,
                "timeframe": timeframe,
                "families": chosen.get("families_used", []),
                "long_rule_text": chosen.get("long_rule_text", ""),
                "short_rule_text": chosen.get("short_rule_text", ""),
                "definition": chosen.get("definition", {}),
                "sharpe": chosen["metrics"]["sharpe"],
                "cagr": chosen["metrics"]["cagr"],
                "dsr": chosen.get("dsr"),
                "fdr_significant": chosen.get("fdr_significant"),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "generalization_validated": gen_state is not None,
            }
            if gen_state:
                manifest["generalization_rate"] = gen_state["generalization"].generalization_rate
                manifest["generalization_verdict"] = gen_state["generalization"].verdict()
            record_discovery_manifest(file_stem, manifest)
            refresh_registry()
            st.session_state["discovery_sent_ok"] = safe_class
            if gen_state:
                st.success(
                    f"Estrategia **{safe_class}** enviada y **guardada de forma permanente** "
                    f"(validada — {gen_state['generalization'].verdict()}) en "
                    f"`strategies/{path.name}` — ya aparece en el selector de estrategias de "
                    f"Backtesting, Optimización y Portfolio, y seguirá ahí cada vez que reinicies "
                    f"el programa (queda escrita como código real, no solo en esta sesión)."
                )
            else:
                st.success(
                    f"Estrategia **{safe_class}** enviada y **guardada de forma permanente** en "
                    f"`strategies/{path.name}` — ya aparece en el selector de estrategias de "
                    f"Backtesting, Optimización y Portfolio, y seguirá ahí cada vez que reinicies "
                    f"el programa (queda escrita como código real, no solo en esta sesión)."
                )
                st.info(
                    "No corriste la validación de generalización antes de enviarla — la estrategia "
                    "quedó guardada igual (no es obligatorio), pero considera correr **🧪 Validar "
                    "generalización** arriba antes de confiar en ella para operar."
                )
        except Exception as e:
            st.error(f"No se pudo generar/enviar la estrategia: {e}")


def _load_data_range(asset: str, timeframe: str, date_from, date_to):
    return load_historical_data(asset, timeframe, date_from=date_from, date_to=date_to)


def _symbol_universe(timeframe: str | None = None) -> list:
    """Universo exclusivamente local para validación cruzada."""
    from research.csv_registry import list_csv_datasets
    from research.data_registry import available_clean_assets
    metas = list_csv_datasets(timeframe) if timeframe else list_csv_datasets()
    assets = sorted({m.get("asset") for m in metas if m.get("asset")})
    clean = available_clean_assets(timeframe) if timeframe else available_clean_assets("1d")
    return sorted(set(assets) | set(clean))


def _run_generalization_validation(chosen: dict, origin_asset: str, timeframe: str, gen_symbols: list) -> None:
    """
    Compila la regla elegida del Hall of Fame a una clase BaseStrategy real
    (mismo camino que usa el Constructor), descarga/carga datos de los
    símbolos elegidos vía el MarketDataProvider existente, y corre
    generalización cruzada + estabilidad de régimen/estacionalidad sobre
    ella. Deja el reporte en session_state para que se muestre debajo.
    """
    from discovery.codegen import compile_definition_to_class
    from discovery.generalization import run_cross_symbol_generalization, run_regime_and_seasonality_test
    from core.types import BacktestConfig

    if not gen_symbols:
        st.warning("Elige al menos un símbolo distinto al de origen para la prueba de generalización.")
        return

    with st.spinner("Compilando la estrategia y descargando datos de validación..."):
        try:
            strategy_class = compile_definition_to_class(
                chosen["definition"], class_name="ValidacionTemporal",
                symbol=origin_asset, timeframe=timeframe,
            )
            strategy = strategy_class()
        except Exception as e:
            st.error(f"No se pudo compilar la estrategia para validar: {e}")
            return

        datasets = {}
        origin_df = _load_data_range(origin_asset, timeframe, None, None)
        if origin_df is not None and not origin_df.empty:
            datasets[origin_asset] = origin_df

        missing = []
        for sym in gen_symbols:
            try:
                df_sym = load_historical_data(sym, timeframe)
            except Exception as e:
                logger.warning(f"[Discovery] Falló la descarga de {sym} {timeframe} durante validación cruzada: {e}")
                df_sym = None
            if df_sym is not None and not df_sym.empty:
                datasets[sym] = df_sym
            else:
                missing.append(sym)
        if missing:
            st.warning(f"Sin datos disponibles para: {', '.join(missing)} — se excluyeron de la prueba.")
        if len(datasets) < 2:
            st.error(
                "No hay suficientes símbolos con datos disponibles para correr la prueba de "
                "generalización (se necesita el símbolo de origen + al menos uno más)."
            )
            return

        config = BacktestConfig(initial_capital=100_000.0)
        try:
            gen_report = run_cross_symbol_generalization(
                strategy, datasets, origin_symbol=origin_asset, config=config, timeframe=timeframe,
            )
            stability_report = None
            if origin_asset in datasets:
                stability_report = run_regime_and_seasonality_test(
                    strategy, datasets[origin_asset], config=config,
                    asset=origin_asset, timeframe=timeframe,
                )
        except Exception as e:
            st.error(f"La prueba de generalización falló: {e}")
            return

    st.session_state[f"gen_report_{chosen['strategy_id']}"] = {
        "generalization": gen_report,
        "stability": stability_report,
        "symbols_tested": list(datasets.keys()),
    }
    st.success("Validación de generalización completada — revisa el reporte abajo.")


def _show_generalization_report(gen_state: dict) -> None:
    gen_report = gen_state["generalization"]
    stability_report = gen_state.get("stability")

    st.markdown("#### 🧪 Reporte de generalización")
    c1, c2 = st.columns(2)
    c1.metric(
        "Generaliza a otros símbolos",
        f"{gen_report.generalization_rate * 100:.0f}%",
        help=f"{gen_report.n_profitable}/{gen_report.n_tested} símbolos rentables en total (incluye el de origen).",
    )
    c1.caption(f"Veredicto: **{gen_report.verdict()}**")

    rows = [{
        "símbolo": r.symbol,
        "origen": "✅" if r.symbol == gen_report.origin_symbol else "",
        "ok": "Sí" if r.ok else f"No ({r.error})",
        "sharpe": round(r.sharpe, 2) if r.ok else "—",
        "net_profit_%": round(r.net_profit_pct, 2) if r.ok else "—",
        "n_trades": r.total_trades if r.ok else "—",
    } for r in gen_report.results]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    regime_summary = gen_report.regime_generalization_summary()
    if regime_summary:
        st.markdown("##### Generalización por régimen de mercado (excluye símbolo de origen)")
        st.caption(
            "Por cada régimen, en cuántos de los símbolos probados la estrategia fue rentable "
            "ÚNICAMENTE con los trades que cayeron en ese régimen. Un símbolo sin ningún trade en "
            "un régimen dado no cuenta ni a favor ni en contra."
        )
        st.dataframe(pd.DataFrame([{
            "régimen": label,
            "símbolos con trades en este régimen": s["n_symbols_with_trades"],
            "símbolos rentables en este régimen": s["n_symbols_profitable"],
            "tasa de generalización": f"{s['rate']*100:.0f}%",
        } for label, s in regime_summary.items()]), width='stretch', hide_index=True)

        with st.expander("Ver desglose por régimen, símbolo por símbolo"):
            detail_rows = []
            for r in gen_report.results:
                if not r.ok or not r.regime_breakdown:
                    continue
                for label, stats in r.regime_breakdown.items():
                    detail_rows.append({
                        "símbolo": r.symbol, "régimen": label,
                        "net_profit_%": round(stats["net_profit_pct"], 2),
                        "sharpe_proxy": round(stats["sharpe"], 2),
                        "n_trades": stats["n_trades"],
                    })
            st.dataframe(pd.DataFrame(detail_rows), width='stretch', hide_index=True, height=300)

    if stability_report:
        c2.caption(f"Estacionalidad: **{stability_report.seasonality_verdict()}**")
        c2.caption(f"Régimen: **{stability_report.regime_verdict()}**")

        st.markdown("##### Estabilidad mensual (símbolo de origen)")
        months_df = pd.DataFrame([{
            "mes": m.month, "net_profit_%": round(m.net_profit_pct, 2), "n_trades": m.total_trades,
        } for m in stability_report.monthly])
        st.dataframe(months_df, width='stretch', hide_index=True, height=200)

        st.markdown("##### Estabilidad por régimen ADX (tendencia vs. rango)")
        regimes_df = pd.DataFrame([{
            "régimen": r.regime, "net_profit_%": round(r.net_profit_pct, 2),
            "sharpe": round(r.sharpe, 2), "n_trades": r.total_trades,
        } for r in stability_report.regimes])
        st.dataframe(regimes_df, width='stretch', hide_index=True)
