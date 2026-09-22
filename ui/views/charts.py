"""
ui/views/charts.py
Página de Gráficos — visualización de datos de mercado con indicadores.

El rango de velas mostrado es elegible aquí mismo: un rango de fechas
exacto, o todo el histórico disponible (bloqueado para temporalidades
inferiores a 1 hora, donde generaría un volumen de velas inmanejable).
Se pide directamente a MT5 bajo demanda — sin depender de ninguna
descarga manual previa.
"""
from datetime import date, timedelta

import streamlit as st

from market_data.mt5_provider import history_gap_warning, is_sub_hourly_timeframe
from strategies import STRATEGY_REGISTRY, list_strategies
from visualization.charts import candlestick_chart, indicators_chart
from core.data_cleaning import data_quality_summary, detect_suspicious_gaps, remove_suspicious_gaps
from research.data_registry import register_clean_dataset
from research.raw_data_registry import register_raw_dataset
from ui.state import AppState
from ui.components.data_source import set_selected_source, SOURCE_CLEAN, SOURCE_CSV
from research.csv_registry import register_csv, list_csv_datasets



def _publish_clean_dataset(df, asset: str, timeframe: str, cleaning_info: dict | None = None) -> dict:
    """Publica una serie limpia como dataset histórico oficial de la plataforma."""
    meta = register_clean_dataset(
        df, asset, timeframe, cleaning_info=cleaning_info or {
            "velas_originales": len(df),
            "velas_resultantes": len(df),
            "velas_eliminadas": 0,
            "huecos_eliminados": 0,
        }
    )
    # La fuente limpia vive en research/data_registry. El cache de sesión solo
    # evita una recarga durante la navegación actual; NO reemplazamos el cache
    # MT5 porque eso mezclaría las dos fuentes.
    AppState.cache_data(asset, timeframe, df)
    set_selected_source(SOURCE_CLEAN)
    return meta


def _publish_raw_dataset(df, asset: str, timeframe: str) -> dict:
    """Publica exactamente las filas recibidas de MT5, sin limpieza."""
    meta = register_raw_dataset(df, asset, timeframe, source="Gráficos / CSV local")
    AppState.cache_data(asset, timeframe, df)
    set_selected_source("Datos originales publicados")
    return meta

def render(asset: str, timeframe: str) -> None:
    st.markdown("## Gráficos de Mercado")

    # ── Importación de CSV propios ────────────────────────────────────────
    # Se mantiene separada de MT5: el usuario puede incorporar activos que
    # su broker no ofrece y luego reutilizarlos desde el selector histórico
    # común de Backtesting/Optimización/Research Lab.
    with st.expander("📥 Importar CSV propio", expanded=False):
        st.caption(
            "Carga OHLCV desde tu propio archivo. Se admiten esquemas habituales de MT5/TradingView "
            "(datetime, date+time, open, high, low, close, volume/tick_volume). El archivo se valida "
            "y queda persistido localmente; no se envía a servicios externos."
        )
        c1, c2 = st.columns([1.2, 1.0])
        csv_asset = c1.text_input("Nombre del activo", value=asset, key="csv_import_asset", help="Ej. SOLUSD, DAX40, AAPL")
        csv_timeframe = c2.text_input("Temporalidad", value=timeframe, key="csv_import_tf", help="Ej. 5m, 15m, 1h, 4h, 1d")
        uploaded_csv = st.file_uploader("Archivo CSV", type=["csv"], key="csv_import_file")
        if uploaded_csv is not None and st.button("Importar y registrar CSV", type="primary", key="csv_import_btn"):
            try:
                meta_csv = register_csv(uploaded_csv, csv_asset, csv_timeframe, uploaded_csv.name)
                st.success(
                    f"{meta_csv['asset']} {meta_csv['timeframe']}: {meta_csv['bars']:,} velas importadas "
                    f"({meta_csv['start'][:19]} → {meta_csv['end'][:19]}). "
                    "La serie ya está disponible para el resto de la plataforma."
                )
                st.info("Si el activo/temporalidad no cambia automáticamente en el selector superior, selecciónalo allí y elige 'CSV importados' como fuente histórica.")
            except Exception as exc:
                st.error(f"No se pudo importar el CSV: {exc}")

    col1, col2 = st.columns([3, 1])
    with col2:
        show_signals = st.checkbox("Mostrar señales", value=False, key="ch_signals")
        selected_strategy = st.selectbox(
            "Estrategia (señales)",
            ["—"] + list_strategies(),
            key="ch_strat",
        ) if show_signals else None

    with st.expander("Rango de velas", expanded=True):
        sub_hourly = is_sub_hourly_timeframe(timeframe)
        mode_options = ["Rango de fechas"] if sub_hourly else ["Rango de fechas", "Todo el histórico"]
        mode_key = "ch_range_mode"
        if st.session_state.get(mode_key) not in mode_options:
            # Reinicia el modo si el activo/temporalidad cambió y el modo
            # guardado ("Todo el histórico") ya no es una opción válida
            # (p.ej. se pasó a una temporalidad sub-horaria).
            st.session_state[mode_key] = mode_options[0]

        c1, c2, c3 = st.columns([1.4, 2, 2])
        range_mode = c1.radio("Modo", mode_options, key=mode_key)
        if sub_hourly:
            c1.caption(
                "⚠️ 'Todo el histórico' no está disponible en temporalidades inferiores a "
                "1 hora (el volumen de velas sería inmanejable). Elige un rango de fechas."
            )

        date_from = date_to = None
        if range_mode == "Rango de fechas":
            default_end = date.today()
            default_start = default_end - timedelta(days=365)
            date_from = c2.date_input("Desde", value=default_start, key="ch_date_from")
            date_to = c3.date_input("Hasta", value=default_end, key="ch_date_to")


    with st.spinner(
        "Cargando histórico CSV local...",
    ):
        df, load_error, diagnostics = _load_data(asset, timeframe, date_from, date_to)
    if df is None or df.empty:
        if load_error:
            st.error(f"No se pudieron obtener velas locales para **{asset} {timeframe}**: {load_error}")
        else:
            st.error(
                f"La fuente de datos no devolvió velas para **{asset} {timeframe}** en el rango "
                "elegido. Verifica el nombre del símbolo y que el mercado tenga histórico para "
                "esa temporalidad."
            )
        if diagnostics:
            with st.expander(f"🩺 Diagnóstico — {len(diagnostics)} tramo(s) pedidos a MT5", expanded=True):
                st.caption(
                    "Cada fila es un tramo de fechas pedido a MT5 al recorrer el histórico hacia "
                    "atrás. Si TODOS los tramos salen en 0 velas desde el principio, MT5 no tiene "
                    "ni un solo dato sincronizado para este símbolo/temporalidad (revisa el nombre "
                    "exacto del símbolo o ábrelo una vez en el terminal). Si los tramos recientes "
                    "traen velas y de golpe empiezan a salir en 0, probablemente llegaste al límite "
                    "real de historial que ese bróker conserva en su servidor para este símbolo — "
                    "eso no lo puede sortear ningún software, ni el propio terminal MT5 a mano."
                )
                st.dataframe(diagnostics, width='stretch', hide_index=True)
        if st.button("🔄 Recargar CSV local", key=f"ch_retry_{asset}_{timeframe}"):
            _load_data.clear()
            st.rerun()
        return

    # Si el usuario ya pidió depurar huecos sospechosos para este activo/
    # temporalidad en esta sesión, se usa la serie limpia (tramo continuo
    # verificado) para TODO lo que sigue: gráfico, indicadores, info del
    # dataset y resumen de calidad -- no solo para el reporte.
    clean_key = f"ch_clean_{asset}_{timeframe}"
    cleaned_info = st.session_state.get(clean_key)
    if cleaned_info is not None:
        df, gap_clean_info = cleaned_info
    else:
        gap_clean_info = None

    if range_mode == "Rango de fechas" and date_from:
        gap_msg = history_gap_warning(df, date_from, asset, timeframe)
        if gap_msg:
            st.warning(gap_msg)

    # Generar señales e indicadores si se seleccionó estrategia
    indicators = {}
    signals = None
    if show_signals and selected_strategy and selected_strategy != "—":
        strategy_class = STRATEGY_REGISTRY.get(selected_strategy)
        if strategy_class:
            try:
                strategy = strategy_class()
                signals_df = strategy.generate_signals(df)
                signals = signals_df.get("signal")
                for col in strategy.get_indicator_columns():
                    if col in signals_df.columns:
                        indicators[col] = signals_df[col]
            except Exception as e:
                st.warning(f"Error generando señales: {e}")

    fig = candlestick_chart(
        df=df,
        title=f"{asset} — {timeframe}",
        signals=signals,
        height=520,
    )
    st.plotly_chart(fig, width='stretch')

    if show_signals and indicators:
        fig_ind = indicators_chart(indicators, height=280)
        st.plotly_chart(fig_ind, width='stretch')

    # Info del dataset
    with st.expander("ℹ Información del dataset"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Velas", f"{len(df):,}")
        col2.metric("Inicio", str(df.index[0].date()))
        col3.metric("Fin", str(df.index[-1].date()))
        col4.metric("Días totales", (df.index[-1] - df.index[0]).days)
        if diagnostics:
            st.caption(
                f"'Todo el histórico' recorrió {len(diagnostics)} tramo(s) hacia atrás antes de "
                "parar (se detiene tras varios tramos vacíos seguidos, asumiendo que ahí termina "
                "lo que el bróker conserva). Detalle tramo por tramo:"
            )
            st.dataframe(diagnostics, width='stretch', hide_index=True)

    # Publicación de la fuente original: NO modifica df.
    with st.expander("📦 Publicar fuente original", expanded=False):
        st.info(
            "Esta publicación conserva exactamente las filas recibidas de MT5: "
            "no elimina duplicados, huecos ni velas. Es útil cuando la limpieza "
            "descarta información que quieres investigar por separado."
        )
        if st.button(
            "📤 Publicar datos originales (sin limpiar) en toda la plataforma",
            key=f"ch_publish_raw_{asset}_{timeframe}",
        ):
            try:
                meta_raw = _publish_raw_dataset(df, asset, timeframe)
                st.success(
                    f"{asset} {timeframe}: publicados {meta_raw['bars']:,} registros originales. "
                    "Ahora puedes seleccionar 'Datos originales publicados' en las herramientas históricas."
                )
            except Exception as exc:
                st.error(f"No se pudo publicar el dataset original: {exc}")

    # Calidad de datos: huecos sospechosos dentro de horario de mercado
    # (posible vela faltante del feed), distinto de los cierres normales
    # de fin de semana/feriado (esos ya no se ven en el gráfico de arriba).
    summary = data_quality_summary(df, timeframe, symbol=asset)
    quality_title = f"🔍 Calidad de datos — {summary['huecos_sospechosos']} hueco(s) sospechoso(s)"
    if cleaned_info is not None:
        quality_title += " — datos ya depurados"
    with st.expander(quality_title, expanded=summary["huecos_sospechosos"] > 0 or cleaned_info is not None):
        if cleaned_info is not None:
            st.success(
                f"Ya se depuró esta serie en esta sesión: se descartaron "
                f"{gap_clean_info['velas_eliminadas']:,} vela(s) alrededor de "
                f"{gap_clean_info['huecos_eliminados']} hueco(s) sospechoso(s), quedando "
                f"{gap_clean_info['velas_resultantes']:,} vela(s) limpias y continuas "
                f"(de {gap_clean_info['velas_originales']:,} originales)."
            )
            if st.button("↩ Revertir — volver a los datos originales (con huecos)", key="ch_clean_revert"):
                del st.session_state[clean_key]
                st.rerun()

        elif summary["huecos_sospechosos"] == 0:
            st.success(
                "Sin huecos sospechosos dentro de horario de mercado. Los espacios de "
                "fin de semana/feriado no cuentan como error: MT5 simplemente no genera "
                "velas cuando el mercado está cerrado, y el gráfico ya los oculta visualmente."
            )
            st.caption("La serie puede enviarse como **datos verificados** al Research Lab sin modificar ninguna vela.")
            if st.button("🚀 Publicar datos limpios en toda la plataforma", key="ch_publish_verified", type="primary"):
                try:
                    meta = _publish_clean_dataset(df, asset, timeframe, {
                        "velas_originales": len(df), "velas_resultantes": len(df),
                        "velas_eliminadas": 0, "huecos_eliminados": 0,
                        "verified_without_removal": True,
                    })
                    st.success(
                        f"{asset} {timeframe} publicado como dataset limpio ({meta['bars']:,} velas). "
                        "Ahora puede seleccionarse desde Datos en las herramientas históricas."
                    )
                except Exception as exc:
                    st.error(f"No se pudo publicar el dataset: {exc}")
        else:
            st.warning(
                f"Se detectaron {summary['huecos_sospechosos']} hueco(s) que NO se explican "
                f"por un cierre normal de fin de semana/festivo (~{summary['velas_faltantes_estimadas']} "
                "velas posiblemente faltantes). Esto puede sesgar retornos, volatilidad y "
                "cualquier métrica estadística calculada sobre esta serie."
            )
            gaps = detect_suspicious_gaps(df, timeframe)
            st.dataframe(gaps, width='stretch', hide_index=True)

            st.markdown(
                "**Eliminar huecos** — elimina únicamente las filas correspondientes al hueco detectado; "
                "no recorta automáticamente las velas vecinas y no inventa/interpola datos."
            )
            if st.button("🧹 Eliminar huecos", key="ch_clean_gaps", type="primary"):
                df_clean, info = remove_suspicious_gaps(df, timeframe, keep="longest")
                st.session_state[clean_key] = (df_clean, info)
                st.rerun()

        if cleaned_info is not None:
            st.caption(
                "La depuración queda en la vista actual hasta que la publiques. Al publicar, "
                "la serie limpia se registra como fuente histórica reutilizable por todas las "
                "herramientas de análisis de CapitalQuant."
            )
            if st.button("🚀 Publicar datos limpios en toda la plataforma",
                          key="ch_publish_clean", type="primary"):
                try:
                    meta = _publish_clean_dataset(
                        df, asset, timeframe, gap_clean_info or {
                            "velas_originales": len(df),
                            "velas_resultantes": len(df),
                            "velas_eliminadas": 0,
                            "huecos_eliminados": 0,
                        }
                    )
                    st.success(
                        f"{asset} {timeframe} publicado ({meta['bars']:,} velas). "
                        "La fuente limpia ya está disponible en Research Lab, Backtesting, "
                        "Descubridor, Optimización, Portfolio, Régimen y Constructores."
                    )
                except Exception as exc:
                    st.error(f"No se pudo publicar el dataset: {exc}")


@st.cache_data(ttl=120, show_spinner=False)
def _load_data(asset: str, timeframe: str, date_from, date_to):
    """Carga exclusivamente el CSV local registrado para el activo/TF."""
    from ui.components.data_source import load_historical_data, SOURCE_CSV
    try:
        df = load_historical_data(asset, timeframe, source=SOURCE_CSV, date_from=date_from, date_to=date_to)
        if df is None or df.empty:
            return df, f"No existe un CSV local para {asset} {timeframe}.", []
        return df, None, []
    except Exception as e:
        return None, str(e), []
