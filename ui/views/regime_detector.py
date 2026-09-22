"""
ui/views/regime_detector.py
Herramienta "Régimen de Mercado" — corre el detector de régimen
(motor Aletheia, causal, sin look-ahead) sobre el activo/temporalidad
elegidos y muestra el régimen vigente vela a vela sobre el precio,
además del historial de segmentos.

La calibración del detector se AUTO-OPTIMIZA por activo (ver
`regime/calibration.py`): no hay sliders que mover a mano. El resultado
se cachea por activo/timeframe/rango de fechas y es la MISMA calibración
que usan los filtros de régimen de "Backtesting", "Optimización" y el
"Descubridor Genético" (ver `ui/components/regime_filter.py`) — un único
punto de verdad para todo el sistema.
"""
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_data.mt5_provider import get_mt5_provider, history_gap_warning
from regime.service import REGIME_LABELS, compute_regime_summary
from regime.calibration import get_cached_calibration
from visualization.theme import base_layout, GREEN, RED, ACCENT, GRID
from ui.components.data_source import render_data_source_selector, load_historical_data

_REGIME_COLORS = {
    "Tendencia Alcista": GREEN,
    "Tendencia Bajista": RED,
    "Consolidación": ACCENT,
}


def render(asset: str, timeframe: str) -> None:
    st.markdown("## Régimen de Mercado")
    data_source = render_data_source_selector(timeframe, key_prefix="regime_source", compact=True)
    st.caption(f"Fuente histórica activa: **{data_source}**")
    st.caption(
        "Clasifica cada vela como Tendencia Alcista, Tendencia Bajista o "
        "Consolidación, de forma 100% causal (nunca usa información futura). "
        "La calibración se auto-optimiza para este activo (ver panel de abajo) "
        "y es la misma que alimenta el filtro de régimen en Backtesting, "
        "Optimización y el Descubridor Genético."
    )

    with st.expander("Rango de fechas", expanded=True):
        c1, c2, c3 = st.columns([1.4, 2, 2])
        range_mode = c1.radio(
            "Modo", ["Rango de fechas", "Máxima historia"],
            key="rg_range_mode",
        )
        date_from = date_to = None
        if range_mode == "Rango de fechas":
            default_end = date.today()
            default_start = default_end - timedelta(days=365 * 3)
            date_from = c2.date_input("Desde", value=default_start, key="rg_date_from")
            date_to = c3.date_input("Hasta", value=default_end, key="rg_date_to")
        else:
            c2.info("Se pedirá a MT5 todo el histórico disponible para este activo/temporalidad.")

    df, load_error = _load_data(asset, timeframe, date_from, date_to)
    if df is None or df.empty:
        st.error(
            f"La fuente de datos no devolvió velas para **{asset} {timeframe}** en el rango elegido"
            f"{f': {load_error}' if load_error else ''}."
        )
        return

    if range_mode == "Rango de fechas" and date_from:
        gap_msg = history_gap_warning(df, date_from, asset, timeframe)
        if gap_msg:
            st.warning(gap_msg)

    from config.settings import TIMEFRAMES
    ann_factor = TIMEFRAMES.get(timeframe, {}).get("ann_factor", 252)

    force_recal = st.session_state.pop("_rg_force_recal", False)
    with st.spinner("Auto-calibrando el detector para este activo..."):
        calib = get_cached_calibration(df, ann_factor, asset, timeframe, force_recompute=force_recal)

    with st.expander("Calibración automática de este activo", expanded=False):
        st.caption(
            "El sistema prueba ~27 configuraciones del detector sobre el historial de "
            f"**{asset} {timeframe}**, valida cada una en 4 sub-periodos separados (con purga "
            "entre ellos) para exigir que funcione de forma consistente en el tiempo -- no solo "
            "en un tramo de suerte -- y penaliza alejarse del default salvo que la mejora sea "
            "clara. El resultado es la MISMA calibración que usan Backtesting, Optimización y "
            "el Descubridor Genético para este activo: no hay nada que ajustar a mano."
        )
        b1, b2, b3 = st.columns(3)
        b1.metric(
            "Resultado", "Configuración por defecto" if calib.is_default else "Configuración optimizada",
            help="'Configuración por defecto' significa que ninguna alternativa probada superó "
                 "al default con un margen que no pudiera explicarse por ruido entre sub-periodos "
                 "-- el sistema prefiere quedarse con el default ante la duda.",
        )
        b2.metric(
            "Candidatos evaluados",
            f"{calib.n_candidates_valid}/{calib.n_candidates_evaluated}",
            help="Cuántas de las ~27 configuraciones del grid produjeron una clasificación "
                 "utilizable en al menos un sub-periodo (las demás fueron degeneradas para "
                 "este activo/rango: muy pocos segmentos o un régimen casi vacío).",
        )
        b3.metric(
            "Puntaje ganador vs. default",
            f"{calib.final_score:+.2f} / {calib.default_score:+.2f}",
            help="Puntaje de separación de retornos futuros entre régimen alcista/bajista, "
                 "ajustado por estabilidad de la clasificación y penalizado por 'parpadeo' de "
                 "régimen -- promediado (con penalización por inconsistencia) entre los 4 "
                 "sub-periodos. Mayor es mejor; no es un indicador de rentabilidad.",
        )
        if not calib.is_default:
            st.json(calib.overrides, expanded=False)
        if calib.fold_scores:
            st.caption(
                "Consistencia entre sub-periodos (puntaje por tramo, deben ser parecidos entre "
                f"sí): {', '.join(f'{s:+.2f}' for s in calib.fold_scores)}"
            )
        if st.button("🔁 Recalibrar ahora", key="rg_recal_btn",
                     help="Fuerza un recálculo ignorando la caché -- útil tras cargar historial "
                          "nuevo o si quieres verificar que el resultado es estable."):
            st.session_state["_rg_force_recal"] = True
            st.rerun()

    with st.spinner("Detectando régimen..."):
        try:
            summary = compute_regime_summary(df, ann_factor=ann_factor, overrides=calib.overrides)
        except Exception as e:
            st.error(f"No se pudo calcular el régimen: {e}")
            return

    current_regime = summary.regime_series.iloc[-1] if len(summary.regime_series) else None
    current_conf = summary.confidence_series.iloc[-1] if len(summary.confidence_series) else None
    current_strength = summary.strength_series.iloc[-1] if len(summary.strength_series) else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Régimen actual", current_regime or "Sin clasificar")
    c2.metric(
        "Confianza del segmento", f"{current_conf:.1%}" if current_conf is not None else "—",
        help="Qué tan clara y consistente fue la señal que confirmó este segmento "
             "(fuerza + consistencia + tamaño de muestra + ajuste con la duración "
             "esperada por la cadena de Markov de microestados). Un régimen recién "
             "confirmado empieza con confianza moderada aunque sea inequívoco: aún "
             "tiene poco historial dentro del segmento.",
    )
    c3.metric(
        "Fuerza actual (vela a vela)",
        f"{current_strength:+.2f}" if current_strength is not None else "—",
        help="Señal continua en [-1, 1] (precio + microestados) EN LA ÚLTIMA VELA, sin "
             "esperar a que se confirme un segmento nuevo. Útil como lectura de qué tan "
             "lejos está el mercado de cambiar de régimen ahora mismo, más rápida que la "
             "confianza del segmento pero también más ruidosa.",
    )
    last_seg = summary.classifications[-1].segment if summary.classifications else None
    c4.metric("Duración del régimen actual", f"{last_seg.duration_bars} velas" if last_seg else "—")

    fig = _regime_price_chart(df, summary.regime_series, asset, timeframe)
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Historial de segmentos")
    if not summary.segments_df.empty:
        st.dataframe(summary.segments_df, width='stretch', hide_index=True, height=350)
        st.download_button(
            "CSV", summary.segments_df.to_csv(index=False),
            f"regimen_{asset}_{timeframe}.csv", "text/csv",
        )
    else:
        st.info("No se detectaron segmentos suficientes con este rango de datos.")


def _regime_price_chart(df: pd.DataFrame, regime_series: pd.Series,
                         asset: str, timeframe: str, height: int = 550) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Precio", increasing_line_color=GREEN, decreasing_line_color=RED,
    ))

    for regime_name, color in _REGIME_COLORS.items():
        mask = regime_series == regime_name
        if not mask.any():
            continue
        # Sombrea en bandas verticales las zonas de este régimen usando
        # el rango de precio de cada tramo contiguo.
        blocks = _contiguous_blocks(mask)
        for start, end in blocks:
            fig.add_vrect(
                x0=df.index[start], x1=df.index[min(end, len(df.index) - 1)],
                fillcolor=color, opacity=0.08, line_width=0,
            )

    layout = base_layout(title=f"{asset} {timeframe} — Régimen de Mercado", height=height)
    layout["xaxis_rangeslider_visible"] = False
    fig.update_layout(**layout)
    return fig


def _contiguous_blocks(mask: pd.Series) -> list[tuple[int, int]]:
    """Índices posicionales [start, end) de tramos contiguos donde mask=True."""
    vals = mask.fillna(False).to_numpy()
    blocks = []
    start = None
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        elif not v and start is not None:
            blocks.append((start, i))
            start = None
    if start is not None:
        blocks.append((start, len(vals)))
    return blocks


def _load_data(asset: str, timeframe: str, date_from, date_to):
    try:
        return load_historical_data(asset, timeframe, date_from=date_from, date_to=date_to), None
    except Exception as e:
        return None, str(e)
