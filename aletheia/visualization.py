"""
Módulo 6 — Visualización
===========================

Genera un gráfico interactivo (Plotly, exportado a HTML autocontenido)
con:

  - Velas (candlestick) del histórico.
  - Segmentos coloreados por régimen (bandas de fondo).
  - Línea temporal de segmentos con su información al pasar el cursor
    (equivalente interactivo a "hacer clic sobre un segmento").
  - Microestados como marcadores coloreados bajo el precio.
  - Heatmap de la matriz de transición de probabilidades.

Se genera un archivo .html independiente que se puede abrir directamente
en el navegador — no requiere un servidor.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .microstate_classifier import MICROSTATE_COLORS
from .regime_detector import RegimeClassification
from .transition_matrix import TransitionModel

REGIME_COLORS = {
    "Tendencia Alcista": "rgba(0, 200, 83, 0.15)",
    "Tendencia Bajista": "rgba(244, 67, 54, 0.15)",
    "Consolidación": "rgba(255, 235, 59, 0.12)",
    "Sin clasificar": "rgba(158, 158, 158, 0.10)",
}


def plot_regime_chart(df: pd.DataFrame,
                       classifications: list[RegimeClassification],
                       state_series: pd.Series | None = None,
                       output_path: str = "outputs/regime_chart.html",
                       title: str = "Aletheia Regime Engine") -> str:
    """
    Construye el gráfico principal: velas + bandas de régimen + microestados.
    Guarda un HTML autocontenido en `output_path` y devuelve la ruta.
    """
    cols = {c.lower(): c for c in df.columns}
    o, h, l, c = (df[cols[k]] for k in ("open", "high", "low", "close"))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.82, 0.18],
        vertical_spacing=0.03,
        subplot_titles=("Precio + Régimen por Segmento", "Microestados"),
    )

    fig.add_trace(
        go.Candlestick(x=df.index, open=o, high=h, low=l, close=c, name="OHLC"),
        row=1, col=1,
    )

    # Bandas de régimen como shapes de fondo + trazas invisibles para hover
    shapes = []
    for cl in classifications:
        seg = cl.segment
        color = REGIME_COLORS.get(cl.regime, "rgba(158,158,158,0.1)")
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=seg.start_time, x1=seg.end_time, y0=0, y1=1,
            fillcolor=color, line=dict(width=0), layer="below",
        ))
        hover = (
            f"<b>{cl.regime}</b><br>"
            f"Inicio: {seg.start_time}<br>Fin: {seg.end_time}<br>"
            f"Duración: {seg.duration_bars} velas<br>"
            f"Polaridad media: {seg.avg_polarity:.3f}<br>"
            f"Entropía: {seg.entropy:.3f}<br>"
            f"Confianza: {seg.confidence:.2%}<br>"
            f"Estados dominantes: {seg.dominant_states}"
        )
        fig.add_trace(
            go.Scatter(
                x=[seg.start_time, seg.end_time], y=[h.max(), h.max()],
                mode="markers", marker=dict(opacity=0), showlegend=False,
                hovertext=hover, hoverinfo="text",
            ),
            row=1, col=1,
        )

    if state_series is not None:
        colors = [MICROSTATE_COLORS.get(int(s), "#999999") if pd.notna(s) else "#FFFFFF"
                  for s in state_series]
        fig.add_trace(
            go.Scatter(
                x=state_series.index, y=state_series.values,
                mode="markers", marker=dict(color=colors, size=6),
                name="Microestado",
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title=title,
        shapes=shapes,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=800,
        legend=dict(orientation="h"),
    )

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_transition_heatmap(model: TransitionModel,
                             output_path: str = "outputs/transition_heatmap.html",
                             title: str = "Matriz de Transición de Microestados") -> str:
    """Genera un heatmap interactivo de la matriz de transición."""
    labels = [str(i) for i in range(model.n_states)]
    fig = go.Figure(data=go.Heatmap(
        z=model.matrix, x=labels, y=labels,
        colorscale="Viridis", colorbar=dict(title="P(siguiente)"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Estado siguiente",
        yaxis_title="Estado actual",
        template="plotly_dark",
        height=600,
    )
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def _build_regime_figure(df: pd.DataFrame,
                          classifications: list[RegimeClassification],
                          state_series: pd.Series | None,
                          title: str) -> go.Figure:
    """Construye la figura de régimen compartida entre la exportación a
    archivo HTML y el embebido web (evita duplicar la lógica)."""
    cols = {c.lower(): c for c in df.columns}
    o, h, l, c = (df[cols[k]] for k in ("open", "high", "low", "close"))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.82, 0.18],
        vertical_spacing=0.03,
        subplot_titles=("Precio + Régimen por Segmento", "Microestados"),
    )
    fig.add_trace(
        go.Candlestick(x=df.index, open=o, high=h, low=l, close=c, name="OHLC"),
        row=1, col=1,
    )

    shapes = []
    for cl in classifications:
        seg = cl.segment
        color = REGIME_COLORS.get(cl.regime, "rgba(158,158,158,0.1)")
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=seg.start_time, x1=seg.end_time, y0=0, y1=1,
            fillcolor=color, line=dict(width=0), layer="below",
        ))
        hover = (
            f"<b>{cl.regime}</b><br>"
            f"Inicio: {seg.start_time}<br>Fin: {seg.end_time}<br>"
            f"Duración: {seg.duration_bars} velas<br>"
            f"Polaridad media: {seg.avg_polarity:.3f}<br>"
            f"Entropía: {seg.entropy:.3f}<br>"
            f"Confianza: {seg.confidence:.2%}<br>"
            f"Estados dominantes: {seg.dominant_states}"
        )
        fig.add_trace(
            go.Scatter(
                x=[seg.start_time, seg.end_time], y=[h.max(), h.max()],
                mode="markers", marker=dict(opacity=0), showlegend=False,
                hovertext=hover, hoverinfo="text",
            ),
            row=1, col=1,
        )

    if state_series is not None:
        colors = [MICROSTATE_COLORS.get(int(s), "#999999") if pd.notna(s) else "#FFFFFF"
                  for s in state_series]
        fig.add_trace(
            go.Scatter(
                x=state_series.index, y=state_series.values,
                mode="markers", marker=dict(color=colors, size=6),
                name="Microestado",
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title=title,
        shapes=shapes,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=620,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h"),
        font=dict(family="Chakra Petch, sans-serif", color="#e8e8e8"),
    )
    return fig


def regime_chart_div(df: pd.DataFrame,
                      classifications: list[RegimeClassification],
                      state_series: pd.Series | None = None,
                      title: str = "Aletheia Regime Engine") -> str:
    """
    Igual que `plot_regime_chart` pero devuelve un <div> HTML embebible
    (sin las etiquetas <html>/<head> ni una copia de plotly.js), pensado
    para insertarse directamente en una página web que ya cargó
    plotly.js una sola vez (vía CDN) en su plantilla base.
    """
    fig = _build_regime_figure(df, classifications, state_series, title)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def transition_heatmap_div(model: TransitionModel,
                            title: str = "Matriz de Transición de Microestados") -> str:
    """Versión embebible (div) del heatmap de transición, para la web."""
    labels = [str(i) for i in range(model.n_states)]
    fig = go.Figure(data=go.Heatmap(
        z=model.matrix, x=labels, y=labels,
        colorscale="Viridis", colorbar=dict(title="P(siguiente)"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Estado siguiente",
        yaxis_title="Estado actual",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=40, r=20, t=50, b=30),
        font=dict(family="Chakra Petch, sans-serif", color="#e8e8e8"),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})
