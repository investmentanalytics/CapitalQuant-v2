"""
visualization/optimization_charts.py
Gráficos para el módulo de optimización avanzada.

Incluye: WFO por ventanas, Monte Carlo fan chart,
superficie 3D de sensibilidad, robustness scatter.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimization.walk_forward import WFOResults
from optimization.monte_carlo import MonteCarloResults
from optimization.sensitivity import SensitivityResult
from visualization.theme import (
    base_layout, GREEN, RED, ACCENT, TEXT, BORDER,
    SURFACE, GRID, COLORS, BG,
)


# ---------------------------------------------------------------------------
# Walk-Forward — equity OOS + ventanas
# ---------------------------------------------------------------------------

def wfo_results_chart(wfo: WFOResults, height: int = 500) -> go.Figure:
    """
    Muestra la equity out-of-sample combinada de todas las ventanas
    junto con el score train vs test por ventana.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=["Equity Out-of-Sample (combinada)", "Score Train vs Test por Ventana"],
        row_heights=[0.6, 0.4],
    )

    # --- Equity OOS combinada ---
    if not wfo.combined_oos_equity.empty:
        fig.add_trace(go.Scatter(
            x=list(range(len(wfo.combined_oos_equity))),
            y=wfo.combined_oos_equity.values,
            name="Equity OOS",
            line=dict(color=ACCENT, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(201,169,97,0.10)",
        ), row=1, col=1)

    # --- Barras train vs test ---
    valid = [w for w in wfo.windows if w.test_results is not None]
    window_ids    = [w.window_id for w in valid]
    train_scores  = [max(w.train_score, 0) for w in valid]
    test_scores   = [max(w.test_score,  0) for w in valid]

    fig.add_trace(go.Bar(
        x=window_ids, y=train_scores,
        name="Train Score",
        marker_color=ACCENT,
        opacity=0.7,
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=window_ids, y=test_scores,
        name="Test Score",
        marker_color=GREEN,
        opacity=0.85,
    ), row=2, col=1)

    er = wfo.efficiency_ratio
    er_color = GREEN if er >= 0.7 else (ACCENT if er >= 0.4 else RED)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.95,
        text=f"Efficiency Ratio: <b>{er:.2f}</b> — {wfo.robustness_label()}",
        showarrow=False,
        font=dict(color=er_color, size=13),
        align="right",
    )

    layout = base_layout(height=height)
    layout["barmode"] = "group"
    layout["xaxis2"] = dict(title="Ventana", gridcolor=GRID, tickvals=window_ids)
    layout["yaxis2"] = dict(title="Score", gridcolor=GRID)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Monte Carlo — fan chart de percentiles
# ---------------------------------------------------------------------------

def monte_carlo_fan_chart(mc: MonteCarloResults, height: int = 480) -> go.Figure:
    """
    Fan chart de Monte Carlo mostrando bandas de percentiles:
    P5–P95, P25–P75, mediana.
    """
    fig = go.Figure()
    x = list(range(mc.n_periods + 1))

    # Banda exterior P5–P95 (más tenue)
    if mc.equity_p5 is not None and mc.equity_p95 is not None:
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=list(mc.equity_p95.values) + list(mc.equity_p5.values[::-1]),
            fill="toself",
            fillcolor="rgba(201,169,97,0.09)",
            line=dict(width=0),
            name="P5–P95",
            showlegend=True,
        ))

    # Banda interior P25–P75 (más densa)
    if mc.equity_p25 is not None and mc.equity_p75 is not None:
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=list(mc.equity_p75.values) + list(mc.equity_p25.values[::-1]),
            fill="toself",
            fillcolor="rgba(201,169,97,0.20)",
            line=dict(width=0),
            name="P25–P75",
            showlegend=True,
        ))

    # Mediana
    if mc.equity_p50 is not None:
        fig.add_trace(go.Scatter(
            x=x, y=mc.equity_p50.values,
            name="Mediana (P50)",
            line=dict(color=ACCENT, width=2.5),
        ))

    # Línea de capital inicial
    fig.add_hline(
        y=mc.initial_capital,
        line_dash="dash",
        line_color=BORDER,
        annotation_text="Capital inicial",
        annotation_font_color=TEXT,
    )

    # Anotaciones de probabilidad
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.95,
        text=(
            f"Prob. ganancia: <b style='color:{GREEN}'>{mc.prob_profit:.1f}%</b><br>"
            f"Prob. pérdida >50%: <b style='color:{RED}'>{mc.prob_ruin:.1f}%</b>"
        ),
        showarrow=False,
        font=dict(color=TEXT, size=12),
        align="left",
        bgcolor=SURFACE,
        bordercolor=BORDER,
        borderwidth=1,
    )

    layout = base_layout(title=f"Monte Carlo — {mc.n_simulations:,} Simulaciones", height=height)
    layout["yaxis_tickprefix"] = "$"
    layout["yaxis_tickformat"] = ",.0f"
    layout["xaxis_title"] = "Períodos"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Distribución final de Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo_histogram(mc: MonteCarloResults, height: int = 300) -> go.Figure:
    """Histograma de capital final de todas las simulaciones."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=mc.final_capitals,
        nbinsx=50,
        marker_color=ACCENT,
        opacity=0.75,
        name="Capital final",
    ))

    # Línea del capital inicial
    fig.add_vline(x=mc.initial_capital, line_dash="dash", line_color=TEXT,
                  annotation_text="Capital inicial", annotation_font_color=TEXT)
    fig.add_vline(x=mc.p50, line_dash="dash", line_color=GREEN,
                  annotation_text=f"P50: ${mc.p50:,.0f}", annotation_font_color=GREEN)

    layout = base_layout(title="Distribución de Capital Final", height=height)
    layout["xaxis_tickprefix"] = "$"
    layout["xaxis_tickformat"] = ",.0f"
    layout["yaxis_title"] = "Frecuencia"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Sensibilidad 1D
# ---------------------------------------------------------------------------

def sensitivity_1d_chart(result: SensitivityResult, height: int = 350) -> go.Figure:
    """Gráfico de línea mostrando variación del score al cambiar un parámetro."""
    fig = go.Figure()

    scores = result.scores
    values = result.param1_values
    best_idx = int(np.argmax(scores))

    fig.add_trace(go.Scatter(
        x=values, y=scores,
        name=result.objective,
        line=dict(color=ACCENT, width=2),
        mode="lines+markers",
        marker=dict(size=5, color=ACCENT),
    ))

    # Marcar el mejor valor
    fig.add_trace(go.Scatter(
        x=[values[best_idx]],
        y=[scores[best_idx]],
        name="Óptimo",
        mode="markers",
        marker=dict(size=12, color=GREEN, symbol="star"),
    ))

    layout = base_layout(
        title=f"Sensibilidad: {result.param1_name}  {result.objective}",
        height=height,
    )
    layout["xaxis_title"] = result.param1_name
    layout["yaxis_title"] = result.objective.capitalize()
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Sensibilidad 2D — superficie 3D
# ---------------------------------------------------------------------------

def sensitivity_2d_surface(result: SensitivityResult, height: int = 500) -> go.Figure:
    """Superficie 3D de sensibilidad para dos parámetros simultáneos."""
    if result.scores.ndim != 2:
        return go.Figure()

    fig = go.Figure(go.Surface(
        x=result.param2_values,
        y=result.param1_values,
        z=np.where(result.scores < -100, np.nan, result.scores),
        colorscale=[
            [0.0, RED],
            [0.5, SURFACE],
            [1.0, GREEN],
        ],
        colorbar=dict(
            title=result.objective,
            titlefont=dict(color=TEXT),
            tickfont=dict(color=TEXT),
            bgcolor=BG,
        ),
        hovertemplate=(
            f"<b>{result.param1_name}</b>: %{{y}}<br>"
            f"<b>{result.param2_name}</b>: %{{x}}<br>"
            f"<b>{result.objective}</b>: %{{z:.3f}}<extra></extra>"
        ),
    ))

    layout = base_layout(
        title=f"Mapa 3D: {result.param1_name} × {result.param2_name}",
        height=height,
    )
    layout["scene"] = dict(
        xaxis=dict(title=result.param2_name, gridcolor=GRID, backgroundcolor=BG),
        yaxis=dict(title=result.param1_name, gridcolor=GRID, backgroundcolor=BG),
        zaxis=dict(title=result.objective,   gridcolor=GRID, backgroundcolor=BG),
        bgcolor=BG,
    )
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Sensibilidad 2D — heatmap plano
# ---------------------------------------------------------------------------

def sensitivity_2d_heatmap(result: SensitivityResult, height: int = 400) -> go.Figure:
    """Heatmap 2D de sensibilidad — más legible que la superficie 3D."""
    if result.scores.ndim != 2:
        return go.Figure()

    z = np.where(result.scores < -100, np.nan, result.scores)

    fig = go.Figure(go.Heatmap(
        x=result.param2_values,
        y=result.param1_values,
        z=z,
        colorscale=[[0, RED], [0.5, SURFACE], [1, GREEN]],
        colorbar=dict(title=result.objective, tickfont=dict(color=TEXT)),
        hovertemplate=(
            f"<b>{result.param1_name}</b>: %{{y}}<br>"
            f"<b>{result.param2_name}</b>: %{{x}}<br>"
            f"<b>{result.objective}</b>: %{{z:.3f}}<extra></extra>"
        ),
    ))

    # Marcar el óptimo
    best_i, best_j = np.unravel_index(np.nanargmax(z), z.shape)
    fig.add_trace(go.Scatter(
        x=[result.param2_values[best_j]],
        y=[result.param1_values[best_i]],
        mode="markers",
        marker=dict(size=14, color=ACCENT, symbol="star", line=dict(color=TEXT, width=1)),
        name="Óptimo",
    ))

    layout = base_layout(
        title=f"Heatmap: {result.param1_name} × {result.param2_name}",
        height=height,
    )
    layout["xaxis_title"] = result.param2_name
    layout["yaxis_title"] = result.param1_name
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Robustness scatter — top trials
# ---------------------------------------------------------------------------

def robustness_scatter(
    history_df: pd.DataFrame,
    x_metric: str = "sharpe",
    y_metric: str = "max_dd_pct",
    height: int = 400,
) -> go.Figure:
    """
    Scatter plot de los mejores trials de la optimización
    para visualizar la frontera de eficiencia riesgo/retorno.
    """
    if history_df is None or history_df.empty:
        return go.Figure()

    # Intentar usar métricas del user_attr si están disponibles
    x_col = x_metric if x_metric in history_df.columns else "value"
    y_col = y_metric if y_metric in history_df.columns else "value"

    fig = go.Figure(go.Scatter(
        x=history_df[x_col],
        y=history_df[y_col],
        mode="markers",
        marker=dict(
            color=history_df["value"] if "value" in history_df.columns else ACCENT,
            colorscale=[[0, RED], [0.5, ACCENT], [1, GREEN]],
            size=7,
            opacity=0.7,
            showscale=True,
            colorbar=dict(title="Score", tickfont=dict(color=TEXT)),
        ),
        hovertemplate=(
            f"<b>{x_col}</b>: %{{x:.3f}}<br>"
            f"<b>{y_col}</b>: %{{y:.3f}}<extra></extra>"
        ),
        name="Trials",
    ))

    layout = base_layout(height=height)
    layout["xaxis"] = dict(title=x_col, gridcolor=GRID)
    layout["yaxis"] = dict(title=y_col, gridcolor=GRID)
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def cpcv_results_chart(results, height: int = 480) -> go.Figure:
    """
    Visualiza el resultado de CPCV (optimization/cpcv.py):
    - Izquierda: dispersión IS (train) vs OOS (test) de TODOS los
      candidatos en TODAS las particiones. La diagonal punteada marca
      "IS == OOS" (generalización perfecta); los puntos que caen muy por
      debajo de la diagonal muestran overfitting (rinde bien in-sample,
      mal out-of-sample).
    - Derecha: distribución del score OOS del candidato "mejor in-sample"
      en cada partición — si esta distribución está centrada muy por
      debajo del score OOS promedio de TODOS los candidatos, es más
      evidencia de que elegir "el mejor in-sample" generaliza peor que
      elegir al azar (justamente lo que mide el PBO).
    """
    if not results or not results.per_result:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["In-Sample vs Out-of-Sample (todos los candidatos)",
                         "OOS del candidato 'mejor in-sample' por partición"],
    )

    is_scores = [r.is_score for r in results.per_result]
    oos_scores = [r.oos_score for r in results.per_result]
    splits = [r.split_id for r in results.per_result]

    fig.add_trace(go.Scatter(
        x=is_scores, y=oos_scores, mode="markers",
        marker=dict(
            color=splits, colorscale=[[0, ACCENT], [1, GREEN]],
            size=6, opacity=0.6, showscale=False,
        ),
        name="Candidatos",
        hovertemplate="IS: %{x:.3f}<br>OOS: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    finite = [v for v in is_scores + oos_scores if np.isfinite(v)]
    if finite:
        lo, hi = min(finite), max(finite)
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color=RED, width=1.5, dash="dash"),
            name="IS = OOS", showlegend=True,
        ), row=1, col=1)

    if results.oos_scores_of_is_best:
        fig.add_trace(go.Histogram(
            x=results.oos_scores_of_is_best,
            marker_color=ACCENT, opacity=0.75, name="OOS (mejor IS)",
            showlegend=False,
        ), row=1, col=2)
        mean_all_oos = float(np.mean(list(results.mean_oos_by_candidate.values()))) \
            if results.mean_oos_by_candidate else None
        if mean_all_oos is not None:
            fig.add_vline(x=mean_all_oos, line=dict(color=GREEN, width=2, dash="dot"),
                          row=1, col=2)

    layout = base_layout(
        title=f"CPCV — {results.summary()}", height=height,
    )
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Score In-Sample", row=1, col=1)
    fig.update_yaxes(title_text="Score Out-of-Sample", row=1, col=1)
    fig.update_xaxes(title_text="Score OOS", row=1, col=2)
    return fig

