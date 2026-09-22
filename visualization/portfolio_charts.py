"""
visualization/portfolio_charts.py
Gráficos específicos del módulo de portfolio multi-activo.

Incluye: pie, treemap, sankey, heatmap de correlación,
equity consolidada, descomposición de riesgo.
"""
from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from portfolio.manager import PortfolioResults
from visualization.theme import (
    base_layout, GREEN, RED, ACCENT, TEXT, BORDER,
    SURFACE, GRID, COLORS, BG,
)


# ---------------------------------------------------------------------------
# Equity consolidada del portfolio
# ---------------------------------------------------------------------------

def portfolio_equity_chart(
    pr: PortfolioResults,
    show_components: bool = True,
    height: int = 500,
) -> go.Figure:
    """
    Equity curve del portfolio + curvas individuales por componente
    + drawdown.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.65, 0.35],
    )

    # Componentes individuales (tenues)
    if show_components and pr.component_equity:
        for i, (name, equity) in enumerate(pr.component_equity.items()):
            color = COLORS[i % len(COLORS)]
            fig.add_trace(go.Scatter(
                x=equity.index, y=equity.values,
                name=name.split("::")[-1],
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.45,
                showlegend=True,
            ), row=1, col=1)

    # Portfolio consolidado (destacado)
    fig.add_trace(go.Scatter(
        x=pr.equity_curve.index,
        y=pr.equity_curve.values,
        name="Portfolio Total",
        line=dict(color=ACCENT, width=3),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.07)",
    ), row=1, col=1)

    # Drawdown del portfolio
    if pr.drawdown_series is not None and not pr.drawdown_series.empty:
        fig.add_trace(go.Scatter(
            x=pr.drawdown_series.index,
            y=pr.drawdown_series.values,
            name="Drawdown",
            line=dict(color=RED, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(248,81,73,0.12)",
            showlegend=False,
        ), row=2, col=1)

    layout = base_layout(title="Portfolio — Equity Curve", height=height)
    layout["yaxis_tickprefix"] = "$"
    layout["yaxis_tickformat"] = ",.0f"
    layout["yaxis2"] = dict(
        gridcolor=GRID, linecolor=BORDER, ticksuffix="%",
        title="Drawdown %",
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Pie chart de asignación de capital
# ---------------------------------------------------------------------------

def capital_allocation_pie(
    pr: PortfolioResults,
    group_by: str = "asset",  # "asset" | "strategy" | "component"
    height: int = 400,
) -> go.Figure:
    """Pie chart de asignación de capital por activo, estrategia o componente."""

    if group_by == "asset":
        labels = list(pr.asset_weights.keys())
        values = [pr.asset_weights[k] * 100 for k in labels]
        title = "Asignación por Activo"

    elif group_by == "strategy":
        # Agrupar por nombre de estrategia (sin activo)
        strat_weights: Dict[str, float] = {}
        for comp, w in pr.component_weights.items():
            _, strat = comp.split("::", 1)
            strat_weights[strat] = strat_weights.get(strat, 0) + w
        labels = list(strat_weights.keys())
        values = [strat_weights[k] * 100 for k in labels]
        title = "Asignación por Estrategia"

    else:  # component
        labels = list(pr.component_weights.keys())
        values = [pr.component_weights[k] * 100 for k in labels]
        title = "Asignación por Componente"

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=COLORS[:len(labels)], line=dict(color=BG, width=2)),
        textinfo="label+percent",
        textfont=dict(color=TEXT, size=12),
        hovertemplate="<b>%{label}</b><br>Peso: %{value:.1f}%<extra></extra>",
    ))

    layout = base_layout(title=title, height=height)
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Treemap de asignación
# ---------------------------------------------------------------------------

def capital_allocation_treemap(pr: PortfolioResults, height: int = 400) -> go.Figure:
    """Treemap jerárquico activo  estrategia  peso."""

    ids, labels, parents, values = [], [], [], []

    # Raíz
    ids.append("Portfolio")
    labels.append("Portfolio")
    parents.append("")
    values.append(100.0)

    # Activos
    for asset, aw in pr.asset_weights.items():
        ids.append(asset)
        labels.append(asset)
        parents.append("Portfolio")
        values.append(round(aw * 100, 2))

    # Componentes
    for comp, cw in pr.component_weights.items():
        asset, strat = comp.split("::", 1)
        ids.append(comp)
        labels.append(strat)
        parents.append(asset)
        values.append(round(cw * 100, 2))

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=values,
            colorscale=[[0, SURFACE], [1, ACCENT]],
            line=dict(color=BG, width=2),
        ),
        texttemplate="<b>%{label}</b><br>%{value:.1f}%",
        hovertemplate="<b>%{label}</b><br>Peso: %{value:.1f}%<extra></extra>",
    ))

    layout = base_layout(title="Mapa de Capital", height=height)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Sankey diagram — flujo de capital
# ---------------------------------------------------------------------------

def capital_sankey(pr: PortfolioResults, height: int = 450) -> go.Figure:
    """
    Diagrama Sankey que muestra el flujo de capital:
    Portfolio  Activo  Estrategia
    """
    nodes = ["Portfolio"]
    assets = list(pr.asset_weights.keys())
    components = list(pr.component_weights.keys())

    nodes += assets
    nodes += [comp.split("::")[-1] + f" ({comp.split('::')[0]})" for comp in components]

    node_idx = {n: i for i, n in enumerate(nodes)}

    sources, targets, values_flow = [], [], []

    # Portfolio  Activos
    for asset, aw in pr.asset_weights.items():
        sources.append(node_idx["Portfolio"])
        targets.append(node_idx[asset])
        values_flow.append(round(aw * 100, 2))

    # Activos  Componentes
    for comp, cw in pr.component_weights.items():
        asset = comp.split("::")[0]
        strat = comp.split("::")[-1]
        node_label = strat + f" ({asset})"
        sources.append(node_idx[asset])
        targets.append(node_idx[node_label])
        values_flow.append(round(cw * 100, 2))

    node_colors = [ACCENT] + COLORS[:len(assets)] + COLORS[:len(components)]

    fig = go.Figure(go.Sankey(
        node=dict(
            label=nodes,
            color=node_colors[:len(nodes)],
            line=dict(color=BORDER, width=0.5),
            pad=20,
            thickness=20,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values_flow,
            color="rgba(88,166,255,0.2)",
        ),
    ))

    layout = base_layout(title="Flujo de Capital", height=height)
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Heatmap de correlación
# ---------------------------------------------------------------------------

def correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlación",
    height: int = 400,
) -> go.Figure:
    """Heatmap de correlación con escala divergente rojo-azul."""
    if corr_matrix is None or corr_matrix.empty:
        return go.Figure()

    labels = list(corr_matrix.columns)
    z = corr_matrix.values.round(2)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=[
            [0.0,  "#ff4d4d"],   # -1  rojo (alta correlación negativa)
            [0.5,  SURFACE],     #  0  neutro
            [1.0,  "#f2c200"],   # +1  oro (alta correlación positiva)
        ],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11, color=TEXT),
        hovertemplate="<b>%{y} × %{x}</b><br>Correlación: %{z:.3f}<extra></extra>",
    ))

    layout = base_layout(title=title, height=height)
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    layout["xaxis"] = dict(tickfont=dict(size=10, color=TEXT))
    layout["yaxis"] = dict(tickfont=dict(size=10, color=TEXT), autorange="reversed")
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Descomposición de riesgo — barras
# ---------------------------------------------------------------------------

def risk_decomposition_chart(
    risk_by_asset: Dict[str, float],
    risk_by_strategy: Dict[str, float],
    height: int = 350,
) -> go.Figure:
    """Gráfico de barras comparando riesgo por activo y por estrategia."""

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Riesgo por Activo", "Riesgo por Componente"],
    )

    # Por activo
    assets = list(risk_by_asset.keys())
    asset_risks = [abs(v) * 100 for v in risk_by_asset.values()]
    fig.add_trace(go.Bar(
        x=assets, y=asset_risks,
        marker_color=COLORS[:len(assets)],
        name="Por Activo",
        showlegend=False,
    ), row=1, col=1)

    # Por componente
    comps = [c.split("::")[-1] for c in risk_by_strategy.keys()]
    comp_risks = [abs(v) * 100 for v in risk_by_strategy.values()]
    fig.add_trace(go.Bar(
        x=comps, y=comp_risks,
        marker_color=COLORS[:len(comps)],
        name="Por Componente",
        showlegend=False,
    ), row=1, col=2)

    layout = base_layout(title="Descomposición de Riesgo (Volatilidad Ponderada %)", height=height)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Tabla comparativa de componentes
# ---------------------------------------------------------------------------

def component_comparison_table(summary_df: pd.DataFrame) -> go.Figure:
    """Tabla visual comparativa de todos los componentes del portfolio."""
    if summary_df is None or summary_df.empty:
        return go.Figure()

    cols_display = {
        "component":    "Componente",
        "weight":       "Peso %",
        "net_profit_%": "Net Profit %",
        "cagr_%":       "CAGR %",
        "sharpe":       "Sharpe",
        "max_dd_%":     "Max DD %",
        "win_rate_%":   "Win Rate %",
        "trades":       "Trades",
    }

    available = [c for c in cols_display if c in summary_df.columns]
    df = summary_df[available].copy()

    def _color_cell(col, vals):
        colors = []
        for v in vals:
            try:
                fv = float(v)
                if col in ("net_profit_%", "cagr_%", "sharpe", "win_rate_%"):
                    colors.append(GREEN if fv > 0 else RED)
                elif col == "max_dd_%":
                    colors.append(RED if fv < -20 else (
                        "#ffe14d" if fv < -10 else GREEN))
                else:
                    colors.append(TEXT)
            except Exception:
                colors.append(TEXT)
        return colors

    cell_colors = []
    for col in available:
        cell_colors.append(_color_cell(col, df[col].values))

    fig = go.Figure(go.Table(
        header=dict(
            values=[cols_display[c] for c in available],
            fill_color=SURFACE,
            font=dict(color=TEXT, size=12),
            align="center",
            line_color=BORDER,
            height=35,
        ),
        cells=dict(
            values=[df[c].values for c in available],
            fill_color=BG,
            font_color=cell_colors,
            align="center",
            line_color=BORDER,
            height=30,
        ),
    ))

    fig = go.Figure(go.Table(
        header=dict(
            values=[cols_display[c] for c in available],
            fill_color=SURFACE,
            font=dict(color=TEXT, size=12),
            align="center",
        ),
        cells=dict(
            values=[df[c].values for c in available],
            fill_color=BG,
            font=dict(color=TEXT, size=11),
            align="center",
        ),
    ))

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=0, r=0, t=0, b=0),
        height=max(200, 35 + 30 * len(df)),
    )
    return fig
