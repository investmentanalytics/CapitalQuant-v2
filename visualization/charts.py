"""
visualization/charts.py
Gráficos principales de CapitalQuant.

Todos construidos con Plotly puro — sin dependencias de Streamlit.
Los gráficos se pasan a st.plotly_chart() en la capa UI.
"""
from __future__ import annotations
from typing import Optional, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization.theme import base_layout, get_active_theme, GREEN, RED, ACCENT, TEXT, BORDER, SURFACE, GRID, COLORS
from core.data_cleaning import missing_calendar_dates


def _apply_gapless_xaxis(fig: go.Figure, index: pd.DatetimeIndex, row: int = 1, col: int = 1) -> None:
    """
    Oculta del EJE del gráfico las fechas completas sin ninguna vela
    (fines de semana, feriados de mercado) para que las velas queden
    pegadas sin espacios en blanco -- el efecto que se busca en NASDAQ,
    Oro, etc. cuando se grafica directo desde MT5.

    Esto es puramente visual (Plotly `rangebreaks`): NO reindexa, NO
    interpola ni modifica el DataFrame usado para el análisis
    estadístico -- ese sigue viviendo con su timestamp real intacto.
    """
    faltantes = missing_calendar_dates(index)
    if faltantes:
        fig.update_xaxes(rangebreaks=[dict(values=faltantes)], row=row, col=col)


# ---------------------------------------------------------------------------
# Candlestick con indicadores y señales
# ---------------------------------------------------------------------------

def candlestick_chart(
    df: pd.DataFrame,
    title: str = "",
    indicators: Optional[dict] = None,
    signals: Optional[pd.Series] = None,
    trades_df: Optional[pd.DataFrame] = None,
    height: int = 600,
    live_mode: bool = False,
    chart_uid: str = "chart",
    crosshair: bool = False,
) -> go.Figure:
    # Colores según el tema activo (claro/oscuro) elegido en la barra
    # superior — se resuelven aquí, no al importar el módulo, para que
    # alternar de tema se refleje sin reiniciar la app.
    _t = get_active_theme()
    GREEN, RED, TEXT, BORDER, SURFACE, GRID = (
        _t["green"], _t["red"], _t["text"], _t["border"], _t["surface"], _t["grid"]
    )
    """
    Gráfico de velas con indicadores overlay, señales de entrada/salida,
    SL, TP y líneas de operaciones.

    Parameters
    ----------
    df : DataFrame OHLCV
    title : Título del gráfico
    indicators : {nombre: pd.Series} para overlay en precio
    signals : Series de señales (1, -1, 0). Solo se usa para marcar entradas
        si NO se pasa `trades_df` (si se pasa trades_df, las entradas/salidas
        se dibujan a partir de ahí para no duplicar marcadores).
    trades_df : DataFrame de operaciones con columnas entry_date, exit_date,
        entry_price, stop_loss, take_profit, net_pnl y opcionalmente
        `is_open` (True para la posición abierta actual, cuyo SL/TP se
        resalta con una caja y etiquetas de precio).
    live_mode : si True, fija `uirevision` (constante) para que el pan/zoom
        del usuario NO se resetee cada vez que Streamlit vuelve a dibujar el
        gráfico en un auto-refresh, y activa `dragmode='pan'` para navegar
        arrastrando en vez de seleccionar (estilo TradingView).
    chart_uid : identificador estable del gráfico, usado como `uirevision`
        cuando `live_mode=True` — debe ser el mismo entre refrescos para que
        el zoom se conserve, y distinto entre gráficos distintos (activo/
        estrategia) para que cada uno arranque con su propia vista.
    crosshair : si True, activa líneas guía (spikes) en ambos ejes que
        siguen al cursor — el retículo estilo TradingView — y prepara el
        estilo de las herramientas de dibujo (`newshape`) para cuando la
        UI agregue los botones `drawline`/`drawrect`/`drawcircle` a la
        barra de herramientas de Plotly (`modeBarButtonsToAdd`, se
        configura al llamar `st.plotly_chart(..., config=...)`, no aquí).
    """
    # Panel de volumen solo si hay datos
    has_volume = "volume" in df.columns and df["volume"].sum() > 0
    row_heights = [0.75, 0.25] if has_volume else [1.0]
    n_rows = 2 if has_volume else 1

    specs = [[{"secondary_y": True}]] + ([[{"secondary_y": False}]] if has_volume else [])
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        specs=specs,
    )

    # --- Velas ---
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        increasing_line_color=GREEN,
        decreasing_line_color=RED,
        increasing_fillcolor=GREEN,
        decreasing_fillcolor=RED,
        line=dict(width=1),
        name="OHLC",
        showlegend=False,
    ), row=1, col=1, secondary_y=False)

    # --- Indicadores: YA NO se dibujan aquí ---
    # Se movieron a un gráfico separado (ver `indicators_chart` más abajo) para
    # que las velas, entradas/salidas y líneas de SL/TP queden legibles sin
    # líneas de indicadores superpuestas. `indicators` se conserva como
    # parámetro (ignorado si se pasa) solo por compatibilidad hacia atrás.
    any_secondary_used = False

    # --- Señales por transición (solo si no hay trades_df) ---
    # Marcar cada vela donde la señal cruda está activa produce un "enjambre"
    # de flechas cuando la señal se mantiene varias velas seguidas. En vez de
    # eso, solo marcamos el punto donde la señal CAMBIA (nueva entrada).
    if trades_df is None and signals is not None and not signals.empty:
        s = signals.reindex(df.index).fillna(0)
        entries = s[(s != 0) & (s != s.shift(1).fillna(0))]
        longs = df.loc[entries[entries == 1].index]
        shorts = df.loc[entries[entries == -1].index]

        if not longs.empty:
            fig.add_trace(go.Scatter(
                x=longs.index, y=longs["low"] * 0.998,
                mode="markers",
                marker=dict(symbol="triangle-up", size=11, color=GREEN,
                            line=dict(width=1, color=GREEN)),
                name="Entrada long",
                showlegend=True,
            ), row=1, col=1)

        if not shorts.empty:
            fig.add_trace(go.Scatter(
                x=shorts.index, y=shorts["high"] * 1.002,
                mode="markers",
                marker=dict(symbol="triangle-down", size=11, color=RED,
                            line=dict(width=1, color=RED)),
                name="Entrada short",
                showlegend=True,
            ), row=1, col=1)

    # --- Trades históricos: entrada, salida, línea de operación y SL/TP ---
    #
    # RENDIMIENTO: la versión anterior llamaba fig.add_shape() 1-3 veces POR
    # TRADE (línea de operación + SL + TP), es decir hasta ~3.000 shapes
    # individuales en un backtest con 1.000 trades. Cada add_shape() es una
    # entrada de layout independiente que Plotly tiene que serializar y el
    # navegador tiene que renderizar por separado — esto es lo que hacía que
    # el gráfico de precio tardara varios segundos (a veces se sentía
    # "congelado") en backtests con muchos trades.
    #
    # Aquí, en cambio, TODAS las líneas de operación se agrupan en un puñado
    # de trazas go.Scatter (una por color/estilo: operación ganadora,
    # perdedora, SL, TP), separando cada segmento con un punto `None` en x/y
    # — la técnica estándar de Plotly para dibujar miles de segmentos de
    # línea independientes como una sola traza. Mismo resultado visual,
    # una fracción del costo de construcción y de renderizado.
    MAX_TRADE_LINES = 400  # por encima de esto, las líneas de conexión dejan
    # de aportar (se amontonan visualmente) y cuestan más de lo que valen;
    # los marcadores de entrada/salida (ya vectorizados abajo) siguen
    # mostrando todos los trades sin excepción.

    if trades_df is not None and not trades_df.empty:
        last_x = df.index[-1]
        entry_x_long, entry_y_long = [], []
        entry_x_short, entry_y_short = [], []
        exit_x_win, exit_y_win = [], []
        exit_x_loss, exit_y_loss = [], []

        win_line_x, win_line_y = [], []
        loss_line_x, loss_line_y = [], []
        sl_line_x, sl_line_y = [], []
        tp_line_x, tp_line_y = [], []

        n_trades_total = len(trades_df)
        draw_connector_lines = n_trades_total <= MAX_TRADE_LINES

        for _, trade in trades_df.iterrows():
            is_open = bool(trade.get("is_open", False))
            raw_direction = trade.get("direction", 1)
            if isinstance(raw_direction, str):
                direction = 1 if raw_direction.lower().startswith("long") else -1
            else:
                direction = int(raw_direction)
            net_pnl = trade.get("net_pnl", 0) or 0
            color = GREEN if net_pnl >= 0 else RED
            x1 = last_x if is_open else trade["exit_date"]
            entry_x, entry_y = trade["entry_date"], trade["entry_price"]

            if direction == 1:
                entry_x_long.append(entry_x); entry_y_long.append(entry_y)
            else:
                entry_x_short.append(entry_x); entry_y_short.append(entry_y)

            if not is_open:
                exit_price = trade.get("exit_price", entry_y)
                if net_pnl >= 0:
                    exit_x_win.append(trade["exit_date"]); exit_y_win.append(exit_price)
                else:
                    exit_x_loss.append(trade["exit_date"]); exit_y_loss.append(exit_price)

            # Siempre se dibuja la línea de conexión de una posición ABIERTA
            # (normalmente 0 o 1) — el cap solo aplica a trades cerrados,
            # que son los que pueden llegar a miles.
            if draw_connector_lines or is_open:
                target_x, target_y = (win_line_x, win_line_y) if net_pnl >= 0 else (loss_line_x, loss_line_y)
                target_x += [entry_x, x1, None]
                target_y += [entry_y, entry_y, None]

                sl, tp = trade.get("stop_loss"), trade.get("take_profit")
                if sl is not None and not pd.isna(sl):
                    sl_line_x += [entry_x, x1, None]
                    sl_line_y += [sl, sl, None]
                if tp is not None and not pd.isna(tp):
                    tp_line_x += [entry_x, x1, None]
                    tp_line_y += [tp, tp, None]

            # Posición abierta: caja riesgo/beneficio resaltada + etiquetas
            # (como mucho una posición abierta por estrategia — costo fijo,
            # no escala con el número de trades históricos).
            if is_open:
                sl, tp = trade.get("stop_loss"), trade.get("take_profit")
                if sl is not None and not pd.isna(sl):
                    fig.add_shape(
                        type="rect", x0=entry_x, x1=last_x,
                        y0=entry_y, y1=sl,
                        fillcolor=RED, opacity=0.08, line_width=0, row=1, col=1,
                    )
                    fig.add_annotation(
                        x=last_x, y=sl, text=f"SL {sl:.5g}", showarrow=False,
                        xanchor="left", font=dict(color=RED, size=11),
                        bgcolor=SURFACE, row=1, col=1,
                    )
                if tp is not None and not pd.isna(tp):
                    fig.add_shape(
                        type="rect", x0=entry_x, x1=last_x,
                        y0=entry_y, y1=tp,
                        fillcolor=GREEN, opacity=0.08, line_width=0, row=1, col=1,
                    )
                    fig.add_annotation(
                        x=last_x, y=tp, text=f"TP {tp:.5g}", showarrow=False,
                        xanchor="left", font=dict(color=GREEN, size=11),
                        bgcolor=SURFACE, row=1, col=1,
                    )
                entry_label = "LONG" if direction == 1 else "SHORT"
                fig.add_annotation(
                    x=last_x, y=entry_y,
                    text=f"{entry_label} @ {entry_y:.5g}",
                    showarrow=False, xanchor="left", font=dict(color=color, size=11),
                    bgcolor=SURFACE, row=1, col=1,
                )

        if win_line_x:
            fig.add_trace(go.Scatter(
                x=win_line_x, y=win_line_y, mode="lines",
                line=dict(color=GREEN, width=1, dash="dot"),
                connectgaps=False, hoverinfo="skip", showlegend=False,
            ), row=1, col=1)
        if loss_line_x:
            fig.add_trace(go.Scatter(
                x=loss_line_x, y=loss_line_y, mode="lines",
                line=dict(color=RED, width=1, dash="dot"),
                connectgaps=False, hoverinfo="skip", showlegend=False,
            ), row=1, col=1)
        if sl_line_x:
            fig.add_trace(go.Scatter(
                x=sl_line_x, y=sl_line_y, mode="lines",
                line=dict(color=RED, width=0.8, dash="dash"),
                connectgaps=False, hoverinfo="skip", showlegend=False,
            ), row=1, col=1)
        if tp_line_x:
            fig.add_trace(go.Scatter(
                x=tp_line_x, y=tp_line_y, mode="lines",
                line=dict(color=GREEN, width=0.8, dash="dash"),
                connectgaps=False, hoverinfo="skip", showlegend=False,
            ), row=1, col=1)
        if not draw_connector_lines:
            fig.add_annotation(
                text=(f"Líneas de conexión/SL/TP ocultas para {n_trades_total} trades "
                      f"(> {MAX_TRADE_LINES}) — los marcadores de entrada/salida siguen "
                      f"mostrando todos los trades."),
                xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False,
                font=dict(color=TEXT, size=11), align="center",
            )

        if entry_x_long:
            fig.add_trace(go.Scatter(
                x=entry_x_long, y=entry_y_long, mode="markers",
                marker=dict(symbol="triangle-up", size=11, color=GREEN, line=dict(width=1, color=GREEN)),
                name="Entrada long", showlegend=True,
            ), row=1, col=1)
        if entry_x_short:
            fig.add_trace(go.Scatter(
                x=entry_x_short, y=entry_y_short, mode="markers",
                marker=dict(symbol="triangle-down", size=11, color=RED, line=dict(width=1, color=RED)),
                name="Entrada short", showlegend=True,
            ), row=1, col=1)
        if exit_x_win:
            fig.add_trace(go.Scatter(
                x=exit_x_win, y=exit_y_win, mode="markers",
                marker=dict(symbol="circle", size=8, color=GREEN, line=dict(width=1.5, color=SURFACE)),
                name="Salida (ganadora)", showlegend=True,
            ), row=1, col=1)
        if exit_x_loss:
            fig.add_trace(go.Scatter(
                x=exit_x_loss, y=exit_y_loss, mode="markers",
                marker=dict(symbol="circle", size=8, color=RED, line=dict(width=1.5, color=SURFACE)),
                name="Salida (perdedora)", showlegend=True,
            ), row=1, col=1)

    # --- Volumen ---
    if has_volume:
        colors = [GREEN if c >= o else RED
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"],
            marker_color=colors,
            marker_opacity=0.6,
            name="Volume",
            showlegend=False,
        ), row=2, col=1)

    layout = base_layout(title=title, height=height)
    layout["xaxis_rangeslider_visible"] = False
    # yaxis2 = eje secundario del panel de precio (osciladores tipo CCI,
    # flags de régimen, etc. — ver bloque de indicadores arriba). Se deja
    # sin grid propio para no duplicar líneas sobre las velas, y sin
    # etiquetas cuando ningún indicador lo usó (para no mostrar un eje
    # vacío/confuso).
    layout["yaxis2"] = dict(
        overlaying="y", side="right", showgrid=False,
        showticklabels=any_secondary_used,
        title="Oscilador" if any_secondary_used else None,
        zeroline=False,
    )
    if has_volume:
        layout["xaxis2"] = dict(gridcolor=GRID, linecolor=BORDER, showgrid=True)
        layout["yaxis3"] = dict(gridcolor=GRID, linecolor=BORDER, showgrid=True, title="Volume")
    if live_mode:
        # uirevision constante entre refrescos = Plotly conserva el zoom/pan
        # que el usuario ya había hecho en vez de volver a autoescalar cada
        # vez que llega una vela nueva (que es el "hundido en autoescala"
        # que se veía antes). dragmode='pan' + hovermode más ligero para que
        # se sienta como arrastrar en TradingView en vez de un rectángulo de
        # selección.
        layout["uirevision"] = chart_uid
        layout["dragmode"] = "pan"
        layout["hovermode"] = "x unified"
    if crosshair:
        # Retículo estilo TradingView: línea vertical+horizontal punteada
        # que sigue al cursor, con la etiqueta del valor en cada eje.
        layout["hovermode"] = "x unified"
        layout["xaxis"] = {
            **layout.get("xaxis", {}),
            "showspikes": True, "spikemode": "across", "spikesnap": "cursor",
            "spikedash": "dot", "spikethickness": 1, "spikecolor": ACCENT,
        }
        layout["yaxis"] = {
            **layout.get("yaxis", {}),
            "showspikes": True, "spikemode": "across", "spikesnap": "cursor",
            "spikedash": "dot", "spikethickness": 1, "spikecolor": ACCENT,
        }
        # Estilo de las herramientas de medida/dibujo (líneas, rectángulos,
        # círculos) que la UI habilita vía modeBarButtonsToAdd.
        layout["newshape"] = dict(line_color=ACCENT, line_width=1.5, opacity=0.85)
    fig.update_layout(**layout)

    # Eliminar del eje los espacios en blanco de fin de semana/feriados
    # (NASDAQ, Oro, etc.) sin tocar los datos usados en el análisis.
    _apply_gapless_xaxis(fig, df.index, row=1, col=1)

    return fig


# ---------------------------------------------------------------------------
# Equity Curve + Drawdown
# ---------------------------------------------------------------------------

def _downsample_minmax(series: pd.Series, max_points: int = 6000) -> pd.Series:
    """
    Reduce una serie a como mucho `max_points` puntos SIN perder picos ni
    valles (a diferencia de un stride simple `series[::n]`, que puede saltar
    justo por encima del máximo drawdown o el pico de equity real y
    mostrar una curva más suave/optimista de lo que realmente fue).

    Divide la serie en buckets de tamaño aproximadamente igual y de cada
    uno conserva el primer, el mínimo, el máximo y el último punto (en
    orden temporal, sin duplicados) — el mismo principio que usan las
    librerías de graficado financiero (p.ej. LTTB/min-max decimation) para
    que miles de puntos se vean prácticamente idénticos con una fracción
    del costo de renderizado. Curvas de equity de backtests intradía con
    cientos de miles de velas son las que más se benefician: sin esto,
    Plotly tiene que serializar y el navegador dibujar cada punto.
    """
    n = len(series)
    if n <= max_points:
        return series

    n_buckets = max(1, max_points // 4)
    bucket_size = int(np.ceil(n / n_buckets))
    keep_positions: set[int] = {0, n - 1}
    for start in range(0, n, bucket_size):
        end = min(start + bucket_size, n)
        if end <= start:
            continue
        bucket_vals = series.values[start:end]
        keep_positions.add(start + int(np.argmin(bucket_vals)))
        keep_positions.add(start + int(np.argmax(bucket_vals)))
        keep_positions.add(end - 1)
    ordered = sorted(keep_positions)
    return series.iloc[ordered]


def equity_curve_chart(
    curves: List[dict],
    height: int = 500,
    show_drawdown: bool = True,
) -> go.Figure:
    """
    Gráfico de curva de capital con drawdown opcional.

    Parameters
    ----------
    curves : Lista de dicts con keys:
        - name: str
        - equity: pd.Series
        - drawdown: pd.Series (opcional)
        - best: bool (resaltar como mejor)
    """
    n_rows = 2 if show_drawdown else 1
    row_heights = [0.65, 0.35] if show_drawdown else [1.0]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )

    for i, curve in enumerate(curves):
        color = COLORS[i % len(COLORS)]
        is_best = curve.get("best", False)
        opacity = 1.0 if is_best else 0.65
        width = 2.5 if is_best else 1.2

        equity_plot = _downsample_minmax(curve["equity"])

        fig.add_trace(go.Scatter(
            x=equity_plot.index,
            y=equity_plot.values,
            name=curve["name"],
            line=dict(color=color, width=width),
            opacity=opacity,
            fill="tozeroy" if len(curves) == 1 else None,
            fillcolor=f"rgba{tuple(list(_hex_to_rgb(color)) + [0.08])}" if len(curves) == 1 else None,
        ), row=1, col=1)

        if show_drawdown and "drawdown" in curve and curve["drawdown"] is not None:
            dd_plot = _downsample_minmax(curve["drawdown"])
            fig.add_trace(go.Scatter(
                x=dd_plot.index,
                y=dd_plot.values,
                name=f"DD {curve['name']}",
                line=dict(color=color, width=1),
                fill="tozeroy",
                fillcolor=f"rgba({','.join(map(str, _hex_to_rgb(color)))},0.15)",
                showlegend=False,
            ), row=2, col=1)

    layout = base_layout(height=height)
    layout["yaxis_tickprefix"] = "$"
    layout["yaxis_tickformat"] = ",.0f"
    if show_drawdown:
        layout["yaxis2"] = dict(
            gridcolor=GRID, linecolor=BORDER, ticksuffix="%",
            title="Drawdown", gridwidth=0.5,
        )
    fig.update_layout(**layout)

    return fig


# ---------------------------------------------------------------------------
# Histograma de distribución de trades
# ---------------------------------------------------------------------------

def trades_histogram(trades_df: pd.DataFrame, height: int = 350) -> go.Figure:
    """Distribución de ganancias/pérdidas por operación."""
    if trades_df is None or trades_df.empty or "net_pnl" not in trades_df.columns:
        return go.Figure()

    pnls = trades_df["net_pnl"].dropna()
    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    fig = go.Figure()

    if not losses.empty:
        fig.add_trace(go.Histogram(
            x=losses, name="Pérdidas",
            marker_color=RED, opacity=0.75,
            nbinsx=20,
        ))

    if not wins.empty:
        fig.add_trace(go.Histogram(
            x=wins, name="Ganancias",
            marker_color=GREEN, opacity=0.75,
            nbinsx=20,
        ))

    # Línea de media
    if not pnls.empty:
        mean_val = pnls.mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color=ACCENT,
            annotation_text=f"Media: ${mean_val:,.2f}",
            annotation_font_color=ACCENT,
        )

    layout = base_layout(title="Distribución de Operaciones", height=height)
    layout["barmode"] = "overlay"
    layout["xaxis_tickprefix"] = "$"
    fig.update_layout(**layout)

    return fig


# ---------------------------------------------------------------------------
# Panel de indicadores (separado del gráfico de precio)
# ---------------------------------------------------------------------------

def indicators_chart(indicators: dict, height: int = 280, title: str = "Indicadores") -> go.Figure:
    """
    Gráfico independiente para las líneas de indicadores de una estrategia
    (SMA/EMA, RSI, MACD, ADX, bandas, osciladores, etc.), separado a
    propósito del gráfico de velas + operaciones para que ambos queden
    legibles: aquí no compiten por espacio ni escala con el precio.

    Igual que en el gráfico de precio, los indicadores cuya escala se aleja
    mucho de la mayoría (p.ej. un oscilador 0-100 junto a una media móvil en
    escala de precio) se separan automáticamente a un eje Y secundario.

    Parameters
    ----------
    indicators : {nombre: pd.Series}
    """
    GRID = get_active_theme()["grid"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not indicators:
        fig.update_layout(**base_layout(title=title, height=height))
        return fig

    # Los indicadores pueden incluir banderas booleanas (por ejemplo,
    # condiciones de régimen). Plotly puede dibujarlas, pero pandas/numpy no
    # admite cuantiles sobre bool en todas las combinaciones de versiones:
    # internamente el cálculo termina intentando hacer True - False.
    # Convertimos aquí a series numéricas y descartamos explícitamente bool,
    # objetos y valores infinitos. Así el gráfico de indicadores nunca puede
    # tumbar el Research Lab/Descubridor por una serie auxiliar no numérica.
    clean = {}
    for name, raw in indicators.items():
        if raw is None:
            continue
        try:
            series = raw if isinstance(raw, pd.Series) else pd.Series(raw)
            if pd.api.types.is_bool_dtype(series.dtype):
                continue
            series = pd.to_numeric(series, errors="coerce")
            series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if not series.empty:
                clean[name] = series
        except Exception:
            continue

    if not clean:
        fig.update_layout(**base_layout(title=title, height=height))
        return fig

    # Escala "típica" de cada serie (rango intercuartílico como referencia
    # robusta a outliers puntuales). Todas las series llegan aquí como
    # numéricas, por lo que no existe resta booleana en pandas/numpy.
    scales = {}
    for name, series in clean.items():
        q75 = float(series.quantile(0.75))
        q25 = float(series.quantile(0.25))
        median = float(series.median())
        scale = max(q75 - q25, abs(median) * 1e-6, 1e-9)
        scales[name] = scale
    ref_scale = sorted(scales.values())[len(scales) // 2]  # escala mediana como referencia del eje primario

    any_secondary_used = False
    for i, (name, series) in enumerate(clean.items()):
        color = COLORS[i % len(COLORS)]
        ratio = scales[name] / ref_scale if ref_scale else 1.0
        use_secondary = ratio > 6 or ratio < 1 / 6
        any_secondary_used = any_secondary_used or use_secondary
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name=name,
            line=dict(color=color, width=1.5),
            opacity=0.9,
        ), secondary_y=use_secondary)

    layout = base_layout(title=title, height=height)
    fig.update_layout(**layout)
    fig.update_yaxes(gridcolor=GRID, secondary_y=False)
    if any_secondary_used:
        fig.update_yaxes(showgrid=False, secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# RSI chart
# ---------------------------------------------------------------------------

def rsi_chart(df: pd.DataFrame, height: int = 200) -> go.Figure:
    """Gráfico de RSI con zonas de sobrecompra/sobreventa."""
    if "rsi" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi"],
        name="RSI",
        line=dict(color=ACCENT, width=1.5),
    ))

    for level, color, label in [(70, RED, "Sobrecompra"), (30, GREEN, "Sobreventa"), (50, BORDER, "")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color,
                      annotation_text=label if label else "",
                      annotation_font_color=color, line_width=0.8)

    # Colorear zona entre 30 y 70
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(201,169,97,0.07)", line_width=0)

    layout = base_layout(height=height)
    layout["yaxis"] = dict(
        gridcolor=GRID, range=[0, 100],
        tickvals=[0, 30, 50, 70, 100],
    )
    fig.update_layout(**layout)

    return fig


# ---------------------------------------------------------------------------
# Historial de optimización
# ---------------------------------------------------------------------------

def optimization_history_chart(history_df: pd.DataFrame, objective: str = "sharpe", height: int = 300) -> go.Figure:
    """Evolución del score durante la optimización."""
    if history_df is None or history_df.empty or "value" not in history_df.columns:
        return go.Figure()

    best_so_far = history_df["value"].cummax()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["trial"], y=history_df["value"],
        name="Score trial",
        mode="markers",
        marker=dict(color=ACCENT, size=4, opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=history_df["trial"], y=best_so_far,
        name="Mejor hasta ahora",
        line=dict(color=GREEN, width=2),
    ))

    layout = base_layout(title=f"Historial de Optimización ({objective})", height=height)
    layout["xaxis_title"] = "Trial"
    layout["yaxis_title"] = objective.capitalize()
    fig.update_layout(**layout)

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple:
    """Convierte color hex a tupla RGB."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
