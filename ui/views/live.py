"""
ui/views/live.py
Mercado en Vivo — el MISMO motor de backtesting (engine/backtester.py) corre
en tiempo real sobre el histórico reciente, así que las entradas/salidas/SL/TP
que se ven aquí son exactamente las que produciría un backtest con la misma
estrategia y los mismos parámetros — no una reconstrucción aproximada aparte.

Fuente de datos: la misma que el resto de la plataforma — únicamente MT5
real (ver market_data/mt5_provider.py); sin conexión se sirve el caché
local. Se pide un número de velas acotado (elegible en el propio módulo)
para que la primera carga sea rápida en cualquier activo/temporalidad; esa
lectura queda cacheada en segundo plano y se amplía sola en cargas
posteriores.

Este módulo es puramente de VISUALIZACIÓN (solo lectura): muestra cómo se
comporta la estrategia sobre el activo elegido en tiempo real, pero no
envía ninguna orden real ni tiene panel de ejecución.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.types import BacktestConfig
from engine.backtester import BacktestEngine
from market_data.mt5_provider import get_mt5_provider
from market_data.local_provider import LocalDataProvider
from strategies import STRATEGY_REGISTRY, list_strategies
from config.settings import BACKTEST_DEFAULTS
from ui.state import AppState
from visualization.charts import candlestick_chart, indicators_chart

_DEFAULT_VISIBLE_BARS = 300


def render(asset: str, timeframe: str) -> None:
    st.markdown("## Mercado en Vivo")
    st.caption(
        "Motor en vivo: misma estrategia, mismos parámetros y el mismo motor de ejecución "
        "(`engine/backtester.py`) que en Backtesting — las entradas, salidas, Stop Loss y Take "
        "Profit que ves aquí son las que el backtest habría generado sobre estas mismas velas. "
        "**Solo visualización** — tú ejecutas la operación manualmente en tu bróker."
    )

    strategies = list_strategies()
    if not strategies:
        st.info("No hay estrategias registradas todavía. Crea una en el Constructor o descúbrela "
                 "en el Descubridor Genético.")
        return

    local = LocalDataProvider()
    mt5 = get_mt5_provider()

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        strategy_name = st.selectbox("Estrategia", strategies, key="live_selected_strategy")
    with col2:
        n_bars_load = st.selectbox(
            "Velas a cargar de MT5", [500, 1000, 3000, 5000, 10000], index=1,
            help="Cuántas velas pedir a MT5 para este activo/temporalidad. Se cachean en "
                 "segundo plano, así que la próxima carga es instantánea.",
        )
    with col3:
        auto_refresh = st.checkbox("Auto-actualizar", value=True)
    with col4:
        visible_bars = st.selectbox("Zoom inicial (velas)", [150, 300, 500, 1000], index=1)

    if not mt5.is_connected:
        st.warning(
            "MT5 no está conectado (conéctalo desde el panel lateral) — se muestra el último "
            "caché de fondo disponible, que puede no incluir las velas más recientes."
        )

    refresh_secs = st.select_slider(
        "Frecuencia de actualización", options=[10, 15, 30, 60, 120], value=30,
        help="Cada cuántos segundos se refresca la última vela y se revalida la señal.",
    ) if auto_refresh else None

    @st.fragment(run_every=refresh_secs if auto_refresh else None)
    def _live_fragment():
        _render_live_body(asset, timeframe, strategy_name, int(visible_bars),
                           int(n_bars_load), mt5, local)

    _live_fragment()


def _render_live_body(asset: str, timeframe: str, strategy_name: str,
                       visible_bars: int, n_bars_load: int, mt5, local: LocalDataProvider) -> None:
    df, load_error = _load_recent_history(asset, timeframe, n_bars_load, mt5.is_connected)
    if df is None or df.empty:
        if mt5.is_connected:
            st.error(
                f"MT5 no devolvió velas para **{asset} {timeframe}**{f': {load_error}' if load_error else ''}. "
                "Verifica que el símbolo exista para tu bróker (revisa el nombre exacto en el "
                "selector de activo) y que el mercado tenga histórico para esa temporalidad."
            )
        else:
            st.error(
                f"No hay datos cacheados para {asset} {timeframe} y MT5 no está conectado. "
                "Conéctalo desde el panel lateral para pedirlos directamente."
            )
        return

    strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
    if strategy_cls is None:
        st.error(f"La estrategia '{strategy_name}' ya no está disponible (¿fue eliminada?).")
        return

    live_cfg = AppState.get_live_config(strategy_name, asset, timeframe)
    params = live_cfg["params"] if live_cfg else {}
    bt_config_kwargs = live_cfg["config"] if live_cfg else dict(BACKTEST_DEFAULTS)

    try:
        strategy = strategy_cls(**params)
        sig_df = strategy.generate_signals(df)
    except Exception as e:
        st.error(f"La estrategia '{strategy_name}' falló al generar señales: {e}")
        return

    try:
        config = BacktestConfig(**bt_config_kwargs)
        engine = BacktestEngine(config)
        results = engine.run(sig_df, strategy_name=strategy_name, asset=asset, timeframe=timeframe)
    except Exception as e:
        st.error(f"El motor de backtest falló al reproducir las entradas en vivo: {e}")
        return

    trades_df = _trades_to_frame(results.trades, len(sig_df))
    open_trade = trades_df[trades_df["is_open"]] if not trades_df.empty else trades_df
    open_pos = open_trade.iloc[-1] if not open_trade.empty else None

    updated_at = df.index[-1]
    now = pd.Timestamp.now(tz=updated_at.tzinfo) if updated_at.tzinfo else pd.Timestamp.now()
    src = f"{mt5.source_name} (en vivo)" if mt5.is_connected else "caché local"
    st.caption(
        f"**{asset} · {timeframe} · {strategy_name}** — {len(df):,} velas cacheadas "
        f"(fuente: {src}). Última vela: **{updated_at}** · revalidado {now.strftime('%H:%M:%S')}."
    )

    last_close = float(sig_df["close"].iloc[-1])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio actual", f"{last_close:.5f}" if last_close < 100 else f"{last_close:.2f}")

    if open_pos is not None:
        direction_label = "🟢 LONG" if open_pos["direction"] == 1 else "🔴 SHORT"
        unrealized = (last_close - open_pos["entry_price"]) * open_pos["direction"]
        unrealized_pct = (unrealized / open_pos["entry_price"]) * 100 if open_pos["entry_price"] else 0.0
        m2.metric("Posición abierta", direction_label, f"{unrealized_pct:+.2f}% no realizado")
        m3.metric("Stop Loss", f"{open_pos['stop_loss']:.5f}" if pd.notna(open_pos["stop_loss"]) else "—")
        m4.metric("Take Profit", f"{open_pos['take_profit']:.5f}" if pd.notna(open_pos["take_profit"]) else "—")
        st.info(
            f"Posición abierta desde **{open_pos['entry_date']}** a **{open_pos['entry_price']:.5f}** "
            f"— reproducida por el mismo motor de backtest, todavía vigente (no ha tocado SL ni TP)."
        )
    else:
        m2.metric("Posición abierta", "— (sin posición)")
        m3.metric("Stop Loss", "—")
        m4.metric("Take Profit", "—")

    n_closed = int((~trades_df["is_open"]).sum()) if not trades_df.empty else 0
    if n_closed:
        wins = int(trades_df.loc[~trades_df["is_open"], "net_pnl"].gt(0).sum())
        st.caption(f"{n_closed} operaciones cerradas en el histórico visible · {wins} ganadoras ({wins/n_closed*100:.0f}%).")

    indicators = {c: sig_df[c] for c in strategy.get_indicator_columns() if c in sig_df.columns}

    fig = candlestick_chart(
        df=sig_df, title=f"{asset} · {timeframe} · {strategy_name}",
        signals=None,
        trades_df=trades_df, height=600,
        live_mode=True, chart_uid=f"live_{asset}_{timeframe}_{strategy_name}",
    )
    if len(sig_df) > visible_bars:
        fig.update_xaxes(range=[sig_df.index[-visible_bars], sig_df.index[-1]])

    st.plotly_chart(
        fig, width='stretch',
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    if indicators:
        st.plotly_chart(
            indicators_chart(indicators, height=260,
                              title=f"Indicadores — {strategy_name}"),
            width='stretch',
            config={"displaylogo": False},
        )

    st.caption(
        "▲ / ▼ = entrada long/short (idénticas a las del backtest) · ● = salida (verde = ganadora, "
        "roja = perdedora). Líneas discontinuas verde/roja = Take Profit / Stop Loss de cada "
        "operación. La posición abierta actual se resalta con su caja de riesgo/beneficio y las "
        "etiquetas de precio a la derecha. Arrastra para desplazarte hacia todo el histórico "
        "cacheado y usa la rueda del ratón para hacer zoom — tu posición de cámara se conserva "
        "entre actualizaciones."
    )

    if mt5.is_connected:
        st.divider()


def _trades_to_frame(trades: list, n_bars: int) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = [t.to_dict() for t in trades]
    tf = pd.DataFrame(rows)
    tf["direction"] = tf["direction"].map(lambda d: 1 if str(d).lower().startswith("long") else -1)
    tf["is_open"] = False
    if len(trades) and trades[-1].exit_reason == "end":
        tf.loc[tf.index[-1], "is_open"] = True
    return tf


@st.cache_data(ttl=15, show_spinner=False)
def _load_recent_history(asset: str, timeframe: str, n_bars: int, mt5_connected: bool):
    """
    Pide directamente a MT5 las últimas `n_bars` velas de asset/timeframe.
    No depende de ninguna descarga manual previa: si no hay nada cacheado
    todavía, MT5Provider.get_data descarga ese tramo bajo demanda y lo deja
    cacheado en segundo plano; si ya hay caché fresco lo reutiliza. Si el
    activo no está en el Market Watch del terminal, MT5Provider lo activa
    solo antes de pedir el histórico.
    Devuelve (df, mensaje_error_o_None).
    """
    mt5 = get_mt5_provider()
    local = LocalDataProvider()
    if mt5_connected:
        try:
            df = mt5.get_data(asset, timeframe, n_bars=n_bars)
            if df is not None and not df.empty:
                return df, None
        except Exception as e:
            return local.get_data(asset, timeframe), str(e)
    return local.get_data(asset, timeframe), None
