"""
ui/app.py
CapitalQuant 3.3 — Entry point.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

try:
    # Envuelto en try/except únicamente para permitir que este mismo archivo
    # se ejecute embebido dentro del router de la suite (main.py), que ya
    # llama a set_page_config una vez. Al ejecutar este archivo solo
    # (streamlit run ui/app.py), el comportamiento es exactamente el mismo
    # de siempre. No se modifica ninguna otra lógica.
    st.set_page_config(
        page_title="CapitalQuant",
        page_icon=":material/candlestick_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except st.errors.StreamlitAPIException:
    pass

if "cq_theme" not in st.session_state:
    st.session_state.cq_theme = "light"

# ─────────────────────────────────────────────────────────────────────────
# Variables de tema — modo claro (blanco/amarillo original) y modo oscuro.
# Todo el CSS de abajo consume estas variables con var(--cq-*), así que
# alternar de tema es solo cambiar este bloque de valores; ningún otro
# selector necesita tocarse.
# ─────────────────────────────────────────────────────────────────────────
_CQ_THEMES = {
    "light": {
        "bg": "#ffffff", "bg-2": "#fbfaf3", "surface": "#ffffff", "surface-2": "#f6f5ec",
        "border": "#e3e1d3", "border-2": "#cfcdb8", "text": "#14140c", "muted": "#6e6c5e",
        "accent": "#b8860b", "accent-2": "#f2c200", "accent-dim": "#e6d9a8",
        "green": "#1a9850", "red": "#d1403d",
        "glow": "rgba(184,134,11,0.22)", "glow-soft": "rgba(184,134,11,0.10)",
    },
    "dark": {
        "bg": "#0f0e0a", "bg-2": "#17150f", "surface": "#181611", "surface-2": "#1f1c15",
        "border": "#332f22", "border-2": "#4a4531", "text": "#f5f3e9", "muted": "#a8a48f",
        "accent": "#f2c200", "accent-2": "#ffd633", "accent-dim": "#7a6a2a",
        "green": "#2ecc71", "red": "#e5544f",
        "glow": "rgba(242,194,0,0.22)", "glow-soft": "rgba(242,194,0,0.12)",
    },
}
_cq_v = _CQ_THEMES[st.session_state.cq_theme]

_cq_root_vars = f"""
    :root {{
        --cq-bg:        {_cq_v["bg"]};
        --cq-bg-2:      {_cq_v["bg-2"]};
        --cq-surface:   {_cq_v["surface"]};
        --cq-surface-2: {_cq_v["surface-2"]};
        --cq-border:    {_cq_v["border"]};
        --cq-border-2:  {_cq_v["border-2"]};
        --cq-text:      {_cq_v["text"]};
        --cq-muted:     {_cq_v["muted"]};
        --cq-accent:    {_cq_v["accent"]};
        --cq-accent-2:  {_cq_v["accent-2"]};
        --cq-accent-dim:{_cq_v["accent-dim"]};
        --cq-green:     {_cq_v["green"]};
        --cq-red:       {_cq_v["red"]};
        --cq-glow:      {_cq_v["glow"]};
        --cq-glow-soft: {_cq_v["glow-soft"]};
        --cq-mono: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, monospace;
    }}
"""

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

    /* ══════════════════════════════════════════════════════════════
       CapitalQuant — identidad "Terminal de Mercado"
       Blanco/negro · amarillo · monoespaciado, denso, paneles con
       barra de título, todo en mayúsculas técnicas. Claro u oscuro
       según st.session_state.cq_theme (variables inyectadas justo
       debajo, ver _cq_root_vars).
       ══════════════════════════════════════════════════════════════ */
""" + _cq_root_vars + """

    html, body, .stApp {
        background-color: var(--cq-bg) !important;
        background-image: none;
        color: var(--cq-text);
    }
    .stApp, .stApp * { font-family: var(--cq-mono) !important; }
    /* Excepción: los íconos de Streamlit (flechas de expander, popovers, etc.)
       usan una fuente de ligaduras (Material Symbols). Forzarles la fuente
       mono rompe el ícono y muestra el texto crudo de la ligadura
       ("arrow_drop_down", "expand_more"...) superpuesto sobre las etiquetas.
       Los popovers/tooltips de Streamlit se renderizan en un portal FUERA de
       .stApp, así que la excepción debe aplicarse también sin ese prefijo. */
    [data-testid="stIconMaterial"],
    span[data-testid="stIconMaterial"],
    i.material-icons,
    i.material-symbols-outlined,
    [class*="material-symbols"],
    [class*="material-icons"] {
        font-family: 'Material Symbols Outlined', 'Material Icons' !important;
    }
    /* El botón disparador de un popover (ej. "Buscar otro activo") trae de
       fábrica una flechita decorativa que a veces no logra cargar la fuente
       de íconos (queda como texto crudo "expand_more" montado sobre la
       etiqueta). No aporta nada aquí, así que se oculta por completo y se
       deja el botón centrado y de una sola línea. */
    [data-testid="stPopover"] button [data-testid="stIconMaterial"] {
        display: none !important;
    }
    [data-testid="stPopover"] button p {
        white-space: nowrap;
    }

    /* ── Estados deshabilitados — Streamlit los pinta con sus grises/blancos
       nativos, que rompían el modo oscuro (botones y campos "en blanco"
       horribles mientras la app rehace el render tras una acción). Se fuerzan
       a la misma paleta que el resto de la terminal en ambos temas. ── */
    .stButton > button:disabled, .stButton > button[disabled],
    .stDownloadButton > button:disabled {
        background: var(--cq-surface-2) !important;
        color: var(--cq-muted) !important;
        border: 1px solid var(--cq-border) !important;
        opacity: 0.6 !important;
    }
    [data-testid="stPopover"] > div > button,
    [data-testid="stPopover"] button {
        background: var(--cq-surface-2) !important;
        color: var(--cq-text) !important;
        border: 1px solid var(--cq-border-2) !important;
        border-radius: 2px !important;
    }
    [data-testid="stPopover"] button:hover {
        border-color: var(--cq-accent) !important; color: var(--cq-accent) !important;
    }
    [data-testid="stPopoverBody"] {
        background: var(--cq-surface) !important; border-color: var(--cq-border) !important;
    }
    .stSelectbox [disabled], .stTextInput input:disabled, .stDateInput input:disabled,
    .stNumberInput input:disabled, textarea:disabled,
    [data-baseweb="select"][aria-disabled="true"] > div {
        background-color: var(--cq-surface-2) !important;
        color: var(--cq-muted) !important;
        -webkit-text-fill-color: var(--cq-muted) !important;
        border-color: var(--cq-border) !important;
        opacity: 0.75 !important;
    }
    /* El propio contenedor del selectbox/date/text input, no solo el input
       interno, también recibe un fondo blanco nativo al deshabilitarse. */
    div[data-testid="stSelectbox"], div[data-testid="stTextInput"],
    div[data-testid="stDateInput"], div[data-testid="stNumberInput"] {
        background: transparent !important;
    }
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stDateInput"] > div > div,
    div[data-testid="stTextInput"] > div > div {
        background-color: var(--cq-surface) !important;
    }

    /* Todo el texto de la terminal en mayúsculas técnicas, como la referencia */
    .stMarkdown p, .stMarkdown li, label, .stButton > button,
    [data-testid="stMetricLabel"], .stSelectbox, .stTabs [data-baseweb="tab"] {
        letter-spacing: 0.01em;
    }

    /* ── Tipografía — títulos técnicos, todo mono, sin degradados ── */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: var(--cq-mono) !important; font-weight: 700;
        letter-spacing: 0.02em; color: var(--cq-text);
        text-transform: uppercase; font-size: 1.05rem !important;
    }
    h1 {
        border-bottom: 1px solid var(--cq-border); padding-bottom: 8px;
        color: var(--cq-accent); position: relative;
    }
    h1::before { content: "// "; color: var(--cq-accent-dim); }
    .stMarkdown h4, .stMarkdown h5 {
        font-family: var(--cq-mono) !important; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; font-size: 0.72rem; color: var(--cq-accent);
    }
    code, .stCode, .stDataFrame, [data-testid="stMetricValue"], .stNumberInput input {
        font-family: var(--cq-mono) !important;
    }

    /* ── Sidebar nativa de Streamlit — deshabilitada por completo ──
       La navegación ahora vive en una barra superior sólida (no
       plegable), así que la sidebar nativa se oculta sin dejar
       ningún control de plegado/desplegado que pueda fallar. */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* ── Botones — planos, esquinas rectas, estilo consola de terminal ── */
    .stButton > button {
        background: var(--cq-surface-2) !important; border: 1px solid var(--cq-border-2) !important;
        color: var(--cq-text) !important; border-radius: 2px; font-weight: 600;
        font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em;
        transition: border-color .1s ease, color .1s ease, background .1s ease;
    }
    .stButton > button:hover {
        border-color: var(--cq-accent) !important; color: var(--cq-accent) !important;
        background: var(--cq-surface) !important;
    }
    .stButton > button:active { transform: translateY(1px); }
    button[kind="primary"] {
        background: var(--cq-accent) !important;
        border: 1px solid var(--cq-accent) !important;
        color: #0a0800 !important; font-weight: 800 !important;
        box-shadow: none !important;
    }
    button[kind="primary"]:hover {
        background: var(--cq-accent-2) !important; border-color: var(--cq-accent-2) !important;
    }

    /* ── Barra superior — franja negra sólida, línea amarilla inferior,
       tipo "ticker bar" de terminal financiera ──
       Se apoya en st.container(key=...), que Streamlit expone como
       una clase .st-key-<key> real y anidable.
       Nota: se usa position:fixed (no sticky) porque, anidada dentro del
       árbol de bloques de Streamlit (flex/columnas), "sticky" dependía de
       cuál ancestro cuenta como su contenedor de scroll y dejaba de
       pegarse al bajar del todo. "fixed" la ancla siempre al viewport real
       del navegador, sin ambigüedad. Como sale del flujo normal, el
       contenido de la página se empuja hacia abajo con el padding-top de
       .block-container (ver más abajo) para que nada quede tapado. */
    .st-key-cq_topbar {
        position: fixed !important; top: 0; left: 0; right: 0; width: 100%;
        z-index: 999;
        background: var(--cq-bg-2);
        border-bottom: 2px solid var(--cq-accent);
        box-shadow: 0 1px 8px rgba(20,20,12,0.06);
        padding: 8px 6px 0 6px; margin: 0;
    }
    .cq-brand { display: flex; align-items: center; gap: 8px; }
    .cq-brand-mark {
        width: 28px; height: 28px; border-radius: 2px; flex-shrink: 0;
        background: var(--cq-accent);
        display: flex; align-items: center; justify-content: center;
    }
    .cq-brand-mark span {
        font-family: var(--cq-mono); font-weight: 800; font-size: 15px; color: #0a0800;
    }
    .cq-brand-name {
        font-family: var(--cq-mono); font-size: 1.05rem; font-weight: 800;
        line-height: 1.1; color: var(--cq-text); margin: 0; text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .cq-brand-name b { color: var(--cq-accent); font-weight: 800; }
    .cq-brand-tag {
        font-size: 9px; letter-spacing: .14em; text-transform: uppercase;
        color: var(--cq-muted); margin: 0;
    }
    .cq-status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 10px; border-radius: 2px; font-size: 11px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.04em;
        border: 1px solid var(--cq-border-2); background: var(--cq-surface-2); white-space: nowrap;
        float: right;
    }
    .cq-status-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
    .cq-status-on  { color: #0a0800; border-color: var(--cq-green); background: var(--cq-green); }
    .cq-status-on  .cq-status-dot { background: #0a0800; }
    .cq-status-off { color: var(--cq-accent); background: var(--cq-accent-dim); border-color: var(--cq-accent-dim); }
    .cq-status-off .cq-status-dot { background: var(--cq-accent); }

    /* Fila de navegación — pestañas planas, subrayado amarillo sólido */
    .st-key-cq_nav_row .stButton > button {
        border-radius: 0; border: none;
        border-bottom: 2px solid transparent; background: transparent;
        font-size: 11.5px; padding: 8px 4px; box-shadow: none;
    }
    .st-key-cq_nav_row .stButton > button:hover {
        background: var(--cq-surface-2); box-shadow: none;
        border-bottom: 2px solid var(--cq-accent-dim); color: var(--cq-accent);
    }
    .st-key-cq_nav_row .stButton > button[kind="primary"] {
        background: transparent !important; box-shadow: none !important;
        color: var(--cq-accent) !important; border-bottom: 2px solid var(--cq-accent) !important;
    }

    /* Segunda fila — mercado/sesión, más compacta */
    .st-key-cq_subrow { padding-bottom: 8px; border-top: 1px solid var(--cq-border); padding-top: 8px; }
    .st-key-cq_subrow label p {
        font-size: 9.5px !important; letter-spacing: .1em; text-transform: uppercase;
        color: var(--cq-accent-dim) !important;
    }
    .st-key-cq_subrow .stButton > button { font-size: 11.5px; padding: 6px 10px; }

    /* ── Inputs — planos y rectos ── */
    .stSelectbox > div > div, .stNumberInput > div > div > input,
    .stTextInput > div > div > input, .stDateInput input, textarea {
        background-color: var(--cq-surface) !important; border-color: var(--cq-border-2) !important;
        color: var(--cq-text) !important; border-radius: 2px !important;
        font-size: 12.5px !important;
    }
    .stSelectbox > div > div:focus-within, .stTextInput > div > div > input:focus {
        border-color: var(--cq-accent) !important; box-shadow: 0 0 0 1px var(--cq-accent) !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: var(--cq-accent) !important; border-radius: 2px;
    }
    .stSlider [data-baseweb="slider"] > div > div { background: var(--cq-accent) !important; }

    hr { border-color: var(--cq-border); }
    details {
        background: var(--cq-surface); border: 1px solid var(--cq-border);
        border-radius: 2px;
    }
    summary { color: var(--cq-text) !important; }

    /* ── Tabs — subrayado amarillo sólido, plano ── */
    .stTabs [data-baseweb="tab"] {
        color: var(--cq-muted); border-bottom: 2px solid transparent;
        font-family: var(--cq-mono) !important; font-weight: 700;
        text-transform: uppercase; font-size: 11.5px; letter-spacing: 0.05em;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--cq-accent); border-bottom-color: var(--cq-accent);
    }

    /* ── Tarjetas de métricas — panel rectangular, barra superior amarilla,
       exactamente como los recuadros "AVANZADORES / DECLIVE / ..." de
       la referencia ── */
    [data-testid="metric-container"] {
        background: var(--cq-surface); border: 1px solid var(--cq-border);
        border-top: 2px solid var(--cq-accent); border-radius: 0; padding: 10px 12px;
        box-shadow: none;
        transition: border-color .15s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: var(--cq-accent-dim); border-top-color: var(--cq-accent);
    }
    [data-testid="stMetricLabel"] {
        text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.68rem;
        color: var(--cq-muted) !important;
    }
    [data-testid="stMetricValue"] { color: var(--cq-text) !important; font-weight: 700 !important; }

    .stProgress > div > div > div { background: var(--cq-accent); }

    /* ── Paneles genéricos (containers con borde) — barra de título estilo
       ventana de terminal, como en la referencia ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0 !important; border-color: var(--cq-border) !important;
        background: var(--cq-surface) !important;
    }
    div[data-testid="stExpander"] {
        border-radius: 0 !important; border: 1px solid var(--cq-border) !important;
        background: var(--cq-surface) !important;
    }
    div[data-testid="stExpander"] summary {
        text-transform: uppercase; font-size: 11.5px; letter-spacing: 0.06em;
        font-weight: 700; color: var(--cq-accent) !important;
        background: var(--cq-surface-2);
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--cq-border) !important; }

    /* ── Scrollbar personalizado ── */
    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-track { background: var(--cq-bg); }
    ::-webkit-scrollbar-thumb { background: var(--cq-border-2); border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--cq-accent); }

    /* ── Ocultar chrome nativo de Streamlit (menú, footer, nav multipágina) ── */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { display: none !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    nav[data-testid="stSidebarNav"] { display: none !important; }
    ul[data-testid="stSidebarNavItems"] { display: none !important; }
    .stPageLink { display: none !important; }
    section[data-testid="stSidebarNavHeader"] { display: none !important; }

    .block-container {
        padding-top: 250px !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    [data-testid="stAppViewContainer"] { padding: 0 !important; }
    [data-testid="stMain"] { padding: 0 !important; }

    /* ── Arreglo del ícono superpuesto en los paneles desplegables ──
       El glifo de la fuente de íconos (flecha de expandir/colapsar) a
       veces no carga y el navegador muestra su texto crudo ("arrow_...")
       encimado sobre el título del panel. En vez de depender de esa
       fuente, se oculta el glifo por completo y se dibuja una flecha
       propia en CSS puro (no depende de ninguna fuente externa, así que
       nunca puede volver a superponerse). ── */
    div[data-testid="stExpander"] summary {
        position: relative !important; padding-right: 30px !important;
    }
    div[data-testid="stExpander"] summary svg,
    div[data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    div[data-testid="stExpander"] summary [data-testid*="Icon"] {
        font-size: 0 !important; line-height: 0 !important;
        color: transparent !important; width: 0 !important; height: 0 !important;
        overflow: hidden !important;
    }
    div[data-testid="stExpander"] summary::after {
        content: ""; position: absolute; right: 10px; top: 50%;
        width: 7px; height: 7px;
        border-right: 2px solid var(--cq-accent); border-bottom: 2px solid var(--cq-accent);
        transform: translateY(-70%) rotate(45deg);
        transition: transform .18s ease;
    }
    details[open] > summary::after { transform: translateY(-30%) rotate(-135deg); }

    /* ── Iluminaciones — estilo lujoso, discretas, tono dorado ──
       Se apoyan en las variables --cq-glow / --cq-glow-soft ya definidas,
       sin tocar la paleta base negro/hueso/amarillo. ── */
    .st-key-cq_topbar {
        box-shadow: 0 2px 22px -4px var(--cq-glow), 0 1px 0 var(--cq-accent);
    }
    .cq-brand-mark {
        box-shadow: 0 0 14px 1px var(--cq-glow), inset 0 0 6px rgba(255,255,255,0.25);
    }
    h1 { text-shadow: 0 0 18px var(--cq-glow-soft); }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        box-shadow: 0 0 0 1px transparent, 0 8px 24px -18px rgba(20,20,12,0.18);
        transition: box-shadow .2s ease, border-color .2s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: var(--cq-accent) !important;
        box-shadow: 0 0 22px -8px var(--cq-glow), 0 8px 24px -18px rgba(20,20,12,0.18);
    }
    div[data-testid="stExpander"] summary {
        transition: color .15s ease, text-shadow .15s ease;
    }
    div[data-testid="stExpander"] summary:hover {
        color: var(--cq-accent-2) !important;
        text-shadow: 0 0 10px var(--cq-glow-soft);
    }
    .stButton > button:hover {
        box-shadow: 0 0 16px -2px var(--cq-glow);
    }
    button[kind="primary"] {
        box-shadow: 0 0 18px -3px var(--cq-glow) !important;
    }
    button[kind="primary"]:hover {
        box-shadow: 0 0 26px -3px var(--cq-glow) !important;
    }
    .st-key-cq_nav_row .stButton > button[kind="primary"] {
        text-shadow: 0 0 12px var(--cq-glow-soft);
    }
    [data-testid="metric-container"]:hover {
        box-shadow: 0 0 20px -6px var(--cq-glow);
    }
    .cq-status-on {
        box-shadow: 0 0 10px 0 rgba(46,204,113,0.45);
    }
    .cq-status-off {
        box-shadow: 0 0 10px 0 var(--cq-glow-soft);
    }
    .stSelectbox > div > div:focus-within, .stTextInput > div > div > input:focus {
        box-shadow: 0 0 0 1px var(--cq-accent), 0 0 14px -2px var(--cq-glow) !important;
    }
</style>
""", unsafe_allow_html=True)

from ui.state import AppState
AppState.init()

from ui.components.sidebar import render_sidebar
page, asset, timeframe = render_sidebar()

if page == "Gráficos":
    from ui.views.charts import render
    render(asset, timeframe)
elif page == "Research Lab":
    from ui.views.research_lab import render
    render(asset, timeframe)
elif page == "Constructor":
    from ui.views.strategy_builder import render
    render(asset, timeframe)
elif page == "Descubridor Genético":
    from ui.views.discovery import render
    render(asset, timeframe)
elif page == "Backtesting":
    from ui.views.backtesting import render
    render(asset, timeframe)
elif page == "Régimen de Mercado":
    from ui.views.regime_detector import render
    render(asset, timeframe)
elif page == "Mercado en Vivo":
    from ui.views.live import render
    render(asset, timeframe)
elif page == "Portfolio":
    from ui.views.portfolio import render
    render()
elif page == "Optimización":
    from ui.views.optimization import render
    render(asset, timeframe)
elif page == "Pine Script":
    from ui.views.pine_generator import render
    render(asset, timeframe)
