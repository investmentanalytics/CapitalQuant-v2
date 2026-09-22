"""Barra superior global para la edición GitHub de CapitalQuant.

No existe conexión MT5. El universo de activos y temporalidades se descubre
exclusivamente desde CSV versionados en data/csv_datasets y CSV importados en
la sesión.
"""
import streamlit as st
from config.settings import TIMEFRAMES
from research.csv_registry import list_csv_datasets
from research.data_registry import list_clean_datasets
from ui.components.ticker_tape import render_ticker_tape

_NAV_ICONS = {"Gráficos":"◆","Constructor":"◈","Descubridor Genético":"✦","Backtesting":"▣","Régimen de Mercado":"◔","Mercado en Vivo":"●","Portfolio":"◫","Optimización":"⚙","Research Lab":"◇","Pine Script":"⌘"}
_PAGES = ["Gráficos", "Research Lab", "Constructor", "Descubridor Genético", "Backtesting", "Régimen de Mercado", "Mercado en Vivo", "Portfolio", "Optimización", "Pine Script"]
_TF_LABELS = {"1m":"1 minuto","5m":"5 minutos","15m":"15 minutos","30m":"30 minutos","1h":"1 Hora","2h":"2 Horas","4h":"4 Horas","6h":"6 Horas","12h":"12 Horas","1d":"Diario","1w":"Semanal","1M":"Mensual"}

@st.fragment(run_every=30)
def _ticker_tape_fragment():
    render_ticker_tape()

def _available(asset):
    metas = list_csv_datasets()
    return sorted({m["timeframe"] for m in metas if m.get("asset") == asset}, key=lambda x: list(_TF_LABELS).index(x) if x in _TF_LABELS else 999)

def render_sidebar() -> tuple[str, str, str]:
    topbar = st.container(key="cq_topbar")
    with topbar:
        _ticker_tape_fragment()
        brand_col, status_col = st.columns([5,1.4], vertical_alignment="center")
        with brand_col:
            st.markdown("<div class='cq-brand'><div class='cq-brand-mark'><span>Q</span></div><div><p class='cq-brand-name'>Capital<b>Quant</b></p><p class='cq-brand-tag'>Elite Quant Research Lab · CSV LOCAL</p></div></div>", unsafe_allow_html=True)
        with status_col:
            st.markdown("<span class='cq-status-pill cq-status-on'><span class='cq-status-dot'></span>DATOS LOCALES</span>", unsafe_allow_html=True)
            is_dark = st.session_state.get("cq_theme", "light") == "dark"
            if st.button("☀" if is_dark else "☾", key="cq_theme_toggle", width="stretch"):
                st.session_state.cq_theme = "light" if is_dark else "dark"; st.rerun()
        if "cq_active_page" not in st.session_state:
            st.session_state.cq_active_page = _PAGES[0]
        nav = st.container(key="cq_nav_row")
        with nav:
            cols = st.columns(len(_PAGES))
            for col,p in zip(cols,_PAGES):
                with col:
                    if st.button(f"{_NAV_ICONS[p]}  {p}", key=f"nav_btn_{p}", width="stretch", type="primary" if st.session_state.cq_active_page==p else "secondary"):
                        st.session_state.cq_active_page=p; st.rerun()
        page = st.session_state.cq_active_page
        metas = list_csv_datasets()
        symbols = sorted({m.get("asset") for m in metas if m.get("asset")})
        clean_assets = sorted({m.get("asset") for m in list_clean_datasets() if m.get("asset")})
        symbols = sorted(set(symbols)|set(clean_assets))
        if not symbols:
            symbols = ["SIN_CSV"]
        current = st.session_state.get("selected_asset", symbols[0])
        if current not in symbols: current=symbols[0]
        c_asset,c_tf,c_reload,c_clear=st.columns([1.5,1.5,1.5,1.2], vertical_alignment="bottom")
        with c_asset:
            selected_asset=st.selectbox("Activo", symbols, index=symbols.index(current), key="sidebar_asset", help="Activos disponibles desde CSV locales/versionados en el repositorio.")
        st.session_state.selected_asset=selected_asset
        tfs=_available(selected_asset)
        if not tfs:
            tfs=sorted(set(m.get("timeframe") for m in metas if m.get("timeframe"))) or ["1d"]
        current_tf=st.session_state.get("selected_timeframe", tfs[0])
        if current_tf not in tfs: current_tf=tfs[0]
        with c_tf:
            selected_tf=st.selectbox("Temporalidad", tfs, index=tfs.index(current_tf), format_func=lambda x:_TF_LABELS.get(x,x), key="sidebar_tf")
        st.session_state.selected_timeframe=selected_tf
        with c_reload:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            if st.button("Forzar recarga", key="sidebar_btn_recarga", width="stretch"):
                st.cache_data.clear(); st.rerun()
        with c_clear:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            if st.button("Limpiar sesión", key="sidebar_btn_limpiar", width="stretch"):
                from ui.state import AppState
                AppState.clear_all(); st.rerun()
        st.caption("Fuente única: CSV local. Añade archivos a **data/csv_datasets/** con formato `ACTIVO_TEMPORALIDAD.csv`, por ejemplo `XAUUSD_4h.csv`, o impórtalos desde Gráficos.")
    return page, selected_asset, selected_tf
