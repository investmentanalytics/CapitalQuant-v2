"""
visualization/theme.py
Tema visual centralizado para todos los gráficos Plotly.

Un único lugar para cambiar colores, fuentes y estilos
de toda la plataforma.
"""
from config.settings import THEME, THEME_DARK, STRATEGY_COLORS

COLORS = STRATEGY_COLORS


def get_active_theme() -> dict:
    """Tema Plotly a usar en este render: sigue el modo claro/oscuro elegido
    en la barra superior (st.session_state.cq_theme). Se resuelve en cada
    llamada (en vez de una sola vez al importar el módulo) para que cambiar
    de modo se refleje de inmediato en los gráficos sin reiniciar la app."""
    try:
        import streamlit as st
        if st.session_state.get("cq_theme") == "dark":
            return THEME_DARK
    except Exception:
        pass
    return THEME


# Compatibilidad hacia atrás: módulos que importan estas constantes directo
# (BG, TEXT, etc.) siguen funcionando, fijas al tema claro por defecto.
# El resto del código nuevo debería usar get_active_theme() o base_layout().
BG       = THEME["bg"]
SURFACE  = THEME["surface"]
BORDER   = THEME["border"]
TEXT     = THEME["text"]
ACCENT   = THEME["accent"]
GREEN    = THEME["green"]
RED      = THEME["red"]
YELLOW   = THEME["yellow"]
GRID     = THEME["grid"]


def base_layout(title: str = "", height: int = 500) -> dict:
    """Layout base para todos los gráficos Plotly, adaptado al tema activo
    (claro u oscuro)."""
    t = get_active_theme()
    bg, surface, border, text, grid = t["bg"], t["surface"], t["border"], t["text"], t["grid"]
    return dict(
        title=dict(text=title, font=dict(color=text, size=14)),
        height=height,
        paper_bgcolor=bg,
        plot_bgcolor=surface,
        font=dict(color=text, family="Inter, sans-serif", size=12),
        xaxis=dict(
            gridcolor=grid, gridwidth=0.5,
            linecolor=border, tickcolor=border,
            showgrid=True, zeroline=False,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            gridcolor=grid, gridwidth=0.5,
            linecolor=border, tickcolor=border,
            showgrid=True, zeroline=False,
        ),
        legend=dict(
            bgcolor=surface,
            bordercolor=border,
            borderwidth=1,
            font=dict(color=text),
        ),
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode="x unified",
    )
