"""Costos de backtest para la edición local CSV.

Sin MT5 no existe un perfil de spread/contrato del broker. El motor utiliza
el modelo explícito de comisión/slippage configurado por el usuario.
"""
import streamlit as st

def render_cost_toggle(symbol: str, key_prefix: str):
    st.caption("Costos realistas de MT5 no están disponibles en la edición GitHub. Se usa comisión/slippage explícitos del backtest.")
    return False, None
