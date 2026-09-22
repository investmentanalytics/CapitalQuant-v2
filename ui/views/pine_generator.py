from __future__ import annotations
import streamlit as st
from strategies import list_discovered_strategies
from pine_generator import generate_pine_script, validate_definition_for_pine


def render(asset: str, timeframe: str) -> None:
    st.markdown("## Pine Script — TradingView")
    st.caption("Exporta estrategias descubiertas/guardadas como Pine Script v6, conservando la regla DNF y el esquema de ejecución de CapitalQuant.")

    items = [x for x in list_discovered_strategies() if x.get("file_exists") and x.get("definition")]
    if not items:
        st.info("No hay estrategias de Research Lab con definición persistida. Guarda una estrategia desde Research Lab.")
        return

    labels = [
        f"{m.get('class_name', m.get('stem',''))} · {m.get('asset', '')} {m.get('timeframe', '')} · {m.get('objective','') or m.get('fitness_objective','')}"
        for m in items
    ]
    idx = st.selectbox("Estrategia", range(len(items)), format_func=lambda i: labels[i], key="pine_strategy")
    stem, meta = items[idx].get("stem"), items[idx]
    definition = meta["definition"]

    unsupported = validate_definition_for_pine(definition)
    if unsupported:
        st.warning("La estrategia contiene condiciones que requieren adaptación Pine: " + ", ".join(unsupported))

    script = generate_pine_script(
        definition,
        strategy_name=meta.get("class_name", stem),
        asset=meta.get("asset", asset),
        timeframe=meta.get("timeframe", timeframe),
    )
    st.success("Pine Script v6 generado.")
    st.download_button("⬇️ Descargar .pine", data=script, file_name=f"{stem}.pine", mime="text/plain", width="stretch")
    st.code(script, language="pine")
    st.caption(
        "La exportación reproduce la regla descubierta, el conflicto LONG/SHORT y los niveles ATR. "
        "TradingView y CapitalQuant pueden diferir en fills, spread, comisión, slippage y redondeo de contratos; valida siempre esas hipótesis antes de comparar métricas."
    )
