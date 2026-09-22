"""
ui/components/metrics_grid.py
Componente reutilizable de grid de métricas institucionales.

Se usa en Backtesting, Portfolio y Optimización.
"""
import streamlit as st
from typing import Optional


def render_metrics_grid(metrics: dict, n_cols: int = 4) -> None:
    """
    Renderiza un grid de métricas con formato institucional.

    Parameters
    ----------
    metrics : dict {label: value_str}
    n_cols : número de columnas del grid
    """
    items = list(metrics.items())
    rows = [items[i:i + n_cols] for i in range(0, len(items), n_cols)]

    for row in rows:
        cols = st.columns(len(row))
        for col, (label, value) in zip(cols, row):
            _metric_card(col, label, value)


def _metric_card(container, label: str, value: str, delta: Optional[str] = None) -> None:
    """Tarjeta individual de métrica con estilo personalizado."""
    # Determinar color por valor
    color = "#f3f1ea"  # neutral (texto base)
    try:
        raw = value.replace("$", "").replace("%", "").replace(",", "").strip()
        fval = float(raw)
        positive_metrics = {"Net Profit", "Net Profit %", "CAGR", "Sharpe",
                            "Sortino", "Calmar", "Profit Factor", "Win Rate",
                            "Expectancy", "Avg Win"}
        negative_metrics = {"Max Drawdown %"}
        if any(m.lower() in label.lower() for m in positive_metrics):
            color = "#2ecc71" if fval > 0 else "#ff4d4d"
        elif any(m.lower() in label.lower() for m in negative_metrics):
            color = "#ff4d4d" if fval < -20 else ("#ffe14d" if fval < -10 else "#2ecc71")
    except Exception:
        pass

    container.markdown(
        f"""
        <div style="
            background: #101012;
            border: 1px solid #2a2a26;
            border-left: 3px solid #f2c200;
            border-radius: 8px;
            padding: 12px 14px;
            text-align: center;
            height: 72px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.35);
        ">
            <div style="
                color: #8f8c82;
                font-size: 10px;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 4px;
            ">{label}</div>
            <div style="
                color: {color};
                font-family: 'JetBrains Mono', monospace;
                font-size: 18px;
                font-weight: 600;
                line-height: 1;
            ">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparison_row(
    results_list: list,
    labels: list,
    metric_keys: list,
) -> None:
    """
    Tabla comparativa de múltiples resultados de backtest.
    Usado en Portfolio para comparar componentes.
    """
    if not results_list or not labels:
        return

    import pandas as pd

    rows = []
    for label, res in zip(labels, results_list):
        row = {"Componente": label}
        for key in metric_keys:
            row[key] = getattr(res, key, "—")
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)
