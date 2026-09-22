"""Selector histórico exclusivamente local para la edición GitHub."""
from __future__ import annotations
import streamlit as st
from research.data_registry import list_clean_datasets, load_clean_dataset
from research.raw_data_registry import list_raw_datasets, load_raw_dataset
from research.csv_registry import list_csv_datasets, load_csv_dataset

SOURCE_CSV = "CSV locales"
SOURCE_CLEAN = "Datos limpios / verificados"
SOURCE_RAW = "Datos originales publicados"
SOURCE_OPTIONS = [SOURCE_CSV, SOURCE_CLEAN, SOURCE_RAW]


def get_selected_source() -> str:
    value = st.session_state.get("historical_data_source", SOURCE_CSV)
    return value if value in SOURCE_OPTIONS else SOURCE_CSV


def set_selected_source(source: str) -> None:
    st.session_state["historical_data_source"] = source if source in SOURCE_OPTIONS else SOURCE_CSV


def render_data_source_selector(timeframe, *, key_prefix: str, compact=False) -> str:
    csvs = list_csv_datasets(timeframe) if timeframe else list_csv_datasets()
    clean = list_clean_datasets(timeframe) if timeframe else list_clean_datasets()
    raw = list_raw_datasets(timeframe) if timeframe else list_raw_datasets()
    current = get_selected_source()
    with st.container(border=True):
        st.markdown("### Datos")
        c1, c2, c3 = st.columns([2.2, 2.8, 2.2])
        source = c1.selectbox(
            "Fuente histórica", SOURCE_OPTIONS,
            index=SOURCE_OPTIONS.index(current), key=f"{key_prefix}_data_source",
            help="Esta edición no utiliza MetaTrader 5. Los datos proceden exclusivamente de CSV locales y datasets publicados."
        )
        set_selected_source(source)
        if source == SOURCE_CSV:
            assets = sorted({x.get("asset") for x in csvs})
            c2.success("Disponibles: " + ", ".join(assets) if assets else "No hay CSV disponibles")
            c3.metric("CSV locales", len(csvs))
        elif source == SOURCE_CLEAN:
            c2.success("Disponibles: " + ", ".join(sorted({x.get('asset') for x in clean})) if clean else "No hay datasets limpios")
            c3.metric("Datasets limpios", len(clean))
        else:
            c2.success("Disponibles: " + ", ".join(sorted({x.get('asset') for x in raw})) if raw else "No hay datasets originales")
            c3.metric("Originales publicados", len(raw))
    return source


def load_historical_data(asset, timeframe, *, source=None, date_from=None, date_to=None, force_download=False):
    source = source or get_selected_source()
    if source == SOURCE_CLEAN:
        return load_clean_dataset(asset, timeframe, date_from, date_to)
    if source == SOURCE_RAW:
        return load_raw_dataset(asset, timeframe, date_from, date_to)
    return load_csv_dataset(asset, timeframe, date_from, date_to)


def is_clean_available(asset, timeframe):
    return asset in {x.get("asset") for x in list_clean_datasets(timeframe)}
