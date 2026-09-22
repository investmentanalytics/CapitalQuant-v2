"""Fuente histórica local de CapitalQuant.

Versión GitHub: MT5 está deliberadamente desactivado. Toda la plataforma
consume únicamente datasets CSV locales/importados y los registros limpios
que se generan a partir de ellos.
"""
from .local_provider import LocalDataProvider
from .mt5_provider import MT5Provider, get_mt5_provider, MT5_TIMEFRAMES, MT5_LABELS, MT5_ANN_FACTORS
from research.csv_registry import load_csv_dataset, list_csv_datasets
from research.data_registry import load_clean_dataset
from research.raw_data_registry import load_raw_dataset


def get_data(symbol: str, timeframe: str, *, force_download: bool = False,
             date_from=None, date_to=None):
    """Carga exclusivamente datos locales.

    Prioridad: CSV registrado/bundled -> dataset limpio -> dataset original.
    No existe descarga ni conexión de red.
    """
    df = load_csv_dataset(symbol, timeframe, date_from, date_to)
    if df is not None and not df.empty:
        return df
    df = load_clean_dataset(symbol, timeframe, date_from, date_to)
    if df is not None and not df.empty:
        return df
    return load_raw_dataset(symbol, timeframe, date_from, date_to)


def get_historical_data(symbol: str, timeframe: str, *, source: str = "csv",
                        force_download: bool = False, date_from=None, date_to=None):
    if source == "clean":
        return load_clean_dataset(symbol, timeframe, date_from, date_to)
    if source == "raw":
        return load_raw_dataset(symbol, timeframe, date_from, date_to)
    return load_csv_dataset(symbol, timeframe, date_from, date_to)
