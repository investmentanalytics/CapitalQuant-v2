"""Compatibilidad histórica: proveedor MT5 DESACTIVADO en la edición GitHub.

Se conserva la interfaz mínima para que módulos antiguos puedan importar sus
símbolos y helpers sin instalar MetaTrader5. Ninguna función abre conexión ni
realiza llamadas de red.
"""
from __future__ import annotations
import pandas as pd
from research.csv_registry import list_csv_datasets, load_csv_dataset
from research.data_registry import load_clean_dataset

MT5_TIMEFRAMES = {"1m":"1 minuto", "5m":"5 minutos", "15m":"15 minutos", "30m":"30 minutos", "1h":"1 Hora", "4h":"4 Horas", "1d":"Diario", "1w":"Semanal", "1M":"Mensual"}
MT5_LABELS = MT5_TIMEFRAMES.copy()
MT5_ANN_FACTORS = {"1m":525600, "5m":105120, "15m":35040, "30m":17520, "1h":8760, "4h":2190, "1d":365, "1w":52, "1M":12}


def is_sub_hourly_timeframe(timeframe: str) -> bool:
    return timeframe in {"1m", "5m", "15m", "30m"}


def history_gap_warning(df, date_from, asset, timeframe):
    return None


class MT5Provider:
    name = "CSV local"
    source_name = "CSV local"
    last_error = "MetaTrader 5 está desactivado en esta edición."
    last_diagnostics = []

    @property
    def is_connected(self):
        return False

    def connect(self):
        return False

    def disconnect(self):
        return None

    def get_available_symbols(self):
        return sorted({m["asset"] for m in list_csv_datasets() if m.get("asset")})

    def get_data(self, symbol, timeframe, *, force_download=False, date_from=None, date_to=None, n_bars=None):
        df = load_csv_dataset(symbol, timeframe, date_from, date_to)
        if df is None:
            df = load_clean_dataset(symbol, timeframe, date_from, date_to)
        if df is not None and n_bars and len(df) > n_bars:
            df = df.tail(n_bars)
        return df

    def get_raw_mt5(self):
        return None

    def get_symbol_info(self, symbol):
        return None

    def account_info(self):
        return None

    def is_available(self, symbol, timeframe):
        return load_csv_dataset(symbol, timeframe) is not None


def get_mt5_provider() -> MT5Provider:
    return MT5Provider()
