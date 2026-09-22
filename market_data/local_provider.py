"""
market_data/local_provider.py
Proveedor de datos locales — respaldado por caché SQLite (market_data/cache_db.py)
en vez de archivos parquet sueltos.

Es la MISMA idea que el motor de datos de Aletheia: toda vela que entra por
MT5Provider se guarda aquí una sola vez, indexada por (symbol, timeframe,
timestamp). Cualquier módulo de la plataforma (Descubridor, Backtesting,
Optimización, Mercado en Vivo) puede pedir después cualquier sub-rango de
esas velas sin volver a tocar el terminal MT5 — de ahí que un rango ya
descargado "funcione en cualquier lado sin problema".
"""
from typing import Optional
from pathlib import Path

import pandas as pd
from loguru import logger

from market_data.base import BaseDataProvider
from market_data import cache_db
from core.validators import normalize_ohlcv
from config.settings import DATA_DIR


class LocalDataProvider(BaseDataProvider):
    """Lee/escribe datos OHLCV desde el caché SQLite local."""

    name = "Caché Local (SQLite)"

    def __init__(self, data_dir: Optional[Path] = None):
        # Se conserva el parámetro por compatibilidad de firma; el caché
        # SQLite vive en config.settings.DATA_DIR/capitalquant_cache.db.
        self.data_dir = data_dir or DATA_DIR

    def is_available(self, symbol: str, timeframe: str) -> bool:
        if timeframe == "4h":
            return self.is_available(symbol, "1h")
        mn, mx = cache_db.cached_range(symbol, timeframe)
        return mn is not None

    def get_data(
        self,
        symbol: str,
        timeframe: str,
        *,
        force_download: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Carga TODO el histórico cacheado para symbol/timeframe.
        Si timeframe es '4h', se resamplea desde '1h' (MT5 no siempre expone 4h)."""
        if timeframe == "4h":
            return self._get_4h(symbol)

        df = cache_db.fetch_ohlcv(symbol, timeframe)
        if df.empty:
            logger.debug(f"[LocalProvider] Sin caché para {symbol} {timeframe}")
            return None
        try:
            df = normalize_ohlcv(df)
        except Exception:
            pass
        return df

    def get_range(
        self, symbol: str, timeframe: str,
        date_from: Optional[pd.Timestamp] = None, date_to: Optional[pd.Timestamp] = None,
    ) -> Optional[pd.DataFrame]:
        """Igual que get_data pero consultando directamente el rango en SQL
        (más eficiente cuando el caché acumulado es grande)."""
        def _to_ts(d):
            if d is None:
                return None
            ts = pd.Timestamp(d)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            return int(ts.timestamp())

        start_ts = _to_ts(date_from)
        end_ts = _to_ts(date_to)
        df = cache_db.fetch_ohlcv(symbol, timeframe, start_ts, end_ts)
        return df if not df.empty else None

    def _get_4h(self, symbol: str) -> Optional[pd.DataFrame]:
        df_1h = self.get_data(symbol, "1h")
        if df_1h is None or df_1h.empty:
            return None
        if df_1h.index.tz is None:
            df_1h = df_1h.copy()
            df_1h.index = df_1h.index.tz_localize("UTC")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df_4h = df_1h.resample("4h", origin="epoch").agg(agg).dropna(subset=["close"])
        return df_4h

    def save(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Guarda (upsert) el DataFrame en el caché SQLite."""
        n = cache_db.upsert_ohlcv(symbol, timeframe, df)
        logger.info(f"[LocalProvider] Cacheadas {n} velas de {symbol} {timeframe}")

    # -- utilidades de introspección del caché de fondo (sin pantalla manual) --

    @staticmethod
    def cached_series() -> pd.DataFrame:
        """Todas las series (símbolo, temporalidad, rango, nº de velas) ya
        descargadas y listas para reutilizar en cualquier módulo."""
        return cache_db.list_cached_series()

    @staticmethod
    def delete(symbol: str, timeframe: str) -> None:
        cache_db.delete_series(symbol, timeframe)

    @staticmethod
    def cached_span(symbol: str, timeframe: str):
        """(inicio, fin) cacheados en UTC, o (None, None) si no hay nada."""
        mn, mx = cache_db.cached_range(symbol, timeframe)
        if mn is None:
            return None, None
        return pd.to_datetime(mn, unit="s", utc=True), pd.to_datetime(mx, unit="s", utc=True)
