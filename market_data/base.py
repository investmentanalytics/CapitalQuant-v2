"""
market_data/base.py
Interfaz abstracta para proveedores de datos de mercado.

Cualquier nueva fuente de datos implementa BaseDataProvider
y queda disponible automáticamente en toda la plataforma.
"""
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class BaseDataProvider(ABC):
    """Interfaz que deben implementar todos los proveedores de datos."""

    name: str = "Base Provider"

    @abstractmethod
    def get_data(
        self,
        symbol: str,
        timeframe: str,
        *,
        force_download: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Retorna DataFrame OHLCV para el símbolo y temporalidad dados.

        Parameters
        ----------
        symbol : str
            Identificador del activo (ej. "BTCUSD", "XAUUSD")
        timeframe : str
            Temporalidad ("1d", "4h", "1h", "15m", "5m")
        force_download : bool
            Si True, ignora caché local y descarga de nuevo.

        Returns
        -------
        pd.DataFrame o None si no hay datos disponibles.
        """
        ...

    @abstractmethod
    def is_available(self, symbol: str, timeframe: str) -> bool:
        """Verifica si hay datos disponibles sin descargar."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
