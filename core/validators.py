"""
core/validators.py
Validación de DataFrames OHLCV.

Centraliza la detección temprana de datos malformados
antes de que lleguen al engine o a las estrategias.
"""
import pandas as pd

from core.data_cleaning import clean_ohlcv


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


class DataValidationError(ValueError):
    """Error específico de validación de datos de mercado."""
    pass


def validate_ohlcv(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Valida que un DataFrame tenga el formato OHLCV correcto.
    Lanza DataValidationError con mensaje descriptivo si hay problemas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar.
    name : str
        Nombre del activo/fuente para mensajes de error claros.
    """
    if df is None:
        raise DataValidationError(f"[{name}] DataFrame es None.")

    if df.empty:
        raise DataValidationError(f"[{name}] DataFrame está vacío.")

    missing = REQUIRED_COLUMNS - set(df.columns.str.lower())
    if missing:
        raise DataValidationError(
            f"[{name}] Faltan columnas requeridas: {sorted(missing)}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"[{name}] El índice debe ser DatetimeIndex. "
            f"Tipo actual: {type(df.index).__name__}"
        )

    n_null = df[["open", "high", "low", "close"]].isnull().sum().sum()
    if n_null > len(df) * 0.1:
        raise DataValidationError(
            f"[{name}] Demasiados valores nulos en OHLC: {n_null} "
            f"({n_null/len(df)*100:.1f}%). Revisar calidad de datos."
        )

    if len(df) < 50:
        raise DataValidationError(
            f"[{name}] Datos insuficientes: {len(df)} velas. "
            f"Mínimo recomendado: 50 velas."
        )


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza columnas y tipos de un DataFrame OHLCV.
    Retorna copia limpia sin modificar el original.
    """
    df = df.copy()

    # Normalizar nombres de columnas
    df.columns = [c.lower().strip() for c in df.columns]

    # Renombrar variantes conocidas
    renames = {
        "adj_close": "close",
        "adj close": "close",
        "vol":       "volume",
    }
    df.rename(columns=renames, inplace=True)

    # Asegurar tipos numéricos
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    # Índice temporal
    df.index = pd.to_datetime(df.index)
    df.index.name = "datetime"
    df.sort_index(inplace=True)

    # Eliminar filas con OHLC nulos
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # Limpieza de integridad (duplicados, orden, velas inválidas) -- ver
    # core/data_cleaning.py. Deliberadamente NO rellena huecos de mercado
    # cerrado: eso se maneja solo a nivel visual en visualization/charts.py.
    df = clean_ohlcv(df)

    return df
