"""
market_data/cache_db.py
Caché SQLite de velas OHLCV — misma filosofía que el motor de datos de
Aletheia: cada vela descargada de MT5 se guarda aquí una sola vez y se
puede reconsultar por cualquier rango de fechas sin volver a golpear el
terminal MT5. Esto es lo que permite:

  - Pedir un rango de fechas una vez (desde Gráficos, Descubridor,
    Backtesting, Optimización o Mercado en Vivo) y que quede disponible
    de fondo para el resto de módulos, sin ningún paso manual de descarga.
  - Que "no hay datos disponibles" solo ocurra si de verdad MT5 no tiene
    ese rango — nunca por perder lo ya leído antes.
  - Listar y reutilizar cualquier serie ya cacheada (símbolo+temporalidad)
    de forma transparente para el usuario.

Basado en tablas SQLite con clave primaria (symbol, timeframe, timestamp)
e inserciones idempotentes (UPSERT) — descargar el mismo rango dos veces
nunca duplica ni corrompe datos.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

from config.settings import DATA_DIR

DB_PATH = DATA_DIR / "capitalquant_cache.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    timestamp   INTEGER NOT NULL,
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup ON ohlcv(symbol, timeframe, timestamp);
"""

_local = threading.local()


def get_connection() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        _local.conn = conn
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript(_SCHEMA)
    conn.commit()


@contextmanager
def transaction():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def upsert_ohlcv(symbol: str, timeframe: str, df: pd.DataFrame) -> int:
    """Guarda (o actualiza) todas las velas de `df` para symbol/timeframe.
    Idempotente: volver a guardar el mismo rango no duplica filas."""
    if df is None or df.empty:
        return 0
    rows = []
    skipped = 0
    for ts, row in df.iterrows():
        try:
            o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
        except (TypeError, ValueError):
            skipped += 1
            continue
        # Fila incompleta (NaN/None en cualquier precio) — insertarla rompería
        # la restricción NOT NULL de la tabla y, con executemany, TODO el lote
        # junto con ella. Se descarta solo esa vela en vez de fallar entero.
        if any(pd.isna(v) for v in (o, h, l, c)):
            skipped += 1
            continue
        vol = getattr(row, "volume", 0.0)
        vol = 0.0 if vol is None or pd.isna(vol) else float(vol)
        rows.append({
            "symbol": symbol, "timeframe": timeframe,
            "timestamp": int(ts.timestamp()),
            "open": o, "high": h, "low": l, "close": c, "volume": vol,
        })
    if skipped:
        logger.debug(f"[cache_db] {symbol} {timeframe}: {skipped} velas descartadas por datos incompletos")
    if not rows:
        return 0
    with transaction() as conn:
        conn.executemany(
            """
            INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (:symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume)
            ON CONFLICT(symbol, timeframe, timestamp) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, volume=excluded.volume
            """,
            rows,
        )
    return len(rows)


def fetch_ohlcv(symbol: str, timeframe: str,
                 start_ts: Optional[int] = None, end_ts: Optional[int] = None) -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM ohlcv WHERE symbol=? AND timeframe=?"
    params: list = [symbol, timeframe]
    if start_ts is not None:
        query += " AND timestamp >= ?"
        params.append(start_ts)
    if end_ts is not None:
        query += " AND timestamp <= ?"
        params.append(end_ts)
    query += " ORDER BY timestamp ASC"
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("datetime").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def fetch_last_n(symbol: str, timeframe: str, n: int = 2) -> pd.DataFrame:
    """Últimas `n` velas cacheadas para symbol/timeframe, vía SQL LIMIT —
    a diferencia de fetch_ohlcv(), NO carga todo el histórico a memoria.
    Pensado para lecturas frecuentes y livianas (ej. la cinta de precios)."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM ohlcv WHERE symbol=? AND timeframe=? "
        "ORDER BY timestamp DESC LIMIT ?",
        (symbol, timeframe, n),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("datetime").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def cached_range(symbol: str, timeframe: str) -> tuple[Optional[int], Optional[int]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT MIN(timestamp) AS mn, MAX(timestamp) AS mx FROM ohlcv WHERE symbol=? AND timeframe=?",
        (symbol, timeframe),
    ).fetchone()
    if row is None or row["mn"] is None:
        return None, None
    return row["mn"], row["mx"]


def list_cached_series() -> pd.DataFrame:
    """Todas las series (símbolo, temporalidad) ya descargadas y disponibles
    para reutilizar sin volver a pedir nada a MT5."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT symbol, timeframe, COUNT(*) as n_bars,
               MIN(timestamp) as start_ts, MAX(timestamp) as end_ts
        FROM ohlcv GROUP BY symbol, timeframe ORDER BY symbol, timeframe
        """
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["symbol", "timeframe", "n_bars", "start", "end"])
    out = []
    for r in rows:
        out.append({
            "symbol": r["symbol"], "timeframe": r["timeframe"], "n_bars": r["n_bars"],
            "start": pd.to_datetime(r["start_ts"], unit="s", utc=True),
            "end": pd.to_datetime(r["end_ts"], unit="s", utc=True),
        })
    return pd.DataFrame(out)


def delete_series(symbol: str, timeframe: str) -> None:
    with transaction() as conn:
        conn.execute("DELETE FROM ohlcv WHERE symbol=? AND timeframe=?", (symbol, timeframe))


init_db()
